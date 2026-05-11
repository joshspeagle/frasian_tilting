"""WidthExperiment: mean CI width (union of regions) on a (theta_true, w) grid.

**Normal-Normal-only by construction** (audit P1 K.4 / Path A). The
experiment hard-codes `model = NormalNormalModel(sigma=self.sigma)` and
the `(theta, w)` grid maps to a Gaussian prior; same caveat as
`CoverageExperiment`. Non-NN models do not enter this experiment.

For each cell (TiltingScheme x TestStatistic) and each (theta_true, w):
  1. Generate `n_reps` samples D ~ N(theta_true, sigma).
  2. Compute the (1-α) CI regions via
     `tilting.confidence_regions(alpha, [D], model, prior, statistic)`
     and record `sum(hi - lo for lo, hi in regions)` (union width).
  3. Mean width = average; SE = sample standard error. Also record
     mean region count per cell (≥ 1; > 1 indicates multimodal-p
     regimes such as Dyn-WALDO under conflict).

This is the "Dynamic-WALDO width" measurement when run on
`(power_law[dynamic_numerical], waldo)`. Union semantics replace the
prior convex-hull width to honour the actual CI structure produced by
multimodal p-values; for single-region cells the two coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar

import numpy as np

from .._parallel import parallel_map
from .._registry import register_experiment
from ..config import Config
from ..diagnostics.base import Diagnostic
from ..diagnostics.width_table import MeanWidthDiagnostic
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..simulation.raw import generate_normal_D_samples
from ..statistics.base import TestStatistic
from ..tilting.base import TiltingScheme
from .base import ExperimentContext, RawResult
from .coverage import _call_with_config, _sigma0_from_w


def _width_replicate(
    D: float,
    *,
    alpha: float,
    model: Any,
    prior: Any,
    tilting: Any,
    statistic: Any,
    config: Config,
) -> tuple[float, int] | None:
    """Compute one replicate's (union_width, n_regions) or None if unsupported.

    Top-level for joblib pickling. The caller aggregates `width_se` and
    `mean_n_regions` from the returned tuples.
    """
    try:
        regions = _call_with_config(
            tilting.confidence_regions,
            alpha,
            np.asarray([D]),
            model,
            prior,
            statistic,
            config=config,
        )
    except NotImplementedError:
        return None
    width = float(sum(hi - lo for lo, hi in regions))
    return (width, int(len(regions)))


@register_experiment(name="width", brief="docs/methods/width_experiment.md")
@dataclass(frozen=True)
class WidthExperiment:
    """Mean CI width on a (theta_true, w) grid."""

    name: ClassVar[str] = "width"
    sigma: float = 1.0
    mu0: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def setup(self, config: Config) -> ExperimentContext:
        return ExperimentContext(
            config=config,
            grid={
                "theta_grid": config.theta_grid.to_array(),
                "w_grid": config.w_grid.to_array(),
            },
            rng_seed=config.seed + 1,  # different RNG stream from coverage
            metadata={
                "sigma": self.sigma,
                "mu0": self.mu0,
                "alpha": config.alpha,
                "n_reps": config.n_reps,
            },
        )

    def run_cell(
        self, ctx: ExperimentContext, tilting: TiltingScheme, statistic: TestStatistic
    ) -> RawResult:
        theta_grid = ctx.grid["theta_grid"]
        w_grid = ctx.grid["w_grid"]
        alpha = ctx.config.alpha
        n_reps = ctx.config.n_reps

        model = NormalNormalModel(sigma=self.sigma)
        rng = np.random.default_rng(ctx.rng_seed)
        raw = generate_normal_D_samples(
            name="width",
            model=model,
            theta_grid=theta_grid,
            n_reps=n_reps,
            rng=rng,
            seed=ctx.rng_seed,
        )

        n_theta = theta_grid.size
        n_w = w_grid.size
        # Audit P1 K.1: initialise with NaN (was `np.empty`, which
        # leaves uninitialised garbage on cells that early-exit).
        # `np.nanmean` downstream is then unbiased.
        mean_width = np.full((n_theta, n_w), np.nan, dtype=np.float64)
        width_se = np.full((n_theta, n_w), np.nan, dtype=np.float64)
        mean_n_regions = np.full((n_theta, n_w), np.nan, dtype=np.float64)

        n_jobs = int(getattr(ctx.config, "n_jobs", 1))
        for j, w in enumerate(w_grid):
            sigma0 = _sigma0_from_w(float(w), self.sigma)
            prior = NormalDistribution(loc=self.mu0, scale=sigma0)
            for i in range(n_theta):
                D_arr = raw.D[i, :]
                fn = partial(
                    _width_replicate,
                    alpha=alpha,
                    model=model,
                    prior=prior,
                    tilting=tilting,
                    statistic=statistic,
                    config=ctx.config,
                )
                # Same `parallel_map` discipline as coverage. Per-rep
                # tuples come back as `(width, n_regions)` or None for
                # unsupported cells.
                results = parallel_map(fn, list(D_arr), n_jobs=n_jobs)
                if any(r is None for r in results):
                    continue  # unsupported → leave NaN row

                widths = np.array([w_ for (w_, _) in results], dtype=np.float64)
                n_regions_arr = np.array([n for (_, n) in results], dtype=np.float64)
                # Audit P1 K.1: use `np.nanmean` / `np.nanstd` to be
                # robust against per-rep failures that fill `widths[k]`
                # with NaN (e.g. a single brentq bracket failure
                # mid-loop that we want to skip rather than abort).
                mean_width[i, j] = float(np.nanmean(widths))
                if np.sum(~np.isnan(widths)) >= 2:
                    width_se[i, j] = float(
                        np.nanstd(widths, ddof=1) / np.sqrt(n_reps)
                    )
                else:
                    width_se[i, j] = np.nan
                mean_n_regions[i, j] = float(np.nanmean(n_regions_arr))

        cell_name = getattr(tilting, "cell_name", tilting.name)
        stat_cell_name = getattr(statistic, "cell_name", statistic.name)
        return RawResult(
            experiment=self.name,
            tilting=cell_name,
            statistic=stat_cell_name,
            arrays={
                "theta_grid": theta_grid,
                "w_grid": w_grid,
                "mean_width": mean_width,
                "width_se": width_se,
                "mean_n_regions": mean_n_regions,
            },
            metadata={
                "raw_fingerprint": raw.fingerprint(),
                "alpha": alpha,
                "n_reps": n_reps,
                "sigma": self.sigma,
                "mu0": self.mu0,
                "selector": getattr(getattr(tilting, "selector", None), "name", None),
            },
        )

    def diagnostics(self) -> list[Diagnostic]:
        return [MeanWidthDiagnostic()]
