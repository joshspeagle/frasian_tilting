"""WidthExperiment: mean CI width (union of regions) on a (theta_true, w) grid.

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
from typing import Any

import numpy as np

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
from .coverage import _sigma0_from_w


@register_experiment(name="width", brief="docs/methods/width_experiment.md")
@dataclass(frozen=True)
class WidthExperiment:
    """Mean CI width on a (theta_true, w) grid."""

    name: str = "width"
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
        mean_width = np.empty((n_theta, n_w), dtype=np.float64)
        width_se = np.empty_like(mean_width)
        mean_n_regions = np.empty_like(mean_width)

        for j, w in enumerate(w_grid):
            sigma0 = _sigma0_from_w(float(w), self.sigma)
            prior = NormalDistribution(loc=self.mu0, scale=sigma0)
            for i in range(n_theta):
                widths = np.empty(n_reps, dtype=np.float64)
                n_regions_arr = np.empty(n_reps, dtype=np.float64)
                supported = True
                for k in range(n_reps):
                    D = raw.D[i, k]
                    try:
                        regions = tilting.confidence_regions(
                            alpha,
                            np.asarray([D]),
                            model,
                            prior,
                            statistic,
                        )
                    except NotImplementedError:
                        supported = False
                        break
                    widths[k] = float(sum(hi - lo for lo, hi in regions))
                    n_regions_arr[k] = float(len(regions))
                if not supported:
                    mean_width[i, j] = np.nan
                    width_se[i, j] = np.nan
                    mean_n_regions[i, j] = np.nan
                else:
                    mean_width[i, j] = float(widths.mean())
                    width_se[i, j] = float(widths.std(ddof=1) / np.sqrt(n_reps))
                    mean_n_regions[i, j] = float(n_regions_arr.mean())

        cell_name = getattr(tilting, "cell_name", tilting.name)
        return RawResult(
            experiment=self.name,
            tilting=cell_name,
            statistic=statistic.name,
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
