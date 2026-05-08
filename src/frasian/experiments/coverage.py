"""CoverageExperiment: empirical frequentist coverage on a (theta, w) grid.

**Normal-Normal-only by construction** (audit P1 K.4 / Path A). The
experiment hard-codes `model = NormalNormalModel(sigma=self.sigma)` and
`prior = NormalDistribution(loc=mu0, scale=sigma0_from_w(w))`. The
`(theta, w)` grid is well-defined only on the conjugate Gaussian
sandbox: `w = sigma0^2/(sigma^2 + sigma0^2)` is a Gaussian-only
quantity, and the data-generating distribution is `N(theta_true, sigma)`.
Bernoulli / Beta and other non-NN models do not enter this experiment;
extending requires a model-protocol-level abstraction over the
"prior axis" (out of scope today).

For each cell (TiltingScheme x TestStatistic) and each (theta_true, w):
  1. Generate `n_reps` samples D ~ N(theta_true, sigma).
  2. Compute the (1 - alpha) CI *regions* from each D via
     `tilting.confidence_regions(alpha, [D], model, prior, statistic)`.
  3. Coverage = fraction of replicates where `theta_true` lies in any
     of the returned regions (union semantics; for single-region cells
     this is the standard coverage check).

The uniform CI interface routes through the tilting: `IdentityTilting`
delegates to the bare statistic; `PowerLawTilting` resolves its own
selector (fixed-η static or dynamic-η per θ) before inverting the
tilted p-value. Multi-region union semantics replaces the prior
convex-hull check so that Dyn-WALDO (which can produce two regions
under conflict) is evaluated against the actual CI it constructs.

Conventions:
  - Canonical sigma = 1 (configurable via `Config.from_overrides`).
  - w in (0, 1) is varied directly; the implied prior is
    `N(mu0=0, sigma0 = sqrt(w/(1-w)) * sigma)`.
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
from ..diagnostics.coverage_table import CoverageRateDiagnostic
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..simulation.raw import generate_normal_D_samples
from ..statistics.base import TestStatistic
from ..tilting.base import TiltingScheme
from .base import ExperimentContext, RawResult


def _sigma0_from_w(w: float, sigma: float) -> float:
    """Inverse of the `weight` map at fixed sigma."""
    if not (0.0 < w < 1.0):
        raise ValueError(f"w must lie in (0, 1), got {w!r}")
    return float(np.sqrt(w / (1.0 - w)) * sigma)


def _coverage_replicate(
    D: float,
    *,
    alpha: float,
    model: Any,
    prior: Any,
    tilting: Any,
    statistic: Any,
    config: Config,
) -> list[tuple[float, float]] | None:
    """Compute one replicate's CI regions or return None if unsupported.

    Top-level so it pickles cleanly under joblib loky workers. Returns
    a list of `(lo, hi)` region pairs (single-element for non-multi-
    region cells) or `None` if the cell raised `NotImplementedError`
    (the inner-loop sentinel that triggers the bucket NaN row).
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
    return [(float(lo), float(hi)) for lo, hi in regions]


def _call_with_config(fn, *args: Any, config: Config, **kwargs: Any) -> Any:
    """Invoke ``fn`` passing ``config=config`` only if its signature accepts it.

    All in-tree TiltingScheme `confidence_regions` methods adopt the
    ``config`` kw-only arg so the Config-derived dynamic-CI scan
    parameters drive computation. Third-party plugin schemes that
    haven't yet adopted the kwarg are tolerated by introspection here:
    they fall back to the pre-fix selector-derived defaults. Skeptic
    Phase 5 vector #2.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if "config" in sig.parameters:
        kwargs["config"] = config
    return fn(*args, **kwargs)


@register_experiment(name="coverage", brief="docs/methods/coverage_experiment.md")
@dataclass(frozen=True)
class CoverageExperiment:
    """Frequentist coverage on a (theta_true, w) grid."""

    name: ClassVar[str] = "coverage"
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
            rng_seed=config.seed,
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
            name="coverage",
            model=model,
            theta_grid=theta_grid,
            n_reps=n_reps,
            rng=rng,
            seed=ctx.rng_seed,
        )

        n_theta = theta_grid.size
        n_w = w_grid.size
        # Audit P1 K.1: initialise with NaN, not `np.empty` (which
        # leaves uninitialised garbage). If a cell early-exits via an
        # unhandled exception, the (i, j) slot stays NaN — downstream
        # `np.nanmean` etc. then ignore it cleanly. Pre-fix the slot
        # would be filled with whatever was in memory at allocation.
        coverage = np.full((n_theta, n_w), np.nan, dtype=np.float64)
        coverage_se = np.full((n_theta, n_w), np.nan, dtype=np.float64)

        n_jobs = int(getattr(ctx.config, "n_jobs", 1))
        for j, w in enumerate(w_grid):
            sigma0 = _sigma0_from_w(float(w), self.sigma)
            prior = NormalDistribution(loc=self.mu0, scale=sigma0)
            for i in range(n_theta):
                theta_true = float(theta_grid[i])
                D_arr = raw.D[i, :]  # shape (n_reps,)

                # Dispatch the n_reps replicates via `parallel_map`. With
                # `n_jobs=1` (default) this is a plain list comp — byte-
                # identical to the legacy loop. With `n_jobs>1` joblib
                # spreads replicates across worker processes; the
                # `_coverage_replicate` helper at module scope pickles
                # cleanly. Per-replicate D values come from a pre-
                # generated raw stream so reordering evaluation does
                # not affect numerical output.
                fn = partial(
                    _coverage_replicate,
                    alpha=alpha,
                    model=model,
                    prior=prior,
                    tilting=tilting,
                    statistic=statistic,
                    config=ctx.config,
                )
                regions_per_rep = parallel_map(fn, list(D_arr), n_jobs=n_jobs)

                if any(r is None for r in regions_per_rep):
                    # Cell that does not support CI inversion: leave NaN.
                    continue

                hits = sum(
                    1
                    for regions in regions_per_rep
                    if any(lo <= theta_true <= hi for lo, hi in regions)
                )
                p = hits / n_reps
                coverage[i, j] = p
                # Audit P1 K.2: drop the 1e-12 SE floor. Pre-fix
                # `max(p*(1-p), 1e-12)` returned a fake non-zero SE
                # (~1e-7/sqrt(n_reps)) at p=0 / p=1, hiding the
                # honest "no MC variation observed" signal. The
                # downstream consumer (CoverageRateDiagnostic /
                # plotting) treats SE=0 as "skip error bar" and
                # SE>0 as "plot SE band"; the floor blurred this
                # boundary. Now SE is exactly 0 at extreme p,
                # which downstream can detect explicitly.
                coverage_se[i, j] = float(np.sqrt(p * (1.0 - p) / n_reps))

        cell_name = getattr(tilting, "cell_name", tilting.name)
        stat_cell_name = getattr(statistic, "cell_name", statistic.name)
        return RawResult(
            experiment=self.name,
            tilting=cell_name,
            statistic=stat_cell_name,
            arrays={
                "theta_grid": theta_grid,
                "w_grid": w_grid,
                "coverage": coverage,
                "coverage_se": coverage_se,
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
        return [CoverageRateDiagnostic()]
