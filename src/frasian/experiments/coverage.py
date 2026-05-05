"""CoverageExperiment: empirical frequentist coverage on a (theta, w) grid.

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
from typing import Any, ClassVar

import numpy as np

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
        coverage = np.empty((n_theta, n_w), dtype=np.float64)
        coverage_se = np.empty_like(coverage)

        for j, w in enumerate(w_grid):
            sigma0 = _sigma0_from_w(float(w), self.sigma)
            prior = NormalDistribution(loc=self.mu0, scale=sigma0)
            for i in range(n_theta):
                hits = 0
                theta_true = float(theta_grid[i])
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
                        # Cell that does not support CI inversion: record NaN.
                        coverage[i, j] = np.nan
                        coverage_se[i, j] = np.nan
                        break
                    if any(lo <= theta_true <= hi for lo, hi in regions):
                        hits += 1
                else:
                    p = hits / n_reps
                    coverage[i, j] = p
                    # Wald-binomial SE; clipped to avoid 0 at extremes.
                    coverage_se[i, j] = float(np.sqrt(max(p * (1.0 - p), 1e-12) / n_reps))

        cell_name = getattr(tilting, "cell_name", tilting.name)
        return RawResult(
            experiment=self.name,
            tilting=cell_name,
            statistic=statistic.name,
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
