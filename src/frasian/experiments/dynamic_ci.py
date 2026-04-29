"""DynamicCIExperiment: empirical behavior of dynamic-eta CIs.

For each cell (TiltingScheme x TestStatistic) and each (theta_true, w):
  1. Generate `n_reps` samples D ~ N(theta_true, sigma).
  2. For each D, compute the *dynamic* CI via the tilting scheme's
     `dynamic_tilted_confidence_interval` (eta = eta*(|Delta(theta)|)
     varying per theta, found by NumericalEtaSelector then interpolated).
  3. Record three quantities per cell on the (theta_true, w) grid:
     - empirical coverage (fraction of dynamic CIs containing theta_true)
     - mean total width (sum of region widths, averaged over D samples)
     - mean region count (the dynamic p-value can be multimodal)

The legacy `dynamic_tilted_ci` (`legacy/src/frasian/tilting.py:508-611`)
is the spec; this experiment makes its behavior measurable on the
framework's standard diagnostic surface so future tilting families can
be compared against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._registry import register_experiment
from ..config import Config
from ..diagnostics.base import Diagnostic
from ..diagnostics.dynamic_ci_table import DynamicCITableDiagnostic
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..simulation.raw import generate_normal_D_samples
from ..statistics.base import TestStatistic
from ..tilting.base import TiltingScheme
from ..tilting.eta_selectors import NumericalEtaSelector
from .base import ExperimentContext, RawResult
from .coverage import _sigma0_from_w


@register_experiment(name="dynamic_ci",
                      brief="docs/methods/dynamic_ci_experiment.md")
@dataclass(frozen=True)
class DynamicCIExperiment:
    """Coverage / width / region-count of dynamic-η CIs on a (θ_true, w) grid."""

    name: str = "dynamic_ci"
    sigma: float = 1.0
    mu0: float = 0.0
    # Resolution knobs for the inner CI scan; smaller = faster, less accurate.
    n_grid: int = 201
    coarse_n: int = 15
    metadata: dict[str, Any] = field(default_factory=dict)

    def setup(self, config: Config) -> ExperimentContext:
        return ExperimentContext(
            config=config,
            grid={
                "theta_grid": config.theta_grid.to_array(),
                "w_grid": config.w_grid.to_array(),
            },
            rng_seed=config.seed + 2,  # different stream from coverage/width
            metadata={
                "sigma": self.sigma, "mu0": self.mu0,
                "alpha": config.alpha, "n_reps": config.n_reps,
                "n_grid": self.n_grid, "coarse_n": self.coarse_n,
            },
        )

    def run_cell(self, ctx: ExperimentContext, tilting: TiltingScheme,
                 statistic: TestStatistic) -> RawResult:
        theta_grid = ctx.grid["theta_grid"]
        w_grid = ctx.grid["w_grid"]
        alpha = ctx.config.alpha
        n_reps = ctx.config.n_reps

        n_theta = theta_grid.size
        n_w = w_grid.size
        coverage = np.full((n_theta, n_w), np.nan)
        coverage_se = np.full_like(coverage, np.nan)
        mean_width = np.full_like(coverage, np.nan)
        width_se = np.full_like(coverage, np.nan)
        mean_regions = np.full_like(coverage, np.nan)

        # Cells whose tilting lacks the dynamic bridge (every stub) produce
        # all-NaN; the diagnostic preserves these.
        if not hasattr(tilting, "dynamic_tilted_confidence_interval"):
            return RawResult(
                experiment=self.name,
                tilting=tilting.name,
                statistic=statistic.name,
                arrays={
                    "theta_grid": theta_grid, "w_grid": w_grid,
                    "coverage": coverage, "coverage_se": coverage_se,
                    "mean_width": mean_width, "width_se": width_se,
                    "mean_regions": mean_regions,
                },
                metadata={"alpha": alpha, "supported": False,
                          "reason": "tilting lacks dynamic_tilted_confidence_interval"},
            )

        model = NormalNormalModel(sigma=self.sigma)
        rng = np.random.default_rng(ctx.rng_seed)
        raw = generate_normal_D_samples(
            name="dynamic_ci", model=model, theta_grid=theta_grid,
            n_reps=n_reps, rng=rng, seed=ctx.rng_seed,
        )
        selector = NumericalEtaSelector(sigma=self.sigma, mu0=self.mu0)

        for j, w in enumerate(w_grid):
            sigma0 = _sigma0_from_w(float(w), self.sigma)
            prior = NormalDistribution(loc=self.mu0, scale=sigma0)
            for i in range(n_theta):
                hits = 0
                widths = np.empty(n_reps, dtype=np.float64)
                regions_count = np.empty(n_reps, dtype=np.float64)
                supported = True
                for k in range(n_reps):
                    D = float(raw.D[i, k])
                    try:
                        regions, total, n_reg = (
                            tilting.dynamic_tilted_confidence_interval(
                                alpha, D, model, prior, statistic.name,
                                selector,
                                n_grid=self.n_grid, coarse_n=self.coarse_n,
                            )
                        )
                    except (NotImplementedError, ValueError, RuntimeError):
                        supported = False
                        break
                    widths[k] = total
                    regions_count[k] = n_reg
                    if any(lo <= float(theta_grid[i]) <= hi
                            for lo, hi in regions):
                        hits += 1
                if not supported:
                    continue
                p = hits / n_reps
                coverage[i, j] = p
                coverage_se[i, j] = float(
                    np.sqrt(max(p * (1.0 - p), 1e-12) / n_reps)
                )
                mean_width[i, j] = float(widths.mean())
                width_se[i, j] = float(widths.std(ddof=1) / np.sqrt(n_reps))
                mean_regions[i, j] = float(regions_count.mean())

        return RawResult(
            experiment=self.name,
            tilting=tilting.name,
            statistic=statistic.name,
            arrays={
                "theta_grid": theta_grid,
                "w_grid": w_grid,
                "coverage": coverage,
                "coverage_se": coverage_se,
                "mean_width": mean_width,
                "width_se": width_se,
                "mean_regions": mean_regions,
            },
            metadata={
                "raw_fingerprint": raw.fingerprint(),
                "alpha": alpha,
                "n_reps": n_reps,
                "sigma": self.sigma,
                "mu0": self.mu0,
                "n_grid": self.n_grid,
                "coarse_n": self.coarse_n,
                "supported": True,
            },
        )

    def diagnostics(self) -> list[Diagnostic]:
        return [DynamicCITableDiagnostic()]
