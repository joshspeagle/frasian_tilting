"""ConfidenceDistributionExperiment: distributional analogue of coverage/width.

For each cell `(TiltingScheme x TestStatistic)` and each `(theta_true, w)`
on the standard grid:
  1. Draw `n_reps` data points D ~ N(theta_true, sigma).
  2. For each D:
     - Build the cell's CD via `cd.build_cd_from_pvalue(...)`.
     - Build the reference Wald CD via `cd.from_closed_form.wald_cd(D, sigma)`.
     - Record per-replicate summaries:
       - `cd_median(D)` — the CD's 0.5-quantile
       - `cd_width_95(D)` — the CD's 95% interval width
       - `w1_to_wald_cd(D)` — W₁ between this cell's CD and Wald CD
       - `nonmonotone(D)` — 1 if `signed_confidence` is non-monotone
  3. Aggregate over `n_reps` → per-cell `(theta_true, w)` arrays of
     means + standard errors.

Output `RawResult.arrays`, all shape `(n_theta, n_w)`:
  - `cd_median`, `cd_median_se`
  - `cd_width_95`, `cd_width_95_se`
  - `w1_to_wald_cd`, `w1_to_wald_cd_se`
  - `nonmonotone_fraction`

The diagnostic produces 4 heatmaps per cell: median, 95-width, W₁-to-Wald,
non-monotone fraction.

References
----------
Schweder, T. & Hjort, N. L. (2016). *Confidence, Likelihood, Probability*.
Cambridge UP. Ch. 4 (confidence curves) and Ch. 9 (worked examples).

Singh, K., Xie, M. & Strawderman, W. E. (2007). "Confidence
distribution (CD) — distribution estimator of a parameter." *IMS
Lecture Notes — Monograph Series* 54: 132–150.

Olkin, I. & Pukelsheim, F. (1982). "The distance between two random
vectors with given dispersion matrices." *Linear Algebra and its
Applications* 48: 257–263.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .._registry import register_experiment
from ..cd.distances import wasserstein_1
from ..cd.from_closed_form import wald_cd
from ..cd.from_pvalue import build_cd_from_pvalue
from ..config import Config
from ..diagnostics.base import Diagnostic
from ..diagnostics.cd_summary import CDSummaryDiagnostic
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..simulation.raw import generate_normal_D_samples
from ..statistics.base import TestStatistic
from ..tilting.base import TiltingScheme
from .base import ExperimentContext, RawResult
from .coverage import _sigma0_from_w


@register_experiment(
    name="confidence_distribution", brief="docs/methods/confidence_distribution_experiment.md"
)
@dataclass(frozen=True)
class ConfidenceDistributionExperiment:
    """Per-cell distributional summaries on a (theta_true, w) grid."""

    name: ClassVar[str] = "confidence_distribution"
    sigma: float = 1.0
    mu0: float = 0.0
    n_grid_cd: int = 401
    half_width_sigma_cd: float = 8.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def setup(self, config: Config) -> ExperimentContext:
        return ExperimentContext(
            config=config,
            grid={
                "theta_grid": config.theta_grid.to_array(),
                "w_grid": config.w_grid.to_array(),
            },
            rng_seed=config.seed + 3,  # different stream from coverage/width/smoothness
            metadata={
                "sigma": self.sigma,
                "mu0": self.mu0,
                "alpha": config.alpha,
                "n_reps": config.n_reps,
                "n_grid_cd": self.n_grid_cd,
                "half_width_sigma_cd": self.half_width_sigma_cd,
            },
        )

    def run_cell(
        self, ctx: ExperimentContext, tilting: TiltingScheme, statistic: TestStatistic
    ) -> RawResult:
        theta_grid = ctx.grid["theta_grid"]
        w_grid = ctx.grid["w_grid"]
        n_reps = ctx.config.n_reps

        model = NormalNormalModel(sigma=self.sigma)
        rng = np.random.default_rng(ctx.rng_seed)
        raw = generate_normal_D_samples(
            name="confidence_distribution",
            model=model,
            theta_grid=theta_grid,
            n_reps=n_reps,
            rng=rng,
            seed=ctx.rng_seed,
        )

        n_theta = theta_grid.size
        n_w = w_grid.size
        cd_median = np.full((n_theta, n_w), np.nan)
        cd_median_se = np.full_like(cd_median, np.nan)
        cd_width_95 = np.full_like(cd_median, np.nan)
        cd_width_95_se = np.full_like(cd_median, np.nan)
        w1_to_wald_cd = np.full_like(cd_median, np.nan)
        w1_to_wald_cd_se = np.full_like(cd_median, np.nan)
        nonmonotone_fraction = np.full_like(cd_median, np.nan)

        for j, w in enumerate(w_grid):
            sigma0 = _sigma0_from_w(float(w), self.sigma)
            prior = NormalDistribution(loc=self.mu0, scale=sigma0)

            for i in range(n_theta):
                medians = np.empty(n_reps, dtype=np.float64)
                widths = np.empty(n_reps, dtype=np.float64)
                w1s = np.empty(n_reps, dtype=np.float64)
                nonmono = np.zeros(n_reps, dtype=np.float64)
                supported = True
                for k in range(n_reps):
                    D = float(raw.D[i, k])
                    try:
                        cd = build_cd_from_pvalue(
                            tilting,
                            statistic,
                            D,
                            model,
                            prior,
                            n_grid=self.n_grid_cd,
                            half_width_sigma=self.half_width_sigma_cd,
                        )
                        wald_ref = wald_cd(
                            D,
                            self.sigma,
                            theta_grid=cd.theta_grid,
                        )
                    except (NotImplementedError, ValueError, RuntimeError):
                        supported = False
                        break
                    medians[k] = cd.median()
                    lo, hi = cd.interval(0.05)
                    widths[k] = hi - lo
                    w1s[k] = wasserstein_1(cd, wald_ref)
                    nonmono[k] = 0.0 if cd.is_monotone_inversion() else 1.0

                if not supported:
                    continue
                cd_median[i, j] = medians.mean()
                cd_median_se[i, j] = float(medians.std(ddof=1) / np.sqrt(n_reps))
                cd_width_95[i, j] = widths.mean()
                cd_width_95_se[i, j] = float(widths.std(ddof=1) / np.sqrt(n_reps))
                w1_to_wald_cd[i, j] = w1s.mean()
                w1_to_wald_cd_se[i, j] = float(w1s.std(ddof=1) / np.sqrt(n_reps))
                nonmonotone_fraction[i, j] = nonmono.mean()

        cell_name = getattr(tilting, "cell_name", tilting.name)
        return RawResult(
            experiment=self.name,
            tilting=cell_name,
            statistic=statistic.name,
            arrays={
                "theta_grid": theta_grid,
                "w_grid": w_grid,
                "cd_median": cd_median,
                "cd_median_se": cd_median_se,
                "cd_width_95": cd_width_95,
                "cd_width_95_se": cd_width_95_se,
                "w1_to_wald_cd": w1_to_wald_cd,
                "w1_to_wald_cd_se": w1_to_wald_cd_se,
                "nonmonotone_fraction": nonmonotone_fraction,
            },
            metadata={
                "raw_fingerprint": raw.fingerprint(),
                "alpha": ctx.config.alpha,
                "n_reps": n_reps,
                "sigma": self.sigma,
                "mu0": self.mu0,
                "n_grid_cd": self.n_grid_cd,
                "selector": getattr(getattr(tilting, "selector", None), "name", None),
            },
        )

    def diagnostics(self) -> list[Diagnostic]:
        return [CDSummaryDiagnostic()]
