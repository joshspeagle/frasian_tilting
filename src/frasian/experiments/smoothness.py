"""SmoothnessExperiment: quantify η*(|Δ|) regularity.

For each cell (TiltingScheme x TestStatistic) and each |Δ| on a fine grid:
  1. Pick η* via NumericalEtaSelector (minimizes tilted CI width).
  2. Record the resulting CI endpoints (L, U) and width.

Outputs: arrays(abs_delta_grid, eta_star, ci_lower, ci_upper) per cell.

For tiltings with no parameter to sweep (e.g. `IdentityTilting`), the
diagnostic records a degenerate constant row — the bare-statistic CI at
each |Δ| — for use as a smoothness reference baseline.

Downstream `SmoothnessDiagnostic` computes Lipschitz, TV, discontinuity
count, and spectral roughness — making the user's "power-law tilting
produces sharp transitions" hypothesis falsifiable. Cells whose
(tilting, statistic) pair lacks a tilted-CI bridge produce NaN rows;
the diagnostic preserves them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._registry import register_experiment
from ..config import Config
from ..diagnostics.base import Diagnostic
from ..diagnostics.smoothness_metrics import SmoothnessDiagnostic
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..statistics.base import TestStatistic
from ..tilting.base import TiltingContext, TiltingScheme
from ..tilting.eta_selectors import NumericalEtaSelector, _D_from_abs_delta
from .base import ExperimentContext, RawResult


@register_experiment(name="smoothness", brief="docs/methods/smoothness_experiment.md")
@dataclass(frozen=True)
class SmoothnessExperiment:
    """Sweep |Δ| at fixed (w, α, σ) and record η*(|Δ|) plus CI endpoints."""

    name: str = "smoothness"
    sigma: float = 1.0
    mu0: float = 0.0
    w: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def setup(self, config: Config) -> ExperimentContext:
        # |Δ| grid lives at config.delta_grid; default 11 points 0..5 in fast,
        # 51 points 0..5 in default. The publication-grade figure uses 401
        # points, which is what `--default` (no --fast) gives via Config.default.
        return ExperimentContext(
            config=config,
            grid={"abs_delta_grid": config.delta_grid.to_array()},
            rng_seed=config.seed,
            metadata={"w": self.w, "sigma": self.sigma, "mu0": self.mu0, "alpha": config.alpha},
        )

    def run_cell(
        self, ctx: ExperimentContext, tilting: TiltingScheme, statistic: TestStatistic
    ) -> RawResult:
        delta_grid = ctx.grid["abs_delta_grid"]
        alpha = ctx.config.alpha
        n = delta_grid.size

        eta_star = np.full(n, np.nan)
        ci_lo = np.full(n, np.nan)
        ci_hi = np.full(n, np.nan)

        cell_name = getattr(tilting, "cell_name", tilting.name)
        sigma0 = float(np.sqrt(self.w / max(1.0 - self.w, 1e-12)) * self.sigma)
        prior = NormalDistribution(loc=self.mu0, scale=sigma0)
        model = NormalNormalModel(sigma=self.sigma)

        # IdentityTilting has nothing to sweep — record the bare-statistic
        # CI at each |Δ| as the smoothness reference (constant η = 0).
        if tilting.name == "identity":
            for i, abs_delta in enumerate(delta_grid):
                D = _D_from_abs_delta(float(abs_delta), self.w, self.sigma, self.mu0)
                try:
                    lo, hi = tilting.confidence_interval(
                        alpha,
                        np.asarray([D]),
                        model,
                        prior,
                        statistic,
                    )
                    eta_star[i] = 0.0
                    ci_lo[i] = lo
                    ci_hi[i] = hi
                except (NotImplementedError, ValueError, RuntimeError):
                    continue
            return RawResult(
                experiment=self.name,
                tilting=cell_name,
                statistic=statistic.name,
                arrays={
                    "abs_delta_grid": delta_grid,
                    "eta_star": eta_star,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                },
                metadata={
                    "w": self.w,
                    "alpha": alpha,
                    "sigma": self.sigma,
                    "mu0": self.mu0,
                    "supported": True,
                    "selector": "identity",
                    "note": "identity tilting: η fixed at 0; row is the "
                    "bare-statistic CI baseline.",
                },
            )

        # Detect whether this cell supports the tilted-CI bridge. If not,
        # we record NaNs — the diagnostic preserves them.
        if not hasattr(tilting, "tilted_confidence_interval"):
            return RawResult(
                experiment=self.name,
                tilting=cell_name,
                statistic=statistic.name,
                arrays={
                    "abs_delta_grid": delta_grid,
                    "eta_star": eta_star,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                },
                metadata={
                    "w": self.w,
                    "alpha": alpha,
                    "supported": False,
                    "reason": "tilting scheme lacks tilted_confidence_interval",
                },
            )

        selector = NumericalEtaSelector(sigma=self.sigma, mu0=self.mu0)

        for i, abs_delta in enumerate(delta_grid):
            ctx_i = TiltingContext(w=self.w, abs_delta=float(abs_delta), alpha=alpha)
            try:
                eta = selector.select(ctx_i, tilting, statistic=statistic)
                D = _D_from_abs_delta(float(abs_delta), self.w, self.sigma, self.mu0)
                lo, hi = tilting.tilted_confidence_interval(
                    alpha,
                    D,
                    model,
                    prior,
                    eta,
                    statistic.name,
                )
                eta_star[i] = eta
                ci_lo[i] = lo
                ci_hi[i] = hi
            except (NotImplementedError, ValueError, RuntimeError):
                # Cells whose statistic lacks the tilted bridge (e.g. future
                # LRT) leave NaNs in this row.
                continue

        return RawResult(
            experiment=self.name,
            tilting=cell_name,
            statistic=statistic.name,
            arrays={
                "abs_delta_grid": delta_grid,
                "eta_star": eta_star,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            },
            metadata={
                "w": self.w,
                "alpha": alpha,
                "sigma": self.sigma,
                "mu0": self.mu0,
                "supported": True,
                "selector": getattr(getattr(tilting, "selector", None), "name", None),
            },
        )

    def diagnostics(self) -> list[Diagnostic]:
        return [SmoothnessDiagnostic()]
