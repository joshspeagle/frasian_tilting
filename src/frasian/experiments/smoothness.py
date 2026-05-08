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
from functools import partial
from typing import Any, ClassVar

import numpy as np

from .._parallel import parallel_map
from .._registry import register_experiment
from ..config import Config
from ..diagnostics.base import Diagnostic
from ..diagnostics.smoothness_metrics import SmoothnessDiagnostic
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..statistics.base import TestStatistic
from ..tilting.base import TiltingScheme
from ..tilting.eta_selectors import NumericalEtaSelector, _D_from_abs_delta
from .base import ExperimentContext, RawResult


def _smoothness_identity_point(
    abs_delta: float,
    *,
    alpha: float,
    w: float,
    sigma: float,
    mu0: float,
    model: Any,
    prior: Any,
    tilting: Any,
    statistic: Any,
) -> tuple[float, float, float] | None:
    """One |Δ|-point of smoothness's identity branch: (eta=0, lo, hi) or None.

    Top-level so it pickles cleanly for joblib workers.
    """
    D = _D_from_abs_delta(float(abs_delta), w, sigma, mu0)
    try:
        lo, hi = tilting.confidence_interval(
            alpha, np.asarray([D]), model, prior, statistic
        )
    except (NotImplementedError, ValueError, RuntimeError):
        return None
    return (0.0, float(lo), float(hi))


def _smoothness_tilted_point(
    abs_delta: float,
    *,
    alpha: float,
    w: float,
    sigma: float,
    mu0: float,
    model: Any,
    prior: Any,
    tilting: Any,
    statistic: Any,
) -> tuple[float, float, float] | None:
    """One |Δ|-point of smoothness's non-identity tilted branch.

    Constructs a fresh `NumericalEtaSelector` per call (cheap, frozen
    dataclass) so workers don't share state. Returns (eta_star, lo, hi)
    or None if the tilted bridge is unavailable / fails.
    """
    D = _D_from_abs_delta(float(abs_delta), w, sigma, mu0)
    try:
        selector = NumericalEtaSelector()
        eta = selector.select(
            tilting,
            data=np.asarray([D]),
            model=model,
            prior=prior,
            alpha=alpha,
            statistic=statistic,
        )
        lo, hi = tilting.tilted_confidence_interval(
            alpha, D, model, prior, eta, statistic.name
        )
    except (NotImplementedError, ValueError, RuntimeError):
        return None
    return (float(eta), float(lo), float(hi))


@register_experiment(name="smoothness", brief="docs/methods/smoothness_experiment.md")
@dataclass(frozen=True)
class SmoothnessExperiment:
    """Sweep |Δ| at fixed (w, α, σ) and record η*(|Δ|) plus CI endpoints."""

    name: ClassVar[str] = "smoothness"
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
        stat_cell_name = getattr(statistic, "cell_name", statistic.name)
        sigma0 = float(np.sqrt(self.w / max(1.0 - self.w, 1e-12)) * self.sigma)
        prior = NormalDistribution(loc=self.mu0, scale=sigma0)
        model = NormalNormalModel(sigma=self.sigma)

        n_jobs = int(getattr(ctx.config, "n_jobs", 1))
        # IdentityTilting has nothing to sweep — record the bare-statistic
        # CI at each |Δ| as the smoothness reference (constant η = 0).
        if tilting.name == "identity":
            fn = partial(
                _smoothness_identity_point,
                alpha=alpha,
                w=self.w,
                sigma=self.sigma,
                mu0=self.mu0,
                model=model,
                prior=prior,
                tilting=tilting,
                statistic=statistic,
            )
            results = parallel_map(fn, list(delta_grid), n_jobs=n_jobs)
            for i, r in enumerate(results):
                if r is None:
                    continue
                eta_star[i] = r[0]
                ci_lo[i] = r[1]
                ci_hi[i] = r[2]
            return RawResult(
                experiment=self.name,
                tilting=cell_name,
                statistic=stat_cell_name,
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

        # `statistic.force_generic=True` for a non-identity tilting:
        # smoothness sweeps η via the closed-form `tilted_confidence_interval`
        # bridge (Theorem 6/8 — NN+Normal only). There is currently no
        # generic per-η bridge for the smoothness experiment, so refuse
        # the cell with a clear unsupported row rather than silently
        # falling back to the closed form. The bare-statistic dispatch
        # (`(identity, X[generic])`) still works because the identity
        # branch above delegates straight to `statistic.confidence_interval`.
        if getattr(statistic, "force_generic", False):
            return RawResult(
                experiment=self.name,
                tilting=cell_name,
                statistic=stat_cell_name,
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
                    "reason": (
                        "smoothness with non-identity tilting + "
                        "statistic.force_generic=True is not supported: "
                        "the η-sweep bridge `tilted_confidence_interval` "
                        "currently has no generic per-η counterpart. "
                        "Run with the bare statistic + `(identity, X[generic])` "
                        "to exercise the generic path."
                    ),
                },
            )

        # Detect whether this cell supports the tilted-CI bridge. If not,
        # we record NaNs — the diagnostic preserves them.
        if not hasattr(tilting, "tilted_confidence_interval"):
            return RawResult(
                experiment=self.name,
                tilting=cell_name,
                statistic=stat_cell_name,
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

        # Non-identity tilted branch: per-|Δ| select η via NumericalEtaSelector,
        # then compute tilted_confidence_interval. Each |Δ| is independent;
        # fan out via `parallel_map` for consistency with the other 3
        # experiments. Cells whose tilted bridge raises (e.g. future LRT)
        # leave NaN per |Δ|.
        fn = partial(
            _smoothness_tilted_point,
            alpha=alpha,
            w=self.w,
            sigma=self.sigma,
            mu0=self.mu0,
            model=model,
            prior=prior,
            tilting=tilting,
            statistic=statistic,
        )
        results = parallel_map(fn, list(delta_grid), n_jobs=n_jobs)
        for i, r in enumerate(results):
            if r is None:
                continue
            eta_star[i] = r[0]
            ci_lo[i] = r[1]
            ci_hi[i] = r[2]

        return RawResult(
            experiment=self.name,
            tilting=cell_name,
            statistic=stat_cell_name,
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
