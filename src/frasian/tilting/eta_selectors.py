"""η-selectors: strategies for picking η given a tilting context.

The legacy code conflated three solvers (numerical search, closed-form
approximation, learned MLP) inside `tilting.py`. Here they live behind
the `EtaSelector` protocol and can be swapped via configuration.

Step 5 ships only `NumericalEtaSelector`. The closed-form approximation
and the learned-MLP variants land in a later step alongside their tests
and briefs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from ..config import Config
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..statistics.base import TestStatistic
from .base import TiltingContext, TiltingScheme


def _D_from_abs_delta(abs_delta: float, w: float, sigma: float, mu0: float
                      ) -> float:
    """Invert Delta = (1 - w)(mu0 - D)/sigma with the convention Delta >= 0."""
    return float(mu0 - abs_delta * sigma / max(1.0 - w, 1e-12))


@dataclass(frozen=True)
class _NamedStatistic:
    """Minimal `.name`-only adapter for selector calls that only need dispatch.

    `dynamic_tilted_confidence_interval` knows the statistic by name (the
    cell-evaluator dispatch in PowerLawTilting is keyed on `statistic.name`)
    and does not have a concrete `TestStatistic` instance handy. Wrapping
    the name in this adapter satisfies the selector's type contract without
    pulling in the registry.
    """
    name: str


@dataclass(frozen=True)
class NumericalEtaSelector:
    """Pick η minimizing the (analytic) CI width for the tilted posterior.

    The width is computed by inverting the tilted p-value at a single
    representative `D` (no Monte-Carlo). For the (power_law, waldo) cell
    this matches the legacy `optimal_eta_numerical` solver up to bracket
    bookkeeping.
    """

    name: str = "numerical"
    sigma: float = 1.0
    mu0: float = 0.0

    def select(self, context: TiltingContext, scheme: TiltingScheme,
               *, statistic: TestStatistic) -> float:
        eta_lo, eta_hi = scheme.admissible_range(context)
        # Cap the upper end at +1 (the Wald limit); anything beyond is
        # mathematically valid but uninteresting in practice.
        eta_hi = min(eta_hi, 1.0 - Config.default().eta_min_buffer)

        sigma0 = float(np.sqrt(context.w / max(1.0 - context.w, 1e-12))
                        * self.sigma)
        prior = NormalDistribution(loc=self.mu0, scale=sigma0)
        model = NormalNormalModel(sigma=self.sigma)
        D = _D_from_abs_delta(context.abs_delta, context.w, self.sigma,
                                self.mu0)

        def width(eta: float) -> float:
            try:
                if hasattr(scheme, "tilted_confidence_interval"):
                    lo, hi = scheme.tilted_confidence_interval(
                        context.alpha, D, model, prior, eta,
                        statistic.name,
                    )
                else:
                    raise NotImplementedError(
                        f"{type(scheme).__name__} does not implement "
                        f"`tilted_confidence_interval` (Step 5 bridge)."
                    )
            except Exception:
                return np.inf
            w_ci = float(hi - lo)
            return w_ci if (w_ci > 0 and np.isfinite(w_ci)) else np.inf

        result = optimize.minimize_scalar(
            width, bounds=(eta_lo, eta_hi), method="bounded",
            options={"xatol": 1e-3},
        )
        return float(result.x)

    def select_grid(self, abs_delta_grid, scheme: TiltingScheme,
                    *, statistic: TestStatistic, w: float, alpha: float
                    ):
        """Vectorised: η* at every |Δ| in `abs_delta_grid`.

        Used by `dynamic_tilted_confidence_interval` to pre-compute a coarse
        η*(|Δ|) lookup that is then interpolated across the fine θ-scan grid.
        Repeating `optimize.minimize_scalar` for every fine-grid θ would be
        prohibitively slow.
        """
        out = np.empty(len(abs_delta_grid), dtype=np.float64)
        for i, ad in enumerate(abs_delta_grid):
            ctx = TiltingContext(w=w, abs_delta=float(ad), alpha=alpha)
            out[i] = self.select(ctx, scheme, statistic=statistic)
        return out
