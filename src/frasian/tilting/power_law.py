"""Power-law tilting (the legacy eta-tilting, ported into the new shape).

  q(theta; eta) ∝ L(theta) * pi(theta)^(1 - eta)

For the conjugate Normal-Normal model this admits the closed form (Theorem 6
in the legacy derivations):

  denom    = 1 - eta * (1 - w)
  mu_eta   = (w*D + (1-eta)*(1-w)*mu0) / denom
  sigma_eta^2 = w * sigma^2 / denom
  w_eta    = w / denom

Identity element is `eta = 0` (recovers the WALDO posterior). The motivating
research observation — the reason this whole framework exists — is that the
*selection* of eta as a function of |Delta| produces a sharp transition
between posterior-driven and likelihood-driven behavior. The smoothness
diagnostic (Step 5) makes that complaint quantitative.

Admissible range is bounded below by the non-negativity of `denom`:
  eta_min = -w/(1-w) + buffer       (variance positive)
  eta_max = +inf in principle; capped at 1 in practice (Wald limit).

This implementation specializes on `NormalNormalModel`. Calling `tilt` with a
non-conjugate-Normal posterior raises `NotImplementedError`, by design — the
generic numerical fallback is a future extension and would obscure the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..config import Config
from ..models.base import Likelihood, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..models.normal_normal import weight as _weight
from .base import ParamSpec, TiltingContext


_ETA_MIN_BUFFER = Config.default().eta_min_buffer


def _require_gaussian(posterior: Posterior, prior: Prior, likelihood: Likelihood
                      ) -> tuple[NormalDistribution, NormalDistribution,
                                 GaussianLikelihood]:
    """Recognize a Normal-Normal context. Raise if any input is not Gaussian."""
    if not isinstance(posterior, NormalDistribution):
        raise NotImplementedError(
            "PowerLawTilting requires a NormalDistribution posterior; "
            f"got {type(posterior).__name__!r}. Numerical fallback is a "
            f"future extension."
        )
    if not isinstance(prior, NormalDistribution):
        raise NotImplementedError(
            "PowerLawTilting requires a NormalDistribution prior; "
            f"got {type(prior).__name__!r}."
        )
    if not isinstance(likelihood, GaussianLikelihood):
        raise NotImplementedError(
            "PowerLawTilting requires a GaussianLikelihood; "
            f"got {type(likelihood).__name__!r}."
        )
    return posterior, prior, likelihood


def _denom(w: float, eta: float) -> float:
    return 1.0 - eta * (1.0 - w)


@register_tilting(name="power_law", brief="docs/methods/power_law.md")
@dataclass(frozen=True)
class PowerLawTilting:
    """The legacy eta-tilting scheme as a `TiltingScheme` implementation."""

    name: str = "power_law"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="eta=0 recovers WALDO; eta=1 recovers Wald.",
    )

    # ----- TiltingScheme protocol -----

    def tilt(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             eta: ArrayLike) -> NormalDistribution:
        """Closed-form tilted Normal posterior (Theorem 6 in legacy docs)."""
        post, pri, lik = _require_gaussian(posterior, prior, likelihood)

        eta_arr = np.asarray(eta, dtype=np.float64)
        if eta_arr.ndim != 0:
            raise NotImplementedError(
                "tilt() expects scalar eta; vectorized eta lands with "
                "the smoothness experiment in Step 5."
            )
        eta_f = float(eta_arr)

        w = _weight(lik.sigma, pri.scale)
        denom = _denom(w, eta_f)
        if denom <= 0.0:
            raise TiltingDomainError(
                f"eta={eta_f!r} drives the tilted-posterior denominator to "
                f"{denom!r} <= 0 with w={w!r}; admissible range is "
                f"({-w/(1-w):+.6g}, inf)."
            )

        mu_eta = (w * lik.D + (1.0 - eta_f) * (1.0 - w) * pri.loc) / denom
        sigma_eta_sq = w * lik.sigma ** 2 / denom
        sigma_eta = float(np.sqrt(sigma_eta_sq))
        return NormalDistribution(loc=float(mu_eta), scale=sigma_eta)

    def path(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             ts: NDArray[np.float64]) -> Iterable[NormalDistribution]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, t)

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        w = context.w
        if not (0.0 < w < 1.0):
            raise ValueError(f"context.w must lie in (0, 1), got {w!r}")
        # denom = 1 - eta*(1 - w) > 0  ⇔  eta < 1/(1 - w); also eta > -w/(1-w)
        # for the equivalent variance-positivity condition with the buffer.
        eta_low = -w / (1.0 - w) + _ETA_MIN_BUFFER
        eta_high = 1.0 / (1.0 - w) - _ETA_MIN_BUFFER
        return (eta_low, eta_high)
