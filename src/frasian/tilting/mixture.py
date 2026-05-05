"""STUB: convex mixture between prior and posterior.

The simplest possible interpolation: q(theta; eta) = (1 - eta) * pi(theta)
+ eta * post(theta). Bounded everywhere, no admissible-range pathology.
For Gaussian inputs the mixture is *not* Gaussian, so closed-form CIs are
not available — the cross-product cell with WaldoStatistic / WaldStatistic
needs a numerical CI inversion based on the mixture cdf.

Included as a baseline for the smoothness diagnostic: a mixture path is
trivially smooth in (mean, var) but produces multi-modal posteriors
when prior and posterior disagree, which may break test-statistic
assumptions. Whether it is *useful* is what we are measuring.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_tilting
from ..models.base import Likelihood, Posterior, Prior
from .base import ParamSpec, TiltingContext


@register_tilting(name="mixture", brief="docs/methods/mixture.md", status="stub")
@dataclass(frozen=True)
class MixtureTilting:
    """STUB. Convex mixture (1-eta)*prior + eta*posterior."""

    name: str = "mixture"
    param_space: ParamSpec = ParamSpec(
        eta_default=1.0,
        eta_identity=1.0,  # eta=1 recovers posterior
        description="STUB: eta in [0, 1]; 0=prior, 1=posterior.",
    )

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> Posterior:
        raise NotImplementedError(
            "MixtureTilting is a stub; see docs/methods/mixture.md. "
            "Output is not Gaussian; impl needs a Distribution wrapper "
            "for two-component mixtures."
        )

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[Posterior]:
        raise NotImplementedError("MixtureTilting is a stub; see docs/methods/mixture.md.")

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        return (0.0, 1.0)
