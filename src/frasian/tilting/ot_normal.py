"""STUB: optimal-transport interpolation between prior and posterior.

For two univariate Gaussians N(mu_a, sigma_a^2) and N(mu_b, sigma_b^2) the
W2 geodesic is a straight line in (mean, std) space:

    mu_t    = (1 - t) * mu_a + t * mu_b
    sigma_t = (1 - t) * sigma_a + t * sigma_b,  t in [0, 1]

Mapped to the framework's tilting parameterisation, the candidate identity
element is `eta = 0` -> posterior, `eta = 1` -> prior (or vice versa,
TBD by /derive). The motivating hypothesis is that this geodesic
produces a smoother eta*(|Delta|) curve than power-law tilting; the
`smoothness` experiment is the gating diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_tilting
from ..models.base import Likelihood, Posterior, Prior
from .base import ParamSpec, TiltingContext


@register_tilting(name="ot_normal", brief="docs/methods/ot_normal.md",
                  status="stub")
@dataclass(frozen=True)
class OTNormalTilting:
    """STUB. Wasserstein-2 geodesic on the Gaussian family."""

    name: str = "ot_normal"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="STUB: t in [0, 1] along W2 geodesic; identity TBD.",
    )

    def tilt(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             eta: ArrayLike) -> Posterior:
        raise NotImplementedError(
            "OTNormalTilting is a stub; see docs/methods/ot_normal.md. "
            "Implementation lands via /propose-method ot_normal."
        )

    def path(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             ts: NDArray[np.float64]) -> Iterable[Posterior]:
        raise NotImplementedError(
            "OTNormalTilting is a stub; see docs/methods/ot_normal.md."
        )

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        # Bounded interpolation parameter — t in [0, 1] is the natural range.
        return (0.0, 1.0)
