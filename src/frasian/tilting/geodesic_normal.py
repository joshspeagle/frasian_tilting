"""STUB: Fisher-Rao geodesic on the univariate Gaussian manifold.

The Gaussian family with the Fisher information metric is hyperbolic
(half-plane model), and geodesics between two Gaussians have a closed
form involving inverse hyperbolic-tangent moves through the (mu, sigma)
half-plane. Compared to the W2 geodesic (`ot_normal`), Fisher-Rao
respects the information-geometric structure rather than the
displacement of mass. The two coincide only when the variances match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_tilting
from ..models.base import Likelihood, Posterior, Prior
from .base import ParamSpec, TiltingContext


@register_tilting(name="geodesic_normal",
                  brief="docs/methods/geodesic_normal.md",
                  status="stub")
@dataclass(frozen=True)
class GeodesicNormalTilting:
    """STUB. Fisher-Rao (information-geometric) geodesic."""

    name: str = "geodesic_normal"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="STUB: t in [0, 1] along Fisher-Rao geodesic.",
    )

    def tilt(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             eta: ArrayLike) -> Posterior:
        raise NotImplementedError(
            "GeodesicNormalTilting is a stub; see "
            "docs/methods/geodesic_normal.md."
        )

    def path(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             ts: NDArray[np.float64]) -> Iterable[Posterior]:
        raise NotImplementedError(
            "GeodesicNormalTilting is a stub; see "
            "docs/methods/geodesic_normal.md."
        )

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        return (0.0, 1.0)
