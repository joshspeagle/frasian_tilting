"""STUB: m-geodesic — convex mixture between posterior and likelihood.

The m-geodesic of information geometry: linear interpolation in density
space between the posterior and the likelihood-as-Gaussian,

    q(theta; eta) = (1 - eta) * post(theta) + eta * L(theta) / Z_L,
    eta in [0, 1].

Following the framework's uniform endpoint contract (matches `power_law`
and `ot`): `eta=0` recovers the posterior (identity element) and `eta=1`
recovers the likelihood-as-Gaussian. See `docs/methods/mixture.md`.

For Gaussian inputs the mixture is *not* Gaussian — it is a two-component
Gaussian mixture, bimodal when prior and likelihood disagree strongly.
Closed-form CIs are not available; the cross-product cell with WALDO
needs numerical inversion of the mixture cdf, and may need HPD-set
semantics when bimodal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_tilting
from ..models.base import Likelihood, Posterior, Prior
from .base import ParamSpec


@register_tilting(name="mixture", brief="docs/methods/mixture.md", status="stub")
@dataclass(frozen=True)
class MixtureTilting:
    """STUB. m-geodesic: q = (1 - eta) * posterior + eta * likelihood.

    Endpoint convention matches the framework: `eta=0` is the identity
    (recovers the posterior); `eta=1` recovers the likelihood-as-Gaussian.
    """

    name: ClassVar[str] = "mixture"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,  # eta=0 recovers posterior
        description="STUB: eta in [0, 1]; 0=posterior, 1=likelihood.",
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
