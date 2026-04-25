"""STUB: exponential-family natural-parameter interpolation.

For exponential-family distributions parameterised by natural parameters
theta_a, theta_b, the canonical interpolation is theta_t = (1-t) theta_a
+ t theta_b. For Gaussians (with natural parameters mu/sigma^2 and
-1/(2 sigma^2)) this yields a Gaussian whose precision interpolates
linearly. Distinct from `power_law` (which interpolates by tempering the
prior) and from `ot_normal` / `geodesic_normal` (which interpolate
geometrically).

Why include it: in non-Normal exponential-family models, exp-family
interpolation is the natural smooth path; the conjugate-Normal sandbox
is the falsifiability check that all four schemes degenerate to
similar Gaussians, with smoothness differences attributable to the
parameterisation rather than the family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_tilting
from ..models.base import Likelihood, Posterior, Prior
from .base import ParamSpec, TiltingContext


@register_tilting(name="exp_family", brief="docs/methods/exp_family.md",
                  status="stub")
@dataclass(frozen=True)
class ExpFamilyTilting:
    """STUB. Natural-parameter linear interpolation."""

    name: str = "exp_family"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="STUB: t in [0, 1] in natural-parameter space.",
    )

    def tilt(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             eta: ArrayLike) -> Posterior:
        raise NotImplementedError(
            "ExpFamilyTilting is a stub; see docs/methods/exp_family.md."
        )

    def path(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             ts: NDArray[np.float64]) -> Iterable[Posterior]:
        raise NotImplementedError(
            "ExpFamilyTilting is a stub; see docs/methods/exp_family.md."
        )

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        return (0.0, 1.0)
