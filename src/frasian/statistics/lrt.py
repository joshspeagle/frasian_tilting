"""STUB: likelihood-ratio test statistic.

  tau_LRT(theta) = -2 * log( L(theta) / max_theta L(theta) ).

For the Normal location family with known variance, tau_LRT reduces to
((D - theta) / sigma)^2 — i.e. exactly the Wald statistic on the
canonical sandbox. The comparison becomes interesting only on
non-Gaussian or non-canonical models, but registering LRT here puts
the (TiltingScheme x LRT) row into the cross-product so the framework
can cleanly extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_statistic
from ..models.base import Model, Prior
from .base import AsymptoticDistribution


@register_statistic(name="lrt", brief="docs/methods/lrt.md", status="stub")
@dataclass(frozen=True)
class LRTStatistic:
    """STUB. Likelihood-ratio test statistic."""

    name: ClassVar[str] = "lrt"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="chi2",
        df=1,
        scale=1.0,
        description="-2 log Lambda ~ chi^2_1 under H0 (Wilks).",
    )

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        raise NotImplementedError("LRTStatistic is a stub; see docs/methods/lrt.md.")

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        raise NotImplementedError("LRTStatistic is a stub; see docs/methods/lrt.md.")

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError("LRTStatistic is a stub; see docs/methods/lrt.md.")

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        raise NotImplementedError("LRTStatistic is a stub; see docs/methods/lrt.md.")
