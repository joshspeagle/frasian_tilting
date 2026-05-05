"""STUB: signed-root LRT statistic (r* / signed root).

  r(theta) = sign(MLE - theta) * sqrt(LRT(theta)).

Asymptotically r ~ N(0, 1). On the Normal location family this equals
the standardised z = (D - theta) / sigma exactly — same CI as Wald.
On non-Normal models r and Wald differ; signed-root has more accurate
small-sample coverage in many cases. Bartlett correction (`bartlett`)
is the related variance shrinkage applied to LRT itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_statistic
from ..models.base import Model, Prior
from .base import AsymptoticDistribution


@register_statistic(name="signed_root", brief="docs/methods/signed_root.md", status="stub")
@dataclass(frozen=True)
class SignedRootStatistic:
    """STUB. Signed-root LRT (r-statistic)."""

    name: str = "signed_root"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="normal",
        df=None,
        scale=1.0,
        description="r ~ N(0, 1) under H0 (Barndorff-Nielsen).",
    )

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> NDArray[np.float64]:
        raise NotImplementedError("SignedRootStatistic is a stub; see docs/methods/signed_root.md.")

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> NDArray[np.float64]:
        raise NotImplementedError("SignedRootStatistic is a stub; see docs/methods/signed_root.md.")

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raise NotImplementedError("SignedRootStatistic is a stub; see docs/methods/signed_root.md.")

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        raise NotImplementedError("SignedRootStatistic is a stub; see docs/methods/signed_root.md.")
