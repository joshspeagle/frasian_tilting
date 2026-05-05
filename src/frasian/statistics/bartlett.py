"""STUB: Bartlett-corrected LRT statistic.

The Bartlett correction adjusts LRT to better match its asymptotic
chi-squared distribution at finite n:

  tau_BCLRT(theta) = LRT(theta) / E[LRT(theta)]

so the corrected statistic has mean equal to df. On the canonical
Normal-location family the LRT ~ chi^2_1 holds *exactly*, so the
correction is trivial; the value lies in non-canonical models.

Implementation note: the cleanest design is `BartlettCorrected` as a
*decorator* over an LRT-like base statistic, rather than a standalone
class. That refactor lands when the base LRT does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_statistic
from ..models.base import Model, Prior
from .base import AsymptoticDistribution


@register_statistic(name="bartlett", brief="docs/methods/bartlett.md", status="stub")
@dataclass(frozen=True)
class BartlettCorrectedLRT:
    """STUB. Bartlett-corrected LRT."""

    name: str = "bartlett"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="chi2",
        df=1,
        scale=1.0,
        description="Bartlett-corrected LRT ~ chi^2_1 to higher order.",
    )

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> NDArray[np.float64]:
        raise NotImplementedError("BartlettCorrectedLRT is a stub; see docs/methods/bartlett.md.")

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> NDArray[np.float64]:
        raise NotImplementedError("BartlettCorrectedLRT is a stub; see docs/methods/bartlett.md.")

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raise NotImplementedError("BartlettCorrectedLRT is a stub; see docs/methods/bartlett.md.")

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        raise NotImplementedError("BartlettCorrectedLRT is a stub; see docs/methods/bartlett.md.")
