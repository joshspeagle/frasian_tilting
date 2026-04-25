"""TestStatistic protocol and supporting types.

Each `TestStatistic` knows how to compute its value at a hypothesized parameter,
its asymptotic null distribution, its p-value, and an acceptance region. The
abstraction is uniform across Wald, WALDO, LRT, signed-root, and Bartlett-
corrected variants — Bartlett is implemented as a decorator over LRT, not as a
separate top-level statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..models.base import Model, Prior


@dataclass(frozen=True)
class AsymptoticDistribution:
    """Description of the asymptotic null distribution of a test statistic."""

    family: str  # "chi2", "normal", "weighted_chi2", ...
    df: float | None = None  # degrees of freedom for chi2-like
    scale: float = 1.0
    description: str = ""


@runtime_checkable
class TestStatistic(Protocol):
    """A test statistic together with its calibration machinery.

    Invariants any implementation must satisfy
    (tests/properties/test_statistic_invariants.py):
        - `pvalue(...)` ∈ [0, 1] for all inputs.
        - Under H0 (data generated at θ₀), `pvalue` is Uniform[0, 1] (KS test).
        - `evaluate` is continuous in `data` away from a measure-zero set.
        - `acceptance_region(alpha, ...)` has α-level frequentist coverage.
    """

    name: str
    asymptotic_null: AsymptoticDistribution

    def evaluate(self, theta0: ArrayLike, data: NDArray[np.float64],
                 model: Model, prior: Prior | None = None
                 ) -> NDArray[np.float64]: ...

    def pvalue(self, theta0: ArrayLike, data: NDArray[np.float64],
               model: Model, prior: Prior | None = None
               ) -> NDArray[np.float64]: ...

    def acceptance_region(self, alpha: float, theta0: ArrayLike,
                          model: Model, prior: Prior | None = None
                          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
