"""TestStatistic protocol and supporting types.

Each `TestStatistic` knows how to compute its value at a hypothesized parameter,
its asymptotic null distribution, its p-value, and an acceptance region. The
abstraction is uniform across Wald, WALDO, LRT, signed-root, and Bartlett-
corrected variants — Bartlett is implemented as a decorator over LRT, not as a
separate top-level statistic.

Statistics also declare which `TiltingScheme`s they accept via
`accepts_tilting`. The default is "any tilting"; pure-likelihood statistics
that ignore the prior (e.g. Wald) override to accept only `IdentityTilting`,
so the runner can gate non-identity cells out of the cross-product instead
of producing numerically-degenerate duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..models.base import Model, Prior

if TYPE_CHECKING:
    from ..tilting.base import TiltingScheme


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

    @property
    def name(self) -> str: ...

    @property
    def asymptotic_null(self) -> AsymptoticDistribution: ...

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> NDArray[np.float64]: ...

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> NDArray[np.float64]: ...

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        """Dual of `acceptance_region`: parameter-space CI given data.

        Returns `(lower, upper)` such that `pvalue(theta, data, ...) >= alpha`
        for all `theta` in the closed interval. Statistics that do not admit a
        natural CI inversion may raise `NotImplementedError`.
        """
        ...

    def accepts_tilting(self, tilting: TiltingScheme) -> bool:
        """Declare whether this statistic supports being paired with `tilting`.

        Default: any tilting is acceptable. Statistics that ignore the prior
        (e.g. Wald) should override to return `True` only for the identity
        tilting — the runner skips cells where this returns `False` and
        records them as `incompatible` in the manifest.
        """
        return True


def accepts_tilting(statistic: TestStatistic, tilting: TiltingScheme) -> bool:
    """Free-function entry point used by the runner.

    Statistics may opt in by defining `accepts_tilting`; absent the method,
    we default to `True` (compatible with any tilting). This avoids forcing
    every concrete statistic to inherit from a common base class — they are
    `Protocol` implementations.
    """
    fn = getattr(statistic, "accepts_tilting", None)
    if fn is None:
        return True
    return bool(fn(tilting))
