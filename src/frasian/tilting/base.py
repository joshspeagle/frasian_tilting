"""TiltingScheme protocol and its companions.

A `TiltingScheme` interpolates between (prior, likelihood, posterior) by some
parameter η. The simplest scheme — `PowerLawTilting` — is the existing
framework's η-tilting (Step 2 ports it). Future schemes (OT, Fisher–Rao
geodesic, mixture, exponential-family path) plug into the same interface.

`EtaSelector` is a separate protocol for choosing η given context. It exists
because the legacy code conflated three solvers (numerical, closed-form
approximation, MLP-learned) inside `tilting.py`; here, all three live behind
the same interface and can be swapped.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._errors import TiltingDomainError  # noqa: F401  (re-export contract)
from ..models.base import Likelihood, Posterior, Prior

if TYPE_CHECKING:
    from ..models.base import Model
    from ..statistics.base import TestStatistic


@dataclass(frozen=True)
class TiltingContext:
    """Non-η parameters that bound or describe a tilting choice."""

    w: float
    abs_delta: float
    alpha: float


@dataclass(frozen=True)
class ParamSpec:
    """Description of a tilting scheme's η parameter space."""

    eta_default: float
    eta_identity: float  # value of η for which `tilt` returns the input posterior
    description: str = ""


@runtime_checkable
class TiltingScheme(Protocol):
    """Family of tilted distributions parameterized by η.

    Invariants any implementation must satisfy (verified per-scheme in
    `tests/properties/test_<scheme_name>_invariants.py` —
    `test_power_law_invariants.py` for the implemented scheme; the
    stub schemes have skipped placeholder tests under the same naming
    convention):
        - `tilt(posterior, ..., eta=param_space.eta_identity)` returns a
          distribution numerically equal to `posterior`.
        - `tilt(...).pdf` is non-negative and integrates to 1 (KS-tolerance).
        - `path(ts)` is continuous in t (W₁ Lipschitz bound).
        - `admissible_range(context)` brackets the η values for which `tilt`
          succeeds; outside this range the implementation MUST raise
          `TiltingDomainError`, never return NaN.
    """

    @property
    def name(self) -> str: ...

    @property
    def param_space(self) -> ParamSpec: ...

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> Posterior: ...

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[Posterior]: ...

    def is_identity(self, eta: float) -> bool: ...

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]: ...

    def confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> tuple[float, float]:
        """Compute the (1-α) CI under this tilting and the given statistic.

        The uniform CI interface — single (lo, hi) tuple summary. For
        cells that may produce multi-region CIs (e.g. dynamic-η Dyn-WALDO
        at low |Δ|), this returns the convex hull `(min lo, max hi)`. For
        the actual region list, use `confidence_regions`.

        Cells whose `(scheme, statistic)` combination is unsupported MUST
        raise `NotImplementedError`.
        """
        ...

    def confidence_regions(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> list[tuple[float, float]]:
        """Return the (possibly multiple) region(s) of the (1-α) CI.

        Default behaviour (and what every static-η cell must satisfy):
        `[confidence_interval(...)]` — a single-element list containing
        the same tuple. Cells whose CI inversion legitimately produces
        multiple disjoint regions (e.g. `power_law` + dynamic-η selector
        at low |Δ|, where the dynamic p-value is multimodal) override
        this to return the actual list, sorted by lower endpoint.

        Consumers (`coverage`, `width`, `confidence_distribution`) should
        prefer this method over `confidence_interval` when union-of-regions
        semantics matter (true coverage, true width, true CD shape).
        """
        ...

    def pvalue(
        self,
        theta: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> NDArray[np.float64]:
        """Selector-aware p-value at hypothesised θ values.

        Default (no tilting): delegate to `statistic.pvalue(θ, data, model, prior)`.
        For a non-identity tilting with a fixed-η static selector: evaluate
        the *tilted* p-value at the selector-resolved η. For a dynamic-η
        selector: use the per-θ varying η = η*(|Δ_θ|).

        This is the universal entry point that the CD constructor uses to
        evaluate `p(θ)` on a fine θ-grid for the Schweder–Hjort density
        construction.

        Cells whose `(scheme, statistic)` combination is unsupported MUST
        raise `NotImplementedError`.
        """
        ...


@runtime_checkable
class EtaSelector(Protocol):
    """Strategy for picking η given a tilting context.

    Implementations: `NumericalEtaSelector` (Brent root-find on CI width),
    `ClosedFormEtaSelector` (analytic approximation), `LearnedEtaSelector`
    (wraps a `LearnedArtifact` such as the monotonic MLP).
    """

    @property
    def name(self) -> str: ...

    def select(
        self, context: TiltingContext, scheme: TiltingScheme, *, statistic: TestStatistic
    ) -> float: ...
