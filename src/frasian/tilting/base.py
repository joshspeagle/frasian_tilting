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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._errors import TiltingDomainError  # noqa: F401  (re-export contract)
from ..models.base import Likelihood, Posterior, Prior

if TYPE_CHECKING:
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

    name: str
    param_space: ParamSpec

    def tilt(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             eta: ArrayLike) -> Posterior: ...

    def path(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             ts: NDArray[np.float64]) -> Iterable[Posterior]: ...

    def is_identity(self, eta: float) -> bool: ...

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]: ...


@runtime_checkable
class EtaSelector(Protocol):
    """Strategy for picking η given a tilting context.

    Implementations: `NumericalEtaSelector` (Brent root-find on CI width),
    `ClosedFormEtaSelector` (analytic approximation), `LearnedEtaSelector`
    (wraps a `LearnedArtifact` such as the monotonic MLP).
    """

    name: str

    def select(self, context: TiltingContext, scheme: TiltingScheme,
               *, statistic: TestStatistic) -> float: ...
