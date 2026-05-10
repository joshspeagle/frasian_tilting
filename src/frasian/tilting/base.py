"""TiltingScheme protocol and its companions.

A `TiltingScheme` interpolates between (prior, likelihood, posterior) by
some parameter η. `PowerLawTilting` is the e-geodesic / log-linear
scheme; `OTTilting` is the W2 geodesic; `FisherRaoTilting` and
`MixtureTilting` are stubs for the Levi-Civita and m-geodesic variants
(see CLAUDE.md for the full taxonomy).

`EtaSelector` is a separate protocol for choosing η given context. It
exists because the legacy code conflated three solvers (numerical,
closed-form approximation, MLP-learned) inside `tilting.py`; here, all
three live behind the same interface and can be swapped.
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
class ParamSpec:
    """Description of a tilting scheme's η parameter space."""

    eta_default: float
    eta_identity: float  # value of η for which `tilt` returns the input posterior
    description: str = ""
    eta_likelihood_only: float | None = None
    """Value of η that makes the tilted inference equivalent to a
    likelihood-only (data-only) procedure — i.e. no contribution from
    the prior. For schemes whose η interpolates posterior↔likelihood
    (power_law, ot, mixture), this is 1.0. For schemes where η doesn't
    parameterize a posterior↔likelihood mixing (identity, future
    schemes), set None — callers that need this value (e.g.
    LearnedDynamicEtaSelector's out-of-training-distribution clamp)
    should treat None as "concept doesn't apply" and skip the clamp.
    """


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
        - For η outside the scheme's admissible parameter region (e.g.
          η that drives a variance non-positive on Normal-Normal),
          `tilt` / `tilted_pvalue` MUST raise `TiltingDomainError`,
          never return NaN. The admissible region is an *internal*
          property of each implementation: the public protocol does
          not expose it. Selectors that need η-bounds compute them
          internally (`NumericalEtaSelector._eta_bounds`); learned
          selectors enforce admissibility via the trained
          `ValidityNet` boundary penalty.
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

    Implementations: `FixedEtaSelector` (constant η),
    `NumericalEtaSelector` (Brent root-find on CI width or integrated p
    at a representative D — post-selection),
    `DynamicNumericalEtaSelector` (per-θ static η, calibrated),
    `LearnedDynamicEtaSelector` (per-θ MLP η, calibrated).

    Signature: `select(...)` accepts `data`, `model`, `prior`, `alpha`,
    `statistic` as keyword arguments and returns a scalar η. Selectors
    that need η-bounds compute them internally — the protocol does not
    require callers to thread a separate context object or query an
    admissible-range API. (Earlier drafts of the framework had a
    `TiltingContext` plus a public `scheme.admissible_range`; both
    were dropped in Phase 3a.)

    **Post-selection / D-conditioning warning.** The signature accepts
    `data`, but selectors split into two semantic flavours:

      * ``is_post_selection = False`` — η is independent of D. The
        WALDO p-value at any fixed η is U[0,1] under H0, so the
        resulting CI is calibrated. `FixedEtaSelector`,
        `DynamicNumericalEtaSelector` (via ``select_grid``), and
        `LearnedDynamicEtaSelector` all satisfy this.
      * ``is_post_selection = True`` — η is chosen by reading
        ``D = data.mean()`` and minimising width / integrated-p AT that
        D. The resulting CI is **post-selection** and undercovers
        nominal level by ~2 points at α=0.05 (see
        `tests/regression/test_post_selection_coverage.py`).
        `NumericalEtaSelector` is the only such implementation.

    The runner does **not** automatically reject post-selection
    selectors; coverage / width experiments should consult
    ``selector.is_post_selection`` to decide whether the resulting
    interval is calibrated or merely a width baseline.
    """

    name: str
    is_dynamic: bool
    is_post_selection: bool

    def select(
        self,
        scheme: TiltingScheme,
        *,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        alpha: float,
        statistic: TestStatistic,
    ) -> float: ...


@runtime_checkable
class DynamicEtaSelector(Protocol):
    """Dynamic selectors expose `select_grid(theta_grid, ...)` returning
    one η per θ.

    `select(...)` on a dynamic selector returns the η at a representative
    θ (e.g. the data sufficient statistic), preserved for callers that
    need a single number.
    """

    name: str
    is_dynamic: bool
    is_post_selection: bool

    def select(
        self,
        scheme: TiltingScheme,
        *,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        alpha: float,
        statistic: TestStatistic,
    ) -> float: ...

    def select_grid(
        self,
        theta_grid: NDArray[np.float64],
        scheme: TiltingScheme,
        *,
        model: Model,
        prior: Prior,
        alpha: float,
        statistic: TestStatistic,
    ) -> NDArray[np.float64]: ...
