"""IdentityTilting — the no-op tilting scheme.

Returns the input posterior unchanged regardless of `eta`. Its role is
to make the framework's `(TiltingScheme x TestStatistic)` matrix uniform:
every test statistic is compatible with `identity`, and the cells
`(identity, wald)` / `(identity, waldo)` recover the plain-statistic CI
without any prior reweighting.

`confidence_interval(...)` delegates straight to
`statistic.confidence_interval(alpha, data, model, prior)`. This is the
mechanism by which the experiments (coverage / width / smoothness) can
call a single uniform interface across all cells.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_tilting
from ..models.base import Likelihood, Model, Posterior, Prior
from ..statistics.base import TestStatistic
from .base import ParamSpec, TiltingContext
from .eta_selectors import FixedEtaSelector

if TYPE_CHECKING:
    from ..config import Config


@register_tilting(name="identity", brief="docs/methods/identity.md")
@dataclass(frozen=True)
class IdentityTilting:
    """No-op tilting: `tilt(...)` returns the input posterior verbatim."""

    name: ClassVar[str] = "identity"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="No-op tilting; eta has no effect.",
    )
    selector: FixedEtaSelector = field(default_factory=lambda: FixedEtaSelector(eta=0.0))

    # ----- TiltingScheme protocol -----

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> Posterior:
        return posterior

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[Posterior]:
        for _ in np.asarray(ts, dtype=np.float64):
            yield posterior

    def is_identity(self, eta: float) -> bool:
        return True

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        return (-np.inf, np.inf)

    # ----- Uniform CI / regions / pvalue interface -----

    def confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: Config | None = None,  # accepted for protocol parity; unused
    ) -> tuple[float, float]:
        """Delegate to the bare statistic — no tilting applied.

        ``config`` is accepted (kw-only) for parity with the dynamic
        schemes (``power_law`` / ``ot``) so callers can pass it
        unconditionally; identity has no dynamic-CI path so the
        argument is unused. Skeptic Phase 5 vector #2 plumbing.
        """
        del config  # silence unused-argument lints
        return statistic.confidence_interval(alpha, data, model, prior)

    def confidence_regions(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: Config | None = None,  # accepted for protocol parity; unused
    ) -> list[tuple[float, float]]:
        """Single-element list around the bare-statistic CI.

        ``config`` is accepted (kw-only) for parity with the dynamic
        schemes; unused here.
        """
        del config
        return [self.confidence_interval(alpha, data, model, prior, statistic)]

    def pvalue(
        self,
        theta: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> NDArray[np.float64]:
        """Delegate to the bare statistic — no tilting applied."""
        return statistic.pvalue(theta, data, model, prior)
