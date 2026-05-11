"""STUB: Fisher-Rao geodesic on a parametric distribution family.

Fisher-Rao is the Riemannian (Levi-Civita) geodesic of a parametric
manifold equipped with the Fisher information metric. It is the
*third* affine connection compatible with the Fisher metric, distinct
from the e-connection (`power_law`'s log-linear path) and the
m-connection (`mixture`'s linear-in-density path) of Amari's dually
flat structure.

On the univariate Gaussian family the Fisher metric makes the manifold
the upper half-plane in `(mu, sigma)` (after rescaling `mu` by sqrt(2)
so the Gaussian curvature is -1), and geodesics between two Gaussians
have a closed form (Costa et al. 2015 Eq. 12). Compared to the W2
geodesic (`ot`), Fisher-Rao respects the *information-geometric*
structure rather than the displacement of mass; the two coincide only
when sigma_a = sigma_b.

Implementation scope (this stub): Gaussian-only closed form first, with
NotImplementedError on non-Gaussian endpoints. A general
ParametricFamily interface (which would let us run Fisher-Rao on
other families) is a follow-up refactor.

Endpoints follow the framework's posterior <-> likelihood convention
(matching `power_law` and `ot`): eta=0 -> posterior, eta=1 ->
likelihood-induced Gaussian N(D, sigma^2).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._registry import register_tilting
from ..models.base import Likelihood, Model, Posterior, Prior
from ..statistics.base import TestStatistic
from .base import ParamSpec

if TYPE_CHECKING:
    from ..config import Config


@register_tilting(name="fisher_rao", brief="docs/methods/fisher_rao.md", status="stub")
@dataclass(frozen=True)
class FisherRaoTilting:
    """STUB. Fisher-Rao (information-geometric) geodesic on the
    Gaussian half-plane. eta=0 -> posterior, eta=1 -> likelihood-as-Gaussian."""

    name: ClassVar[str] = "fisher_rao"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description=(
            "STUB: t in [0, 1] along the Fisher-Rao geodesic between posterior "
            "(t=0) and the likelihood-induced Gaussian N(D, sigma^2) (t=1)."
        ),
    )

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> Posterior:
        raise NotImplementedError("FisherRaoTilting is a stub; see docs/methods/fisher_rao.md.")

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[Posterior]:
        raise NotImplementedError("FisherRaoTilting is a stub; see docs/methods/fisher_rao.md.")

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    # ----- Uniform CI / regions / pvalue interface (audit P0-10) -----

    def confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: "Config | None" = None,
    ) -> tuple[float, float]:
        raise NotImplementedError(
            "FisherRaoTilting.confidence_interval is a stub; see docs/methods/fisher_rao.md."
        )

    def confidence_regions(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: "Config | None" = None,
    ) -> list[tuple[float, float]]:
        raise NotImplementedError(
            "FisherRaoTilting.confidence_regions is a stub; see docs/methods/fisher_rao.md."
        )

    def pvalue(
        self,
        theta: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            "FisherRaoTilting.pvalue is a stub; see docs/methods/fisher_rao.md."
        )
