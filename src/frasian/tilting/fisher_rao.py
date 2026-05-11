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


import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
from scipy import stats as _scalar_scipy_stats

from .. import _jax_setup as _x64  # noqa: F401  -- ensure float64 active
from .._errors import TiltingDomainError
from ..models._dispatch import is_normal_normal
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ._dynamic import dynamic_ci_scan
from ._solvers import brentq_with_doubling
from .base import EtaSelector
from .eta_selectors import FixedEtaSelector

_FORCE_X64 = _x64

# Numerical guards
_VERTICAL_CASE_EPS = 1e-12       # threshold for |mu_a - mu_b| -> vertical-line geodesic
_SIGMA_FLOOR = 1e-300            # absolute floor on sigma along the path


# ----------------------------------------------------------------------
# Closed-form half-plane geodesic helpers
# ----------------------------------------------------------------------


def _fr_geodesic_gaussian_scalar(
    mu_a: float, sigma_a: float, mu_b: float, sigma_b: float, t: float
) -> tuple[float, float]:
    """Constant-speed Fisher-Rao geodesic between two Gaussians at parameter t.

    Returns ``(mu_t, sigma_t)`` along the half-plane geodesic from
    ``(mu_a, sigma_a)`` to ``(mu_b, sigma_b)``. Vertical-line case when
    ``mu_a == mu_b``; circular-arc case otherwise.

    Constant-speed parametrisation (rev 1): on the generic arc,
    ``ds/dphi = 1/sin(phi)`` (NOT constant), so linear-in-phi is NOT
    constant-speed. The correct constant-speed (= arc-length) param
    uses the antiderivative ``s(phi) = ln tan(phi/2)``:

        s(t)  = (1 - t) * ln tan(phi_a/2) + t * ln tan(phi_b/2)
        phi(t) = 2 * arctan(exp(s(t)))

    Derivation: docs/methods/fisher_rao.md "Definition" (rev 1) +
    deriver output Step 5 at docs/superpowers/specs/2026-05-11-fisher-
    rao-deriver-output.md (sympy-verified).
    """
    if not (sigma_a > 0.0 and sigma_b > 0.0):
        raise TiltingDomainError(
            f"FisherRaoTilting requires positive sigmas; got sigma_a={sigma_a!r}, sigma_b={sigma_b!r}."
        )
    sqrt2 = math.sqrt(2.0)
    mu_a_t = mu_a / sqrt2
    mu_b_t = mu_b / sqrt2
    if abs(mu_a_t - mu_b_t) < _VERTICAL_CASE_EPS:
        s_t = sigma_a ** (1.0 - t) * sigma_b ** t
        return float(mu_a), float(s_t)
    c_tilde = ((mu_a_t * mu_a_t - mu_b_t * mu_b_t) + (sigma_a * sigma_a - sigma_b * sigma_b)) \
              / (2.0 * (mu_a_t - mu_b_t))
    r = math.sqrt((mu_a_t - c_tilde) ** 2 + sigma_a * sigma_a)
    phi_a = math.atan2(sigma_a, mu_a_t - c_tilde)
    phi_b = math.atan2(sigma_b, mu_b_t - c_tilde)
    s_a = math.log(math.tan(phi_a / 2.0))
    s_b = math.log(math.tan(phi_b / 2.0))
    s_t = (1.0 - t) * s_a + t * s_b
    phi_t = 2.0 * math.atan(math.exp(s_t))
    mu_t_tilde = c_tilde + r * math.cos(phi_t)
    s_sigma = r * math.sin(phi_t)
    if s_sigma <= _SIGMA_FLOOR:
        raise TiltingDomainError(
            f"FisherRaoTilting: geodesic crossed sigma=0 boundary at t={t!r} "
            f"(numerical instability or out-of-admissible parameter)."
        )
    return float(sqrt2 * mu_t_tilde), float(s_sigma)


def _fr_arc_length_costa(
    mu_a: float, sigma_a: float, mu_b: float, sigma_b: float
) -> float:
    """Closed-form Fisher-Rao arc-length per Costa et al. 2015 Eqs. 5-6.

    d_FR = sqrt(2) * arccosh(1 + ((mu_a-mu_b)^2/2 + (sigma_a-sigma_b)^2) / (2*sigma_a*sigma_b))

    (Note: cited in earlier drafts as "Eq. 12" — that was wrong; Eq. 12
    is the symmetrised KL distance. Correct citation is Eqs. 5-6.)
    """
    arg = 1.0 + ((mu_a - mu_b) ** 2 / 2.0 + (sigma_a - sigma_b) ** 2) / (2.0 * sigma_a * sigma_b)
    return float(math.sqrt(2.0) * math.acosh(arg))


def _fr_geodesic_arc_length_numerical(
    mu_a: float, sigma_a: float, mu_b: float, sigma_b: float, n_steps: int = 10000
) -> float:
    """Trapezoidal arc-length integration along the closed-form geodesic.

    Used by tests to verify that the constant-speed parametrisation
    integrates to the Costa et al. 2015 closed form. The half-plane
    metric ds = sqrt((dmu_tilde^2 + dsigma^2)/sigma^2) where
    mu_tilde = mu/sqrt(2). Multiply by sqrt(2) to get the Fisher arc
    length d_F = sqrt(2) * d_H.
    """
    import numpy as np
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    pts = np.empty((n_steps + 1, 2))
    for i, t in enumerate(ts):
        pts[i] = _fr_geodesic_gaussian_scalar(mu_a, sigma_a, mu_b, sigma_b, float(t))
    sqrt2 = math.sqrt(2.0)
    diffs = np.diff(pts, axis=0)
    sigma_mid = 0.5 * (pts[:-1, 1] + pts[1:, 1])
    diffs[:, 0] /= sqrt2  # mu -> mu_tilde for half-plane metric
    ds = np.sqrt((diffs ** 2).sum(axis=1)) / sigma_mid
    return float(sqrt2 * ds.sum())  # multiply by sqrt(2) for Fisher arc length


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
