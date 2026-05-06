"""Wald test statistic — pure-likelihood, prior-ignoring CI.

  tau_Wald(theta) = ((D - theta) / sigma)^2
  CI:           D ± z_{1-alpha/2} * sigma

Specializes on `NormalNormalModel` (port from legacy `waldo.py:wald_ci`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._registry import register_statistic
from ..models._dispatch import require_model
from ..models.base import Model, Prior
from ..models.normal_normal import NormalNormalModel
from .base import AsymptoticDistribution

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _require_normal_normal(model: Model) -> NormalNormalModel:
    return require_model(model, NormalNormalModel, caller="WaldStatistic")


@register_statistic(name="wald", brief="docs/methods/wald.md")
@dataclass(frozen=True)
class WaldStatistic:
    """Wald statistic for the Normal location family."""

    name: ClassVar[str] = "wald"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="chi2",
        df=1,
        scale=1.0,
        description="(D - theta)^2 / sigma^2 ~ chi^2_1 under H0.",
    )

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        m = _require_normal_normal(model)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = (D - jnp.asarray(theta0, dtype=jnp.float64)) / m.sigma
        return z * z

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        m = _require_normal_normal(model)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = jnp.abs(D - jnp.asarray(theta0, dtype=jnp.float64)) / m.sigma
        # Two-sided p-value: 2 * (1 - Phi(|z|)).
        # jax.scipy.stats.norm has no `sf`, so use 1 - cdf.
        return 2.0 * (1.0 - jsp_stats.norm.cdf(z))

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Return (D_lo, D_hi) such that Wald accepts H0 iff D in [D_lo, D_hi]."""
        m = _require_normal_normal(model)
        z_crit = jsp_stats.norm.ppf(1.0 - alpha / 2.0)
        theta_arr = jnp.asarray(theta0, dtype=jnp.float64)
        return (theta_arr - z_crit * m.sigma, theta_arr + z_crit * m.sigma)

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        """Closed-form Wald CI: D ± z_{1-alpha/2} * sigma."""
        m = _require_normal_normal(model)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = float(jsp_stats.norm.ppf(1.0 - alpha / 2.0))
        half = z * m.sigma
        return (D - half, D + half)

    def accepts_tilting(self, tilting) -> bool:
        """Wald ignores the prior, so non-identity tiltings are degenerate."""
        return getattr(tilting, "name", "") == "identity"
