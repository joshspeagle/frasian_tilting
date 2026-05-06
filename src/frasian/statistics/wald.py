"""Wald test statistic — pure-likelihood, prior-ignoring CI.

  tau_Wald(theta) = (mle(data) - theta)^2 * I(theta)
  Asymptotic null: tau_Wald ~ chi^2_1 under H0 (Wilks).

For NormalNormalModel this reduces to the closed form
  CI = D ± z_{1-alpha/2} * sigma
which is dispatched as the fast path; for any other Model it falls
through to the model-agnostic numerical path that consumes only
`model.mle(data)`, `model.fisher_information(theta)`, and the
chi^2_1 calibration.
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
from ..models.base import Model, Prior
from ..models.normal_normal import NormalNormalModel
from .base import AsymptoticDistribution

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _is_normal_normal(model: Model) -> bool:
    return isinstance(model, NormalNormalModel)


@register_statistic(name="wald", brief="docs/methods/wald.md")
@dataclass(frozen=True)
class WaldStatistic:
    """Wald statistic with closed-form Normal-Normal fast path + generic default.

    The closed-form path uses `D ± z·σ` and `2(1−Φ(|D−θ|/σ))`. The generic
    default path uses `tau = (mle − θ)² · I(θ)` with χ²(1) calibration; it
    works against any `Model` that implements `mle` and `fisher_information`.
    Cross-checks in `tests/regression/test_wald_generic_matches_closed_form.py`
    pin agreement of the two paths within numerical tolerance on Normal-Normal.
    """

    name: ClassVar[str] = "wald"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="chi2",
        df=1,
        scale=1.0,
        description="(D - theta)^2 / sigma^2 ~ chi^2_1 under H0; "
        "generic: (mle - theta)^2 * I(theta) ~ chi^2_1.",
    )

    # ---------- closed-form Normal-Normal path ----------

    def _closed_form_evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: NormalNormalModel
    ) -> jax.Array:
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = (D - jnp.asarray(theta0, dtype=jnp.float64)) / model.sigma
        return z * z

    def _closed_form_pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: NormalNormalModel
    ) -> jax.Array:
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = jnp.abs(D - jnp.asarray(theta0, dtype=jnp.float64)) / model.sigma
        return 2.0 * (1.0 - jsp_stats.norm.cdf(z))

    def _closed_form_acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: NormalNormalModel
    ) -> tuple[jax.Array, jax.Array]:
        z_crit = jsp_stats.norm.ppf(1.0 - alpha / 2.0)
        theta_arr = jnp.asarray(theta0, dtype=jnp.float64)
        return (theta_arr - z_crit * model.sigma, theta_arr + z_crit * model.sigma)

    def _closed_form_confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: NormalNormalModel
    ) -> tuple[float, float]:
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = float(jsp_stats.norm.ppf(1.0 - alpha / 2.0))
        half = z * model.sigma
        return (D - half, D + half)

    # ---------- generic model-agnostic path ----------

    def _generic_evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model
    ) -> jax.Array:
        theta_arr = jnp.asarray(theta0, dtype=jnp.float64)
        mle = jnp.asarray(model.mle(data))
        info = jnp.asarray(model.fisher_information(theta_arr))
        diff = mle - theta_arr
        return diff * diff * info

    def _generic_pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model
    ) -> jax.Array:
        tau = self._generic_evaluate(theta0, data, model)
        # chi^2_1 survival function: 1 - chi2.cdf(tau, df=1).
        return 1.0 - jsp_stats.chi2.cdf(tau, 1)

    def _generic_confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model
    ) -> tuple[float, float]:
        # scipy: brentq lives at the public CI-inversion boundary; numpy/scipy.
        from ..tilting._solvers import brentq_with_doubling

        mle = float(np.asarray(model.mle(data)))

        def f(theta: float) -> float:
            return float(self._generic_pvalue(theta, data, model)) - alpha

        # Bracket outward from the MLE (where the p-value is 1).
        # Use 1/sqrt(I(mle)) as a natural width scale.
        info_at_mle = float(np.asarray(model.fisher_information(mle)))
        if info_at_mle <= 0 or not np.isfinite(info_at_mle):
            half = 1.0
        else:
            half = 4.0 / np.sqrt(info_at_mle)
        lower = brentq_with_doubling(
            f, midpoint=mle, initial_half_width=half, direction=-1
        )
        upper = brentq_with_doubling(
            f, midpoint=mle, initial_half_width=half, direction=+1
        )
        return (lower, upper)

    # ---------- public protocol surface (dispatches) ----------

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        if _is_normal_normal(model):
            return self._closed_form_evaluate(theta0, data, model)  # type: ignore[arg-type]
        return self._generic_evaluate(theta0, data, model)

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        if _is_normal_normal(model):
            return self._closed_form_pvalue(theta0, data, model)  # type: ignore[arg-type]
        return self._generic_pvalue(theta0, data, model)

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Return (D_lo, D_hi) such that Wald accepts H0 iff D in [D_lo, D_hi].

        Closed-form Normal-Normal only — the generic path inverts in
        `theta`-space (`confidence_interval`), not in data-space. Calling
        `acceptance_region` against a non-Normal model raises.
        """
        if _is_normal_normal(model):
            return self._closed_form_acceptance_region(alpha, theta0, model)  # type: ignore[arg-type]
        raise NotImplementedError(
            f"WaldStatistic.acceptance_region (data-space) is only "
            f"available for NormalNormalModel; got {type(model).__name__}. "
            f"Use confidence_interval(...) for the generic theta-space inversion."
        )

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        if _is_normal_normal(model):
            return self._closed_form_confidence_interval(alpha, data, model)  # type: ignore[arg-type]
        return self._generic_confidence_interval(alpha, data, model)

    def accepts_tilting(self, tilting) -> bool:
        return getattr(tilting, "name", "") == "identity"
