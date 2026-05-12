"""Likelihood-ratio test statistic — pure-likelihood, prior-ignoring CI.

  tau_LRT(theta) = -2 * [ log L(theta) - log L(theta_hat) ]
  Asymptotic null: tau_LRT ~ chi^2_1 under H0 (Wilks 1938).

On `NormalNormalModel` the loglikelihood is exactly quadratic with
curvature `1/sigma^2`, so the LRT collapses identically (not just
asymptotically) to Wald:

  tau_LRT(theta)  ==  ((D - theta)/sigma)^2  ==  tau_Wald(theta).

The closed-form Normal-Normal fast path therefore dispatches into
Wald's NN math; the generic path computes `tau_LRT` from
`model.likelihood(data).loglik(...)` calls and uses `chi^2_1`
calibration with brentq inversion.

`lrt` is the frequentist half of the LRT pair (the Bayesian /
tilted-posterior half is `lrto`). It ignores the prior and only
accepts `identity` tilting, matching `wald`'s contract.

See `docs/methods/lrt.md` for the derivation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr as _ndtr, ndtri as _ndtri

# Tolerance for the tau-non-negativity guard in `_generic_evaluate`.
# A `tau` between `-_GENERIC_TAU_NEG_TOL` and 0 is silently clamped (FP
# cancellation near the mode); a `tau` below `-_GENERIC_TAU_NEG_TOL`
# triggers a `RuntimeWarning` because it implies `loglik(theta) >
# loglik(theta_hat)` — i.e. `model.mle` did not actually maximise the
# likelihood (skeptic finding #3).
_GENERIC_TAU_NEG_TOL = 1e-8

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._registry import register_statistic
from ..models._dispatch import is_normal_normal
from ..models.base import Model, Prior
from ..models.normal_normal import NormalNormalModel  # noqa: F401  (legacy field-access)
from .base import AsymptoticDistribution

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _is_normal_normal(model: Model) -> bool:
    return is_normal_normal(model)


@register_statistic(name="lrt", brief="docs/methods/lrt.md")
@dataclass(frozen=True)
class LRTStatistic:
    """Likelihood-ratio statistic with closed-form NN fast path + generic default.

    The NN closed-form path is *identical* to Wald's by Derivation Step 3
    of `docs/methods/lrt.md` (`tau_LRT == tau_Wald` exactly on NN). The
    generic default path computes `tau = -2 [loglik(theta_hat) - loglik(theta)]`
    using `model.mle`, `model.likelihood(data).loglik`, and chi^2(1)
    calibration; it works against any `Model` with a one-dim parameter and
    a unimodal loglikelihood. Cross-checks in
    `tests/regression/test_lrt_matches_wald.py` pin agreement of the
    closed-form and generic paths within numerical tolerance on NN.
    """

    name: ClassVar[str] = "lrt"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="chi2",
        df=1,
        scale=1.0,
        description="-2 [loglik(theta_hat) - loglik(theta)] ~ chi^2_1 under H0 "
        "(Wilks 1938); exact on NormalNormalModel.",
    )
    # See `WaldStatistic.force_generic` — flag exists for path-coverage
    # debugging only; the two paths agree to within numerical tolerance
    # on NN, so production CI estimation should leave this False.
    force_generic: bool = False

    @property
    def cell_name(self) -> str:
        """Discriminator for the cache key + manifest. Default closed-form
        cell is `lrt`; `force_generic=True` flips it to `lrt[generic]`."""
        return f"{self.name}[generic]" if self.force_generic else self.name

    # ---------- closed-form Normal-Normal path ----------
    #
    # tau_LRT(theta) == ((D - theta)/sigma)^2 == tau_Wald(theta) exactly
    # on NormalNormalModel (Derivation Step 3, docs/methods/lrt.md). The
    # closed-form NN pvalue / CI / acceptance-region formulas are
    # therefore identical to Wald's; we reproduce them here rather than
    # importing WaldStatistic to keep the call graph flat and to avoid
    # circular re-dispatch.

    def _closed_form_evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: NormalNormalModel
    ) -> jax.Array:
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = (D - jnp.asarray(theta0, dtype=jnp.float64)) / model.sigma
        return z * z

    def _closed_form_pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: NormalNormalModel
    ) -> NDArray[np.float64]:
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = np.abs(D - np.asarray(theta0, dtype=np.float64)) / model.sigma
        return 2.0 * (1.0 - _ndtr(z))

    def _closed_form_acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: NormalNormalModel
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        z_crit = float(_ndtri(1.0 - alpha / 2.0))
        theta_arr = np.asarray(theta0, dtype=np.float64)
        return (theta_arr - z_crit * model.sigma, theta_arr + z_crit * model.sigma)

    def _closed_form_confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: NormalNormalModel
    ) -> tuple[float, float]:
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = float(_ndtri(1.0 - alpha / 2.0))
        half = z * model.sigma
        return (D - half, D + half)

    # ---------- generic model-agnostic path ----------

    def _generic_evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model
    ) -> jax.Array:
        theta_arr = jnp.asarray(theta0, dtype=jnp.float64)
        mle = jnp.asarray(model.mle(data))
        lik = model.likelihood(data)
        ll_theta = jnp.asarray(lik.loglik(theta_arr))
        ll_mle = jnp.asarray(lik.loglik(mle))
        tau = -2.0 * (ll_theta - ll_mle)
        # Non-negativity guard. By definition of the MLE,
        # `ll_theta <= ll_mle`, so `tau >= 0`. FP cancellation near
        # the mode produces small negatives that are safe to clamp;
        # a *significantly* negative `tau` means `model.mle` did not
        # actually maximise the likelihood — surface that as a warning
        # rather than silently returning a wrong p-value.
        tau_np = np.asarray(tau, dtype=np.float64)
        worst = float(np.nanmin(tau_np)) if tau_np.size else 0.0
        if worst < -_GENERIC_TAU_NEG_TOL:
            warnings.warn(
                f"LRTStatistic._generic_evaluate: tau={worst:.3e} < "
                f"-{_GENERIC_TAU_NEG_TOL:.0e}; this implies loglik(theta) > "
                f"loglik(mle) and model.mle is not returning the true MLE. "
                f"Clamping to 0; downstream p-value/CI will be incorrect.",
                RuntimeWarning,
                stacklevel=2,
            )
        return jnp.maximum(tau, 0.0)

    def _generic_pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model
    ) -> jax.Array:
        tau = self._generic_evaluate(theta0, data, model)
        # chi^2_1 survival function.
        return 1.0 - jsp_stats.chi2.cdf(tau, 1)

    def _generic_confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model
    ) -> tuple[float, float]:
        # brentq inversion at the public CI-inversion boundary.
        from .._errors import BracketingFailed
        from ..tilting._solvers import brentq_with_doubling

        mle = float(np.asarray(model.mle(data)))
        support_lo, support_hi = model.support()

        def f(theta: float) -> float:
            # Clamp to model support — same pattern as Wald._generic_confidence_interval.
            theta_safe = max(float(support_lo), min(float(support_hi), theta))
            return float(self._generic_pvalue(theta_safe, data, model)) - alpha

        # Bracket-width estimation — mirrors `WaldStatistic._generic_confidence_interval`.
        # Three regimes:
        # 1. I(mle) finite & > 0 & not absurd: 4 / sqrt(I(mle)) is the
        #    natural Wald half-width. LRT and Wald share the quadratic
        #    approximation *asymptotically* (Derivation Step 2:
        #    `tau_LRT = tau_Wald + O_p(n^{-1/2})`); on the NN sandbox
        #    they coincide exactly. Off-NN at small n with a skewed
        #    loglikelihood the LRT level set can be asymmetric around
        #    the MLE, so the same half-width on both sides is a
        #    starting guess that brentq's doubling will expand as
        #    needed — slow but correct.
        # 2. I(mle) singular (bounded-support boundary): fall back to a
        #    fraction of the model support range.
        # 3. Numerical pathologies (NaN, negative): default to 1.0.
        info_at_mle = float(np.asarray(model.fisher_information(mle)))
        if np.isfinite(support_lo) and np.isfinite(support_hi):
            support_width = max(float(support_hi) - float(support_lo), 1e-6)
            width_cap = support_width / 2.0
        else:
            width_cap = float("inf")
        if info_at_mle > 0 and np.isfinite(info_at_mle):
            half_from_fisher = 4.0 / np.sqrt(info_at_mle)
        else:
            half_from_fisher = 1.0
        half = min(max(half_from_fisher, 1e-3), width_cap)
        try:
            lower = brentq_with_doubling(
                f, midpoint=mle, initial_half_width=half, direction=-1
            )
        except BracketingFailed:
            lower = float(support_lo)
        try:
            upper = brentq_with_doubling(
                f, midpoint=mle, initial_half_width=half, direction=+1
            )
        except BracketingFailed:
            upper = float(support_hi)
        return (
            max(lower, float(support_lo)),
            min(upper, float(support_hi)),
        )

    # ---------- public protocol surface (dispatches) ----------

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        if not self.force_generic and _is_normal_normal(model):
            return self._closed_form_evaluate(theta0, data, model)  # type: ignore[arg-type]
        return self._generic_evaluate(theta0, data, model)

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        if not self.force_generic and _is_normal_normal(model):
            return self._closed_form_pvalue(theta0, data, model)  # type: ignore[arg-type]
        return self._generic_pvalue(theta0, data, model)

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Return (D_lo, D_hi) such that LRT accepts H0 iff D in [D_lo, D_hi].

        Closed-form Normal-Normal only — the generic path inverts in
        `theta`-space (`confidence_interval`), not in data-space. Calling
        `acceptance_region` against a non-Normal model — or against an
        NN model with `force_generic=True` — raises, mirroring Wald.
        """
        if not self.force_generic and _is_normal_normal(model):
            return self._closed_form_acceptance_region(alpha, theta0, model)  # type: ignore[arg-type]
        if self.force_generic and _is_normal_normal(model):
            raise NotImplementedError(
                "LRTStatistic.acceptance_region (data-space) has no generic "
                "path; got force_generic=True. Use confidence_interval(...) "
                "for the generic theta-space inversion, or unset force_generic "
                "to recover the closed-form NN data-space region."
            )
        raise NotImplementedError(
            f"LRTStatistic.acceptance_region (data-space) is only "
            f"available for NormalNormalModel; got {type(model).__name__}. "
            f"Use confidence_interval(...) for the generic theta-space inversion."
        )

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        if not self.force_generic and _is_normal_normal(model):
            return self._closed_form_confidence_interval(alpha, data, model)  # type: ignore[arg-type]
        return self._generic_confidence_interval(alpha, data, model)

    def accepts_tilting(self, tilting) -> bool:
        return getattr(tilting, "name", "") == "identity"
