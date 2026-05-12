"""Rao score statistic — third member of the asymptotic trinity.

  tau_Score(theta) = U(theta)^2 / I(theta)
  U(theta) = d/dtheta log L(theta; data)   (score function)
  I(theta) = model.fisher_information(theta)
  Asymptotic null: tau_Score ~ chi^2_1 under H0 (Rao 1948).

On `NormalNormalModel` the loglikelihood is exactly quadratic, so the
score test collapses identically (not just asymptotically) to Wald
and LRT:

  tau_Score(theta) == ((D - theta)/sigma)^2 == tau_Wald == tau_LRT.

The closed-form Normal-Normal fast path dispatches into Wald's NN
math; the generic path computes `tau_Score` via `jax.grad` on
`model.likelihood(data).loglik` for U and
`model.fisher_information(theta)` for I, with `chi^2_1` calibration
and brentq inversion.

`score` is the frequentist half of the `(score, scoreo)` pair (the
Bayesian / posterior-score half is `scoreo`). It ignores the prior
and only accepts `identity` tilting, matching `wald` and `lrt`'s
contracts.

See `docs/methods/score.md` for the derivation.
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

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._registry import register_statistic
from ..models._dispatch import is_normal_normal
from ..models.base import Model, Prior
from ..models.normal_normal import NormalNormalModel  # noqa: F401  (legacy field-access)
from .base import AsymptoticDistribution

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _is_normal_normal(model: Model) -> bool:
    return is_normal_normal(model)


@register_statistic(name="score", brief="docs/methods/score.md")
@dataclass(frozen=True)
class ScoreStatistic:
    """Rao score statistic with closed-form NN fast path + generic default.

    The NN closed-form path is *identical* to Wald's by Derivation Step 2
    of `docs/methods/score.md` (`tau_Score == tau_Wald` exactly on NN). The
    generic default path computes
    `tau = U(theta)^2 / I(theta)` using `jax.grad` on
    `model.likelihood(data).loglik` for U and
    `model.fisher_information(theta)` for I, with `chi^2_1` calibration.
    The score statistic's main advantage over LRT/Wald is that it
    does **not** require `model.mle` — it evaluates entirely at
    `theta_0` (Engle 1984 §3.1).
    """

    name: ClassVar[str] = "score"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="chi2",
        df=1,
        scale=1.0,
        description="U(theta)^2 / I(theta) ~ chi^2_1 under H0 "
        "(Rao 1948); exact on NormalNormalModel by the trinity-collapse "
        "argument (Derivation Step 2 of docs/methods/score.md).",
    )
    # See `WaldStatistic.force_generic` — flag exists for path-coverage
    # debugging only; the two paths agree to within numerical tolerance
    # on NN, so production CI estimation should leave this False.
    force_generic: bool = False

    @property
    def cell_name(self) -> str:
        """Discriminator for the cache key + manifest. Default closed-form
        cell is `score`; `force_generic=True` flips it to `score[generic]`."""
        return f"{self.name}[generic]" if self.force_generic else self.name

    # ---------- closed-form Normal-Normal path ----------
    #
    # tau_Score(theta) == ((D - theta)/sigma)^2 == tau_Wald(theta) exactly
    # on NormalNormalModel (Derivation Step 2, docs/methods/score.md). The
    # closed-form NN pvalue / CI / acceptance-region formulas are
    # therefore identical to Wald's and LRT's; we reproduce them here
    # rather than importing the other statistics to keep the call graph
    # flat and to avoid circular re-dispatch.

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
    #
    # tau_Score = U(theta)^2 / I(theta) with
    #   U = jax.grad(loglik)(theta)
    #   I = model.fisher_information(theta)
    # No MLE needed (the score's headline advantage over Wald/LRT).

    @staticmethod
    def _score_value(
        theta: jax.Array, lik
    ) -> jax.Array:
        """U(theta) = d/dtheta loglik(theta; data) via JAX autodiff.

        `lik` is a `Likelihood` object built from `model.likelihood(data)`;
        the data are closed over so `jax.grad` differentiates w.r.t. theta.
        """
        return jax.grad(lambda th: jnp.asarray(lik.loglik(th)).sum())(theta)

    def _generic_evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model
    ) -> jax.Array:
        theta_arr = jnp.asarray(theta0, dtype=jnp.float64)
        lik = model.likelihood(data)
        # `jax.grad` requires a scalar input -> scalar output. Use `vmap`
        # for array inputs.
        if theta_arr.ndim == 0:
            U = self._score_value(theta_arr, lik)
        else:
            U = jax.vmap(lambda th: self._score_value(th, lik))(theta_arr)
        I = jnp.asarray(model.fisher_information(theta_arr), dtype=jnp.float64)
        # Non-negativity: U^2 >= 0 and I > 0 by regularity, so tau >= 0
        # exactly. No FP cancellation guard needed (unlike lrt where
        # tau = -2[ll(theta) - ll(mle)] can produce tiny negatives near
        # the mode).
        #
        # I-degeneracy guard (skeptic finding #5): `model.fisher_information`
        # is contractually `> 0` and finite. A non-positive or non-finite
        # value would make `U^2/I` `inf`/`nan`/negative; surface that as a
        # warning rather than letting a chi^2 survival of `nan` produce a
        # silent `p = 0` (or worse, NaN that propagates through brentq).
        I_np = np.asarray(I, dtype=np.float64)
        if I_np.size and (np.any(~np.isfinite(I_np)) or np.any(I_np <= 0.0)):
            warnings.warn(
                f"ScoreStatistic._generic_evaluate: model.fisher_information "
                f"returned a non-positive or non-finite value (min={float(np.min(I_np)):.3e}). "
                f"This violates the regularity-condition contract; tau and "
                f"the downstream p-value/CI will be incorrect.",
                RuntimeWarning,
                stacklevel=2,
            )
        return (U * U) / I

    def _generic_pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model
    ) -> jax.Array:
        tau = self._generic_evaluate(theta0, data, model)
        # chi^2_1 survival function.
        return 1.0 - jsp_stats.chi2.cdf(tau, 1)

    def _generic_confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model
    ) -> tuple[float, float]:
        from .._errors import BracketingFailed
        from ..tilting._solvers import brentq_with_doubling

        # The score doesn't require MLE for its statistic computation,
        # but it DOES need a brentq seed midpoint. Use `model.mle(data)`
        # as the bracket centre — this matches Wald/LRT and is the
        # natural choice since the score CI is symmetric around the
        # zero of U on regular models.
        mle = float(np.asarray(model.mle(data)))
        support_lo, support_hi = model.support()

        def f(theta: float) -> float:
            theta_safe = max(float(support_lo), min(float(support_hi), theta))
            return float(self._generic_pvalue(theta_safe, data, model)) - alpha

        # Same bracket-width estimation as LRT: 4/sqrt(I(mle)) is the
        # Wald half-width, which the score test shares asymptotically
        # (and exactly on NN).
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
        boundary_hit_lo = False
        boundary_hit_hi = False
        try:
            lower = brentq_with_doubling(
                f, midpoint=mle, initial_half_width=half, direction=-1
            )
        except BracketingFailed:
            # Skeptic finding #4: previously a silent fallback to
            # `support_lo`; now warn so callers can annotate metadata
            # with the boundary-hit (mirrors waldo's behaviour).
            lower = float(support_lo)
            boundary_hit_lo = True
        try:
            upper = brentq_with_doubling(
                f, midpoint=mle, initial_half_width=half, direction=+1
            )
        except BracketingFailed:
            upper = float(support_hi)
            boundary_hit_hi = True
        if boundary_hit_lo or boundary_hit_hi:
            sides = [s for s, hit in (("lower", boundary_hit_lo),
                                       ("upper", boundary_hit_hi)) if hit]
            warnings.warn(
                f"ScoreStatistic._generic_confidence_interval: bracket "
                f"exhausted on the {' and '.join(sides)} side(s); "
                f"returning model.support() boundary at alpha={alpha!r}. "
                f"This may be a true open CI or a numerical pathology "
                f"(e.g. flat score over a wide region, or score CI off-"
                f"centre from the MLE on a non-NN model).",
                UserWarning,
                stacklevel=2,
            )
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
            return jnp.asarray(self._closed_form_pvalue(theta0, data, model))  # type: ignore[arg-type]
        return self._generic_pvalue(theta0, data, model)

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Return (D_lo, D_hi) such that Score accepts H0 iff D in [D_lo, D_hi].

        Closed-form Normal-Normal only — the generic path inverts in
        `theta`-space (`confidence_interval`), not in data-space. Calling
        `acceptance_region` against a non-Normal model — or against an
        NN model with `force_generic=True` — raises, mirroring Wald/LRT.
        """
        if not self.force_generic and _is_normal_normal(model):
            return self._closed_form_acceptance_region(alpha, theta0, model)  # type: ignore[arg-type]
        if self.force_generic and _is_normal_normal(model):
            raise NotImplementedError(
                "ScoreStatistic.acceptance_region (data-space) has no generic "
                "path; got force_generic=True. Use confidence_interval(...) "
                "for the generic theta-space inversion, or unset force_generic "
                "to recover the closed-form NN data-space region."
            )
        raise NotImplementedError(
            f"ScoreStatistic.acceptance_region (data-space) is only "
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
