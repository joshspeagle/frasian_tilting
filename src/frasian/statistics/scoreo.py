"""SCOREO statistic — Bayesian / posterior score test.

    tau_Scoreo(theta0; data, prior) = U_post(theta0)^2 / I_post(theta0)

where `U_post(theta) = d/dtheta log pi(theta|data)` is the
posterior score and `I_post(theta) = -d^2/dtheta^2 log
pi(theta|data)` is the observed posterior information
(Tierney-Kadane 1986). The WALDO-style counterpart of `score`.

On `NormalNormalModel + NormalDistribution` the log-posterior is
quadratic with constant curvature `1/sigma_n^2` and
`U_post(theta) = -(theta - mu_n)/sigma_n^2`, so

    tau_Scoreo(theta) == (theta - mu_n)^2 / sigma_n^2
                      == tau_WALDO(theta) == tau_LRTO(theta)

exactly (Derivation Step 3 of `docs/methods/scoreo.md`). The
closed-form NN+Normal fast path therefore reuses WALDO's
`Phi(b - a) + Phi(-a - b)` formula — same machinery as
`LRTOStatistic`.

The generic path computes `U_post = jax.grad(posterior.logpdf)
(theta_0)` and `I_post = -jax.hessian(posterior.logpdf)(theta_0)`
via JAX autodiff; no MAP-finding is required (cf. `lrto` which
needs `theta_MAP`). MC calibration mirrors WALDO/LRTO with CRN
seed discipline so brentq probes nest cleanly.

`scoreo.accepts_tilting(*)` returns `True` (prior-aware by
design), mirroring `waldo` and `lrto`.

See `docs/methods/scoreo.md` for the derivation.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._errors import BracketingFailed
from .._registry import register_statistic
from ..models._dispatch import is_normal_normal
from ..models.base import Model, Prior
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel, posterior_params  # noqa: F401
from .base import AsymptoticDistribution
from .waldo import (
    _closed_form_pvalue_scalar,
    _is_normal_normal_n1,
    _is_normal_normal_pair,
)

if TYPE_CHECKING:
    from ..tilting.base import TiltingScheme

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


@register_statistic(name="scoreo", brief="docs/methods/scoreo.md")
@dataclass(frozen=True)
class ScoreoStatistic:
    """Bayesian / posterior score statistic with closed-form NN+Normal
    fast path + generic MC numerical path.

    Generic-path knobs (`n_mc`, `seed`) mirror `WaldoStatistic` and
    `LRTOStatistic`. The closed-form NN+Normal path is mathematically
    identical to WALDO's (Derivation Step 3); `force_generic=True`
    runs the model-agnostic MC path for path-coverage debugging.
    """

    name: ClassVar[str] = "scoreo"
    asymptotic_null: AsymptoticDistribution = field(
        default_factory=lambda: AsymptoticDistribution(
            family="weighted_chi2",
            df=1,
            scale=1.0,
            description="U_post(theta)^2 / I_post(theta) = "
            "(mu_n - theta)^2 / sigma_n^2 ~ w * ncx2_1(lambda(theta)) "
            "(closed-form NN+Normal; same as WALDO/LRTO); generic: MC ref "
            "under H_0. NOT chi^2_1 in general — see scoreo.md Step 5.",
        )
    )

    n_mc: int = 2000
    seed: int = 0xC0FFEE
    force_generic: bool = False

    @property
    def cell_name(self) -> str:
        """Discriminator for cache key + manifest: `scoreo` (closed-form)
        vs `scoreo[generic]` (force_generic=True)."""
        return f"{self.name}[generic]" if self.force_generic else self.name

    # ---------- closed-form Normal-Normal+Normal path ----------
    #
    # tau_Scoreo == tau_WALDO == tau_LRTO exactly on NN+Normal
    # (Derivation Step 3). Reuse WALDO's module-level helpers
    # (_pvalue_components, _closed_form_pvalue_scalar) — the math
    # lives in one place.

    def _closed_form_evaluate(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: NormalNormalModel,
        prior: NormalDistribution,
    ) -> jax.Array:
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, sigma_n, _ = posterior_params(D, prior.loc, model.sigma, prior.scale)
        diff = mu_n - jnp.asarray(theta0, dtype=jnp.float64)
        return diff * diff / (sigma_n**2)

    def _closed_form_pvalue(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: NormalNormalModel,
        prior: NormalDistribution,
    ) -> NDArray[np.float64]:
        from scipy.special import ndtr as _ndtr

        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n_arr, _, w = posterior_params(D, prior.loc, model.sigma, prior.scale)
        mu_n = float(mu_n_arr)
        theta_arr = np.asarray(theta0, dtype=np.float64)
        a = np.abs(mu_n - theta_arr) / (w * model.sigma)
        b = (1.0 - w) * (prior.loc - theta_arr) / (w * model.sigma)
        return _ndtr(b - a) + _ndtr(-a - b)

    def _closed_form_confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: NormalNormalModel,
        prior: NormalDistribution,
    ) -> tuple[float, float]:
        from ..tilting._solvers import brentq_with_doubling

        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n_arr, _, w = posterior_params(D, prior.loc, model.sigma, prior.scale)
        mu_n_f = float(mu_n_arr)
        sigma = model.sigma
        mu0 = prior.loc

        def f(theta: float) -> float:
            return _closed_form_pvalue_scalar(theta, D, mu_n_f, w, mu0, sigma) - alpha

        half = 4.0 * sigma
        lower = brentq_with_doubling(f, midpoint=mu_n_f, initial_half_width=half, direction=-1)
        upper = brentq_with_doubling(f, midpoint=mu_n_f, initial_half_width=half, direction=+1)
        return (lower, upper)

    def _closed_form_acceptance_region(
        self,
        alpha: float,
        theta0: ArrayLike,
        model: NormalNormalModel,
        prior: NormalDistribution,
    ) -> tuple[jax.Array, jax.Array]:
        from ..tilting._solvers import brentq_with_doubling

        theta_arr = np.atleast_1d(np.asarray(theta0, dtype=np.float64))
        D_lo = np.empty_like(theta_arr)
        D_hi = np.empty_like(theta_arr)
        sigma = model.sigma
        mu0 = prior.loc
        for i, theta_val in enumerate(theta_arr):
            theta_f = float(theta_val)

            def f(D_val: float, _theta_f: float = theta_f) -> float:
                mu_n_arr_i, _, w_i = posterior_params(D_val, mu0, sigma, prior.scale)
                return _closed_form_pvalue_scalar(
                    _theta_f, D_val, float(mu_n_arr_i), w_i, mu0, sigma
                ) - alpha

            half = 4.0 * sigma
            D_lo[i] = brentq_with_doubling(
                f, midpoint=theta_f, initial_half_width=half, direction=-1
            )
            D_hi[i] = brentq_with_doubling(
                f, midpoint=theta_f, initial_half_width=half, direction=+1
            )
        return (
            jnp.asarray(D_lo if D_lo.size > 1 else D_lo.reshape(())),
            jnp.asarray(D_hi if D_hi.size > 1 else D_hi.reshape(())),
        )

    # ---------- generic model-agnostic path ----------

    @staticmethod
    def _is_gaussian_posterior(posterior) -> bool:
        """True iff `posterior` is a `NormalDistribution`.

        Same opt-in-by-class-identity discipline as `LRTOStatistic`
        (lrto skeptic finding #1): a future location-scale wrapper
        (Student-t, lognormal) that happens to have `.loc`/`.scale`
        attributes would NOT route through the WALDO-formula fast
        path. New Gaussian classes must explicitly join this check.
        """
        return isinstance(posterior, NormalDistribution)

    @staticmethod
    def _posterior_score_and_info(posterior, theta0_f: float) -> tuple[float, float]:
        """U_post(theta_0), I_post(theta_0) via JAX autodiff.

        Returns `(U, I)` as plain Python floats (the brentq closure
        and MC inner loops operate on scalars).

        `posterior.logpdf` must be JAX-traceable. For a Gaussian
        posterior the gradient is `-(theta - loc)/scale^2` and the
        Hessian is the constant `-1/scale^2`.
        """
        def _logpdf_scalar(th):
            return jnp.asarray(posterior.logpdf(th)).sum()

        theta = jnp.float64(theta0_f)
        U = float(jax.grad(_logpdf_scalar)(theta))
        # Observed info: -d^2/dtheta^2 log pi.
        I = float(-jax.grad(jax.grad(_logpdf_scalar))(theta))
        return U, I

    def _generic_evaluate(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
    ) -> jax.Array:
        if prior is None:
            raise ValueError("ScoreoStatistic.evaluate requires a prior (got None).")
        posterior = model.posterior(data, prior)
        # Skeptic finding #6: detect scalar input BEFORE atleast_1d so
        # the shape-restoration check matches `_generic_pvalue`'s pattern.
        theta_arr_input = np.asarray(theta0, dtype=np.float64)
        scalar_input = (theta_arr_input.ndim == 0)
        theta_arr = np.atleast_1d(theta_arr_input)
        tau = np.empty_like(theta_arr)
        worst_nonpos_I = float("inf")
        for i, t in enumerate(theta_arr):
            U, I = self._posterior_score_and_info(posterior, float(t))
            if not np.isfinite(I) or I <= 0.0:
                worst_nonpos_I = min(worst_nonpos_I, I if np.isfinite(I) else 0.0)
                tau[i] = np.nan
            else:
                tau[i] = (U * U) / I
        if worst_nonpos_I != float("inf"):
            # I_post non-positive at some test theta: posterior log-density
            # is locally concave-up / inflecting, which means theta is NOT
            # in a Gaussian-like basin. Surface this rather than silently
            # propagating NaN through the brentq closure (parallels
            # `score`'s I-guard and `lrto`'s wrong-MAP guard).
            warnings.warn(
                f"ScoreoStatistic._generic_evaluate: I_post(theta) <= 0 "
                f"(min={worst_nonpos_I:.3e}) — the posterior log-density "
                f"is locally not concave at the test theta, so tau_Scoreo "
                f"is undefined there. Returning NaN; downstream p-value "
                f"will be unusable at affected theta.",
                RuntimeWarning,
                stacklevel=2,
            )
        if scalar_input:
            return jnp.asarray(float(tau[0]))
        return jnp.asarray(tau)

    @staticmethod
    def _stable_seed(
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        alpha: float,
        base_seed: int,
    ) -> int:
        """CRN seed: cross-process stable, theta-INDEPENDENT.

        Same construction as `WaldoStatistic._stable_seed` /
        `LRTOStatistic._stable_seed`.
        """
        h = hashlib.blake2b(digest_size=8)
        h.update(np.ascontiguousarray(data, dtype=np.float64).tobytes())
        h.update(repr(model.fingerprint()).encode("utf-8"))
        h.update(repr(prior.fingerprint()).encode("utf-8"))
        h.update(np.float64(alpha).tobytes())
        h.update(np.int64(base_seed).tobytes())
        return int.from_bytes(h.digest()[:4], "little", signed=False)

    def _generic_mc_reference(
        self,
        theta0_f: float,
        n_obs: int,
        model: Model,
        prior: Prior,
        n_mc: int,
        derived_seed: int,
        *,
        is_gaussian: bool,
    ) -> NDArray[np.float64]:
        """Sample n_mc tau_Scoreo values under H_0 ~ likelihood(.|theta_0).

        `is_gaussian`: caller-supplied flag (hoisted at the
        CI/pvalue level, mirroring LRTO).

        - **Gaussian fast path**: for a Gaussian posterior
          `pi_{D'} = N(mu, sigma^2)`, `U_post(theta_0) =
          -(theta_0 - mu)/sigma^2`, `I_post = 1/sigma^2`, so
          `tau_Scoreo = (theta_0 - mu)^2 / sigma^2` — same numpy
          expression as WALDO/LRTO. Vectorised via
          `posterior_moments_batch`.

        - **Per-row path**: per-draw JAX-grad/hessian. Includes the
          same I-non-positivity guard as `_generic_evaluate`
          (RuntimeWarning + NaN row).

        Returns shape `(n_mc,)`.
        """
        from ..models.base import (
            posterior_moments_batch as _moments_batch,
            sample_data_batch as _sample_batch,
        )

        rng = np.random.default_rng(int(derived_seed))
        D_batch = _sample_batch(model, float(theta0_f), rng, int(n_mc), int(n_obs))

        if is_gaussian:
            mu_arr, var_arr = _moments_batch(model, D_batch, prior)
            finite = (var_arr > 0.0) & np.isfinite(var_arr) & np.isfinite(mu_arr)
            with np.errstate(invalid="ignore", divide="ignore"):
                diff = mu_arr - float(theta0_f)
                tau = diff * diff / np.where(finite, var_arr, 1.0)
            return np.where(finite, tau, np.nan)

        # Per-row JAX-autodiff path.
        tau = np.empty(int(n_mc), dtype=np.float64)
        any_nonpos_I = False
        for i in range(int(n_mc)):
            try:
                post_i = model.posterior(D_batch[i], prior)
                U_i, I_i = self._posterior_score_and_info(post_i, float(theta0_f))
                if not np.isfinite(I_i) or I_i <= 0.0:
                    tau[i] = np.nan
                    any_nonpos_I = True
                else:
                    tau_i = (U_i * U_i) / I_i
                    tau[i] = tau_i if np.isfinite(tau_i) else np.nan
            except Exception:
                tau[i] = np.nan
        if any_nonpos_I:
            warnings.warn(
                "ScoreoStatistic._generic_mc_reference: at least one MC "
                "draw produced I_post(theta_0) <= 0 (non-concave posterior). "
                "Returning NaN for those rows; the p-value uses the "
                "remaining finite draws.",
                RuntimeWarning,
                stacklevel=2,
            )
        return tau

    def _generic_pvalue(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
        *,
        derived_seed: int | None = None,
        obs_state: tuple[bool, object] | None = None,
    ) -> jax.Array:
        """MC empirical p-value with `(k+1)/(n+1)` continuity correction.

        `obs_state`: optional `(is_gaussian, posterior_obs)` tuple
        (theta-INDEPENDENT) hoisted out of the CI brentq loop so we
        don't re-detect / re-construct the posterior on every probe.
        """
        if prior is None:
            raise ValueError("ScoreoStatistic.pvalue requires a prior (got None).")
        data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        if data_arr.ndim != 1:
            raise NotImplementedError(
                "ScoreoStatistic.pvalue currently expects 1-D data; got "
                f"data.ndim={data_arr.ndim}."
            )
        n_obs = int(data_arr.size)

        if obs_state is None:
            posterior = model.posterior(data_arr, prior)
            is_gaussian = self._is_gaussian_posterior(posterior)
        else:
            is_gaussian, posterior = obs_state

        if derived_seed is None:
            derived_seed = self._stable_seed(data_arr, model, prior, 0.0, self.seed)

        theta_arr_input = np.asarray(theta0, dtype=np.float64)
        scalar_input = (theta_arr_input.ndim == 0)
        theta_arr = np.atleast_1d(theta_arr_input)

        # tau_obs(theta) = U_post(theta)^2 / I_post(theta) at each
        # test theta — uses the (theta-dependent) JAX-grad/hessian of
        # the OBSERVED posterior.
        #
        # Skeptic findings #1, #2, #3: previously this branch raised
        # ValueError on I_post(theta) <= 0, which leaked past the brentq
        # closure (only catches BracketingFailed) and crashed CI
        # inversion. Unified policy now matches `_generic_evaluate`'s
        # contract: set NaN + RuntimeWarning; downstream brentq's non-
        # finite-midpoint guard converts cleanly to BracketingFailed and
        # the CI-level UserWarning surfaces the failure.
        tau_obs = np.empty(theta_arr.shape, dtype=np.float64)
        obs_nonpos = False
        for i, t in enumerate(theta_arr):
            U_i, I_i = self._posterior_score_and_info(posterior, float(t))
            if not np.isfinite(I_i) or I_i <= 0.0:
                tau_obs[i] = np.nan
                obs_nonpos = True
            else:
                tau_obs[i] = (U_i * U_i) / I_i
        if obs_nonpos:
            warnings.warn(
                "ScoreoStatistic._generic_pvalue: observed posterior has "
                "I_post(theta) <= 0 at one or more test thetas (non-"
                "concave). Returning NaN for those entries; downstream "
                "CI inversion will fall back to model.support() with a "
                "secondary UserWarning.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Per-theta MC reference (each theta gets its own H_0
        # reference; CRN seed shared so brentq probes nest cleanly).
        p_out = np.empty(theta_arr.shape, dtype=np.float64)
        any_mc_collapse = False
        for i, theta_f in enumerate(theta_arr):
            if not np.isfinite(tau_obs[i]):
                # Already flagged; propagate NaN p.
                p_out[i] = np.nan
                continue
            tau_ref = self._generic_mc_reference(
                float(theta_f), n_obs, model, prior, self.n_mc, derived_seed,
                is_gaussian=is_gaussian,
            )
            finite = np.isfinite(tau_ref)
            n_eff = int(finite.sum())
            if n_eff == 0:
                # All MC draws produced I_post <= 0 — degenerate region.
                # Skeptic finding #3: previously raised, crashing CI.
                p_out[i] = np.nan
                any_mc_collapse = True
                continue
            n_more_extreme = float(np.sum(tau_ref[finite] >= tau_obs[i]))
            p_out[i] = (n_more_extreme + 1.0) / (float(n_eff) + 1.0)
        if any_mc_collapse:
            warnings.warn(
                "ScoreoStatistic._generic_pvalue: every MC reference draw "
                "produced a non-finite tau at one or more test thetas. "
                "Returning NaN; downstream CI inversion will fall back to "
                "model.support() with a secondary UserWarning.",
                RuntimeWarning,
                stacklevel=2,
            )

        if scalar_input:
            return jnp.asarray(float(p_out[0]))
        return jnp.asarray(p_out)

    def _generic_confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
    ) -> tuple[float, float]:
        if prior is None:
            raise ValueError(
                "ScoreoStatistic.confidence_interval requires a prior (got None)."
            )
        from ..tilting._solvers import brentq_with_doubling

        data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        if data_arr.ndim != 1:
            raise NotImplementedError(
                "ScoreoStatistic.confidence_interval expects 1-D data; got "
                f"data.ndim={data_arr.ndim}."
            )
        derived_seed = self._stable_seed(data_arr, model, prior, 0.0, self.seed)
        support_lo, support_hi = model.support()

        posterior = model.posterior(data, prior)
        is_gaussian = self._is_gaussian_posterior(posterior)
        var_post = float(np.asarray(posterior.var()))
        mu_post = float(np.asarray(posterior.mean()))
        sigma_post = float(np.sqrt(max(var_post, 1e-300)))
        obs_state = (is_gaussian, posterior)

        def f(theta: float) -> float:
            theta_safe = max(float(support_lo), min(float(support_hi), theta))
            return float(
                self._generic_pvalue(
                    theta_safe, data, model, prior,
                    derived_seed=derived_seed,
                    obs_state=obs_state,
                )
            ) - alpha

        half = max(4.0 * sigma_post, 1e-3)
        boundary_hit_lo = False
        boundary_hit_hi = False
        try:
            lower = brentq_with_doubling(
                f, midpoint=mu_post, initial_half_width=half, direction=-1
            )
        except BracketingFailed:
            lower = float(support_lo)
            boundary_hit_lo = True
        try:
            upper = brentq_with_doubling(
                f, midpoint=mu_post, initial_half_width=half, direction=+1
            )
        except BracketingFailed:
            upper = float(support_hi)
            boundary_hit_hi = True
        if boundary_hit_lo or boundary_hit_hi:
            sides = [s for s, hit in (("lower", boundary_hit_lo),
                                       ("upper", boundary_hit_hi)) if hit]
            warnings.warn(
                f"ScoreoStatistic._generic_confidence_interval: bracket "
                f"exhausted on the {' and '.join(sides)} side(s); "
                f"returning model.support() boundary at alpha={alpha!r}.",
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
        if not self.force_generic and _is_normal_normal_n1(model, prior, data):
            return self._closed_form_evaluate(theta0, data, model, prior)  # type: ignore[arg-type]
        return self._generic_evaluate(theta0, data, model, prior)

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        if not self.force_generic and _is_normal_normal_n1(model, prior, data):
            return jnp.asarray(self._closed_form_pvalue(theta0, data, model, prior))  # type: ignore[arg-type]
        return self._generic_pvalue(theta0, data, model, prior)

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Data-space accept region (closed-form NN+Normal only)."""
        if not self.force_generic and _is_normal_normal_pair(model, prior):
            return self._closed_form_acceptance_region(alpha, theta0, model, prior)  # type: ignore[arg-type]
        if self.force_generic and _is_normal_normal_pair(model, prior):
            raise NotImplementedError(
                "ScoreoStatistic.acceptance_region (data-space) has no generic "
                "path; got force_generic=True. Use confidence_interval(...) "
                "for the generic theta-space inversion, or unset force_generic."
            )
        raise NotImplementedError(
            f"ScoreoStatistic.acceptance_region (data-space) is only "
            f"available for the closed-form Normal-Normal pair; got "
            f"model={type(model).__name__}, prior={type(prior).__name__}."
        )

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        if not self.force_generic and _is_normal_normal_n1(model, prior, data):
            return self._closed_form_confidence_interval(alpha, data, model, prior)  # type: ignore[arg-type]
        return self._generic_confidence_interval(alpha, data, model, prior)

    def accepts_tilting(self, tilting: TiltingScheme) -> bool:
        return True
