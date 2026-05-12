"""LRTO statistic — Bayesian / posterior likelihood-ratio test.

    tau_LRTO(theta0; data, prior) = -2 [ log pi(theta0 | data)
                                         - log pi(theta_MAP | data) ]

where `pi(. | data)` is the posterior produced by
`model.posterior(data, prior)` and `theta_MAP = argmax pi(theta |
data)`. The WALDO-style counterpart of `lrt`.

On `NormalNormalModel + NormalDistribution` the log-posterior is
quadratic with curvature `1 / sigma_n^2` and `theta_MAP = mu_n`, so

    tau_LRTO(theta0)  ==  (mu_n - theta0)^2 / sigma_n^2  ==  tau_WALDO(theta0)

exactly (Derivation Step 3 of `docs/methods/lrto.md`). The closed-
form NN+Normal fast path therefore reuses WALDO's
`Phi(b - a) + Phi(-a - b)` formula (the module-level helpers
`_pvalue_components` and `_closed_form_pvalue_scalar` are imported
from `waldo` so the math lives in one place).

The generic path finds `theta_MAP` via `scipy.optimize.minimize_scalar`
on `-posterior.logpdf` over `model.support()`, then computes
`tau_LRTO` from the posterior log-density gap. MC calibration
(D2) mirrors WALDO's CRN seed discipline so brentq probes nest
cleanly across CI inversion.

`lrto.accepts_tilting(*)` returns `True` (prior-aware by design),
mirroring `waldo`'s contract. The (TiltingScheme x lrto) cell calls
`lrto` against the tilted posterior produced by `tilting.tilt(...)`.

See `docs/methods/lrto.md` for the derivation.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as _optimize
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

# Tolerance for the tau-non-negativity guard in `_generic_evaluate`
# (parallels `lrt.py`). A `tau` slightly below 0 is FP cancellation
# near the mode; significantly below 0 means the mode-finder returned
# a non-maximiser, which surfaces as a `RuntimeWarning` rather than
# silently being clamped.
_GENERIC_TAU_NEG_TOL = 1e-8


@register_statistic(name="lrto", brief="docs/methods/lrto.md")
@dataclass(frozen=True)
class LRTOStatistic:
    """Bayesian / posterior LRT statistic with closed-form NN+Normal fast
    path + generic Monte-Carlo numerical path.

    Generic-path knobs (`n_mc`, `seed`) mirror `WaldoStatistic`'s
    defaults. The closed-form NN+Normal path is mathematically
    identical to WALDO's (Derivation Step 3 of `docs/methods/lrto.md`);
    `force_generic=True` runs the model-agnostic MC path for path-
    coverage debugging — the two agree within MC noise on NN.
    """

    name: ClassVar[str] = "lrto"
    asymptotic_null: AsymptoticDistribution = field(
        default_factory=lambda: AsymptoticDistribution(
            family="weighted_chi2",
            df=1,
            scale=1.0,
            description="-2 [log pi(theta|D) - log pi(theta_MAP|D)] = "
            "(mu_n - theta)^2 / sigma_n^2 ~ w * ncx2_1(lambda(theta)) "
            "(closed-form NN+Normal; same as WALDO); generic: MC ref under H_0. "
            "NOT chi^2_1 in general — see lrto.md Derivation Step 5.",
        )
    )

    n_mc: int = 2000
    seed: int = 0xC0FFEE
    # Flips dispatch to the generic MC path on NN+Normal; see WALDO
    # `force_generic` for the path-coverage-debugging rationale.
    force_generic: bool = False

    @property
    def cell_name(self) -> str:
        """Discriminator for cache key + manifest: `lrto` (closed-form) vs
        `lrto[generic]` (force_generic=True)."""
        return f"{self.name}[generic]" if self.force_generic else self.name

    # ---------- closed-form Normal-Normal+Normal path ----------
    #
    # tau_LRTO == tau_WALDO exactly on NN+Normal (Derivation Step 3),
    # so the closed-form p-value, CI, and acceptance-region formulas
    # coincide with WALDO's. We reuse WALDO's module-level helpers
    # rather than re-implementing them to keep the math in one place.

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
    def _find_theta_map(posterior, support: tuple[float, float]) -> float:
        """Locate `argmax posterior.logpdf` within `support` (Derivation Step 7).

        Fast-path: any `Distribution` with a `loc` attribute (currently
        `NormalDistribution` and any future symmetric/unimodal family
        whose mode equals its location parameter) returns
        `float(loc)` directly. This skips the scipy.optimize call,
        which matters in the generic-MC reference loop where the
        mode-finder runs once per MC draw (n_mc=2000 calls per brentq
        probe; ~3-5 ms each makes the unoptimised path unusably slow).

        Slow-path: `scipy.optimize.minimize_scalar(..., method="bounded")`
        on `-logpdf`. For bounded supports the support endpoints are
        the bracket; for unbounded supports the bracket is
        `posterior.mean() +/- 10*sqrt(posterior.var())`. Assumes the
        (tilted) posterior is unimodal — see Failure modes.
        """
        if hasattr(posterior, "loc"):
            return float(np.asarray(posterior.loc))
        lo, hi = float(support[0]), float(support[1])
        finite = np.isfinite(lo) and np.isfinite(hi)
        if finite:
            res = _optimize.minimize_scalar(
                lambda th: -float(np.asarray(posterior.logpdf(np.float64(th)))),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-9},
            )
            return float(res.x)
        # Unbounded — warm-start at the posterior mean (exact for
        # Gaussian; reasonable for any unimodal continuous posterior).
        center = float(np.asarray(posterior.mean()))
        scale = float(np.sqrt(max(float(np.asarray(posterior.var())), 1e-12)))
        warm_lo = center - 10.0 * scale
        warm_hi = center + 10.0 * scale
        res = _optimize.minimize_scalar(
            lambda th: -float(np.asarray(posterior.logpdf(np.float64(th)))),
            bounds=(warm_lo, warm_hi),
            method="bounded",
            options={"xatol": 1e-9},
        )
        return float(res.x)

    def _generic_evaluate(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
    ) -> jax.Array:
        if prior is None:
            raise ValueError("LRTOStatistic.evaluate requires a prior (got None).")
        posterior = model.posterior(data, prior)
        theta_arr = np.asarray(theta0, dtype=np.float64)
        theta_map = self._find_theta_map(posterior, model.support())
        ll_theta = np.asarray(posterior.logpdf(np.asarray(theta_arr, dtype=np.float64)),
                              dtype=np.float64)
        ll_map = float(np.asarray(posterior.logpdf(np.float64(theta_map))))
        tau = -2.0 * (ll_theta - ll_map)
        # Non-negativity guard (parallels lrt._generic_evaluate). By
        # definition of theta_MAP, tau >= 0; significantly negative tau
        # means the mode-finder returned a non-maximiser.
        worst = float(np.nanmin(tau)) if tau.size else 0.0
        if worst < -_GENERIC_TAU_NEG_TOL:
            warnings.warn(
                f"LRTOStatistic._generic_evaluate: tau={worst:.3e} < "
                f"-{_GENERIC_TAU_NEG_TOL:.0e}; this implies logpdf(theta) > "
                f"logpdf(theta_MAP) and the mode-finder did not converge "
                f"to the true MAP. Clamping to 0; downstream p-value/CI "
                f"will be incorrect.",
                RuntimeWarning,
                stacklevel=2,
            )
        return jnp.asarray(np.maximum(tau, 0.0))

    @staticmethod
    def _stable_seed(
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        alpha: float,
        base_seed: int,
    ) -> int:
        """CRN seed: cross-process stable, theta-INDEPENDENT.

        Same construction as `WaldoStatistic._stable_seed` — hash
        (data, model.fingerprint, prior.fingerprint, alpha, base_seed)
        via blake2b. Sharing the seed across brentq probes makes the
        MC empirical p-value a deterministic function of theta.
        """
        h = hashlib.blake2b(digest_size=8)
        h.update(np.ascontiguousarray(data, dtype=np.float64).tobytes())
        h.update(repr(model.fingerprint()).encode("utf-8"))
        h.update(repr(prior.fingerprint()).encode("utf-8"))
        h.update(np.float64(alpha).tobytes())
        h.update(np.int64(base_seed).tobytes())
        return int.from_bytes(h.digest()[:4], "little", signed=False)

    @staticmethod
    def _posterior_is_gaussian_like(model: Model, prior: Prior) -> bool:
        """Detect whether `model.posterior(data, prior)` returns a
        location-scale Gaussian-style Distribution (has `.loc` and
        `.scale`). When True, the MC reference can use the vectorised
        fast path that mirrors WALDO's `posterior_moments_batch`
        machinery: `tau_LRTO = (mu_n - theta_0)^2 / sigma_n^2` (which
        equals WALDO's `t` on NN). ~50-200x faster than the
        per-row Python loop on NormalNormalModel + NormalDistribution.
        """
        try:
            arr = np.zeros(1, dtype=np.float64)
            probe = model.posterior(arr, prior)
        except Exception:
            return False
        return hasattr(probe, "loc") and hasattr(probe, "scale")

    def _generic_mc_reference(
        self,
        theta0_f: float,
        n_obs: int,
        model: Model,
        prior: Prior,
        n_mc: int,
        derived_seed: int,
    ) -> NDArray[np.float64]:
        """Sample n_mc tau values under H_0 ~ likelihood(.|theta_0).

        Two paths:

        - **Gaussian-posterior fast path** (`posterior` returns a
          Distribution with `.loc` and `.scale`): vectorised via
          `posterior_moments_batch`. For a Gaussian posterior
          `pi(.|D') = N(mu, sigma^2)`, `theta_MAP_{D'} = mu` and

              tau_LRTO = -2 [log pi(theta_0) - log pi(mu)]
                       = (mu - theta_0)^2 / sigma^2

          which is the same single-line numpy expression as WALDO's
          `t`. ~50-200x speedup on NN at n_mc=2000.

        - **Generic path**: per-row Python loop with
          `scipy.optimize.minimize_scalar` for `theta_MAP` and
          per-row `logpdf` evaluations. O(n_mc * mode_find +
          2 * logpdf) — slow for non-conjugate posteriors.

        Returns shape `(n_mc,)` with NaN for any draw that produces a
        non-finite posterior summary.
        """
        from ..models.base import (
            posterior_moments_batch as _moments_batch,
            sample_data_batch as _sample_batch,
        )

        rng = np.random.default_rng(int(derived_seed))
        D_batch = _sample_batch(model, float(theta0_f), rng, int(n_mc), int(n_obs))

        if self._posterior_is_gaussian_like(model, prior):
            mu_arr, var_arr = _moments_batch(model, D_batch, prior)
            finite = (var_arr > 0.0) & np.isfinite(var_arr) & np.isfinite(mu_arr)
            with np.errstate(invalid="ignore", divide="ignore"):
                diff = mu_arr - float(theta0_f)
                tau = diff * diff / np.where(finite, var_arr, 1.0)
            return np.where(finite, tau, np.nan)

        # Generic (non-Gaussian-posterior) path.
        support = model.support()
        tau = np.empty(int(n_mc), dtype=np.float64)
        for i in range(int(n_mc)):
            try:
                post_i = model.posterior(D_batch[i], prior)
                map_i = self._find_theta_map(post_i, support)
                ll0 = float(np.asarray(post_i.logpdf(np.float64(theta0_f))))
                ll_map = float(np.asarray(post_i.logpdf(np.float64(map_i))))
                tau_i = -2.0 * (ll0 - ll_map)
                if not np.isfinite(tau_i):
                    tau[i] = np.nan
                else:
                    tau[i] = max(tau_i, 0.0)
            except Exception:
                tau[i] = np.nan
        return tau

    def _generic_pvalue(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
        *,
        derived_seed: int | None = None,
        obs_state: tuple[float, NDArray[np.float64]] | None = None,
    ) -> jax.Array:
        """MC empirical p-value with `(k+1)/(n+1)` continuity correction.

        `obs_state`: optional `(theta_map_obs, logpdf_at_map_obs)` —
        theta-INDEPENDENT, hoisted out of the CI brentq loop to avoid
        re-finding the MAP at every probe.
        """
        if prior is None:
            raise ValueError("LRTOStatistic.pvalue requires a prior (got None).")
        data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        if data_arr.ndim != 1:
            raise NotImplementedError(
                "LRTOStatistic.pvalue currently expects 1-D data; got "
                f"data.ndim={data_arr.ndim}."
            )
        n_obs = int(data_arr.size)

        # Hoist observed MAP + log-density-at-MAP (theta-independent).
        if obs_state is None:
            posterior = model.posterior(data_arr, prior)
            theta_map_obs = self._find_theta_map(posterior, model.support())
            ll_map_obs = float(np.asarray(posterior.logpdf(np.float64(theta_map_obs))))
        else:
            theta_map_obs, ll_map_obs_arr = obs_state
            ll_map_obs = float(ll_map_obs_arr)
            posterior = model.posterior(data_arr, prior)
        if not np.isfinite(ll_map_obs):
            raise ValueError(
                "LRTOStatistic._generic_pvalue: posterior.logpdf at the MAP is "
                "non-finite; cannot define tau_LRTO. data.shape="
                f"{data_arr.shape!r}."
            )

        if derived_seed is None:
            derived_seed = self._stable_seed(data_arr, model, prior, 0.0, self.seed)

        theta_arr_input = np.asarray(theta0, dtype=np.float64)
        scalar_input = (theta_arr_input.ndim == 0)
        theta_arr = np.atleast_1d(theta_arr_input)
        # tau_obs(theta) = -2 [log pi(theta|D) - log pi(theta_MAP_obs|D)]
        ll_theta_obs = np.asarray(
            posterior.logpdf(np.asarray(theta_arr, dtype=np.float64)),
            dtype=np.float64,
        )
        tau_obs = np.maximum(-2.0 * (ll_theta_obs - ll_map_obs), 0.0)

        # Per-theta MC reference (each theta gets its own H_0 reference;
        # CRN seed shared so brentq probes nest cleanly).
        p_out = np.empty(theta_arr.shape, dtype=np.float64)
        for i, theta_f in enumerate(theta_arr):
            tau_ref = self._generic_mc_reference(
                float(theta_f), n_obs, model, prior, self.n_mc, derived_seed
            )
            finite = np.isfinite(tau_ref)
            n_eff = int(finite.sum())
            if n_eff == 0:
                raise ValueError(
                    f"LRTOStatistic._generic_pvalue: every MC reference draw "
                    f"produced a non-finite tau at theta0={float(theta_f)!r}."
                )
            n_more_extreme = float(np.sum(tau_ref[finite] >= tau_obs[i]))
            p_out[i] = (n_more_extreme + 1.0) / (float(n_eff) + 1.0)

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
                "LRTOStatistic.confidence_interval requires a prior (got None)."
            )
        from ..tilting._solvers import brentq_with_doubling

        data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        if data_arr.ndim != 1:
            raise NotImplementedError(
                "LRTOStatistic.confidence_interval expects 1-D data; got "
                f"data.ndim={data_arr.ndim}."
            )
        # CRN seed (theta- AND alpha-independent so nested CIs share draws).
        derived_seed = self._stable_seed(data_arr, model, prior, 0.0, self.seed)
        support_lo, support_hi = model.support()

        # Hoisted observed MAP — theta-INDEPENDENT, computed once.
        posterior = model.posterior(data, prior)
        theta_map_obs = self._find_theta_map(posterior, model.support())
        ll_map_obs = float(np.asarray(posterior.logpdf(np.float64(theta_map_obs))))
        var_post = float(np.asarray(posterior.var()))
        sigma_post = float(np.sqrt(max(var_post, 1e-300)))
        obs_state = (theta_map_obs, np.float64(ll_map_obs))

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
                f, midpoint=theta_map_obs, initial_half_width=half, direction=-1
            )
        except BracketingFailed:
            lower = float(support_lo)
            boundary_hit_lo = True
        try:
            upper = brentq_with_doubling(
                f, midpoint=theta_map_obs, initial_half_width=half, direction=+1
            )
        except BracketingFailed:
            upper = float(support_hi)
            boundary_hit_hi = True
        if boundary_hit_lo or boundary_hit_hi:
            sides = [s for s, hit in (("lower", boundary_hit_lo),
                                       ("upper", boundary_hit_hi)) if hit]
            warnings.warn(
                f"LRTOStatistic._generic_confidence_interval: bracket "
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
        """Data-space accept region for theta0 at level alpha.

        n=1 closed form on NN+Normal only; no generic data-space path.
        Mirrors `WaldoStatistic.acceptance_region` exactly.
        """
        if not self.force_generic and _is_normal_normal_pair(model, prior):
            return self._closed_form_acceptance_region(alpha, theta0, model, prior)  # type: ignore[arg-type]
        if self.force_generic and _is_normal_normal_pair(model, prior):
            raise NotImplementedError(
                "LRTOStatistic.acceptance_region (data-space) has no generic "
                "path; got force_generic=True. Use confidence_interval(...) "
                "for the generic theta-space inversion, or unset force_generic "
                "to recover the closed-form NN+Normal data-space region."
            )
        raise NotImplementedError(
            f"LRTOStatistic.acceptance_region (data-space) is only "
            f"available for the closed-form Normal-Normal pair; got "
            f"model={type(model).__name__}, prior={type(prior).__name__}. "
            f"Use confidence_interval(...) for the generic theta-space inversion."
        )

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        if not self.force_generic and _is_normal_normal_n1(model, prior, data):
            return self._closed_form_confidence_interval(alpha, data, model, prior)  # type: ignore[arg-type]
        return self._generic_confidence_interval(alpha, data, model, prior)

    def accepts_tilting(self, tilting: TiltingScheme) -> bool:
        return True
