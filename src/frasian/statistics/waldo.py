"""WALDO statistic — Bayesian-frequentist hybrid via the posterior mean.

The closed-form Normal-Normal+Normal p-value (Theorem 3 in the legacy
derivations, ported verbatim from `legacy/src/frasian/waldo.py:115`):

  a(theta) = |mu_n - theta| / (w * sigma)
  b(theta) = (1 - w) * (mu0 - theta) / (w * sigma)
  p(theta) = Phi(b - a) + Phi(-a - b)

The generic model-agnostic path uses

  t(D, theta) = (mu_post - theta)^2 / sigma_post^2
  p(theta; D) = MC tail probability under H_0 ~ likelihood(.|theta_0)

and inverts via `brentq_with_doubling` on theta-space. Works against
any `Model` with `sample_data`, `posterior`, and any `Prior` that
`model.posterior(data, prior)` accepts. Cross-check vs the closed
form on Normal-Normal lives in
`tests/regression/test_waldo_generic_matches_closed_form.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._registry import register_statistic
from ..models.base import Model, Prior
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel, posterior_params
from .base import AsymptoticDistribution

if TYPE_CHECKING:
    from ..tilting.base import TiltingScheme

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _is_normal_normal_pair(model: Model, prior: Prior | None) -> bool:
    return isinstance(model, NormalNormalModel) and isinstance(prior, NormalDistribution)


def _pvalue_components(
    theta: jax.Array, mu_n: float, mu0: float, w: float, sigma: float
) -> tuple[jax.Array, jax.Array]:
    """a(theta), b(theta) from the closed-form WALDO p-value formula."""
    a = jnp.abs(mu_n - theta) / (w * sigma)
    b = (1.0 - w) * (mu0 - theta) / (w * sigma)
    return a, b


@register_statistic(name="waldo", brief="docs/methods/waldo.md")
@dataclass(frozen=True)
class WaldoStatistic:
    """WALDO statistic with closed-form Normal-Normal+Normal fast path
    + generic Monte-Carlo numerical path.

    Generic path knobs (`n_mc`, `seed`) are dataclass fields with
    defaults; override at construction time:

        WaldoStatistic(n_mc=2000, seed=12345)

    The MC reference distribution at each candidate `theta` is sampled
    deterministically — same `(theta, model.fingerprint(),
    prior.fingerprint(), seed)` ⇒ same MC draws — so `confidence_interval`
    is a deterministic function of its inputs and `brentq` converges
    cleanly within MC tolerance.
    """

    name: ClassVar[str] = "waldo"
    asymptotic_null: AsymptoticDistribution = field(
        default_factory=lambda: AsymptoticDistribution(
            family="weighted_chi2",
            df=1,
            scale=1.0,
            description="(mu_n - theta)^2 / sigma_n^2 ~ w * ncx2_1(lambda(theta)) "
            "(closed-form Normal-Normal); generic: MC reference under H_0.",
        )
    )

    n_mc: int = 500
    seed: int = 0xC0FFEE

    # ---------- closed-form Normal-Normal+Normal path ----------

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
    ) -> jax.Array:
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, _, w = posterior_params(D, prior.loc, model.sigma, prior.scale)
        theta_arr = jnp.asarray(theta0, dtype=jnp.float64)
        a, b = _pvalue_components(theta_arr, float(mu_n), prior.loc, w, model.sigma)
        return jsp_stats.norm.cdf(b - a) + jsp_stats.norm.cdf(-a - b)

    def _closed_form_confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: NormalNormalModel,
        prior: NormalDistribution,
    ) -> tuple[float, float]:
        # scipy: brentq_with_doubling lives at the public CI-inversion boundary.
        from ..tilting._solvers import brentq_with_doubling

        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, _, _ = posterior_params(D, prior.loc, model.sigma, prior.scale)

        def f(theta: float) -> float:
            return float(self._closed_form_pvalue(theta, data, model, prior)) - alpha

        half = 4.0 * model.sigma
        lower = brentq_with_doubling(
            f, midpoint=float(mu_n), initial_half_width=half, direction=-1
        )
        upper = brentq_with_doubling(
            f, midpoint=float(mu_n), initial_half_width=half, direction=+1
        )
        return (lower, upper)

    def _closed_form_acceptance_region(
        self,
        alpha: float,
        theta0: ArrayLike,
        model: NormalNormalModel,
        prior: NormalDistribution,
    ) -> tuple[jax.Array, jax.Array]:
        # scipy: brentq_with_doubling lives at the public CI-inversion boundary.
        from ..tilting._solvers import brentq_with_doubling

        theta_arr = np.atleast_1d(np.asarray(theta0, dtype=np.float64))
        D_lo = np.empty_like(theta_arr)
        D_hi = np.empty_like(theta_arr)
        for i, theta_val in enumerate(theta_arr):

            def f(D_val: float, _theta=theta_val) -> float:
                return float(
                    self._closed_form_pvalue(float(_theta), np.asarray([D_val]), model, prior)
                ) - alpha

            half = 4.0 * model.sigma
            D_lo[i] = brentq_with_doubling(
                f, midpoint=float(theta_val), initial_half_width=half, direction=-1
            )
            D_hi[i] = brentq_with_doubling(
                f, midpoint=float(theta_val), initial_half_width=half, direction=+1
            )
        return (
            jnp.asarray(D_lo if D_lo.size > 1 else D_lo.reshape(())),
            jnp.asarray(D_hi if D_hi.size > 1 else D_hi.reshape(())),
        )

    # ---------- generic model-agnostic path ----------

    def _generic_evaluate(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
    ) -> jax.Array:
        if prior is None:
            raise ValueError("WaldoStatistic.evaluate requires a prior (got None).")
        posterior = model.posterior(data, prior)
        theta_arr = jnp.asarray(theta0, dtype=jnp.float64)
        mu_post = jnp.asarray(posterior.mean(), dtype=jnp.float64)
        var_post = jnp.asarray(posterior.var(), dtype=jnp.float64)
        diff = mu_post - theta_arr
        return diff * diff / jnp.maximum(var_post, 1e-300)

    def _generic_mc_reference(
        self,
        theta0_f: float,
        n_obs: int,
        model: Model,
        prior: Prior,
        n_mc: int,
        derived_seed: int,
    ) -> NDArray[np.float64]:
        """Sample n_mc t-values under H_0 ~ likelihood(.|theta0).

        Each draw constructs the posterior under that synthetic data
        and recomputes the WALDO statistic; this is what gives the
        method its frequentist calibration.
        """
        rng = np.random.default_rng(derived_seed)
        t_samples = np.empty(n_mc, dtype=np.float64)
        for i in range(n_mc):
            D_prime = model.sample_data(theta0_f, rng, n_obs)
            post_prime = model.posterior(D_prime, prior)
            mu = float(np.asarray(post_prime.mean()))
            var = float(np.asarray(post_prime.var()))
            if var <= 0.0 or not np.isfinite(var):
                t_samples[i] = 0.0
            else:
                d = mu - theta0_f
                t_samples[i] = d * d / var
        return t_samples

    def _generic_derived_seed(
        self, theta0_f: float, model: Model, prior: Prior
    ) -> int:
        """Reproducible 32-bit seed derived from (theta, fingerprints, base seed)."""
        h = hash((theta0_f, tuple(model.fingerprint()), tuple(prior.fingerprint())))
        return (int(self.seed) ^ (h & 0xFFFFFFFF)) & 0xFFFFFFFF

    def _generic_pvalue(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
    ) -> jax.Array:
        if prior is None:
            raise ValueError("WaldoStatistic.pvalue requires a prior (got None).")
        theta_f = float(np.asarray(theta0))
        n_obs = int(np.atleast_1d(np.asarray(data, dtype=np.float64)).size)
        t_obs = float(np.asarray(self._generic_evaluate(theta_f, data, model, prior)))
        derived_seed = self._generic_derived_seed(theta_f, model, prior)
        t_ref = self._generic_mc_reference(
            theta_f, n_obs, model, prior, self.n_mc, derived_seed
        )
        # +1 smoothing: empirical p-value with continuity correction; bounded
        # away from 0 and 1 so brentq can bracket cleanly when the observed
        # t lies in the extreme tails of the MC reference.
        p = (float(np.sum(t_ref >= t_obs)) + 1.0) / (float(self.n_mc) + 1.0)
        return jnp.asarray(p)

    def _generic_confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
    ) -> tuple[float, float]:
        if prior is None:
            raise ValueError(
                "WaldoStatistic.confidence_interval requires a prior (got None)."
            )
        # scipy: brentq_with_doubling at the public CI-inversion boundary.
        from ..tilting._solvers import brentq_with_doubling

        posterior = model.posterior(data, prior)
        mu_post = float(np.asarray(posterior.mean()))
        sigma_post = float(np.sqrt(max(float(np.asarray(posterior.var())), 1e-300)))

        def f(theta: float) -> float:
            return float(self._generic_pvalue(theta, data, model, prior)) - alpha

        half = max(4.0 * sigma_post, 1e-3)
        support_lo, support_hi = model.support()
        try:
            lower = brentq_with_doubling(
                f, midpoint=mu_post, initial_half_width=half, direction=-1
            )
        except Exception:
            lower = float(support_lo)
        try:
            upper = brentq_with_doubling(
                f, midpoint=mu_post, initial_half_width=half, direction=+1
            )
        except Exception:
            upper = float(support_hi)
        return (max(lower, float(support_lo)), min(upper, float(support_hi)))

    # ---------- public protocol surface (dispatches) ----------

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        if _is_normal_normal_pair(model, prior):
            return self._closed_form_evaluate(theta0, data, model, prior)  # type: ignore[arg-type]
        return self._generic_evaluate(theta0, data, model, prior)

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        if _is_normal_normal_pair(model, prior):
            return self._closed_form_pvalue(theta0, data, model, prior)  # type: ignore[arg-type]
        return self._generic_pvalue(theta0, data, model, prior)

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if _is_normal_normal_pair(model, prior):
            return self._closed_form_acceptance_region(alpha, theta0, model, prior)  # type: ignore[arg-type]
        raise NotImplementedError(
            f"WaldoStatistic.acceptance_region (data-space) is only "
            f"available for the closed-form Normal-Normal pair; got "
            f"model={type(model).__name__}, prior={type(prior).__name__}. "
            f"Use confidence_interval(...) for the generic theta-space inversion."
        )

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        if _is_normal_normal_pair(model, prior):
            return self._closed_form_confidence_interval(alpha, data, model, prior)  # type: ignore[arg-type]
        return self._generic_confidence_interval(alpha, data, model, prior)

    def accepts_tilting(self, tilting: TiltingScheme) -> bool:
        return True
