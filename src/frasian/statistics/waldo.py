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

Reproducibility & MC discipline
-------------------------------
The generic-path Monte Carlo uses **common random numbers (CRN)**: a
single seed is derived from the *call's* inputs (data + fingerprints
+ alpha + self.seed) — NOT from the candidate theta — via
`hashlib.blake2b` for cross-process stability. A fresh
`np.random.default_rng(seed)` is constructed at every brentq probe;
same seed ⇒ same internal uniform stream ⇒ same Bernoulli/Normal
inverse-CDF draws across theta. The result is that `f(theta)` is a
piecewise-constant (Bernoulli) or smooth (Normal) function of theta
— brentq actually converges instead of locking onto a re-randomised
staircase.

The empirical p-value uses the conservative `(k+1)/(n+1)` continuity
correction. This biases coverage upward by O(1/n_mc) — explicitly a
*conservative* CI, never anti-conservative — and bounds the p-value
strictly inside (0, 1] so brentq can bracket cleanly even at the
extreme tails.

Cost note: each MC draw constructs `model.posterior(D', prior)`. For
the conjugate Beta and Normal cases that's O(1); for hypothetical
non-conjugate posteriors (NUTS / VI), the cost scales linearly in
`n_mc` and the CI inversion cost is O(n_mc * brentq_iters).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as _scalar_scipy_stats

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._errors import BracketingFailed
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


def _is_normal_normal_n1(
    model: Model, prior: Prior | None, data: ArrayLike
) -> bool:
    """The closed-form Normal-Normal+Normal path is correct only for the
    n=1 sandbox (`NormalNormalModel.posterior` collapses ``data`` to its
    mean and uses sigma^2, not sigma^2/n — see the comment at
    ``normal_normal.py:106``). For n>1 callers we route through the
    generic MC path, which uses ``n_obs = data.size`` correctly. Without
    this guard the closed form silently disagrees with the generic path
    on ``data.size > 1``.
    """
    if not _is_normal_normal_pair(model, prior):
        return False
    arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    return arr.size == 1


def _pvalue_components(
    theta: jax.Array, mu_n: float, mu0: float, w: float, sigma: float
) -> tuple[jax.Array, jax.Array]:
    """a(theta), b(theta) from the closed-form WALDO p-value formula."""
    a = jnp.abs(mu_n - theta) / (w * sigma)
    b = (1.0 - w) * (mu0 - theta) / (w * sigma)
    return a, b


def _closed_form_pvalue_scalar(
    theta_f: float,
    D_f: float,
    mu_n_f: float,
    w: float,
    mu0: float,
    sigma: float,
) -> float:
    """Numpy-eager scalar mirror of `_closed_form_pvalue`. Used inside
    brentq closures (~10 us/call vs ~200 us through `jsp_stats.norm.cdf`).
    See `tilting/power_law.py::_tilted_pvalue_numpy_scalar` for the
    same-pattern motivation.
    """
    a = abs(mu_n_f - theta_f) / (w * sigma)
    b = (1.0 - w) * (mu0 - theta_f) / (w * sigma)
    return float(
        _scalar_scipy_stats.norm.cdf(b - a) + _scalar_scipy_stats.norm.cdf(-a - b)
    )


@register_statistic(name="waldo", brief="docs/methods/waldo.md")
@dataclass(frozen=True)
class WaldoStatistic:
    """WALDO statistic with closed-form Normal-Normal+Normal fast path
    + generic Monte-Carlo numerical path.

    Generic path knobs (`n_mc`, `seed`) are dataclass fields with
    defaults; override at construction time:

        WaldoStatistic(n_mc=4000, seed=12345)

    The default `n_mc=2000` gives ~0.022 MC standard error on a
    p-value near 0.5, dropping to ~0.005 at p=0.05. For accurate CI
    inversion at small alpha, use n_mc>=2000.
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

    n_mc: int = 2000
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
        mu_n_arr, _, w = posterior_params(D, prior.loc, model.sigma, prior.scale)
        mu_n_f = float(mu_n_arr)
        sigma = model.sigma
        mu0 = prior.loc

        def f(theta: float) -> float:
            # scipy: hot brentq inner loop — use the numpy-eager scalar
            # mirror (~10 us) instead of routing through jsp_stats.norm
            # (~200 us per call). See power_law.py::tilted_pvalue note.
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
        # scipy: brentq_with_doubling lives at the public CI-inversion boundary.
        from ..tilting._solvers import brentq_with_doubling

        theta_arr = np.atleast_1d(np.asarray(theta0, dtype=np.float64))
        D_lo = np.empty_like(theta_arr)
        D_hi = np.empty_like(theta_arr)
        sigma = model.sigma
        mu0 = prior.loc
        for i, theta_val in enumerate(theta_arr):
            theta_f = float(theta_val)

            def f(D_val: float, _theta_f: float = theta_f) -> float:
                # mu_n depends on D, so recompute per call. Same numpy
                # fast-path discipline as the CI-inversion brentq above.
                mu_n_arr_i, _, w_i = posterior_params(D_val, mu0, sigma, prior.scale)
                return _closed_form_pvalue_scalar(
                    _theta_f, D_val, float(mu_n_arr_i), w_i, mu0, sigma
                ) - alpha

            half = 4.0 * model.sigma
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
        # Degenerate-variance contract: return NaN when var_post <= 0,
        # NOT a tiny floor (which would inflate t to ~10^300 and pair
        # with the mc-reference's `0.0`-on-degenerate behaviour to
        # produce a false p ≈ 1/(n_mc+1)). Both code paths now agree on
        # NaN; consumers treat NaN as "uninformative draw" (mc reference)
        # or "ill-posed observed data" (raises in `_generic_pvalue`).
        safe_var = jnp.where(var_post > 0.0, var_post, 1.0)
        raw = diff * diff / safe_var
        return jnp.where(var_post > 0.0, raw, jnp.nan)

    @staticmethod
    def _stable_seed(
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        alpha: float,
        base_seed: int,
    ) -> int:
        """Cross-process-stable 32-bit seed for the MC reference.

        Python's `hash()` is randomised per-process (`PYTHONHASHSEED`
        defaults to a random value, randomising hashes of strings and
        tuples containing strings). We hash a deterministic byte-
        encoding via `hashlib.blake2b` so the same `(data, model,
        prior, alpha, seed)` tuple always produces the same MC draws,
        regardless of process. The seed is intentionally INDEPENDENT
        of the candidate theta — that's what makes common random
        numbers across brentq probes possible.
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
    ) -> NDArray[np.float64]:
        """Sample n_mc t-values under H_0 ~ likelihood(.|theta0).

        Uses CRN: same `derived_seed` ⇒ same internal uniform stream ⇒
        same inverse-CDF mappings to D' across different theta values.
        On Bernoulli, `Generator.binomial(1, theta_f, ...)` draws
        uniforms and compares to theta_f, so different theta_f produces
        D' = (#{u_i < theta_f}) — a piecewise-constant function of theta
        with the *same* uniforms. On Normal, `Generator.normal(loc=theta_f,
        scale=sigma, ...)` shifts the same Z-draws — smooth in theta_f.
        Either way, brentq sees a deterministic function of theta and
        converges.
        """
        rng = np.random.default_rng(derived_seed)
        t_samples = np.empty(n_mc, dtype=np.float64)
        for i in range(n_mc):
            D_prime = model.sample_data(theta0_f, rng, n_obs)
            post_prime = model.posterior(D_prime, prior)
            mu = float(np.asarray(post_prime.mean()))
            var = float(np.asarray(post_prime.var()))
            if var <= 0.0 or not np.isfinite(var):
                # Degenerate MC draw: posterior is a point mass, t is
                # ill-defined. Mark NaN; `_generic_pvalue` filters NaN
                # from the reference and divides by n_eff. Previously
                # this returned 0.0, which paired with `_generic_evaluate`'s
                # 1e-300 floor to make t_obs huge and t_ref always-zero
                # — collapsing the empirical p to 1/(n_mc+1) regardless
                # of θ. Fixing both halves of the inconsistency together.
                t_samples[i] = np.nan
            else:
                d = mu - theta0_f
                t_samples[i] = d * d / var
        return t_samples

    def _generic_pvalue(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
        *,
        derived_seed: int | None = None,
    ) -> jax.Array:
        """MC empirical p-value with `(k+1)/(n+1)` continuity correction.

        ``derived_seed``: when supplied, used as the MC seed (call this
        path from `_generic_confidence_interval` to enable CRN across
        brentq probes). When ``None``, a stable seed is computed from
        (data, model, prior, alpha=0, self.seed) — the resulting
        single-call p-value is reproducible across processes but does
        NOT share random numbers with any companion call. Use the
        explicit form whenever multiple p-values must agree under CRN.
        """
        if prior is None:
            raise ValueError("WaldoStatistic.pvalue requires a prior (got None).")
        theta_f = float(np.asarray(theta0))
        data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        if data_arr.ndim != 1:
            raise NotImplementedError(
                "WaldoStatistic.pvalue currently expects 1-D data (n=1 sandbox "
                "or n trials of a single Bernoulli/Normal scalar); got "
                f"data.ndim={data_arr.ndim}. Multi-dim data requires the model "
                "to expose an `n_obs(data) -> int` accessor (latent skeptic vector #7)."
            )
        n_obs = int(data_arr.size)
        t_obs = float(np.asarray(self._generic_evaluate(theta_f, data, model, prior)))
        if not np.isfinite(t_obs):
            raise ValueError(
                f"WaldoStatistic._generic_pvalue: observed posterior variance "
                f"is degenerate (var <= 0) at the supplied data, so the test "
                f"statistic is ill-defined. theta0={theta_f!r}, "
                f"data.shape={data_arr.shape!r}."
            )
        if derived_seed is None:
            derived_seed = self._stable_seed(data_arr, model, prior, 0.0, self.seed)
        t_ref = self._generic_mc_reference(
            theta_f, n_obs, model, prior, self.n_mc, derived_seed
        )
        # Drop MC draws with degenerate variance (NaN per the
        # _generic_mc_reference contract) and renormalise.
        finite = np.isfinite(t_ref)
        n_eff = int(finite.sum())
        if n_eff == 0:
            raise ValueError(
                f"WaldoStatistic._generic_pvalue: every MC reference draw "
                f"produced a degenerate posterior at theta0={theta_f!r}. "
                f"Cannot compute an empirical p-value."
            )
        t_ref_clean = t_ref[finite]
        # +1 smoothing: empirical p-value with continuity correction. This
        # is intentionally CONSERVATIVE — coverage is biased upward by
        # O(1/n_eff), never downward. Bounded strictly in (0, 1] so brentq
        # can bracket cleanly even at the extreme tails of the MC reference.
        p = (float(np.sum(t_ref_clean >= t_obs)) + 1.0) / (float(n_eff) + 1.0)
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

        data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        if data_arr.ndim != 1:
            raise NotImplementedError(
                "WaldoStatistic.confidence_interval expects 1-D data; got "
                f"data.ndim={data_arr.ndim}."
            )
        # CRN seed: stable across processes, INDEPENDENT of theta. Threading
        # this seed into every brentq probe makes the MC pvalue use the
        # same internal uniform stream at each theta — `f(theta)` becomes
        # a deterministic function of theta (piecewise-constant for
        # Bernoulli, smooth for Normal) instead of a fresh stochastic
        # process, so brentq actually converges (skeptic finding #1+#13).
        derived_seed = self._stable_seed(data_arr, model, prior, alpha, self.seed)
        support_lo, support_hi = model.support()

        posterior = model.posterior(data, prior)
        mu_post = float(np.asarray(posterior.mean()))
        sigma_post = float(np.sqrt(max(float(np.asarray(posterior.var())), 1e-300)))

        def f(theta: float) -> float:
            # Clamp to model support: brentq's bracket-doubling can
            # probe values outside the parameter space (e.g. theta>1
            # for Bernoulli), where `model.sample_data(theta, ...)`
            # would raise. Clamping makes f flat outside support so
            # brentq returns BracketingFailed cleanly when the CI
            # truly extends to the boundary; the caller's
            # `except BracketingFailed` then yields support_lo / hi.
            theta_safe = max(float(support_lo), min(float(support_hi), theta))
            return float(
                self._generic_pvalue(
                    theta_safe, data, model, prior, derived_seed=derived_seed
                )
            ) - alpha

        half = max(4.0 * sigma_post, 1e-3)
        try:
            lower = brentq_with_doubling(
                f, midpoint=mu_post, initial_half_width=half, direction=-1
            )
        except BracketingFailed:
            # Bracket exhausted at the support boundary — return the
            # boundary explicitly so callers see an honest "open CI".
            # We do NOT swallow other exceptions (silent CI=[support_lo,
            # support_hi] would mask real bugs; skeptic finding #4).
            lower = float(support_lo)
        try:
            upper = brentq_with_doubling(
                f, midpoint=mu_post, initial_half_width=half, direction=+1
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
        if _is_normal_normal_n1(model, prior, data):
            return self._closed_form_evaluate(theta0, data, model, prior)  # type: ignore[arg-type]
        return self._generic_evaluate(theta0, data, model, prior)

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> jax.Array:
        if _is_normal_normal_n1(model, prior, data):
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
        if _is_normal_normal_n1(model, prior, data):
            return self._closed_form_confidence_interval(alpha, data, model, prior)  # type: ignore[arg-type]
        return self._generic_confidence_interval(alpha, data, model, prior)

    def accepts_tilting(self, tilting: TiltingScheme) -> bool:
        return True
