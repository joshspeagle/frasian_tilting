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
same seed ⇒ same internal uniform stream ⇒ same Normal inverse-CDF
draws across theta. The result is that `f(theta)` is a smooth (or,
for future non-NN models with discrete support, piecewise-constant)
function of theta — brentq actually converges instead of locking
onto a re-randomised staircase.

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
from scipy.special import ndtr as _ndtr

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._errors import BracketingFailed
from .._registry import register_statistic
from ..models._dispatch import is_normal_normal
from ..models.base import Model, Prior
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel, posterior_params  # noqa: F401  (legacy field-access)
from .base import AsymptoticDistribution

if TYPE_CHECKING:
    from ..tilting.base import TiltingScheme

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _is_normal_normal_pair(model: Model, prior: Prior | None) -> bool:
    # Audit P1 G.5: model is dispatched via fingerprint() instead of
    # isinstance(NormalNormalModel) so the closed-form path is opt-in
    # by fingerprint contract, not by inheritance.
    return is_normal_normal(model) and isinstance(prior, NormalDistribution)


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
    brentq closures (~1 us/call via scipy.special.ndtr vs ~30 us via
    scipy.stats.norm.cdf's argsreduce wrapper, vs ~200 us through
    jsp_stats.norm.cdf). See `tilting/power_law.py::_tilted_pvalue_numpy_scalar`
    for the same-pattern motivation.
    """
    a = abs(mu_n_f - theta_f) / (w * sigma)
    b = (1.0 - w) * (mu0 - theta_f) / (w * sigma)
    return float(_ndtr(b - a) + _ndtr(-a - b))


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
    # When True, every dispatch site below skips the closed-form
    # Normal-Normal+Normal fast path (Theorem 3) and runs the
    # model-agnostic Monte-Carlo numerical path even on the conjugate
    # NN sandbox. The two paths agree to within MC noise on NN n=1
    # (see `tests/regression/test_waldo_generic_matches_closed_form.py`),
    # so this flag is intended for *path-coverage debugging*, not
    # production CI estimation. The discriminated `cell_name`
    # ensures the cache and manifest don't collide across the two
    # flavours.
    force_generic: bool = False

    @property
    def cell_name(self) -> str:
        """Discriminator for the cache key + manifest. Default closed-form
        cell is `waldo`; `force_generic=True` flips it to `waldo[generic]`."""
        return f"{self.name}[generic]" if self.force_generic else self.name

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
    ) -> NDArray[np.float64]:
        # NumPy + scipy.special.ndtr: ~50x faster than jsp_stats.norm.cdf
        # for the small-array sizes the CD experiment evaluates per replicate.
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
        """t = (mu_post - theta)^2 / var_post; degenerate var -> NaN.

        Pure-numpy hot path: brentq is Python control-flow so any JAX
        scalar wrapping is pure overhead (~50-100 µs per `jnp.asarray`
        on a scalar). Returns `jax.Array` at the public boundary for
        protocol parity, but the math is numpy.
        """
        if prior is None:
            raise ValueError("WaldoStatistic.evaluate requires a prior (got None).")
        posterior = model.posterior(data, prior)
        theta_arr = np.asarray(theta0, dtype=np.float64)
        mu_post = float(np.asarray(posterior.mean()))
        var_post = float(np.asarray(posterior.var()))
        diff = mu_post - theta_arr
        # Degenerate-variance contract: NaN when var_post <= 0, NOT a
        # tiny floor (which would inflate t to ~10^300 and pair with the
        # mc-reference's `0.0`-on-degenerate behaviour to produce a
        # false p ≈ 1/(n_mc+1)). Consumers treat NaN as "uninformative
        # draw" (mc reference) or "ill-posed observed data" (raises in
        # `_generic_pvalue`).
        if var_post > 0.0 and np.isfinite(var_post):
            return jnp.asarray(diff * diff / var_post)
        return jnp.asarray(np.full_like(theta_arr, np.nan))

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

        Vectorised: one `model.sample_data_batch` call returns shape
        (n_mc, n_obs); one `model.posterior_moments_batch` call returns
        per-row (mu, var) arrays. The t-statistic is then a single
        broadcast `(mu - theta0)^2 / var` numpy expression. Replaces the
        former `n_mc`-iteration Python loop (~45 µs per iteration of
        sample + posterior + scalar arithmetic) with two C-level numpy
        ops; ~50-200x speedup at n_mc=2000 on Normal-Normal.

        CRN: a fresh `np.random.default_rng(derived_seed)` per call. Same
        seed ⇒ same internal uniform stream ⇒ same inverse-CDF mappings
        to D' across theta probes (the property survives the batch since
        a single rng call draws the same uniforms internally).

        Degenerate-variance handling: rows with `var <= 0` or non-finite
        var produce NaN t. `_generic_pvalue` filters NaN downstream.
        """
        from ..models.base import (
            posterior_moments_batch as _moments_batch,
            sample_data_batch as _sample_batch,
        )

        rng = np.random.default_rng(int(derived_seed))
        D_batch = _sample_batch(model, float(theta0_f), rng, int(n_mc), int(n_obs))
        mu_arr, var_arr = _moments_batch(model, D_batch, prior)
        finite = (var_arr > 0.0) & np.isfinite(var_arr)
        # Suppress divide-by-zero / invalid warnings for the masked-out
        # rows — we replace them with NaN below.
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = mu_arr - float(theta0_f)
            t = diff * diff / np.where(finite, var_arr, 1.0)
        return np.where(finite, t, np.nan)

    def _generic_pvalue(
        self,
        theta0: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior | None,
        *,
        derived_seed: int | None = None,
        obs_moments: tuple[float, float] | None = None,
    ) -> jax.Array:
        """MC empirical p-value with `(k+1)/(n+1)` continuity correction.

        ``theta0`` accepts a scalar OR a 1-D array. The array path
        (used by the CD experiment, which evaluates the p-value on a
        ~401-pt theta grid per replicate) loops over thetas in Python
        — each theta gets its own MC reference under H_0:theta — but
        each per-theta MC reference is fully vectorised.

        ``derived_seed``: when supplied, used as the MC seed for *every*
        theta (CRN across the array — what `_generic_confidence_interval`
        threads in, and what makes `f(theta)` deterministic for brentq).
        When ``None``, a stable seed is derived from (data, model, prior,
        alpha=0, self.seed). The seed is INDEPENDENT of theta by design.

        ``obs_moments``: optional `(mu_obs, var_obs)`. When supplied,
        skips the (theta-independent) observed-posterior recomputation —
        ~30 µs saved per call. The CI inversion path passes this once-
        computed pair through every brentq probe.
        """
        if prior is None:
            raise ValueError("WaldoStatistic.pvalue requires a prior (got None).")
        data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        if data_arr.ndim != 1:
            raise NotImplementedError(
                "WaldoStatistic.pvalue currently expects 1-D data (n=1 sandbox "
                "or n trials of a single scalar); got "
                f"data.ndim={data_arr.ndim}. Multi-dim data requires the model "
                "to expose an `n_obs(data) -> int` accessor (latent skeptic vector #7)."
            )
        n_obs = int(data_arr.size)

        # Hoist observed moments (theta-independent) — pure-numpy, no JAX.
        if obs_moments is None:
            posterior = model.posterior(data_arr, prior)
            mu_obs = float(np.asarray(posterior.mean()))
            var_obs = float(np.asarray(posterior.var()))
        else:
            mu_obs, var_obs = obs_moments
        if not (np.isfinite(var_obs) and var_obs > 0.0):
            raise ValueError(
                f"WaldoStatistic._generic_pvalue: observed posterior variance "
                f"is degenerate (var <= 0) at the supplied data, so the test "
                f"statistic is ill-defined. var_obs={var_obs!r}, "
                f"data.shape={data_arr.shape!r}."
            )

        if derived_seed is None:
            derived_seed = self._stable_seed(data_arr, model, prior, 0.0, self.seed)

        theta_arr_input = np.asarray(theta0, dtype=np.float64)
        scalar_input = (theta_arr_input.ndim == 0)
        theta_arr = np.atleast_1d(theta_arr_input)
        # t_obs(theta) = (mu_obs - theta)^2 / var_obs — vectorised across thetas.
        diff_obs = mu_obs - theta_arr
        t_obs = diff_obs * diff_obs / var_obs

        # Per-theta MC reference (each theta needs its own H_0 reference;
        # CRN seed shared so brentq probes nest cleanly).
        p_out = np.empty(theta_arr.shape, dtype=np.float64)
        for i, theta_f in enumerate(theta_arr):
            t_ref = self._generic_mc_reference(
                float(theta_f), n_obs, model, prior, self.n_mc, derived_seed
            )
            finite = np.isfinite(t_ref)
            n_eff = int(finite.sum())
            if n_eff == 0:
                raise ValueError(
                    f"WaldoStatistic._generic_pvalue: every MC reference draw "
                    f"produced a degenerate posterior at theta0={float(theta_f)!r}. "
                    f"Cannot compute an empirical p-value."
                )
            # +1 smoothing: conservative continuity correction — coverage
            # biased upward by O(1/n_eff), never downward. Bounded in
            # (0, 1] so brentq can bracket cleanly even at the tails.
            n_more_extreme = float(np.sum(t_ref[finite] >= t_obs[i]))
            p_out[i] = (n_more_extreme + 1.0) / (float(n_eff) + 1.0)

        # Match input shape: scalar in -> scalar out (preserves the
        # historical scalar contract that `_generic_confidence_interval`
        # and the smoke tests rely on).
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
        # CRN seed: stable across processes, INDEPENDENT of theta AND
        # of alpha (audit P1 G.3). Threading this seed into every brentq
        # probe makes the MC pvalue use the same internal uniform stream
        # at each theta — `f(theta)` becomes a deterministic function of
        # theta (smooth for Normal) instead of a fresh stochastic
        # process, so brentq actually converges (skeptic finding
        # #1+#13). Dropping alpha makes two
        # cross-call invocations at different alphas (e.g. 0.05 and
        # 0.10) share the same MC reference draws — the resulting CIs
        # then nest cleanly (broader α → wider CI) instead of jumping
        # at the MC noise level.
        derived_seed = self._stable_seed(data_arr, model, prior, 0.0, self.seed)
        support_lo, support_hi = model.support()

        # Hoisted observed moments — theta-INDEPENDENT, computed once per
        # CI (was previously recomputed inside every brentq probe via
        # `_generic_evaluate` -> `model.posterior`). Threaded into
        # `_generic_pvalue` via the `obs_moments` kwarg below.
        posterior = model.posterior(data, prior)
        mu_post = float(np.asarray(posterior.mean()))
        var_post = float(np.asarray(posterior.var()))
        sigma_post = float(np.sqrt(max(var_post, 1e-300)))
        obs_moments = (mu_post, var_post)

        def f(theta: float) -> float:
            # Clamp to model support: brentq's bracket-doubling can
            # probe values outside the parameter space (e.g. theta
            # outside a bounded support), where
            # `model.sample_data(theta, ...)` would raise. Clamping
            # makes f flat outside support so
            # brentq returns BracketingFailed cleanly when the CI
            # truly extends to the boundary; the caller's
            # `except BracketingFailed` then yields support_lo / hi.
            theta_safe = max(float(support_lo), min(float(support_hi), theta))
            return float(
                self._generic_pvalue(
                    theta_safe, data, model, prior,
                    derived_seed=derived_seed,
                    obs_moments=obs_moments,
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
            # Bracket exhausted at the support boundary — return the
            # boundary explicitly so callers see an honest "open CI".
            # We do NOT swallow other exceptions (silent CI=[support_lo,
            # support_hi] would mask real bugs; skeptic finding #4).
            # Audit P1 G.4: emit a UserWarning so callers / runners can
            # annotate metadata with the boundary-hit; the previous
            # behaviour (silent fallback to support edge) hid the
            # difference between "α-inversion zero at support_lo" and
            # "we never bracketed; boundary returned by default".
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
            import warnings as _w
            sides = [s for s, hit in (("lower", boundary_hit_lo),
                                       ("upper", boundary_hit_hi)) if hit]
            _w.warn(
                f"WaldoStatistic._generic_confidence_interval: bracket "
                f"exhausted on the {' and '.join(sides)} side(s); "
                f"returning model.support() boundary at alpha={alpha!r}. "
                f"This may be a true open CI or a numerical pathology "
                f"(e.g. all MC reference draws degenerate). Callers / "
                f"runners should annotate this in metadata.",
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
            return self._closed_form_pvalue(theta0, data, model, prior)  # type: ignore[arg-type]
        return self._generic_pvalue(theta0, data, model, prior)

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Data-space accept region for theta0 at level alpha.

        **n=1 implicit convention** (audit P0-review #1, P1 G.6): this
        method takes no `data`, so the closed-form NN+Normal derivation
        treats the observation as a single scalar `D` (i.e. n=1). For
        n>1 datasets the data-space region for the sample mean is
        scaled by `sqrt(n)`; the closed form here does not carry a
        sample-size parameter and therefore returns the n=1 region
        unconditionally. This is asymmetric with `evaluate` /
        `pvalue` / `confidence_interval`, which dispatch via
        `_is_normal_normal_n1` (n>1 → generic MC). The asymmetry is
        principled: the dispatch on `_is_normal_normal_pair` here is
        the only honest answer the closed-form derivation can give,
        and lifting to the generic (n>1) regime would require a
        full MC inversion in data space — out of scope. Callers who
        care about n>1 should use `confidence_interval` (theta-space)
        instead. Audit P1 G.6 reclassifies `acceptance_region` as
        an optional protocol method; feature-detect via
        `has_acceptance_region(stat)`.
        """
        if not self.force_generic and _is_normal_normal_pair(model, prior):
            return self._closed_form_acceptance_region(alpha, theta0, model, prior)  # type: ignore[arg-type]
        if self.force_generic and _is_normal_normal_pair(model, prior):
            raise NotImplementedError(
                "WaldoStatistic.acceptance_region (data-space) has no generic "
                "path; got force_generic=True. Use confidence_interval(...) "
                "for the generic theta-space inversion, or unset force_generic "
                "to recover the closed-form NN+Normal data-space region."
            )
        raise NotImplementedError(
            f"WaldoStatistic.acceptance_region (data-space) is only "
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
