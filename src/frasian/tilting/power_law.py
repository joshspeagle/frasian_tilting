"""Power-law tilting (the legacy eta-tilting, ported into the new shape).

  q(theta; eta) ∝ L(theta) * pi(theta)^(1 - eta)

For the conjugate Normal-Normal model this admits the closed form (Theorem 6
in the legacy derivations):

  denom    = 1 - eta * (1 - w)
  mu_eta   = (w*D + (1-eta)*(1-w)*mu0) / denom
  sigma_eta^2 = w * sigma^2 / denom
  w_eta    = w / denom

Identity element is `eta = 0` (recovers the WALDO posterior). The motivating
research observation — the reason this whole framework exists — is that the
*selection* of eta as a function of |Delta| produces a sharp transition
between posterior-driven and likelihood-driven behavior; the smoothness
experiment makes that complaint quantitative.

Admissible range is bounded below by the non-negativity of `denom`:
  eta_min = -w/(1-w) + buffer       (variance positive)
  eta_max = +inf in principle; capped at 1 in practice (Wald limit).

This implementation specializes on `NormalNormalModel`. Calling `tilt` with a
non-conjugate-Normal posterior raises `NotImplementedError`, by design — the
generic numerical fallback is a future extension and would obscure the math.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..config import Config
from ..models._dispatch import is_normal_normal
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..models.normal_normal import weight as _weight
from ..statistics.base import TestStatistic
from .base import EtaSelector, ParamSpec
from .eta_selectors import FixedEtaSelector

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


from functools import partial

# scipy: used by the numpy-eager scalar fast path inside tilted_pvalue.
# JAX dispatch costs ~50-300 us per scalar call even with jit; numpy +
# scipy.stats.norm is ~10 us. The bulk JAX kernel below handles the
# array path (length >= 2) and is autodiff-clean for Phase 4.
from scipy import stats as _scalar_scipy_stats


def _tilted_pvalue_numpy_scalar(
    theta_f: float,
    eta_f: float,
    D_f: float,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> float:
    """Numpy-eager scalar fast path. Mirrors `_tilted_pvalue_kernel` but
    runs on Python floats + scipy.stats.norm. Used when caller passes
    scalar (theta, eta, D) — the brentq inner-loop pattern. ~10 us/call.
    """
    denom = 1.0 - eta_f * (1.0 - w)
    if statistic_name == "wald":
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))
    if statistic_name == "waldo":
        mu_eta = (w * D_f + (1.0 - eta_f) * (1.0 - w) * mu0) / denom
        norm_factor = w * sigma / denom
        a_eta = abs(mu_eta - theta_f) / norm_factor
        b_eta = (1.0 - eta_f) * (1.0 - w) * (mu0 - theta_f) / (denom * norm_factor)
        return float(
            _scalar_scipy_stats.norm.cdf(b_eta - a_eta)
            + _scalar_scipy_stats.norm.cdf(-a_eta - b_eta)
        )
    raise NotImplementedError(
        f"_tilted_pvalue_numpy_scalar not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo'."
    )


@partial(jax.jit, static_argnames=("statistic_name",))
def _tilted_pvalue_kernel(
    theta: jax.Array,
    eta: jax.Array,
    D: jax.Array,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> jax.Array:
    """Pure JAX arithmetic kernel for the tilted p-value.

    Autodiff-clean (no validation, no Python control flow except the
    static `statistic_name` dispatch). Phase 4's learned-eta `loss`
    closes over this directly inside `@jax.jit`. The public
    `PowerLawTilting.tilted_pvalue` validates inputs once, then
    delegates here.

    JIT note (style guide allows: profile shows it pays off): the bulk
    path is called by `dynamic_ci_scan` (theta-grid shape (n_grid,))
    and by selectors / experiments with a small set of recurring
    shapes. JAX's jit cache is process-wide, so within one test sweep
    the first call per (shape, statistic_name) pays ~150 ms compile
    and every subsequent call is ~0.5 ms; eager mode without jit
    would be ~2 ms per call regardless. Across the L0 sweep that
    difference is the ~30 s vs >120 s gap.

    Scalar inputs do NOT reach this kernel — the public method's
    shape dispatch routes them to the numpy fast path (~10 us/call,
    much faster than either jit'd or eager JAX would be on a 0-d).
    """
    denom = 1.0 - eta * (1.0 - w)
    if statistic_name == "wald":
        # eta-independent: 2 * (1 - Phi(|D - theta| / sigma)).
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - jsp_stats.norm.cdf(z))
    if statistic_name == "waldo":
        mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * mu0) / denom
        norm_factor = w * sigma / denom
        a_eta = jnp.abs(mu_eta - theta) / norm_factor
        b_eta = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / (denom * norm_factor)
        return jsp_stats.norm.cdf(b_eta - a_eta) + jsp_stats.norm.cdf(-a_eta - b_eta)
    raise NotImplementedError(
        f"_tilted_pvalue_kernel not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo'."
    )


def _is_gaussian_triple(
    posterior: Posterior, prior: Prior, likelihood: Likelihood
) -> bool:
    """Return True iff (posterior, prior, likelihood) is the Normal-Normal triple
    that admits the Theorem 6 closed form. Used to dispatch between the
    closed-form fast path and the model-agnostic numerical path.
    """
    return (
        isinstance(posterior, NormalDistribution)
        and isinstance(prior, NormalDistribution)
        and isinstance(likelihood, GaussianLikelihood)
    )


def _require_gaussian(
    posterior: Posterior, prior: Prior, likelihood: Likelihood
) -> tuple[NormalDistribution, NormalDistribution, GaussianLikelihood]:
    """Same as the isinstance check, but raises with the legacy error message.

    Retained for the closed-form path's narrowing assertions; the public
    `tilt()` method dispatches via `_is_gaussian_triple` and only this
    function's raising is used post-dispatch (i.e. on the closed-form
    branch where the inputs are already known to be Gaussian).
    """
    if not isinstance(posterior, NormalDistribution):
        raise NotImplementedError(
            "PowerLawTilting closed form requires a NormalDistribution posterior; "
            f"got {type(posterior).__name__!r}."
        )
    if not isinstance(prior, NormalDistribution):
        raise NotImplementedError(
            "PowerLawTilting closed form requires a NormalDistribution prior; "
            f"got {type(prior).__name__!r}."
        )
    if not isinstance(likelihood, GaussianLikelihood):
        raise NotImplementedError(
            "PowerLawTilting closed form requires a GaussianLikelihood; "
            f"got {type(likelihood).__name__!r}."
        )
    return posterior, prior, likelihood


def _denom(w: float, eta: float) -> float:
    return 1.0 - eta * (1.0 - w)


# ----- Generic numerical path (any Distribution-conforming inputs) -----

# Default grid parameters per the deriver's recommendation
# (`/root/.claude/plans/...`, validated atol 1e-7 at N=1024 against
# Theorem 6 on Normal-Normal). N=2048 + k=8 are mildly more conservative;
# the suite picks N=1024 to keep per-call cost ~1 ms on Bernoulli.
_GENERIC_TILT_N_GRID: int = 1024
_GENERIC_TILT_HALF_WIDTH_K: float = 8.0
_GENERIC_TILT_QUANTILE_EPS: float = 1e-4


def _generic_tilt_grid_window(
    posterior: Posterior,
    prior: Prior,
    *,
    support: tuple[float, float],
) -> tuple[float, float]:
    """Pick the (theta_lo, theta_hi) integration window for `_generic_tilt`.

    Strategy: union of the posterior's and prior's near-quantile windows,
    clipped to `model.support()`. Quantile-based on bounded supports
    (Bernoulli's `[0, 1]`, etc.) so we don't pick an arbitrary k·std
    interval that overruns the support; ``mean ± k * std`` fallback
    on unbounded supports where quantile inversion can be expensive
    on a non-Gaussian posterior.

    The window must contain enough mass of *both* the prior and the
    likelihood-shaped posterior so the η ∈ [0, 1] sweep stays well-
    integrated. Posterior alone undercovers at η→1; prior alone
    undercovers at η→0; the union is robust at both ends.
    """
    eps = _GENERIC_TILT_QUANTILE_EPS
    support_lo, support_hi = float(support[0]), float(support[1])

    if np.isfinite(support_lo) and np.isfinite(support_hi):
        # Bounded support: use quantile-based window on both endpoints.
        # Skeptic finding #2: narrow the catch to specific exception types
        # so genuine bugs in `quantile()` (e.g. shape mismatches, internal
        # scipy errors) propagate cleanly. The `(ValueError, RuntimeError,
        # NotImplementedError)` set covers the documented failure modes
        # (improper shape parameters, numerical overflow on extreme Beta,
        # missing quantile implementation).
        try:
            lo = min(
                float(np.asarray(posterior.quantile(eps))),
                float(np.asarray(prior.quantile(eps))),
            )
            hi = max(
                float(np.asarray(posterior.quantile(1.0 - eps))),
                float(np.asarray(prior.quantile(1.0 - eps))),
            )
        except (ValueError, RuntimeError, NotImplementedError):
            # Documented fallback: full support window. Coarser than the
            # quantile window, but still produces a valid integration.
            # Coverage may degrade on very-skewed priors (Beta(101, 1));
            # that's pinned by a regression test.
            lo, hi = support_lo, support_hi
        return (max(lo, support_lo), min(hi, support_hi))

    # Unbounded support: use mean ± k * std for both, take the union.
    try:
        post_mu, post_sigma = float(posterior.mean()), float(np.sqrt(posterior.var()))
        prior_mu, prior_sigma = float(prior.mean()), float(np.sqrt(prior.var()))
    except (TypeError, ValueError, AttributeError) as e:
        raise ValueError(
            "PowerLawTilting._generic_tilt: posterior/prior must expose mean()+var() "
            f"on unbounded supports; got {type(posterior).__name__!r}, "
            f"{type(prior).__name__!r}."
        ) from e
    k = _GENERIC_TILT_HALF_WIDTH_K
    lo = min(post_mu - k * post_sigma, prior_mu - k * prior_sigma)
    hi = max(post_mu + k * post_sigma, prior_mu + k * prior_sigma)
    return (max(lo, support_lo), min(hi, support_hi))


def _generic_tilt(
    posterior: Posterior,
    prior: Prior,
    likelihood: Likelihood,
    eta: float,
    *,
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
):
    """Numerical tilted distribution: log q(theta;eta) ∝ log L(theta) + (1-eta) log pi(theta).

    The deriver verified (`/root/.claude/plans/.../`) that this formula
    reduces to PowerLawTilting's Theorem 6 closed form on Normal-Normal,
    atol 1e-7 at N=1024 — see `tests/regression/test_grid_distribution.py`
    and the cross-check below in `test_power_law_generic_matches_closed_form.py`.

    Returns a `GridDistribution` (conforms to the Distribution protocol).
    """
    from ._grid_distribution import grid_distribution_from_log_density

    theta_lo, theta_hi = _generic_tilt_grid_window(
        posterior, prior, support=support
    )
    if not (theta_lo < theta_hi):
        raise ValueError(
            f"PowerLawTilting._generic_tilt: degenerate grid window "
            f"[{theta_lo}, {theta_hi}] (likely posterior collapsed onto "
            f"support boundary or quantile inversion failed)."
        )
    theta_grid = np.linspace(theta_lo, theta_hi, n_grid)

    # log q_tilt(theta; eta) = log L(theta) + (1-eta) * log pi(theta).
    log_lik = np.asarray(likelihood.loglik(theta_grid), dtype=np.float64)
    log_prior = np.asarray(prior.logpdf(theta_grid), dtype=np.float64)
    log_q = log_lik + (1.0 - float(eta)) * log_prior

    # Replace -inf entries (e.g. theta outside the prior's support on
    # bounded models) with a finite floor so the trapezoidal Z is finite.
    # The grid_distribution_from_log_density helper will exp(log_q -
    # max(log_q)) which sends these to ~0 anyway.
    log_q = np.where(np.isfinite(log_q), log_q, -1e300)

    # Audit P1 H.1: admissibility check. The closed-form Normal-Normal
    # path raises `TiltingDomainError` when (1-w)*(1-eta) drives the
    # tilted-posterior denominator non-positive (the tilted distribution
    # is not normalisable — log q diverges at infinity). The generic
    # path here cannot compute that scalar, but the same divergence
    # surfaces as `argmax(log_q)` lying on a grid boundary: a tilted
    # distribution that grows without bound is window-truncated, and
    # its mode pins to the cut-off. We detect that and refuse rather
    # than silently returning a `GridDistribution` of a non-normalisable
    # density. (The first/last index check has a 1-bin tolerance for
    # legitimate cases where the prior pulls the mode to a support
    # boundary on bounded models — e.g. Beta(1, 1) on Bernoulli.)
    finite_mask = np.isfinite(log_lik) & np.isfinite(log_prior)
    if np.any(finite_mask):
        idx_max = int(np.argmax(np.where(finite_mask, log_q, -np.inf)))
        # On bounded supports (Bernoulli + Beta), the boundary may be
        # the legitimate mode — only flag the divergence when the
        # boundary log-density is BIGGER than the next interior point.
        # That signals the truncation, not a real boundary mode.
        if idx_max == 0 and log_q[0] > log_q[1] + 1e-9:
            raise TiltingDomainError(
                f"PowerLawTilting._generic_tilt: log q_tilt(θ; η={float(eta)!r}) "
                f"is monotonically increasing toward the lower grid edge "
                f"θ={theta_grid[0]!r}. The tilted posterior is not "
                f"normalisable on this support — typically because (1-η) is "
                f"large enough to drive the prior contribution divergent at "
                f"infinity. Reduce |η| or use a bounded prior."
            )
        if idx_max == n_grid - 1 and log_q[-1] > log_q[-2] + 1e-9:
            raise TiltingDomainError(
                f"PowerLawTilting._generic_tilt: log q_tilt(θ; η={float(eta)!r}) "
                f"is monotonically increasing toward the upper grid edge "
                f"θ={theta_grid[-1]!r}. The tilted posterior is not "
                f"normalisable on this support."
            )

    return grid_distribution_from_log_density(
        theta_grid,
        log_q,
        metadata={
            "scheme": "power_law",
            "eta": float(eta),
            "n_grid": int(n_grid),
            "theta_lo": float(theta_lo),
            "theta_hi": float(theta_hi),
        },
    )


# Default knobs for the generic MC tilted-pvalue path. Used only on
# non-Normal-Normal pairings (the closed-form Normal-Normal path
# doesn't need MC). Keep these as module-level constants rather than
# constructor args on PowerLawTilting so the closed-form fast path
# doesn't carry unused knobs in its dataclass surface.
# `_GENERIC_TILTED_PVALUE_BASE_SEED` is sourced from `_generic_pvalue`
# so PowerLaw and OT share the same CRN seed at fixed (data, prior, eta,
# alpha) — enables direct cross-scheme MC comparison in the smoothness
# experiment.
from ._generic_pvalue import (  # noqa: F401
    _GENERIC_TILTED_PVALUE_BASE_SEED,
    _resolve_support,
    _stable_tilted_pvalue_seed,
)

_GENERIC_TILTED_PVALUE_N_MC: int = 200
# The MC inner-loop t-statistic uses a coarser grid than the observed
# t_obs because (a) MC noise dominates over grid-discretisation noise
# at any reasonable n_mc, and (b) per-MC-call cost is the dominant
# wall-time of the CI inversion (~n_mc * brentq_iters * grid_n flops).
# n_grid_mc=256 brings per-CI cost down ~4x with negligible coverage
# impact at n_mc>=200.
_GENERIC_TILTED_PVALUE_N_GRID_MC: int = 256


def _generic_tilted_moments(
    posterior: Posterior,
    prior: Prior,
    likelihood: Likelihood,
    eta: float,
    *,
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
) -> tuple[float, float]:
    """Direct (mean, var) of the tilted distribution on a theta-grid.

    Skips the GridDistribution materialisation (no cdf cache / quantile
    interpolation / sample method) — computes only the first two
    moments, which is all the t-statistic needs. Used as the hot
    inner kernel of `_generic_tilted_pvalue`'s MC reference (called
    n_mc * brentq_iters * 2 times per CI inversion).
    """
    theta_lo, theta_hi = _generic_tilt_grid_window(
        posterior, prior, support=support
    )
    if not (theta_lo < theta_hi):
        # Degenerate window: posterior collapsed onto support boundary.
        # Return a sentinel that t-statistic computation handles cleanly.
        return float(theta_lo), 0.0
    theta_grid = np.linspace(theta_lo, theta_hi, n_grid)
    log_lik = np.asarray(likelihood.loglik(theta_grid), dtype=np.float64)
    log_prior = np.asarray(prior.logpdf(theta_grid), dtype=np.float64)
    log_q = log_lik + (1.0 - float(eta)) * log_prior
    # max-subtract for stability; -inf entries become 0 after exp.
    log_q = np.where(np.isfinite(log_q), log_q, -1e300)
    pdf_unnorm = np.exp(log_q - np.max(log_q))
    Z = float(np.trapezoid(pdf_unnorm, theta_grid))
    if Z <= 0.0 or not np.isfinite(Z):
        return float(theta_lo), 0.0
    pdf = pdf_unnorm / Z
    mean = float(np.trapezoid(theta_grid * pdf, theta_grid))
    m2 = float(np.trapezoid(theta_grid * theta_grid * pdf, theta_grid))
    var = max(m2 - mean * mean, 0.0)
    return mean, var


def _generic_tilted_t_statistic(
    theta_f: float,
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta: float,
    *,
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
) -> float:
    """Compute t = (mu_tilted - theta)^2 / sigma_tilted^2 at observed data.

    Calls `_generic_tilted_moments` (direct moments, no GridDistribution
    materialisation) for speed in the MC inner loop.
    """
    posterior = model.posterior(data, prior)
    likelihood = model.likelihood(data)
    mu, var = _generic_tilted_moments(
        posterior, prior, likelihood, eta, support=support, n_grid=n_grid
    )
    var_safe = max(var, 1e-300)
    diff = mu - theta_f
    return diff * diff / var_safe


def _generic_tilted_mc_reference_batch(
    *,
    theta_f: float,
    n_obs: int,
    model: object,
    prior: Prior,
    eta_f: float,
    support: tuple[float, float],
    n_mc: int,
    n_grid: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], int]:
    """Vectorised MC reference for the power-law tilted p-value.

    Returns `(t_samples, n_collapsed)` with `t_samples` shape `(n_mc,)`.
    Replaces the per-MC-iteration Python loop in `_generic_tilted_pvalue`
    with three batched operations:

      1. `sample_data_batch(model, theta_f, rng, n_mc, n_obs)` →
         `D_batch` shape `(n_mc, n_obs)` in one rng call.
      2. `posterior_moments_batch(model, D_batch, prior)` → per-row
         `(mu_arr, var_arr)`, used to size a single conservative
         theta-grid window covering all rows.
      3. `batch_loglik_grid(model, D_batch, theta_grid)` → per-row
         log-likelihood on the shared grid; combined with
         `prior.logpdf(theta_grid)` and the tilting exponent
         `(1 - eta)` to form per-row tilted log-densities; per-row
         normalisation + trapezoidal moments yields `(mu, var)` per row.

    Per-row windows in the original implementation differed slightly;
    using a single conservative window (max-extent across rows) keeps
    the integration vectorised at the cost of some per-row resolution
    in the wings. Compensated by using the same `n_grid` as the
    original — the per-row windows were always similar in width
    because all rows are draws from the same `likelihood(.|theta_f)`.

    Collapsed-row handling matches the legacy contract: rows whose
    integration normalisation `Z` is non-positive / non-finite OR whose
    derived variance is non-positive contribute `t = 0` (treats them as
    NOT more-extreme — biases the empirical p UP, conservative).
    """
    from ..models.base import (
        batch_loglik_grid as _batch_loglik_grid,
        posterior_moments_batch as _posterior_moments_batch,
        sample_data_batch as _sample_data_batch,
    )

    if n_mc <= 0:
        return np.empty(0, dtype=np.float64), 0
    support_lo, support_hi = float(support[0]), float(support[1])
    D_batch = _sample_data_batch(model, float(theta_f), rng, int(n_mc), int(n_obs))
    mu_arr, var_arr = _posterior_moments_batch(model, D_batch, prior)

    # Single conservative theta-window covering all rows. mu ± 8 sigma_post
    # is the per-row default; union across rows is `min(mu - 8*sigma)` /
    # `max(mu + 8*sigma)`. Clip to support.
    sigma_post_arr = np.sqrt(np.maximum(var_arr, 1e-300))
    half = 8.0 * sigma_post_arr
    finite = np.isfinite(mu_arr) & np.isfinite(sigma_post_arr)
    if not np.any(finite):
        return np.zeros(int(n_mc), dtype=np.float64), int(n_mc)
    lo = max(float(np.nanmin((mu_arr - half)[finite])), support_lo)
    hi = min(float(np.nanmax((mu_arr + half)[finite])), support_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or not (lo < hi):
        # Degenerate window across all rows.
        return np.zeros(int(n_mc), dtype=np.float64), int(n_mc)

    theta_grid = np.linspace(lo, hi, int(n_grid))

    log_lik_batch = _batch_loglik_grid(model, D_batch, theta_grid)  # (n_mc, n_grid)
    log_prior = np.asarray(prior.logpdf(theta_grid), dtype=np.float64)  # (n_grid,)
    log_q = log_lik_batch + (1.0 - float(eta_f)) * log_prior[None, :]
    # Sanitise -inf entries before the max-subtract; keeps exp finite.
    log_q = np.where(np.isfinite(log_q), log_q, -1e300)
    log_q_max = log_q.max(axis=-1, keepdims=True)
    pdf_unnorm = np.exp(log_q - log_q_max)  # (n_mc, n_grid)
    Z = np.trapezoid(pdf_unnorm, theta_grid, axis=-1)  # (n_mc,)
    Z_safe = np.where(Z > 0, Z, 1.0)
    pdf = pdf_unnorm / Z_safe[:, None]
    # Per-row mean and second moment via trapezoidal weighting.
    mean_arr = np.trapezoid(theta_grid[None, :] * pdf, theta_grid, axis=-1)
    m2_arr = np.trapezoid(theta_grid[None, :] ** 2 * pdf, theta_grid, axis=-1)
    var_tilted = np.maximum(m2_arr - mean_arr * mean_arr, 0.0)

    # Collapsed = bad normalisation OR zero variance.
    finite_z = (Z > 0) & np.isfinite(Z)
    finite_var = (var_tilted > 0) & np.isfinite(var_tilted)
    finite_row = finite_z & finite_var
    n_collapsed = int(np.sum(~finite_row))

    diff = mean_arr - float(theta_f)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = diff * diff / np.where(finite_row, var_tilted, 1.0)
    t_samples = np.where(finite_row, t, 0.0)
    return t_samples, n_collapsed


def _compute_obs_moments_per_eta_vec(
    *,
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta_arr: NDArray[np.float64],
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised observed-data tilted moments across an eta-array.

    For each ``eta_arr[i]`` returns the observed tilted distribution's
    `(mu_obs, var_obs)` evaluated against `data` (a single observation
    set). Used by ``_generic_tilted_pvalue_vec`` so the dynamic-η fine
    scan doesn't recompute observed-moments per-eta in a Python loop.

    Implementation: builds a single θ-grid covering the observed
    posterior; computes log_lik on the grid once (theta-independent of
    `eta`); per-eta combines with `(1 - eta) · log_prior(theta_grid)`
    via broadcasting, then per-row trapezoid normalisation +
    moment integration. Single (n_eta, n_grid) numpy block.
    """
    posterior_obs = model.posterior(data, prior)
    mu_post = float(np.asarray(posterior_obs.mean()))
    var_post = float(np.asarray(posterior_obs.var()))
    sigma_post = float(np.sqrt(max(var_post, 1e-300)))

    support_lo, support_hi = float(support[0]), float(support[1])
    half = 8.0 * sigma_post
    lo = max(mu_post - half, support_lo)
    hi = min(mu_post + half, support_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or not (lo < hi):
        # Fallback: if window is degenerate, return mu_post / 0 per eta.
        n_eta = int(np.asarray(eta_arr).size)
        return np.full(n_eta, mu_post), np.zeros(n_eta)
    theta_grid = np.linspace(lo, hi, int(n_grid))

    log_lik = np.asarray(
        model.likelihood(data).loglik(theta_grid), dtype=np.float64
    )  # (n_grid,)
    log_prior = np.asarray(prior.logpdf(theta_grid), dtype=np.float64)  # (n_grid,)
    eta_arr_np = np.asarray(eta_arr, dtype=np.float64)
    # log_q[i, k] = log_lik[k] + (1 - eta_arr[i]) * log_prior[k]
    log_q = log_lik[None, :] + (1.0 - eta_arr_np)[:, None] * log_prior[None, :]
    log_q = np.where(np.isfinite(log_q), log_q, -1e300)
    log_q_max = log_q.max(axis=-1, keepdims=True)
    pdf_unnorm = np.exp(log_q - log_q_max)  # (n_eta, n_grid)
    Z = np.trapezoid(pdf_unnorm, theta_grid, axis=-1)  # (n_eta,)
    Z_safe = np.where(Z > 0, Z, 1.0)
    pdf = pdf_unnorm / Z_safe[:, None]
    mean_arr = np.trapezoid(theta_grid[None, :] * pdf, theta_grid, axis=-1)
    m2_arr = np.trapezoid(theta_grid[None, :] ** 2 * pdf, theta_grid, axis=-1)
    var_arr = np.maximum(m2_arr - mean_arr * mean_arr, 0.0)
    # Where Z was bad, fall back to posterior moments at that eta — the
    # tilted moments are ill-defined; downstream consumers use these to
    # compute t_obs, where var≈0 will yield NaN/inf t_obs and the row's
    # p-value gets the conservative "no extreme draws" upper bound.
    finite = (Z > 0) & np.isfinite(Z)
    mean_arr = np.where(finite, mean_arr, mu_post)
    var_arr = np.where(finite, var_arr, 0.0)
    return mean_arr, var_arr


_VEC_PVALUE_CHUNK_MEM_BUDGET_MB: float = 50.0
"""Target peak memory per `_generic_tilted_pvalue_vec` chunk.

Used to set `chunk_size` adaptively based on `(n_mc, n_grid)`. The
log_lik_3d allocation alone is `chunk × n_mc × n_grid × 8 bytes`;
peak memory is roughly 4-5× that across temporaries (log_q,
pdf_unnorm, pdf, mean_3d, m2_3d). Bumping the budget too high leads
to allocator slowdowns on large tensors (observed ~10x wall-clock
inflation when chunks exceeded ~250 MB during Phase F development).
"""


def _adaptive_chunk_size(n_mc: int, n_grid: int) -> int:
    """Pick a chunk_size keeping the per-chunk log_lik allocation
    around `_VEC_PVALUE_CHUNK_MEM_BUDGET_MB`."""
    # Each log_lik element is float64 = 8 bytes; allow 4x overhead for
    # the broadcast intermediates (log_q, pdf_unnorm, pdf, etc.).
    bytes_per_chunk_row = int(n_mc) * int(n_grid) * 8 * 4
    budget = int(_VEC_PVALUE_CHUNK_MEM_BUDGET_MB * 1024 * 1024)
    chunk = max(budget // max(bytes_per_chunk_row, 1), 1)
    return min(chunk, 256)  # cap at 256 to keep Python loop overhead bounded


def _generic_tilted_pvalue_vec(
    theta_arr: NDArray[np.float64],
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta_arr: NDArray[np.float64],
    statistic_name: str,
    *,
    n_mc: int = _GENERIC_TILTED_PVALUE_N_MC,
    n_grid: int = _GENERIC_TILTED_PVALUE_N_GRID_MC,
    derived_seed: int | None = None,
    alpha: float = 0.0,
    base_seed: int = _GENERIC_TILTED_PVALUE_BASE_SEED,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Vectorised generic MC tilted p-value across (theta_arr, eta_arr).

    Computes p-values for many (theta_i, eta_i) pairs in one call.
    Used by the dynamic-η fine-scan path in
    `dynamic_tilted_confidence_interval_generic` — replaces 401 scalar
    calls (~88 ms each = 35 s) with one batched call (~100-200 ms total).

    Triple-batched memory layout: `D_3d (chunk, n_mc, n_obs)`,
    `log_lik_3d (chunk, n_mc, n_grid)` etc. Chunked along the theta
    axis (default 64) so peak memory stays in single-call-friendly
    territory (~26 MB for the log_lik tensor at the canonical sizes).

    For ``statistic_name="wald"``: eta-independent; delegates to
    scalar Wald per theta in a Python loop (Wald is fast enough that
    vectorisation isn't needed). For ``statistic_name="waldo"``:
    full triple-batched MC reference.

    CRN: a fresh `np.random.default_rng(derived_seed + chunk_offset)`
    per chunk so the MC stream is reproducible AND deterministic w.r.t.
    chunking. Within a chunk, all (theta_i, mc_j) pairs draw from a
    single big `rng.normal/binomial` call shifted by the per-theta
    location parameter (NN/Bernoulli sampling vectorises naturally).
    """
    from ..models.base import (
        batch_loglik_grid as _batch_loglik_grid,
        posterior_moments_batch as _posterior_moments_batch,
    )
    from ..statistics.wald import WaldStatistic

    theta_arr_np = np.asarray(theta_arr, dtype=np.float64)
    eta_arr_np = np.asarray(eta_arr, dtype=np.float64)
    if theta_arr_np.shape != eta_arr_np.shape:
        raise ValueError(
            f"theta_arr and eta_arr must have the same shape; got "
            f"{theta_arr_np.shape!r} and {eta_arr_np.shape!r}."
        )
    n_theta = int(theta_arr_np.size)

    if statistic_name == "wald":
        # Wald is eta-independent; delegate per theta. The scalar generic
        # Wald is already fast (chi²₁ inversion via a single jsp_stats call
        # per theta; cf. WaldStatistic._generic_pvalue).
        wald = WaldStatistic()
        return np.asarray([
            float(np.asarray(wald._generic_pvalue(float(t), data, model)))
            for t in theta_arr_np
        ], dtype=np.float64)

    if statistic_name != "waldo":
        raise NotImplementedError(
            f"_generic_tilted_pvalue_vec not implemented for "
            f"statistic={statistic_name!r}; supported: 'wald', 'waldo'."
        )

    data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if data_arr.ndim != 1:
        raise NotImplementedError(
            f"_generic_tilted_pvalue_vec expects 1-D data; got "
            f"data.ndim={data_arr.ndim}."
        )
    n_obs = int(data_arr.size)
    support = _resolve_support(model, data_arr)
    support_lo, support_hi = float(support[0]), float(support[1])

    if derived_seed is None:
        derived_seed = _stable_tilted_pvalue_seed(
            data_arr, model, prior, 0.0, alpha, base_seed
        )

    # Hoist observed tilted moments per eta — single batched grid integration
    # across the eta-axis. Returns shape (n_theta,) for both mu_obs and var_obs.
    mu_obs_per_theta, var_obs_per_theta = _compute_obs_moments_per_eta_vec(
        data=data_arr,
        model=model,
        prior=prior,
        eta_arr=eta_arr_np,
        support=support,
        n_grid=_GENERIC_TILT_N_GRID,
    )
    var_obs_safe = np.maximum(var_obs_per_theta, 1e-300)
    diff_obs = mu_obs_per_theta - theta_arr_np
    t_obs = diff_obs * diff_obs / var_obs_safe  # (n_theta,)

    # Process in chunks along the theta axis to bound peak memory.
    if chunk_size is None:
        chunk_size = _adaptive_chunk_size(int(n_mc), int(n_grid))
    p_out = np.empty(n_theta, dtype=np.float64)
    for chunk_start in range(0, n_theta, int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), n_theta)
        chunk_n = chunk_end - chunk_start
        theta_chunk = theta_arr_np[chunk_start:chunk_end]
        eta_chunk = eta_arr_np[chunk_start:chunk_end]
        t_obs_chunk = t_obs[chunk_start:chunk_end]

        # Batched sampling: for each theta_i in the chunk, draw n_mc
        # datasets of n_obs observations under H_0:theta_i. The chunk
        # offset in the seed keeps each chunk's stream reproducible
        # AND independent across chunks.
        rng = np.random.default_rng(int(derived_seed) + chunk_start)
        D_3d = _sample_data_batch_per_theta(
            model, theta_chunk, rng, int(n_mc), int(n_obs)
        )  # (chunk_n, n_mc, n_obs)

        # Posterior moments per (chunk_i, mc_j). Reshape to 2D for
        # `posterior_moments_batch`, then reshape back.
        D_flat = D_3d.reshape(chunk_n * int(n_mc), int(n_obs))
        mu_post_flat, var_post_flat = _posterior_moments_batch(model, D_flat, prior)
        mu_post_3d = mu_post_flat.reshape(chunk_n, int(n_mc))
        var_post_3d = var_post_flat.reshape(chunk_n, int(n_mc))

        # Single conservative theta-window for the tilted-density grid,
        # covering all (chunk_i, mc_j) posteriors. Computed from the
        # union of (mu - 8 sigma_post, mu + 8 sigma_post) across rows.
        sigma_post_3d = np.sqrt(np.maximum(var_post_3d, 1e-300))
        finite_mu = np.isfinite(mu_post_3d) & np.isfinite(sigma_post_3d)
        if not np.any(finite_mu):
            p_out[chunk_start:chunk_end] = 1.0  # all collapsed → conservative
            continue
        half_chunk = 8.0 * sigma_post_3d
        lo_chunk = max(
            float(np.nanmin((mu_post_3d - half_chunk)[finite_mu])),
            support_lo,
        )
        hi_chunk = min(
            float(np.nanmax((mu_post_3d + half_chunk)[finite_mu])),
            support_hi,
        )
        if not np.isfinite(lo_chunk) or not np.isfinite(hi_chunk) or not (lo_chunk < hi_chunk):
            p_out[chunk_start:chunk_end] = 1.0
            continue
        theta_grid_chunk = np.linspace(lo_chunk, hi_chunk, int(n_grid))

        # Batched log-likelihood evaluation across rows × grid.
        log_lik_2d = _batch_loglik_grid(model, D_flat, theta_grid_chunk)  # (chunk_n*n_mc, n_grid)
        log_lik_3d = log_lik_2d.reshape(chunk_n, int(n_mc), int(n_grid))
        log_prior = np.asarray(prior.logpdf(theta_grid_chunk), dtype=np.float64)

        # log_q[i, j, k] = log_lik_3d[i, j, k] + (1 - eta_chunk[i]) * log_prior[k]
        log_q = log_lik_3d + (1.0 - eta_chunk[:, None, None]) * log_prior[None, None, :]
        log_q = np.where(np.isfinite(log_q), log_q, -1e300)
        log_q_max = log_q.max(axis=-1, keepdims=True)
        pdf_unnorm = np.exp(log_q - log_q_max)
        Z = np.trapezoid(pdf_unnorm, theta_grid_chunk, axis=-1)  # (chunk_n, n_mc)
        Z_safe = np.where(Z > 0, Z, 1.0)
        pdf = pdf_unnorm / Z_safe[..., None]
        mean_3d = np.trapezoid(theta_grid_chunk[None, None, :] * pdf, theta_grid_chunk, axis=-1)
        m2_3d = np.trapezoid(theta_grid_chunk[None, None, :] ** 2 * pdf, theta_grid_chunk, axis=-1)
        var_tilted_3d = np.maximum(m2_3d - mean_3d * mean_3d, 0.0)

        # Per-row collapse mask (bad normalisation OR zero variance).
        finite_z = (Z > 0) & np.isfinite(Z)
        finite_var = (var_tilted_3d > 0) & np.isfinite(var_tilted_3d)
        finite_row = finite_z & finite_var

        # t per (i, j): (mean_tilted[i, j] - theta_chunk[i])^2 / var_tilted[i, j]
        diff_3d = mean_3d - theta_chunk[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            t_3d = diff_3d * diff_3d / np.where(finite_row, var_tilted_3d, 1.0)
        t_3d = np.where(finite_row, t_3d, 0.0)  # collapsed → t=0 per legacy

        # Empirical p per theta_i with (k+1)/(n+1) smoothing.
        more_extreme = t_3d >= t_obs_chunk[:, None]
        n_more = float_(more_extreme.sum(axis=-1))
        n_eff = finite_row.sum(axis=-1).astype(np.float64)
        # Where n_eff == 0 (all rows collapsed), use conservative p=1.
        with np.errstate(invalid="ignore", divide="ignore"):
            p_chunk = np.where(
                n_eff > 0,
                (n_more + 1.0) / (n_eff + 1.0),
                1.0,
            )
        p_out[chunk_start:chunk_end] = p_chunk
    return p_out


def _sample_data_batch_per_theta(
    model: object,
    theta_arr: NDArray[np.float64],
    rng: np.random.Generator,
    n_mc: int,
    n_obs: int,
) -> NDArray[np.float64]:
    """Draw `n_mc × n_obs` observations under H_0:theta_arr[i] for each i.

    Returns shape ``(n_theta, n_mc, n_obs)``. Vectorised for NN
    (single rng.normal call shifted per theta) and Bernoulli (single
    rng.binomial call with broadcast probability). Falls back to a
    Python loop over theta for models without these properties.
    """
    from ..models.normal_normal import NormalNormalModel
    from ..models.bernoulli import BernoulliModel

    n_theta = int(theta_arr.size)
    n_mc, n_obs = int(n_mc), int(n_obs)
    if isinstance(model, NormalNormalModel):
        # Single rng call returns N(0, sigma²) draws; shift per theta.
        # ~50x faster than n_theta separate rng.normal calls.
        z = rng.normal(loc=0.0, scale=model.sigma, size=(n_theta, n_mc, n_obs))
        return z + theta_arr[:, None, None]
    if isinstance(model, BernoulliModel):
        # rng.binomial accepts an n-shape `p` argument that broadcasts.
        # Broadcast theta to (n_theta, 1, 1) and rely on numpy's
        # broadcasting to fill the (n_theta, n_mc, n_obs) tensor.
        theta_b = np.broadcast_to(theta_arr[:, None, None], (n_theta, n_mc, n_obs))
        return rng.binomial(1, theta_b).astype(np.float64)
    # Generic fallback — Python loop. Slow but correct for any model.
    out = np.empty((n_theta, n_mc, n_obs), dtype=np.float64)
    from ..models.base import sample_data_batch as _sample_data_batch
    for i, theta_f in enumerate(theta_arr):
        out[i] = _sample_data_batch(model, float(theta_f), rng, n_mc, n_obs)
    return out


# Helper alias so the vectorised pvalue formula stays dense at the
# call site; `np.float_` was deprecated in numpy 2.0 — use `.astype`.
def float_(x: NDArray) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def _generic_tilted_pvalue(
    theta: float,
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta: float,
    statistic_name: str,
    *,
    n_mc: int = _GENERIC_TILTED_PVALUE_N_MC,
    derived_seed: int | None = None,
    alpha: float = 0.0,
    base_seed: int = _GENERIC_TILTED_PVALUE_BASE_SEED,
    obs_moments: tuple[float, float] | None = None,
) -> float:
    """Generic MC tilted p-value — works on any (model, prior) pair.

    For ``statistic_name="wald"``: eta-independent, delegates to
    ``WaldStatistic._generic_pvalue`` (the χ²₁-calibrated MLE-based path).

    For ``statistic_name="waldo"``: compute the tilted-WALDO statistic
    ``t(D, theta) = (mu_tilted - theta)^2 / sigma_tilted^2`` at observed
    data D and at n_mc synthetic D' samples drawn from likelihood(.|theta).
    Empirical tail probability with `(k+1)/(n+1)` smoothing.

    The MC reference is **CRN-seeded** (same root cause + fix as the
    Phase 2 WALDO blake2b fix): seed depends on data + fingerprints,
    NOT on theta, so brentq probes share the same internal uniform
    stream and `f(theta)` is a deterministic function of theta.
    """
    from ..statistics.wald import WaldStatistic
    from ..models.distributions import BernoulliLikelihood, GaussianLikelihood

    if statistic_name == "wald":
        # Wald is eta-independent; delegate to the Phase 2 generic path.
        # `data` is already a numpy array.
        return float(np.asarray(WaldStatistic()._generic_pvalue(theta, data, model)))

    if statistic_name != "waldo":
        raise NotImplementedError(
            f"Generic tilted_pvalue not implemented for statistic={statistic_name!r}; "
            f"supported: 'wald', 'waldo'."
        )

    data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if data_arr.ndim != 1:
        raise NotImplementedError(
            "Generic tilted_pvalue currently expects 1-D data; got "
            f"data.ndim={data_arr.ndim}."
        )

    support = _resolve_support(model, data_arr)

    theta_f = float(theta)
    eta_f = float(eta)

    if derived_seed is None:
        derived_seed = _stable_tilted_pvalue_seed(
            data_arr, model, prior, eta_f, alpha, base_seed
        )

    # Observed tilted moments are theta-INDEPENDENT given the data;
    # callers in a hot loop (CI inversion, per-theta pvalue) should
    # hoist them and pass via `obs_moments` to skip the redundant
    # recomputation per call. (Skeptic Phase 3 finding #3.)
    if obs_moments is not None:
        mu_obs, var_obs = obs_moments
    else:
        posterior_obs = model.posterior(data_arr, prior)
        likelihood_obs = model.likelihood(data_arr)
        mu_obs, var_obs = _generic_tilted_moments(
            posterior_obs, prior, likelihood_obs, eta_f,
            support=support, n_grid=_GENERIC_TILT_N_GRID,
        )
    var_obs_safe = max(var_obs, 1e-300)
    diff_obs = mu_obs - theta_f
    t_obs = diff_obs * diff_obs / var_obs_safe

    # MC reference under H_0: theta_0 = theta — vectorised.
    # Replaces the per-iteration Python loop (each iteration:
    # model.sample_data + posterior + likelihood + 256-pt grid
    # integration ~ms) with a single batched sample + a single batched
    # tilted-moments call. ~50–200x speedup on Normal-Normal at n_mc=200.
    rng = np.random.default_rng(derived_seed)
    n_obs = int(data_arr.size)
    t_samples, n_collapsed = _generic_tilted_mc_reference_batch(
        theta_f=theta_f,
        n_obs=n_obs,
        model=model,
        prior=prior,
        eta_f=eta_f,
        support=support,
        n_mc=int(n_mc),
        n_grid=_GENERIC_TILTED_PVALUE_N_GRID_MC,
        rng=rng,
    )
    if n_collapsed > n_mc // 2:
        # More than half the MC samples collapsed — the conservative-
        # direction bias is too large to be useful. Surface as a warning.
        import warnings
        warnings.warn(
            f"_generic_tilted_pvalue: {n_collapsed}/{n_mc} MC samples "
            f"collapsed (theta={theta_f}, eta={eta_f}); empirical p is "
            f"strongly biased upward. Increase data size or reduce eta.",
            RuntimeWarning,
            stacklevel=2,
        )

    # +1 smoothing: conservative empirical p-value (see Phase 2 WALDO).
    p = (float(np.sum(t_samples >= t_obs)) + 1.0) / (float(n_mc) + 1.0)
    return float(p)


def _generic_tilted_confidence_interval(
    alpha: float,
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta: float,
    statistic_name: str,
    *,
    n_mc: int = _GENERIC_TILTED_PVALUE_N_MC,
    base_seed: int = _GENERIC_TILTED_PVALUE_BASE_SEED,
) -> tuple[float, float]:
    """Generic CI inversion: invert `_generic_tilted_pvalue(...) >= alpha` via brentq.

    Single seed per CI call (CRN); brentq sees a deterministic function
    of theta. Bracket-doubling outward from `mu_tilted` (the natural
    centre of the tilted-WALDO distribution at observed data); clamps
    theta to model.support() so out-of-support brentq probes don't
    raise inside `model.sample_data`.
    """
    from .._errors import BracketingFailed
    from ._solvers import brentq_with_doubling

    data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    eta_f = float(eta)
    derived_seed = _stable_tilted_pvalue_seed(
        data_arr, model, prior, eta_f, alpha, base_seed
    )

    support = _resolve_support(model, data_arr)
    support_lo, support_hi = support

    # Hoist observed moments (skeptic finding #3): mu_obs, var_obs are
    # theta-INDEPENDENT — compute once, reuse across every brentq probe
    # via the `obs_moments` kwarg on `_generic_tilted_pvalue`. ~10x speedup
    # on the brentq path.
    posterior_at_obs = model.posterior(data_arr, prior)
    likelihood_at_obs = model.likelihood(data_arr)
    mu_obs, var_obs = _generic_tilted_moments(
        posterior_at_obs, prior, likelihood_at_obs, eta_f,
        support=support, n_grid=_GENERIC_TILT_N_GRID,
    )
    var_obs_safe = max(var_obs, 1e-300)
    sigma_tilted = float(np.sqrt(var_obs_safe))

    def f(theta_val: float) -> float:
        # Clamp to support: brentq's bracket-doubling can probe out-of-
        # support θ where `model.sample_data(theta, ...)` would raise.
        # Clamping makes f flat outside support → BracketingFailed
        # cleanly when the CI extends to the boundary.
        theta_safe = max(support_lo, min(support_hi, float(theta_val)))
        return _generic_tilted_pvalue(
            theta_safe,
            data_arr,
            model,
            prior,
            eta_f,
            statistic_name,
            n_mc=n_mc,
            derived_seed=derived_seed,
            alpha=alpha,
            base_seed=base_seed,
            obs_moments=(mu_obs, var_obs),
        ) - alpha

    half = max(4.0 * sigma_tilted, 1e-3)

    # Skeptic finding #1 — explicit boundary detection. If
    # `f(support_lo) >= 0` (resp. `f(support_hi) >= 0`), the CI extends
    # PAST that boundary. brentq-with-doubling on the clamped flat
    # plateau would just exhaust its 16 doublings and raise
    # BracketingFailed — caught and snapped to support, but with no
    # telemetry. Detect the boundary case explicitly.
    if np.isfinite(support_lo):
        f_at_lower_support = f(support_lo)
        ci_extends_below = (f_at_lower_support >= 0.0)
    else:
        ci_extends_below = False
    if np.isfinite(support_hi):
        f_at_upper_support = f(support_hi)
        ci_extends_above = (f_at_upper_support >= 0.0)
    else:
        ci_extends_above = False

    if ci_extends_below:
        lower = support_lo
    else:
        try:
            lower = brentq_with_doubling(
                f, midpoint=mu_obs, initial_half_width=half, direction=-1
            )
        except BracketingFailed:
            lower = support_lo
    if ci_extends_above:
        upper = support_hi
    else:
        try:
            upper = brentq_with_doubling(
                f, midpoint=mu_obs, initial_half_width=half, direction=+1
            )
        except BracketingFailed:
            upper = support_hi
    return (max(lower, support_lo), min(upper, support_hi))


def _data_to_scalar_D(data: NDArray[np.float64]) -> float:
    """Coerce ``data`` to a single scalar D for the n=1 sandbox.

    The framework's Normal-Normal contract is single-observation
    (CLAUDE.md "1D conjugate Normal-Normal sandbox"). Earlier code
    silently used ``data.mean()`` which produced wrong CI widths for
    n>1 (the effective σ would shrink as σ/√n, not σ). Make the
    assumption explicit: refuse n>1 with a clear message rather than
    silently mis-scaling. Tier 1.5-O8 in the audit.
    """
    arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if arr.size != 1:
        raise NotImplementedError(
            f"Normal-Normal sandbox is single-observation (n=1); got "
            f"data.size={arr.size}. For n>1, use sigma_eff=sigma/sqrt(n) "
            f"and pass data.mean() with a model whose sigma is sigma_eff."
        )
    # Use ``arr.item()`` so shape-(1,1) and other non-flat single-element
    # inputs reduce cleanly. ``float(arr[0])`` would crash with
    # ``TypeError: only 0-dimensional arrays can be converted to Python
    # scalars`` on a shape-(1,1) input where ``arr[0]`` is a shape-(1,)
    # array. Phase 5 skeptic vector #3.
    return float(arr.item())


@register_tilting(name="power_law", brief="docs/methods/power_law.md")
@dataclass(frozen=True)
class PowerLawTilting:
    """The legacy eta-tilting scheme as a `TiltingScheme` implementation.

    Parameterised by an `EtaSelector` that decides the η used at CI
    inversion time. Static selectors (`FixedEtaSelector`,
    `NumericalEtaSelector`) route through `tilted_confidence_interval`;
    dynamic selectors (`DynamicNumericalEtaSelector`) route through
    `dynamic_tilted_confidence_interval` and produce per-θ varying η.
    The cell display name picks up the selector's name when non-default
    so the runner emits distinguishable cells like
    `power_law[dynamic_numerical]`.
    """

    name: ClassVar[str] = "power_law"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="eta=0 recovers WALDO; eta=1 recovers Wald.",
    )
    selector: EtaSelector = field(default_factory=lambda: FixedEtaSelector(eta=0.0))

    @property
    def cell_name(self) -> str:
        """Display name for the runner's cell key.

        `power_law` for the default fixed-η-zero selector (matches the
        identity tilting numerically); `power_law[<selector>]` otherwise.
        """
        sel_name = getattr(self.selector, "name", "")
        if isinstance(self.selector, FixedEtaSelector) and self.selector.eta == 0.0:
            return self.name
        return f"{self.name}[{sel_name}]"

    # ----- TiltingScheme protocol -----

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> Posterior:
        """Tilted distribution `q(theta; eta) ∝ L(theta) * pi(theta)^(1-eta)`.

        Dispatch:
          - **Normal-Normal closed form (Theorem 6)** when (posterior, prior,
            likelihood) is the (NormalDistribution, NormalDistribution,
            GaussianLikelihood) triple. Returns a `NormalDistribution`.
          - **Generic numerical fallback** otherwise. Builds the tilted
            log-density `log L(theta) + (1-eta) * log pi(theta)` on a 1024-
            point theta grid sized from the union of posterior and prior
            quantile windows (clipped to `model.support()`), normalises
            via trapezoidal Z, and wraps the result as a `GridDistribution`.
            Requires `model.support()` to be supplied via the `_support`
            attribute or attribute-of-likelihood; pass through the
            `confidence_regions` / `pvalue` consumer paths instead of
            calling directly when possible.

        Generic-vs-closed-form agreement on Normal-Normal: atol 1e-7 at
        N=1024 (per the deriver-verified Theorem-6 reduction; cross-checked
        in `tests/regression/test_power_law_generic_matches_closed_form.py`).

        Backward-compat note: ``tilt()`` previously returned only
        `NormalDistribution`; the return type widens to `Posterior`
        (the Distribution protocol). Callers that rely on the concrete
        `NormalDistribution` API (`.loc`, `.scale`) on Normal-Normal
        inputs continue to work unchanged — the closed-form branch
        still returns `NormalDistribution`. Generic-path consumers
        must use the protocol surface (`pdf`, `cdf`, `mean`, `var`).
        """
        eta_arr = np.asarray(eta, dtype=np.float64)
        if eta_arr.ndim != 0:
            raise NotImplementedError(
                "tilt() expects scalar eta; vectorised eta is consumed "
                "by the smoothness experiment via repeated scalar calls."
            )
        eta_f = float(eta_arr)

        if _is_gaussian_triple(posterior, prior, likelihood):
            # Closed-form Theorem-6 fast path.
            post, pri, lik = _require_gaussian(posterior, prior, likelihood)
            w = _weight(lik.sigma, pri.scale)
            denom = _denom(w, eta_f)
            if denom <= 0.0:
                raise TiltingDomainError(
                    f"eta={eta_f!r} drives the tilted-posterior denominator "
                    f"to {denom!r} <= 0 with w={w!r}; admissible range is "
                    f"({-w/(1-w):+.6g}, inf)."
                )
            mu_eta = (w * lik.D + (1.0 - eta_f) * (1.0 - w) * pri.loc) / denom
            sigma_eta_sq = w * lik.sigma**2 / denom
            sigma_eta = float(np.sqrt(sigma_eta_sq))
            return NormalDistribution(loc=float(mu_eta), scale=sigma_eta)

        # Generic numerical fallback. Centralised support resolution
        # via `_resolve_support` (works for any model that conforms to
        # the protocol's `support()` plus a likelihood-class fallback).
        from ..models.distributions import BernoulliLikelihood, GaussianLikelihood
        if isinstance(likelihood, BernoulliLikelihood):
            support_attr = (0.0, 1.0)
        elif isinstance(likelihood, GaussianLikelihood):
            support_attr = (-float("inf"), float("inf"))
        else:
            support_attr = (-float("inf"), float("inf"))

        return _generic_tilt(
            posterior, prior, likelihood, eta_f, support=support_attr
        )

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[NormalDistribution]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, t)

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    # ----- (TiltingScheme, TestStatistic) cross-product specialisations -----
    #
    # The smoothness experiment needs `tilted p-value` and `tilted CI` —
    # quantities that depend on *both* the tilting scheme and the test
    # statistic. The cleanest factoring would be multiple dispatch on
    # (scheme, statistic) types; for now we dispatch on `statistic.name`
    # inside the scheme. Documented as a temporary bridge in
    # `docs/methods/power_law.md`. The generalisation lands when a
    # second non-Wald/WALDO statistic gets implemented.

    def tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float | NDArray[np.float64],
        model: object,
        prior: NormalDistribution,
        eta: ArrayLike,
        statistic_name: str,
    ) -> float | jax.Array:
        """p-value of `statistic_name` evaluated against the tilted posterior.

        Specialized for (power_law, waldo) — Theorem 8 closed form — and
        (power_law, wald) — eta-independent two-sided Wald.

        ``eta`` accepts either a scalar (the historical contract) or an
        array broadcastable to ``theta``. The array path is what
        ``dynamic_ci_scan`` (and ``dynamic_tilted_pvalue``) use to evaluate
        a per-θ varying η in one bulk JAX call (Tier 1.3 N1/N3).

        ``D`` accepts either a scalar (the historical contract) or an
        array broadcastable to ``theta_arr``. The closed-form formulas
        are pure broadcasting arithmetic.

        Returns ``jax.Array`` for bulk inputs (length >= 2 in any of
        theta/eta/D); returns Python float for the all-scalar fast
        path. The relaxed scalar return type is deliberate: wrapping a
        scalar in ``jnp.asarray(...)`` costs ~200 us (dwarfs the ~10 us
        scalar arithmetic), defeating the dispatch. Callers that need
        a uniform return type wrap in ``float(...)`` or
        ``jnp.asarray(...)`` at their own boundary; this is what the
        brentq closures in ``tilted_confidence_interval`` and
        ``dynamic_tilted_confidence_interval`` already do, and what
        ``dynamic_tilted_pvalue`` already does via ``np.asarray(...)``.
        See ``docs/jax_style.md``.
        """
        from ..models.normal_normal import NormalNormalModel

        if not is_normal_normal(model):
            raise NotImplementedError(
                "tilted_pvalue currently requires NormalNormalModel; "
                f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "tilted_pvalue currently requires a NormalDistribution prior."
            )
        sigma = model.sigma
        mu0 = prior.loc
        sigma0 = prior.scale
        w = sigma0**2 / (sigma**2 + sigma0**2)

        # Validation runs in numpy (JAX can't raise mid-trace; validation
        # lives outside the autodiff-clean kernel). Convert eta to numpy
        # exactly once for the finiteness + admissibility checks; the
        # JAX kernel re-asarrays internally.
        eta_np = np.asarray(eta, dtype=np.float64)
        mask_invalid = ~np.isfinite(eta_np)
        if np.any(mask_invalid):
            bad_idx = int(np.argmax(mask_invalid))
            raise TiltingDomainError(
                f"PowerLawTilting.tilted_pvalue requires finite eta; "
                f"offending index {bad_idx} eta="
                f"{float(eta_np.flat[bad_idx])!r}"
            )
        denom_np = 1.0 - eta_np * (1.0 - w)
        if np.any(denom_np <= 0.0):
            if eta_np.ndim == 0:
                raise TiltingDomainError(
                    f"eta={float(eta_np)!r} drives denom to "
                    f"{float(denom_np)!r} <= 0 with w={w!r}."
                )
            bad = int(np.argmax(denom_np <= 0.0))
            raise TiltingDomainError(
                f"PowerLawTilting.tilted_pvalue: eta drives denom <= 0 at "
                f"index {bad} (eta={float(eta_np.flat[bad])!r}, "
                f"denom={float(np.asarray(denom_np).flat[bad])!r}, w={w!r})."
            )

        # Shape dispatch: scalar inputs route through the numpy fast path
        # (~10 us per call) returning a Python float; bulk inputs route
        # through the autodiff-clean JAX kernel (~10 us per call after
        # jit warmup; the first call per unique shape pays a ~50 ms
        # compile) returning a jax.Array. The criterion is "all of
        # theta, eta, D are scalar" — when even one is an array, the
        # JAX broadcast handles the result. The return-type union is
        # declared in the signature; callers that need a uniform type
        # convert at their boundary (see `dynamic_tilted_pvalue` ->
        # `np.asarray(...)`).
        theta_np = np.asarray(theta, dtype=np.float64)
        D_np = np.asarray(D, dtype=np.float64)
        if theta_np.size == 1 and eta_np.size == 1 and D_np.size == 1:
            # Scalar fast path: return Python float directly, NOT
            # jnp.asarray(float). Wrapping in jnp.asarray would cost
            # ~200 us per call (dwarfing the ~10 us scalar arithmetic
            # below) and defeat the dispatch.
            return _tilted_pvalue_numpy_scalar(
                # .item() handles shape-(), shape-(1,), shape-(1,1) uniformly;
                # plain float(arr) fails on >0-d single-element arrays.
                float(theta_np.item()),
                float(eta_np.item()),
                float(D_np.item()),
                w,
                mu0,
                sigma,
                statistic_name,
            )
        # Bulk JAX kernel — autodiff-clean, no Python control flow.
        theta_arr = jnp.asarray(theta, dtype=jnp.float64)
        eta_arr = jnp.asarray(eta, dtype=jnp.float64)
        D_arr = jnp.asarray(D, dtype=jnp.float64)
        return _tilted_pvalue_kernel(theta_arr, eta_arr, D_arr, w, mu0, sigma, statistic_name)

    def dynamic_tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float,
        model: object,
        prior: NormalDistribution,
        statistic_name: str,
        eta_at_theta: ArrayLike,
    ) -> NDArray[np.float64]:
        """p(theta) with η varying per θ via a precomputed lookup.

        Caller supplies `eta_at_theta`, the η* value chosen *for each θ*.
        This is what `dynamic_tilted_confidence_interval` does internally:
        run a coarse η-selector, then interpolate η*(|Δ(θ)|) across the
        fine θ scan.
        """
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        eta_arr = np.atleast_1d(np.asarray(eta_at_theta, dtype=np.float64))
        if theta_arr.shape != eta_arr.shape:
            raise ValueError(
                f"theta and eta_at_theta must have the same shape; got "
                f"{theta_arr.shape!r} and {eta_arr.shape!r}."
            )
        # Vectorised path: `tilted_pvalue` now returns jax.Array; convert
        # to numpy at the boundary because `_dynamic.py`'s scan algorithm
        # is numpy-only by design (Brent root-find inner loops). One bulk
        # call replaces the scalar Python loop; an invalid eta in any
        # slot still raises TiltingDomainError with the offending index
        # identified (Tier 1.3 N3).
        out = np.asarray(
            self.tilted_pvalue(theta_arr, D, model, prior, eta_arr, statistic_name),
            dtype=np.float64,
        )
        return out if out.size > 1 else np.asarray(float(out.item()))

    def dynamic_tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: object,
        prior: NormalDistribution,
        statistic_name: str,
        eta_selector,
        n_grid: int = 401,
        coarse_n: int = 25,
        search_mult: float = 8.0,
    ) -> tuple[list[tuple[float, float]], float, int]:
        """Dynamic-η CI: η = η*(|Δ(θ)|) per θ.

        Delegates the scan + α-crossing algorithm to
        `frasian.tilting._dynamic.dynamic_ci_scan` (see that function's
        docstring for the documented step-by-step recipe). The
        scheme-specific bit is the `tilted_pvalue` closure.
        """
        from ..models.normal_normal import NormalNormalModel
        from ._dynamic import dynamic_ci_scan

        if not is_normal_normal(model):
            raise NotImplementedError(
                "dynamic_tilted_confidence_interval currently requires " "NormalNormalModel."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        def _tilted_pvalue_fn(theta: float, eta: float) -> float:
            return float(
                self.tilted_pvalue(
                    theta,
                    D,
                    model,
                    prior,
                    eta,
                    statistic_name,
                )
            )

        def _tilted_pvalue_vec_fn(
            theta_arr: np.ndarray, eta_arr: np.ndarray
        ) -> np.ndarray:
            # Bulk path: tilted_pvalue broadcasts over array eta when
            # eta has the same shape as theta. Single numpy/scipy call
            # replaces the scalar Python loop in dynamic_ci_scan
            # (Tier 1.3 N1).
            return np.asarray(
                self.tilted_pvalue(
                    theta_arr, D, model, prior, eta_arr, statistic_name
                ),
                dtype=np.float64,
            )

        return dynamic_ci_scan(
            tilted_pvalue_fn=_tilted_pvalue_fn,
            tilted_pvalue_vec_fn=_tilted_pvalue_vec_fn,
            alpha=alpha,
            D=D,
            w=w,
            mu0=mu0,
            sigma=sigma,
            eta_selector=eta_selector,
            scheme=self,
            statistic_name=statistic_name,
            n_grid=n_grid,
            coarse_n=coarse_n,
            search_mult=search_mult,
            model_fingerprint=model.fingerprint(),
            prior_fingerprint=prior.fingerprint(),
            model=model,
            prior=prior,
        )

    def dynamic_tilted_confidence_interval_generic(
        self,
        alpha: float,
        D: float,
        data: NDArray[np.float64],
        model: object,
        prior: NormalDistribution,
        statistic_name: str,
        eta_selector,
        n_grid: int = 401,
        coarse_n: int = 25,
        search_mult: float = 8.0,
        n_mc: int = _GENERIC_TILTED_PVALUE_N_MC,
    ) -> tuple[list[tuple[float, float]], float, int]:
        """Dynamic-η CI inversion using the GENERIC (MC) tilted p-value.

        Algorithm:
          1. Coarse-grid η*(θ) selection via the supplied `eta_selector`
             (NumericalEtaSelector internally — closed-form NN, fast).
          2. Interpolate η*(θ) onto the fine θ-grid.
          3. Vectorised generic-MC p-value across all (θ_i, η_i) pairs
             via `_generic_tilted_pvalue_vec`. Single batched call.
          4. **Linear interpolation at each α-crossing** — NOT brentq.
             Fine grid spacing at canonical settings is `16σ/400 = 0.04σ`;
             linear-interp accuracy is `O((0.04σ)² × p-curvature)` ≈
             sub-mm in θ-space, well below MC noise on the p-value.
             brentq would refine each crossing further but each
             refinement is a fresh scalar `_generic_tilted_pvalue` call
             (~100 ms at n_mc=2000) — for ~30 crossings that's ~3s
             additional cost per CI with no measurable accuracy gain.

        Cost: dominated by the vec fine-scan call. At n_mc=2000,
        n_grid=401, n_grid_mc=256: ~1-2 s/CI. With n_jobs=8 parallelism,
        ~30 min per coverage experiment.

        Caveat: with low n_mc (~200), MC noise on the fine scan can
        produce spurious sub-α regions near the true CI boundaries —
        the WaldoStatistic.n_mc default is 2000, which is the safe
        regime at α=0.05. Using `WaldoStatistic(n_mc=...)` lower trades
        accuracy for speed.
        """
        from ..models.normal_normal import NormalNormalModel
        from ._generic_pvalue import _stable_tilted_pvalue_seed
        from .eta_selectors import _NamedStatistic

        if not is_normal_normal(model):
            raise NotImplementedError(
                "dynamic_tilted_confidence_interval_generic currently requires "
                "NormalNormalModel — the dynamic-η selector is NN-only by design "
                "(NumericalEtaSelector inside is closed-form NN); generalising "
                "to non-NN models requires a generic eta selector first."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)

        # CRN seed: stable across processes. Per-CI; brentq isn't used
        # here but the same seed is reused for any future scalar calls.
        derived_seed = _stable_tilted_pvalue_seed(
            np.atleast_1d(np.asarray(data, dtype=np.float64)),
            model, prior, 0.0, alpha, _GENERIC_TILTED_PVALUE_BASE_SEED,
        )

        # 1. Search window — the same Normal-Normal `D ± k·σ` heuristic
        #    used by `dynamic_ci_scan`. With auto-widen on boundary hit.
        for widen in (1.0, 2.0):
            search_half = float(widen * search_mult * sigma)
            theta_lo = D - search_half
            theta_hi = D + search_half
            theta_grid = np.linspace(theta_lo, theta_hi, int(n_grid))

            # 2. Coarse η selection — closed-form NN via NumericalEtaSelector.
            coarse_theta_grid = np.linspace(theta_lo, theta_hi, int(coarse_n))
            coarse_eta = eta_selector.select_grid(
                coarse_theta_grid,
                self,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=_NamedStatistic(statistic_name),
            )
            eta_at_theta = np.interp(theta_grid, coarse_theta_grid, coarse_eta)

            # 3. Fine-scan p-values via the triple-batched MC vec fn.
            data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
            p_theta = _generic_tilted_pvalue_vec(
                theta_grid,
                data_arr,
                model,
                prior,
                eta_at_theta,
                statistic_name,
                n_mc=int(n_mc),
                n_grid=_GENERIC_TILTED_PVALUE_N_GRID_MC,
                derived_seed=derived_seed,
                alpha=alpha,
                base_seed=_GENERIC_TILTED_PVALUE_BASE_SEED,
            )

            # 4. Find α-crossings via linear interpolation; build regions.
            diff = p_theta - alpha
            sgn = np.where(diff >= 0.0, 1, -1).astype(np.int64)
            crossings: list[float] = []
            for i in range(theta_grid.size - 1):
                if sgn[i] != sgn[i + 1]:
                    denom = diff[i] - diff[i + 1]
                    if denom == 0.0:
                        crossings.append(float(0.5 * (theta_grid[i] + theta_grid[i + 1])))
                    else:
                        t = diff[i] / denom
                        crossings.append(
                            float(theta_grid[i] + t * (theta_grid[i + 1] - theta_grid[i]))
                        )

            # Pad with window edges when the boundary is inside the
            # accept region — same logic as `dynamic_ci_scan`.
            entries = list(crossings)
            hit_boundary = False
            if sgn[0] > 0:
                entries = [float(theta_lo)] + entries
                hit_boundary = True
            if sgn[-1] > 0:
                entries = entries + [float(theta_hi)]
                hit_boundary = True

            if not hit_boundary:
                break  # search window large enough; done
            if widen == 2.0:
                # Boundary still hit at 2× search width — escalate.
                from .._errors import BracketingFailed
                raise BracketingFailed(
                    f"dynamic_tilted_confidence_interval_generic: CI extends "
                    f"past search box (±{search_half}·σ around D={D!r}; "
                    f"sigma={sigma!r}). Increase search_mult."
                )

        if len(entries) % 2 != 0:
            from .._errors import BracketingFailed
            raise BracketingFailed(
                f"dynamic_tilted_confidence_interval_generic produced "
                f"odd-parity entries; got {entries!r}. Indicates a missed "
                f"tangential α-touch on the grid. Try a finer theta-grid."
            )

        regions: list[tuple[float, float]] = [
            (entries[i], entries[i + 1]) for i in range(0, len(entries), 2)
        ]
        total = float(sum(hi - lo for lo, hi in regions))
        return regions, total, len(regions)

    # ----- Uniform CI / regions / pvalue interface (called by experiments) -----

    def _require_normal_sandbox(self, model: Model, prior: Prior) -> None:
        """Enforce Normal-Normal sandbox. Used by paths that have no
        generic fallback (currently the dynamic-η selector path —
        `dynamic_ci_scan` builds its θ-window from `D ± search_mult * sigma`
        which is Normal-Normal-flavoured. The static-selector path
        in `confidence_regions` / `confidence_interval` / `pvalue`
        dispatches to `_generic_tilted_*` for non-Normal pairings.)
        """
        from ..models.normal_normal import NormalNormalModel

        if not is_normal_normal(model):
            raise NotImplementedError(
                "PowerLawTilting requires NormalNormalModel; " f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "PowerLawTilting requires a NormalDistribution prior; "
                f"got {type(prior).__name__!r}."
            )

    @staticmethod
    def _is_normal_normal_pair(model: Model, prior: Prior) -> bool:
        """Predicate-only counterpart to `_require_normal_sandbox` for
        `confidence_regions` etc. to dispatch between closed-form fast
        path and generic numerical fallback."""
        from ..models.normal_normal import NormalNormalModel

        return is_normal_normal(model) and isinstance(prior, NormalDistribution)

    def confidence_regions(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: Config | None = None,
    ) -> list[tuple[float, float]]:
        """Selector-aware region list. Single-element for static selectors;
        multi-element for dynamic selectors at conflict-band D where the
        dynamic p-value is multimodal.

        Phase 3c: non-Normal-Normal pairings are now supported via the
        generic numerical path, but ONLY with a static selector. Dynamic
        selectors still require the Normal-Normal sandbox (the dynamic
        scanner builds its θ-window from ``D ± search_mult * sigma``,
        which is Normal-Normal-flavoured; generalising it is a
        separate research item).

        ``config`` (optional, kw-only): when supplied, the dynamic-CI
        scan reads ``n_grid`` / ``coarse_n`` / ``search_mult`` from
        ``Config.dynamic_*``. When ``None`` (default), falls back to
        the selector's own attributes (or the function-default
        constants when the selector lacks them) for backward
        compatibility. Skeptic Phase 5 vector #2: previously the
        Config fields fed only the cache fingerprint, never the
        runtime path; ``config`` plumbing closes that gap.
        """
        # Phase 3c: dispatch on (model, prior) — generic for non-Normal-Normal,
        # closed-form for Normal-Normal. Dynamic selectors require NN
        # for the eta-selector itself (NumericalEtaSelector is NN-only).
        # Path-coverage-debug override: a statistic carrying
        # `force_generic=True` collapses both branches onto the generic
        # MC path even on NN, so the "tilted CI = invert tilted p-value =
        # generic test-statistic computation" architecture is exercised
        # uniformly.
        #
        # NEW (Phase F): dynamic + force_generic IS supported on NN via
        # the triple-batched `_generic_tilted_pvalue_vec` (vectorises 401
        # fine-scan p-values into one call). η selection stays closed-
        # form (NumericalEtaSelector internal); only the CI inversion
        # uses generic MC. ~3 s/CI vs the closed-form ~0.7 ms but
        # tractable; gives an honest path-coverage check on the dynamic
        # cell. For non-NN pairings the dynamic selector still raises.
        force_generic = bool(getattr(statistic, "force_generic", False))
        route_generic = force_generic or not self._is_normal_normal_pair(model, prior)
        if route_generic:
            if getattr(self.selector, "is_dynamic", False):
                if force_generic and self._is_normal_normal_pair(model, prior):
                    # Phase F: triple-batched generic dynamic on NN.
                    if config is not None:
                        n_grid = int(config.dynamic_n_grid)
                        coarse_n = int(config.dynamic_coarse_n)
                        search_mult = float(config.dynamic_search_mult)
                    else:
                        n_grid = int(getattr(self.selector, "n_grid", 401))
                        coarse_n = int(getattr(self.selector, "coarse_n", 25))
                        search_mult = float(getattr(self.selector, "search_mult", 8.0))
                    # Honour `statistic.n_mc` (default _GENERIC_TILTED_PVALUE_N_MC=200).
                    # Dynamic-η fine-scan p-values cross α many times when MC noise
                    # is comparable to alpha; n_mc>=2000 is the safe regime at α=0.05.
                    stat_n_mc = int(getattr(statistic, "n_mc", _GENERIC_TILTED_PVALUE_N_MC))
                    D = _data_to_scalar_D(data)
                    regions, _, _ = self.dynamic_tilted_confidence_interval_generic(
                        alpha,
                        D,
                        np.atleast_1d(np.asarray(data, dtype=np.float64)),
                        model,
                        prior,
                        statistic.name,
                        self.selector,
                        n_grid=n_grid,
                        coarse_n=coarse_n,
                        search_mult=search_mult,
                        n_mc=stat_n_mc,
                    )
                    if not regions:
                        raise RuntimeError(
                            f"dynamic generic CI inversion produced no regions at D={D!r}"
                        )
                    return regions
                if force_generic:
                    raise NotImplementedError(
                        "PowerLawTilting + dynamic-η selector + "
                        "force_generic=True currently only supported on "
                        "NormalNormalModel + NormalDistribution prior. "
                        "Generalising to non-NN models requires a generic "
                        "eta-selector (NumericalEtaSelector inside is "
                        "closed-form NN)."
                    )
                raise NotImplementedError(
                    "PowerLawTilting dynamic-η selector currently requires "
                    "NormalNormalModel + NormalDistribution prior (the dynamic "
                    "scanner is Normal-Normal-flavoured). Use a static selector "
                    "(FixedEtaSelector) for non-Normal-Normal pairings, or "
                    "generalise dynamic_ci_scan to θ-only inputs as a follow-up."
                )
            # Static-selector generic path: pick η, invert tilted-pvalue
            # via brentq with CRN, return single region.
            eta = float(
                self.selector.select(
                    self,
                    data=data,
                    model=model,
                    prior=prior,
                    alpha=alpha,
                    statistic=statistic,
                )
            )
            lo, hi = _generic_tilted_confidence_interval(
                alpha, data, model, prior, eta, statistic.name
            )
            return [(lo, hi)]

        self._require_normal_sandbox(model, prior)
        # Narrow types after the dispatch check (mypy can't infer through it).
        # `cast` is `-O`-safe; the runtime gate is `_require_normal_sandbox` above.
        from ..models.normal_normal import NormalNormalModel

        model = cast(NormalNormalModel, model)
        prior = cast(NormalDistribution, prior)
        D = _data_to_scalar_D(data)
        sigma = model.sigma
        sigma0 = prior.scale
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
            # Config wins when supplied; selector attrs are the legacy
            # fallback (preserves existing behaviour for callers that
            # don't yet thread Config through).
            if config is not None:
                n_grid = int(config.dynamic_n_grid)
                coarse_n = int(config.dynamic_coarse_n)
                search_mult = float(config.dynamic_search_mult)
            else:
                n_grid = int(getattr(self.selector, "n_grid", 401))
                coarse_n = int(getattr(self.selector, "coarse_n", 25))
                search_mult = float(getattr(self.selector, "search_mult", 8.0))
            regions, _, _ = self.dynamic_tilted_confidence_interval(
                alpha,
                D,
                model,
                prior,
                statistic.name,
                self.selector,
                n_grid=n_grid,
                coarse_n=coarse_n,
                search_mult=search_mult,
            )
            if not regions:
                raise RuntimeError(f"dynamic CI inversion produced no regions at D={D!r}")
            return regions

        eta = float(
            self.selector.select(
                self,
                data=data,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=statistic,
            )
        )
        return [
            self.tilted_confidence_interval(
                alpha,
                D,
                model,
                prior,
                eta,
                statistic.name,
            )
        ]

    def confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: Config | None = None,
    ) -> tuple[float, float]:
        """Convex hull of `confidence_regions` — single (lo, hi) summary.

        Multi-region cells (e.g. dynamic-η Dyn-WALDO at low |Δ|) collapse
        to `(min lo, max hi)` here; consumers that need union semantics
        should call `confidence_regions` directly.

        ``config`` is forwarded to `confidence_regions`; see that method
        for the dynamic-CI scan semantics.
        """
        regions = self.confidence_regions(
            alpha, data, model, prior, statistic, config=config
        )
        lo = float(min(r[0] for r in regions))
        hi = float(max(r[1] for r in regions))
        return (lo, hi)

    def pvalue(
        self,
        theta: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> NDArray[np.float64]:
        """Selector-aware p-value at hypothesised θ values.

        Static selector: resolve a single η via `selector.select(...)`,
        evaluate `tilted_pvalue(θ, D, …, η, statistic.name)`.

        Dynamic selector: precompute the coarse η*(|Δ|) lookup once, then
        interpolate to the per-θ |Δ_θ| values in `theta`, and evaluate
        `dynamic_tilted_pvalue` with the resulting `eta_at_theta` array.

        The α used by the dynamic selector defaults to
        `Config.default().alpha` (= 0.05) — overridable via
        `metadata={'alpha': ...}` on the prior or via subclassing if a
        downstream consumer needs different per-α p-value evaluation.

        Phase 3c: non-Normal-Normal pairings supported via the generic
        path with a STATIC selector. Dynamic selectors remain Normal-
        Normal-only here too.
        """
        # Phase 3c: dispatch generic path for non-Normal-Normal pairings.
        # `statistic.force_generic` collapses both branches onto the
        # generic MC path even on NN — same flag used in
        # `confidence_regions`.
        force_generic = bool(getattr(statistic, "force_generic", False))
        route_generic = force_generic or not self._is_normal_normal_pair(model, prior)
        if route_generic:
            if getattr(self.selector, "is_dynamic", False):
                if force_generic:
                    raise NotImplementedError(
                        "PowerLawTilting.pvalue + dynamic-η selector + "
                        "force_generic=True is not supported (dynamic "
                        "scanner is NN-flavoured). Use a static selector."
                    )
                raise NotImplementedError(
                    "PowerLawTilting.pvalue dynamic-η selector currently "
                    "requires NormalNormalModel + NormalDistribution prior."
                )
            from ..config import Config as _Config
            alpha = float(_Config.default().alpha)
            eta = float(
                self.selector.select(
                    self,
                    data=data,
                    model=model,
                    prior=prior,
                    alpha=alpha,
                    statistic=statistic,
                )
            )
            theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
            data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
            # Compute generic tilted pvalue per θ. CRN seed depends on
            # data + fingerprints + eta + alpha (NOT theta), so the
            # per-θ MC samples share the same uniform stream and are
            # comparable across θ.
            derived_seed = _stable_tilted_pvalue_seed(
                data_arr, model, prior, eta, alpha,
                _GENERIC_TILTED_PVALUE_BASE_SEED,
            )
            out = np.empty(theta_arr.shape, dtype=np.float64)
            for i, th in enumerate(theta_arr):
                out[i] = _generic_tilted_pvalue(
                    float(th), data_arr, model, prior, eta, statistic.name,
                    derived_seed=derived_seed, alpha=alpha,
                )
            return out

        self._require_normal_sandbox(model, prior)
        # Narrow types after the dispatch check (mypy can't infer through it).
        # `cast` is `-O`-safe; the runtime gate is `_require_normal_sandbox` above.
        from ..models.normal_normal import NormalNormalModel

        model = cast(NormalNormalModel, model)
        prior = cast(NormalDistribution, prior)
        from ..config import Config
        from .eta_selectors import _NamedStatistic

        alpha = float(Config.default().alpha)
        D = _data_to_scalar_D(data)
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        sigma = model.sigma
        sigma0 = prior.scale
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
            # Phase 3a-1: dynamic selectors are θ-indexed natively. Build a
            # coarse θ-grid spanning the inference θ values and call
            # `select_grid(theta_grid, model=, prior=, ...)` directly.
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            theta_lo = float(theta_arr.min())
            theta_hi = float(theta_arr.max())
            # Pad the bounds slightly so np.interp at the extremes does
            # not extrapolate.
            half_pad = 1e-6 * max(1.0, abs(theta_hi - theta_lo))
            coarse_grid = np.linspace(theta_lo - half_pad, theta_hi + half_pad, coarse_n)
            coarse_eta = self.selector.select_grid(  # type: ignore[attr-defined]
                coarse_grid,
                self,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=_NamedStatistic(statistic.name),
            )
            eta_at_theta = np.interp(theta_arr, coarse_grid, coarse_eta)
            return self.dynamic_tilted_pvalue(
                theta_arr,
                D,
                model,
                prior,
                statistic.name,
                eta_at_theta,
            )

        # Static selector: single η for the whole evaluation.
        eta = float(
            self.selector.select(
                self,
                data=data,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=statistic,
            )
        )
        return self.tilted_pvalue(theta_arr, D, model, prior, eta, statistic.name)

    def tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: object,
        prior: NormalDistribution,
        eta: float,
        statistic_name: str,
    ) -> tuple[float, float]:
        """Numerical CI inversion of `tilted_pvalue` via brentq_with_doubling."""
        from ..models.normal_normal import NormalNormalModel
        from ._solvers import brentq_with_doubling

        if not is_normal_normal(model):
            raise NotImplementedError(
                "tilted_confidence_interval currently requires NormalNormalModel."
            )
        sigma = model.sigma
        sigma0 = prior.scale
        w = sigma0**2 / (sigma**2 + sigma0**2)

        # Use the tilted posterior mean as the bracket midpoint (CI mode for WALDO).
        denom = _denom(w, eta)
        if denom <= 0.0:
            raise TiltingDomainError(f"eta={eta!r} drives denom to {denom!r} <= 0 with w={w!r}.")
        mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * prior.loc) / denom
        if statistic_name == "wald":
            mu_eta = D  # Wald CI is centred on D, not mu_eta.

        def f(theta_val: float) -> float:
            return (
                float(self.tilted_pvalue(float(theta_val), D, model, prior, eta, statistic_name))
                - alpha
            )

        half = 4.0 * sigma
        lo = brentq_with_doubling(f, midpoint=float(mu_eta), initial_half_width=half, direction=-1)
        hi = brentq_with_doubling(f, midpoint=float(mu_eta), initial_half_width=half, direction=+1)
        return (lo, hi)
