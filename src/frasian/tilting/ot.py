"""Optimal-transport (Wasserstein-2) geodesic tilting.

The W2 geodesic between two 1D distributions p and q at parameter
`t in [0, 1]` is the **quantile-mixture**: the distribution whose
quantile function is the linear interpolation of the endpoint
quantiles,

    F_t^{-1}(u) = (1 - t) * F_p^{-1}(u) + t * F_q^{-1}(u),  u in [0, 1].

This is a general result for any two endpoints exposing `quantile`,
which is in the framework's `Distribution` protocol — so OT tilting
applies to *any* (posterior, likelihood-as-distribution) pair, not
just Gaussians. On the Normal-Normal sandbox, the geodesic stays in
the Gaussian family with the closed form

    mu_t = (1 - t) * mu_a + t * mu_b
    sigma_t = (1 - t) * sigma_a + t * sigma_b

(linear in `(mu, sigma)`, *not* in `(mu, sigma^2)`). The tilt method
recognises this Gaussian fast path and returns a `NormalDistribution`
directly; non-Gaussian endpoints fall back to a `QuantileMixturePath`
wrapper that derives `pdf` / `cdf` numerically.

Endpoints follow the framework's posterior <-> likelihood convention
(matching `power_law`): eta=0 -> posterior, eta=1 -> likelihood-induced
Gaussian N(D, sigma^2). At eta=0 the tilted-WALDO p-value reduces to
bare WALDO; at eta=1 it reduces to bare two-sided Wald — so OT is a
*different* (smoother) path between the same WALDO/Wald endpoints
that `power_law` already interpolates between.

For the (ot, waldo) cell on Normal-Normal, the tilted p-value has the
closed form

    s_t        = (w + eta * (1 - w)) * sigma
    mu_t       = (1 - eta) * mu_n + eta * D
    a(theta)   = |mu_t - theta| / s_t
    b(theta)   = (1 - eta) * (1 - w) * (mu0 - theta) / s_t
    p(theta)   = Phi(b - a) + Phi(-a - b)

derived by substituting the OT-tilted (mu_t, s_t) for (mu_n, w*sigma)
in WALDO's two-Gaussian-CDF formula. See the brief at
docs/methods/ot.md for the full derivation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar, cast

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray

# scipy: used by the numpy-eager scalar fast path inside tilted_pvalue.
# Same boundary discipline as power_law.py — see that file's comments
# for the rationale (~10 us scalar vs ~200 us under JAX dispatch).
from scipy import stats as _scalar_scipy_stats

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..models._dispatch import is_normal_normal
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..statistics.base import TestStatistic
from .base import EtaSelector, ParamSpec
from .eta_selectors import FixedEtaSelector
# Cross-module reuse of the grid-distribution τ helpers from power_law.
# These operate on a normalised log-pdf row `(n_grid,)` + θ-grid and so
# work uniformly for any tilted q_η represented as a grid pdf —
# including OT's quantile-mixture once we materialise q_η on the grid.
from .power_law import _grid_tau_lrto, _grid_tau_scoreo
from .quantile_mixture import QuantileMixturePath

if TYPE_CHECKING:
    from ..config import Config

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _ot_tilted_pvalue_numpy_scalar(
    theta_f: float,
    eta_f: float,
    D_f: float,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> float:
    """Numpy-eager scalar fast path. Mirrors `_ot_tilted_pvalue_kernel`
    but runs on Python floats + scipy.stats.norm. Used when caller
    passes scalar (theta, eta, D) — the brentq inner-loop pattern.
    ~10 us/call.
    """
    if statistic_name == "wald":
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))
    if statistic_name in ("waldo", "lrto", "scoreo"):
        # Trinity collapse on NN+Normal: OT's tilted posterior is the W₂
        # geodesic between N(μ_n, σ_n²) and N(D, σ²), a single Gaussian,
        # so τ_LRTO,η = τ_SCOREO,η = τ_WALDO,η identically and the H₀
        # reference under D'~N(θ,σ²) gives the same Φ(b-a)+Φ(-a-b)
        # formula. See docs/notes/2026-05-12-tilted-trinity-derivation.md.
        mu_n = w * D_f + (1.0 - w) * mu0
        mu_t = (1.0 - eta_f) * mu_n + eta_f * D_f
        s_t = (w + eta_f * (1.0 - w)) * sigma
        a = abs(mu_t - theta_f) / s_t
        b = (1.0 - eta_f) * (1.0 - w) * (mu0 - theta_f) / s_t
        return float(
            _scalar_scipy_stats.norm.cdf(b - a) + _scalar_scipy_stats.norm.cdf(-a - b)
        )
    raise NotImplementedError(
        f"_ot_tilted_pvalue_numpy_scalar not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo', 'lrto', 'scoreo'."
    )


@partial(jax.jit, static_argnames=("statistic_name",))
def _ot_tilted_pvalue_kernel(
    theta: jax.Array,
    eta: jax.Array,
    D: jax.Array,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> jax.Array:
    """Pure JAX arithmetic kernel for the OT-tilted p-value.

    Same shape / discipline / jit rationale as
    `power_law._tilted_pvalue_kernel`. Scalar inputs do NOT reach
    this kernel — the public method's shape dispatch routes them to
    the numpy fast path.
    """
    if statistic_name == "wald":
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - jsp_stats.norm.cdf(z))
    if statistic_name in ("waldo", "lrto", "scoreo"):
        # Trinity collapse — see _ot_tilted_pvalue_numpy_scalar above.
        mu_n = w * D + (1.0 - w) * mu0
        mu_t = (1.0 - eta) * mu_n + eta * D
        s_t = (w + eta * (1.0 - w)) * sigma
        a = jnp.abs(mu_t - theta) / s_t
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / s_t
        return jsp_stats.norm.cdf(b - a) + jsp_stats.norm.cdf(-a - b)
    raise NotImplementedError(
        f"_ot_tilted_pvalue_kernel not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo', 'lrto', 'scoreo'."
    )


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


# ----- Generic numerical path (any Distribution-conforming inputs) -----

# Default knobs for the generic MC tilted-pvalue path. Mirrors PowerLaw
# (see `power_law.py::_GENERIC_TILTED_PVALUE_*`). `_GENERIC_TILTED_PVALUE_BASE_SEED`
# is sourced from `_generic_pvalue` so PowerLaw and OT share the same
# CRN seed at fixed (data, prior, eta, alpha) — enables direct cross-
# scheme MC comparison in the smoothness experiment.
from ._generic_pvalue import _GENERIC_TILTED_PVALUE_BASE_SEED  # noqa: F401

_GENERIC_TILT_N_GRID: int = 1024
_GENERIC_TILTED_PVALUE_N_MC: int = 200
_GENERIC_TILTED_PVALUE_N_GRID_MC: int = 256


def _generic_tilt_ot(
    posterior: Posterior,
    likelihood: Likelihood,
    eta: float,
    *,
    model: object,
    data: NDArray[np.float64],
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
):
    """Numerical W2 tilt: QuantileMixturePath between posterior and
    likelihood-as-distribution at parameter ``t = eta``.

    The geodesic *segment* is ``t in [0, 1]``; extrapolation along the
    W2 displacement line is admissible whenever the resulting law is
    still a valid distribution (audit P0-4). The monotonicity check
    that gates extrapolation is enforced by ``QuantileMixturePath``'s
    own ``__post_init__`` validator — this helper only refuses
    non-finite ``eta`` upfront so callers don't pay the
    likelihood-as-distribution construction cost on a doomed call.

    Endpoints:
      - p = posterior (passed in, already a Distribution).
      - q = likelihood_as_distribution(model, data, support, n_grid).

    Both expose `.quantile(u)` so the W2 geodesic is the linear
    interpolation of their quantile functions — that's exactly what
    `QuantileMixturePath` implements. Returns a QuantileMixturePath
    (also a Distribution).
    """
    from ._generic_pvalue import likelihood_as_distribution

    if not np.isfinite(float(eta)):
        raise TiltingDomainError(
            f"OTTilting requires finite eta, got {float(eta)!r}."
        )
    q = likelihood_as_distribution(model, data, support, n_grid=n_grid)
    # Monotonicity probe at construction time (QuantileMixturePath
    # __post_init__) raises ValueError for non-monotone extrapolation;
    # callers in the validity helper / CI inversion already catch
    # ValueError → NaN, so non-Gaussian extrapolation that violates
    # the inverse-density slope balance fails loudly at the right gate.
    return QuantileMixturePath(p=posterior, q=q, t=float(eta))


def _generic_tilted_moments_ot(
    posterior: Posterior,
    likelihood: Likelihood,
    eta: float,
    *,
    model: object,
    data: NDArray[np.float64],
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
) -> tuple[float, float]:
    """(mean, var) of the OT-tilted distribution at parameter t = eta.

    Builds the QuantileMixturePath via `_generic_tilt_ot`; reads off
    `mean()` (closed-form linear-in-t) and `var()` (Gauss-Legendre
    on the quantile, see `quantile_mixture.py::QuantileMixturePath.var`).
    Used as the hot inner kernel of `_generic_tilted_pvalue_ot`.
    """
    qmp = _generic_tilt_ot(
        posterior, likelihood, eta,
        model=model, data=data, support=support, n_grid=n_grid,
    )
    return float(qmp.mean()), float(qmp.var())


def _generic_tilted_t_statistic_ot(
    theta_f: float,
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta: float,
    *,
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
) -> float:
    """t = (mu_tilted - theta)^2 / sigma_tilted^2 at observed data.

    OT analogue of `power_law._generic_tilted_t_statistic`. Constructs
    the posterior + likelihood at observed data, then the tilted
    moments via `_generic_tilted_moments_ot`.
    """
    posterior = model.posterior(data, prior)
    likelihood = model.likelihood(data)
    mu, var = _generic_tilted_moments_ot(
        posterior, likelihood, eta,
        model=model, data=data, support=support, n_grid=n_grid,
    )
    var_safe = max(var, 1e-300)
    diff = mu - theta_f
    return diff * diff / var_safe


def _batched_inverse_cdf(
    grid: NDArray[np.float64],
    cdf_batch: NDArray[np.float64],
    u_query: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-row inverse CDF via numpy `searchsorted` in a tight Python loop.

    `grid` is the SHARED θ-grid of shape `(n_grid,)`. `cdf_batch` is
    monotone-non-decreasing-along-axis-1 with shape `(n_mc, n_grid)`.
    `u_query` has shape `(n_u,)`. Returns shape `(n_mc, n_u)` where
    `output[i, j]` is the linear-interpolated x-value at which
    `cdf_batch[i, :]` crosses `u_query[j]`.

    Why the Python loop: tried a fully-broadcast implementation that
    did `cdf[:, :, None] <= u[None, None, :]` (3.3 MB boolean tensor at
    n_mc=200, n_grid=256, n_u=64) followed by `take_along_axis`. It
    benchmarked **slower** than this loop (~437 ms vs ~339 ms per OT
    CI inversion in steady state) because the per-call allocation +
    memory-bandwidth cost of the 3D tensor and the strided gather
    outweighed the ~1 ms saved on Python-loop overhead. C-level
    `searchsorted(cdf_row, u_query)` is already extremely fast at
    n_grid=256 / n_u=64; 200 loop iterations of it cost <2 ms total.
    """
    n_mc = cdf_batch.shape[0]
    n_u = u_query.shape[0]
    out = np.empty((n_mc, n_u), dtype=np.float64)
    n_grid = cdf_batch.shape[1]
    for i in range(n_mc):
        cdf_row = cdf_batch[i]
        # `side='right'` matches np.interp's convention on monotone
        # input. Clamp to [1, n_grid-1] so idx-1 is valid.
        idx = np.clip(np.searchsorted(cdf_row, u_query, side="right"),
                      1, n_grid - 1)
        cdf_lo = cdf_row[idx - 1]
        cdf_hi = cdf_row[idx]
        denom = cdf_hi - cdf_lo
        denom_safe = np.where(denom > 0, denom, 1.0)
        frac = np.where(denom > 0, (u_query - cdf_lo) / denom_safe, 0.0)
        out[i] = grid[idx - 1] + frac * (grid[idx] - grid[idx - 1])
    return out


# Module-level cache for Gauss-Legendre nodes / weights at the
# canonical n_u=64. `np.polynomial.legendre.leggauss(64)` costs ~700 µs
# per call (eigenvalue decomposition of the 64-point Jacobi matrix);
# pre-computed jnp views skip both the numpy compute AND the per-call
# `jnp.asarray` boundary conversion in the kernel call. Wins ~700-800 µs
# per OT MC reference call vs the legacy "compute every time" form.
from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=8)
def _gauss_legendre_01(n_u: int) -> tuple[jax.Array, jax.Array]:
    """Return `(u01, w01)` Gauss-Legendre nodes/weights mapped to (0,1)
    as `jnp.Array`s. Cached by `n_u` so the n_u=64 case (the only one
    used by OT today) computes once per process.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(n_u))
    return jnp.asarray(0.5 * (nodes + 1.0)), jnp.asarray(0.5 * weights)


@jax.jit
def _ot_tilted_kernel_jit(
    log_lik_batch: jax.Array,
    theta_grid: jax.Array,
    F_post_inv: jax.Array,
    u01: jax.Array,
    w01: jax.Array,
    eta: jax.Array,
    theta_f: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX kernel for OT-tilted MC reference post-likelihood block.

    Inputs (all jnp.Array):
      log_lik_batch (n_mc, n_grid): per-row log-likelihood on the
        shared theta-grid. Caller computes via
        `model.batch_loglik_grid(D_batch, theta_grid)` and converts.
      theta_grid (n_grid,): the shared grid on which the likelihood-
        as-distribution CDF is built.
      F_post_inv (n_mc, n_u): per-row posterior inverse-CDF at the
        Gauss-Legendre nodes, computed externally via
        `model.posterior_quantile_batch(D_batch, prior, u01)`.
      u01 (n_u,): Gauss-Legendre nodes mapped to (0, 1).
      w01 (n_u,): Gauss-Legendre weights scaled to (0, 1).
      eta, theta_f: scalars.

    Returns `(t_samples, n_collapsed_int32)`. The pure-JAX block fuses
    (i) `pdf_lik` normalisation via max-subtract + exp + trapezoid Z,
    (ii) cumulative-trapezoid CDF construction, (iii) per-row
    inverse-CDF lookup at `u01` via `vmap(searchsorted + linear interp)`
    — replaces the python-loop `_batched_inverse_cdf` with vectorised
    XLA, the main pre-JIT bottleneck — (iv) OT W2 quantile mixing,
    (v) Gauss-Legendre 64-pt quadrature for `(m1, m2)`, (vi) collapsed-
    row masking + final t-statistic.

    Closes over no model state — pre-computed `F_post_inv` is the only
    model-specific quantity, so the jit cache hits across cells with
    the same shape signature instead of recompiling per (model, prior).
    """
    n_grid = theta_grid.shape[0]

    # 1. Build the likelihood-as-distribution PDF on the grid.
    log_lik = jnp.where(jnp.isfinite(log_lik_batch), log_lik_batch, -1e300)
    log_lik_max = log_lik.max(axis=-1, keepdims=True)
    pdf = jnp.exp(log_lik - log_lik_max)
    Z = jnp.trapezoid(pdf, theta_grid, axis=-1)
    Z_safe = jnp.where(Z > 0, Z, 1.0)
    pdf = pdf / Z_safe[:, None]

    # 2. Cumulative trapezoid → CDF, normalised to [0, 1] per row.
    dtheta = jnp.diff(theta_grid)
    incr = 0.5 * (pdf[:, :-1] + pdf[:, 1:]) * dtheta[None, :]
    cdf = jnp.concatenate(
        [jnp.zeros((pdf.shape[0], 1)), jnp.cumsum(incr, axis=-1)], axis=-1
    )
    cdf_total = cdf[:, -1:]
    cdf = jnp.clip(cdf / jnp.where(cdf_total > 0, cdf_total, 1.0), 0.0, 1.0)

    # 3. Per-row inverse CDF at u01 via vmap. Replaces the Python
    #    `for i in range(n_mc)` searchsorted loop in the legacy
    #    `_batched_inverse_cdf` — the dominant pre-JIT cost.
    def _inverse_per_row(cdf_row: jax.Array) -> jax.Array:
        idx = jnp.clip(
            jnp.searchsorted(cdf_row, u01, side="right"),
            1,
            n_grid - 1,
        )
        cdf_lo = cdf_row[idx - 1]
        cdf_hi = cdf_row[idx]
        denom = cdf_hi - cdf_lo
        denom_safe = jnp.where(denom > 0, denom, 1.0)
        frac = jnp.where(denom > 0, (u01 - cdf_lo) / denom_safe, 0.0)
        return theta_grid[idx - 1] + frac * (theta_grid[idx] - theta_grid[idx - 1])

    F_lik_inv = jax.vmap(_inverse_per_row)(cdf)  # (n_mc, n_u)

    # 4. OT W2 quantile mixing → tilted distribution's quantile at u01.
    F_mixed = (1.0 - eta) * F_post_inv + eta * F_lik_inv

    # 5. Gauss-Legendre quadrature for (m1, m2).
    m1 = jnp.sum(w01[None, :] * F_mixed, axis=-1)
    m2 = jnp.sum(w01[None, :] * F_mixed * F_mixed, axis=-1)
    var_tilted = jnp.maximum(m2 - m1 * m1, 0.0)

    # 6. Collapsed-row mask + final t.
    finite_z = (Z > 0) & jnp.isfinite(Z)
    finite_var = (var_tilted > 0) & jnp.isfinite(var_tilted)
    finite_row = finite_z & finite_var
    n_collapsed = jnp.sum(~finite_row).astype(jnp.int32)

    diff = m1 - theta_f
    safe_var = jnp.where(finite_row, var_tilted, 1.0)
    t_raw = diff * diff / safe_var
    t_samples = jnp.where(finite_row, t_raw, 0.0)
    return t_samples, n_collapsed


def _generic_tilted_mc_reference_batch_ot(
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
    statistic_name: str = "waldo",
) -> tuple[NDArray[np.float64], int]:
    """Vectorised MC reference for OT-tilted p-value.

    Returns `(t_samples, n_collapsed)` with `t_samples` shape `(n_mc,)`.
    Replaces the per-MC-iteration construction of
    `likelihood_as_distribution` + `QuantileMixturePath` (each ~tens of
    ms) with a single batched pipeline:

      1. `sample_data_batch` → D_batch shape (n_mc, n_obs).
      2. `posterior_moments_batch` → per-row (mu_post, var_post),
         used to size a shared θ-window covering all rows.
      3. `batch_loglik_grid` + per-row trapezoid-normalise + cumulative
         trapezoid → `cdf_lik_batch` shape (n_mc, n_grid). This is the
         likelihood-as-distribution CDF, vectorised across rows.
      4. `posterior_quantile_batch` → per-row F_post^{-1} at the
         Gauss-Legendre nodes, shape (n_mc, n_u).
      5. `_batched_inverse_cdf` → per-row F_lik^{-1} at the same nodes,
         shape (n_mc, n_u).
      6. OT mixing in quantile space:
         F_qmp^{-1}(u) = (1-η)·F_post^{-1}(u) + η·F_lik^{-1}(u).
      7. Gauss-Legendre 64-pt quadrature → per-row `(m1, m2)` →
         `var_tilted = m2 - m1²`, `t = (m1 - θ)² / var_tilted`.

    All steps after sampling are pure-numpy broadcast ops; no Python
    loop over n_mc except the cheap per-row `searchsorted` in
    `_batched_inverse_cdf`. ~500x speedup vs the legacy per-row
    GridDistribution / QuantileMixturePath construction.
    """
    from ..models.base import (
        batch_loglik_grid as _batch_loglik_grid,
        posterior_moments_batch as _posterior_moments_batch,
        posterior_quantile_batch as _posterior_quantile_batch,
        sample_data_batch as _sample_data_batch,
    )

    if n_mc <= 0:
        return np.empty(0, dtype=np.float64), 0
    support_lo, support_hi = float(support[0]), float(support[1])

    # 1. Batched sampling under H_0:theta_f.
    D_batch = _sample_data_batch(model, float(theta_f), rng, int(n_mc), int(n_obs))

    # 2. Per-row posterior moments (used to size the likelihood window
    #    + as the OT W2 mean component below if we wanted closed form).
    mu_post_arr, var_post_arr = _posterior_moments_batch(model, D_batch, prior)
    sigma_post_arr = np.sqrt(np.maximum(var_post_arr, 1e-300))

    # Single shared θ-window covering all rows. For OT we want the
    # likelihood-as-distribution to be well-resolved — its support
    # widens with the observed data (Normal: ~D ± 6σ; future bounded
    # models: their support window). Take the union of
    # (mu_post ± 8 σ_post) ∪ (D_means ± 8 σ) clipped to model support.
    finite = np.isfinite(mu_post_arr) & np.isfinite(sigma_post_arr)
    if not np.any(finite):
        return np.zeros(int(n_mc), dtype=np.float64), int(n_mc)
    half = 8.0 * sigma_post_arr
    lo = max(float(np.nanmin((mu_post_arr - half)[finite])), support_lo)
    hi = min(float(np.nanmax((mu_post_arr + half)[finite])), support_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or not (lo < hi):
        return np.zeros(int(n_mc), dtype=np.float64), int(n_mc)

    theta_grid = np.linspace(lo, hi, int(n_grid))

    # 3. Build per-row log-likelihood on the shared grid (numpy).
    log_lik = _batch_loglik_grid(model, D_batch, theta_grid)  # (n_mc, n_grid)

    # lrto/scoreo branch — needs per-row q_η log-pdf on the θ-grid.
    # Building this via per-replicate QuantileMixturePath.pdf is too
    # slow (~600µs/grid-point × 200 rows × 256 grid = minutes). Instead
    # construct q_η pdf directly in u-space, all vectorised across the
    # n_mc axis:
    #   - F_post^{-1}(u): per-row via `posterior_quantile_batch` (scipy
    #     ndtri for Gaussian — vectorised).
    #   - F_lik^{-1}(u):  per-row via cumulative-trapezoid CDF of
    #     `pdf_lik` + per-row searchsorted (`_batched_inverse_cdf`).
    #   - F_t^{-1}(u) = (1-η)F_post^{-1} + η F_lik^{-1}  (vectorised).
    #   - du/dθ_t at the u-grid via central differences → q_η pdf at
    #     the inverse-CDF-image points θ_mixed[j, :].
    #   - Linear-interpolate the pdf onto the fixed `theta_grid` →
    #     `pdf_qmp` shape (n_mc, n_grid). Take log + dispatch to τ.
    # This is all numpy broadcast ops; no brentq, no per-row Python.
    if statistic_name in ("lrto", "scoreo"):
        from ..models.base import posterior_quantile_batch as _posterior_quantile_batch

        # Build the u-grid (mid-resolution; we don't need the τ helper's
        # full n_grid here because we interpolate back to theta_grid).
        # Higher than `_GENERIC_TILTED_PVALUE_N_GRID_MC=256` so central-
        # difference du/dθ has low tail bias; integrate-to-1 quality
        # validated at ~0.998 in the Task 6 review.
        n_u = 1024
        eps_u = 1.0 / (n_u + 1)
        u_grid = np.linspace(eps_u, 1.0 - eps_u, n_u)

        # F_post^{-1}(u_grid): shape (n_mc, n_u). NN: analytic Gaussian.
        F_post_inv = np.asarray(
            _posterior_quantile_batch(model, D_batch, prior, u_grid),
            dtype=np.float64,
        )

        # F_lik^{-1}(u_grid): build per-row pdf, normalise, cumulative-
        # trapezoid CDF, then `_batched_inverse_cdf` to invert at u_grid.
        log_lik_clean = np.where(np.isfinite(log_lik), log_lik, -1e300)
        log_lik_max = log_lik_clean.max(axis=-1, keepdims=True)
        pdf_lik_unnorm = np.exp(log_lik_clean - log_lik_max)  # (n_mc, n_grid)
        Z_lik = np.trapezoid(pdf_lik_unnorm, theta_grid, axis=-1)
        Z_lik_safe = np.where(Z_lik > 0, Z_lik, 1.0)
        pdf_lik = pdf_lik_unnorm / Z_lik_safe[:, None]
        dtheta = np.diff(theta_grid)  # (n_grid - 1,)
        incr_lik = 0.5 * (pdf_lik[:, :-1] + pdf_lik[:, 1:]) * dtheta[None, :]
        cdf_lik = np.concatenate(
            [np.zeros((pdf_lik.shape[0], 1)), np.cumsum(incr_lik, axis=-1)],
            axis=-1,
        )
        cdf_lik_total = cdf_lik[:, -1:]
        cdf_lik = np.clip(
            cdf_lik / np.where(cdf_lik_total > 0, cdf_lik_total, 1.0),
            0.0,
            1.0,
        )
        F_lik_inv = _batched_inverse_cdf(theta_grid, cdf_lik, u_grid)  # (n_mc, n_u)

        # OT W2 mix in u-space.
        F_mixed = (1.0 - float(eta_f)) * F_post_inv + float(eta_f) * F_lik_inv  # (n_mc, n_u)

        # pdf at F_mixed via du/dθ central differences:
        # f_t(θ_mixed[i]) ≈ (u[i+1] - u[i-1]) / (θ_mixed[i+1] - θ_mixed[i-1]).
        du_central = (u_grid[2:] - u_grid[:-2])[None, :]
        dtheta_central = F_mixed[:, 2:] - F_mixed[:, :-2]  # (n_mc, n_u-2)
        # Endpoint backfill: one-sided difference at the boundaries.
        du_left = (u_grid[1] - u_grid[0])
        du_right = (u_grid[-1] - u_grid[-2])
        dtheta_left = (F_mixed[:, 1] - F_mixed[:, 0])[:, None]
        dtheta_right = (F_mixed[:, -1] - F_mixed[:, -2])[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            pdf_central = np.where(
                dtheta_central > 0, du_central / np.where(dtheta_central > 0, dtheta_central, 1.0), 0.0
            )
            pdf_left = np.where(
                dtheta_left > 0, du_left / np.where(dtheta_left > 0, dtheta_left, 1.0), 0.0
            )
            pdf_right = np.where(
                dtheta_right > 0, du_right / np.where(dtheta_right > 0, dtheta_right, 1.0), 0.0
            )
        # Stack: (n_mc, n_u). pdf_at_mixed[j, i] = density of q_η at F_mixed[j, i].
        pdf_at_mixed = np.concatenate([pdf_left, pdf_central, pdf_right], axis=-1)
        # Sanity: pdf_at_mixed >= 0 (already by construction), finite.
        pdf_at_mixed = np.where(np.isfinite(pdf_at_mixed), pdf_at_mixed, 0.0)

        # Resample to the fixed theta_grid via per-row linear interp.
        # `F_mixed[j, :]` is monotone non-decreasing (W2 geodesic of
        # monotone quantiles) so np.interp is well-defined.
        pdf_qmp = np.zeros((int(n_mc), int(n_grid)), dtype=np.float64)
        for j in range(int(n_mc)):
            # Linear interp; outside the support of θ_mixed[j], np.interp
            # extrapolates to endpoint values — we zero those out below.
            pdf_qmp[j] = np.interp(theta_grid, F_mixed[j], pdf_at_mixed[j])
            outside = (theta_grid < F_mixed[j, 0]) | (theta_grid > F_mixed[j, -1])
            pdf_qmp[j] = np.where(outside, 0.0, pdf_qmp[j])

        # Compute τ per row via the shared grid helpers. Use log-pdf.
        with np.errstate(divide="ignore"):
            log_pdf_qmp = np.where(pdf_qmp > 0, np.log(np.maximum(pdf_qmp, 1e-300)), -1e300)
        tau_fn = _grid_tau_lrto if statistic_name == "lrto" else _grid_tau_scoreo
        t_samples = np.zeros(int(n_mc), dtype=np.float64)
        n_collapsed = 0
        for j in range(int(n_mc)):
            row = log_pdf_qmp[j]
            if not np.any(np.isfinite(row)) or not np.any(pdf_qmp[j] > 0):
                n_collapsed += 1
                continue
            tau_j = tau_fn(row, theta_grid, float(theta_f))
            if not np.isfinite(tau_j):
                n_collapsed += 1
            else:
                t_samples[j] = float(tau_j)
        return t_samples, int(n_collapsed)

    if statistic_name != "waldo":
        raise NotImplementedError(
            f"_generic_tilted_mc_reference_batch_ot not implemented for "
            f"statistic={statistic_name!r}; supported: 'waldo', 'lrto', 'scoreo'."
        )

    # 4. Gauss-Legendre nodes + weights — cached jnp.Array per n_u so
    #    we don't pay the ~700 µs leggauss eigendecomposition or the
    #    per-call `jnp.asarray` conversion in the kernel-input list.
    u01_j, w01_j = _gauss_legendre_01(64)
    u01_np = np.asarray(u01_j)  # numpy view for `posterior_quantile_batch`

    # 5. Per-row posterior inverse-CDF at u01 (numpy or scipy under
    #    the hood — `model.posterior_quantile_batch` for NN uses
    #    scipy.special.ndtri).
    F_post_inv = _posterior_quantile_batch(model, D_batch, prior, u01_np)

    # 6. Hand off to the jit-compiled XLA kernel for the post-loglik
    #    pipeline: `pdf_lik` normalisation, cumulative-trapezoid CDF,
    #    vectorised inverse-CDF lookup at `u01` (replaces the per-row
    #    Python `searchsorted` loop — the dominant pre-JIT cost), OT W2
    #    quantile mixing, Gauss-Legendre moments, collapsed-row masking.
    #    `u01_j` and `w01_j` already jnp.Array (cached); only log_lik,
    #    theta_grid, F_post_inv, and the two scalars need conversion.
    t_samples_j, n_collapsed_j = _ot_tilted_kernel_jit(
        jnp.asarray(log_lik),
        jnp.asarray(theta_grid),
        jnp.asarray(F_post_inv),
        u01_j,
        w01_j,
        jnp.asarray(float(eta_f)),
        jnp.asarray(float(theta_f)),
    )
    t_samples = np.asarray(t_samples_j, dtype=np.float64)
    n_collapsed = int(np.asarray(n_collapsed_j))
    return t_samples, n_collapsed


def _generic_tilted_pvalue_ot(
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
    """Generic MC tilted p-value for OTTilting on any (model, prior).

    Mirrors `power_law._generic_tilted_pvalue` structurally:
    - statistic_name="wald": eta-independent, delegates to WaldStatistic.
    - statistic_name="waldo": MC reference under H_0 via
      `model.sample_data(theta, ...)`, recompute t per draw using
      `_generic_tilted_t_statistic_ot`. Conservative `(k+1)/(n+1)`
      smoothing. CRN-seeded via blake2b stable hash.
    - statistic_name="lrto" / "scoreo": materialise the q_η log-pdf on a
      θ-grid (via `QuantileMixturePath.logpdf` for the observed τ;
      vectorised u-space change-of-variables for the MC replicates —
      see `_generic_tilted_mc_reference_batch_ot`) and dispatch through
      the shared grid-distribution τ helpers (`_grid_tau_lrto` /
      `_grid_tau_scoreo` from power_law.py). On NN+Normal the
      Gaussian q_η yields trinity-collapse τ_LRTO=τ_SCOREO=τ_WALDO.
    """
    from ..statistics.wald import WaldStatistic
    from ._generic_pvalue import _resolve_support, _stable_tilted_pvalue_seed

    if statistic_name == "wald":
        return float(np.asarray(WaldStatistic()._generic_pvalue(theta, data, model)))
    if statistic_name not in ("waldo", "lrto", "scoreo"):
        raise NotImplementedError(
            f"OTTilting generic tilted_pvalue not implemented for "
            f"statistic={statistic_name!r}; supported: 'wald', 'waldo', 'lrto', 'scoreo'."
        )

    data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if data_arr.ndim != 1:
        raise NotImplementedError(
            f"OTTilting generic tilted_pvalue expects 1-D data; got data.ndim={data_arr.ndim}."
        )

    support = _resolve_support(model, data_arr)
    theta_f = float(theta)
    eta_f = float(eta)
    # Audit P0-4: OT is well-defined on the full W2 displacement line.
    # Per-eta admissibility (monotonicity of the linear quantile combo)
    # is enforced inside `_generic_tilt_ot` → `QuantileMixturePath`;
    # only refuse non-finite eta here, before the seed-derivation cost.
    if not np.isfinite(eta_f):
        raise TiltingDomainError(
            f"OTTilting requires finite eta, got {eta_f!r}."
        )

    if derived_seed is None:
        derived_seed = _stable_tilted_pvalue_seed(
            data_arr, model, prior, eta_f, alpha, base_seed
        )

    # Observed tilted τ at (data, theta_f, eta_f).
    if statistic_name == "waldo":
        # Hoist observed moments (skeptic finding from PowerLaw 3c-fix1).
        if obs_moments is not None:
            mu_obs, var_obs = obs_moments
        else:
            posterior_obs = model.posterior(data_arr, prior)
            likelihood_obs = model.likelihood(data_arr)
            mu_obs, var_obs = _generic_tilted_moments_ot(
                posterior_obs, likelihood_obs, eta_f,
                model=model, data=data_arr, support=support,
                n_grid=_GENERIC_TILT_N_GRID,
            )
        var_obs_safe = max(var_obs, 1e-300)
        diff_obs = mu_obs - theta_f
        t_obs = diff_obs * diff_obs / var_obs_safe
    else:
        # lrto / scoreo: build the full normalised observed tilted log-pdf
        # grid on the same θ-window the MC reference will use (so τ_obs
        # and τ_rep are computed at matching grid resolution). For OT,
        # q_η is a QuantileMixturePath; we materialise its log-pdf row
        # by calling `.logpdf(theta_grid)` once.
        posterior_obs = model.posterior(data_arr, prior)
        likelihood_obs = model.likelihood(data_arr)
        # Reuse the same θ-window-sizing strategy as the MC reference:
        # union of (posterior ± 8σ) and the model support, but we use
        # the simpler approach of just calling `_generic_tilt_ot` and
        # evaluating on a fresh grid spanning the posterior + likelihood
        # support. The MC batch uses its own per-call window, but for
        # τ_obs/τ_rep parity, using the posterior moments + support
        # clipping yields a comparable window.
        try:
            mu_post_obs = float(np.asarray(posterior_obs.mean()))
            var_post_obs = float(np.asarray(posterior_obs.var()))
        except (TypeError, ValueError, AttributeError):
            # Conservative: fall back to model support.
            mu_post_obs, var_post_obs = 0.0, 1.0
        sigma_post_obs = float(np.sqrt(max(var_post_obs, 1e-300)))
        lo_obs = max(mu_post_obs - 8.0 * sigma_post_obs, float(support[0]))
        hi_obs = min(mu_post_obs + 8.0 * sigma_post_obs, float(support[1]))
        if not (lo_obs < hi_obs) or not np.isfinite(lo_obs) or not np.isfinite(hi_obs):
            # Degenerate observed window — conservative p=1.
            return 1.0
        theta_grid_obs = np.linspace(
            lo_obs, hi_obs, _GENERIC_TILTED_PVALUE_N_GRID_MC
        )
        try:
            qmp_obs = _generic_tilt_ot(
                posterior_obs, likelihood_obs, eta_f,
                model=model, data=data_arr, support=support,
                n_grid=_GENERIC_TILT_N_GRID,
            )
            log_pdf_obs = np.asarray(
                qmp_obs.logpdf(theta_grid_obs), dtype=np.float64
            )
        except (ValueError, RuntimeError, TiltingDomainError):
            # Non-monotone extrapolation or brentq failure — conservative p=1.
            return 1.0
        if not np.all(np.isfinite(log_pdf_obs)):
            # log-pdf has -inf at out-of-support points → renormalise
            # by setting -inf to a small finite floor relative to max
            # so the τ helpers see a clean grid.
            log_pdf_obs = np.where(np.isfinite(log_pdf_obs), log_pdf_obs, -1e300)
        tau_fn = _grid_tau_lrto if statistic_name == "lrto" else _grid_tau_scoreo
        t_obs = tau_fn(log_pdf_obs, theta_grid_obs, theta_f)
        if not np.isfinite(t_obs):
            # τ_obs out-of-window or non-finite: conservative smoothed p.
            return 1.0 / (float(n_mc) + 1.0)

    # MC reference under H_0:theta_f — vectorised batched pipeline.
    # Replaces a per-iteration `likelihood_as_distribution` (full
    # GridDistribution construction) + `QuantileMixturePath` (Gauss-
    # Legendre on inverse-CDF) loop with batched cumulative-trapezoid
    # CDF + vectorised inverse-CDF lookup. ~500x speedup at n_mc=200.
    # For lrto/scoreo, the batch uses a vectorised u-space change-of-variables
    # construction; the observed τ uses a single-row `QuantileMixturePath.logpdf`.
    rng = np.random.default_rng(derived_seed)
    n_obs = int(data_arr.size)
    t_samples, n_collapsed = _generic_tilted_mc_reference_batch_ot(
        theta_f=theta_f,
        n_obs=n_obs,
        model=model,
        prior=prior,
        eta_f=eta_f,
        support=support,
        n_mc=int(n_mc),
        n_grid=_GENERIC_TILTED_PVALUE_N_GRID_MC,
        rng=rng,
        statistic_name=statistic_name,
    )
    if n_collapsed > n_mc // 2:
        import warnings
        warnings.warn(
            f"OTTilting._generic_tilted_pvalue: {n_collapsed}/{n_mc} MC samples "
            f"collapsed (theta={theta_f}, eta={eta_f}); empirical p is "
            f"strongly biased upward. Increase data size or reduce eta.",
            RuntimeWarning,
            stacklevel=2,
        )
    p = (float(np.sum(t_samples >= t_obs)) + 1.0) / (float(n_mc) + 1.0)
    return float(p)


def _generic_tilted_pvalue_ot_vec(
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
) -> NDArray[np.float64]:
    """Generic MC tilted p-value across (theta_arr, eta_arr) for OT.

    OT analog of ``power_law._generic_tilted_pvalue_vec``. Mirrors the
    same call signature so the dynamic-CI scanner can swap schemes by
    name. Used by ``OTTilting.dynamic_tilted_confidence_interval_ot_generic``
    to fine-scan the dynamic-η p-value at force_generic=True on NN.

    Supports ``statistic_name in {"wald", "waldo", "lrto", "scoreo"}``;
    routing is delegated to the scalar helper. The lrto/scoreo branches
    use per-row grid τ helpers (scalar output by construction), so even
    a future triple-batched vec would still loop over the n_mc axis for
    those statistics — the current Python-loop fallback over n_theta
    has no significant downside relative to a fully-batched version.

    Implementation note: the current OT version is a Python loop over
    (theta_i, eta_i) wrapping the scalar ``_generic_tilted_pvalue_ot``,
    NOT a true triple-batched implementation like power_law's vec
    helper. Each scalar call already vectorises the per-(θ, η) MC
    reference (``_generic_tilted_mc_reference_batch_ot``), so the
    wall-time gap vs a fully-batched OT vec is roughly the n_theta
    Python-loop overhead — small in absolute terms (~ms/iter) but
    n_theta=401 fine-grid points × ~100ms per scalar call adds up
    (~30-40 s/CI). Adequate for sanity checks; for production audits
    a true batched OT vec (mirroring power_law's chunked triple-batch)
    is a tractable follow-up.
    """
    from ._generic_pvalue import _stable_tilted_pvalue_seed

    theta_arr_np = np.asarray(theta_arr, dtype=np.float64)
    eta_arr_np = np.asarray(eta_arr, dtype=np.float64)
    if theta_arr_np.shape != eta_arr_np.shape:
        raise ValueError(
            f"theta_arr and eta_arr must have the same shape; got "
            f"{theta_arr_np.shape!r} and {eta_arr_np.shape!r}."
        )
    data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if derived_seed is None:
        derived_seed = _stable_tilted_pvalue_seed(
            data_arr, model, prior, 0.0, alpha, base_seed
        )

    n_theta = int(theta_arr_np.size)
    p_out = np.empty(n_theta, dtype=np.float64)
    for i in range(n_theta):
        # Per-element CRN: derived_seed + i to keep streams reproducible
        # AND independent across grid points (mirrors power_law's chunked
        # CRN). Use the scalar helper as the inner primitive — already
        # vectorised across the per-(θ, η) MC sample axis.
        p_out[i] = _generic_tilted_pvalue_ot(
            float(theta_arr_np[i]),
            data_arr,
            model,
            prior,
            float(eta_arr_np[i]),
            statistic_name,
            n_mc=int(n_mc),
            derived_seed=int(derived_seed) + i,
            alpha=alpha,
            base_seed=base_seed,
        )
    return p_out


def _generic_tilted_confidence_interval_ot(
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
    """Generic CI inversion for OTTilting via brentq + CRN.

    Mirrors `power_law._generic_tilted_confidence_interval` structurally,
    including the explicit boundary detection from Phase 3c-fix1.
    """
    from .._errors import BracketingFailed
    from ._generic_pvalue import _resolve_support, _stable_tilted_pvalue_seed
    from ._solvers import brentq_with_doubling

    data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    eta_f = float(eta)
    # Skeptic LOW #8: validate eta at function entry rather than letting
    # _generic_tilt_ot raise mid-inversion. Saves the seed-derivation +
    # posterior-construction cost on a doomed call. Audit P0-4: only
    # finite-eta is required upfront — admissibility along the W2
    # displacement line is enforced by `QuantileMixturePath`'s runtime
    # monotonicity probe, not a [0, 1] interval clamp.
    if not np.isfinite(eta_f):
        raise TiltingDomainError(
            f"OTTilting requires finite eta, got {eta_f!r}."
        )
    derived_seed = _stable_tilted_pvalue_seed(
        data_arr, model, prior, eta_f, alpha, base_seed
    )

    support = _resolve_support(model, data_arr)
    support_lo, support_hi = support

    # Hoist observed moments.
    posterior_at_obs = model.posterior(data_arr, prior)
    likelihood_at_obs = model.likelihood(data_arr)
    mu_obs, var_obs = _generic_tilted_moments_ot(
        posterior_at_obs, likelihood_at_obs, eta_f,
        model=model, data=data_arr, support=support,
        n_grid=_GENERIC_TILT_N_GRID,
    )
    var_obs_safe = max(var_obs, 1e-300)
    sigma_tilted = float(np.sqrt(var_obs_safe))

    def f(theta_val: float) -> float:
        theta_safe = max(support_lo, min(support_hi, float(theta_val)))
        return _generic_tilted_pvalue_ot(
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

    # Explicit boundary detection (Phase 3c-fix1).
    if np.isfinite(support_lo):
        ci_extends_below = (f(support_lo) >= 0.0)
    else:
        ci_extends_below = False
    if np.isfinite(support_hi):
        ci_extends_above = (f(support_hi) >= 0.0)
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


@register_tilting(name="ot", brief="docs/methods/ot.md")
@dataclass(frozen=True)
class OTTilting:
    """Wasserstein-2 geodesic tilting (general 1D, Gaussian fast path)."""

    name: ClassVar[str] = "ot"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description=(
            "Position along the W2 displacement line through posterior "
            "(eta=0) and likelihood-induced Gaussian N(D, sigma^2) "
            "(eta=1). The geodesic *segment* is [0, 1]; eta outside that "
            "extrapolates along the same line and is admissible whenever "
            "the resulting distribution is well-defined (Gaussian path: "
            "sigma_t > 0; closed-form pvalue: s_t > 0 ⇔ eta > -w/(1-w))."
        ),
        eta_likelihood_only=1.0,
    )
    selector: EtaSelector = field(default_factory=lambda: FixedEtaSelector(eta=0.0))

    @property
    def cell_name(self) -> str:
        sel_name = getattr(self.selector, "name", "")
        if isinstance(self.selector, FixedEtaSelector) and self.selector.eta == 0.0:
            return self.name
        return f"{self.name}[{sel_name}]"

    # ----- TiltingScheme protocol -----

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> Posterior:
        """W2-geodesic tilt between posterior and likelihood-as-distribution.

        Three dispatch paths:
        - **Gaussian fast path**: closed-form linear interpolation in
          (mu, sigma) when both posterior and likelihood are Gaussian.
          Returns `NormalDistribution`.
        - **Gaussian-likelihood quantile-mixture**: when likelihood is
          `GaussianLikelihood` but posterior isn't (rare on the
          Normal-Normal sandbox). Constructs `q = N(D, sigma)`
          directly and wraps in `QuantileMixturePath`.
        - **Generic numerical (Phase 3d)**: when neither path applies.
          Constructs `q` as a `GridDistribution` from `log L(theta)`
          on the model support; returns `QuantileMixturePath(p=posterior,
          q=q, t=eta)`. Works for arbitrary (model, prior) pairs.

        ``data`` is required for the generic path (to evaluate
        `log L(theta)`). Callers from `confidence_regions` etc.
        plumb it through; direct callers must use the
        `data`-aware OT generic helpers.
        """
        eta_arr = np.asarray(eta, dtype=np.float64)
        if eta_arr.ndim != 0:
            raise NotImplementedError(
                "tilt() expects scalar eta; vectorised eta is consumed via "
                "repeated scalar calls (see `path`)."
            )
        t = float(eta_arr)
        if not np.isfinite(t):
            raise TiltingDomainError(f"OTTilting requires finite eta, got {t!r}.")

        # Gaussian fast path: linear interpolation in (mu, sigma). The W2
        # displacement line is well-defined for any finite t, but the
        # output is only a valid Gaussian when sigma_t > 0.
        # Audit P1 H.6: `prior` is intentionally unused on this path —
        # OT interpolates between posterior and the likelihood-induced
        # Gaussian; the prior is already absorbed in `posterior`. The
        # parameter stays in the protocol signature for uniformity
        # with `power_law`/`mixture` etc. that DO consume the prior.
        if isinstance(posterior, NormalDistribution) and isinstance(likelihood, GaussianLikelihood):
            mu_a, sigma_a = posterior.loc, posterior.scale
            mu_b, sigma_b = float(likelihood.D), float(likelihood.sigma)
            sigma_t = (1.0 - t) * sigma_a + t * sigma_b
            if sigma_t <= 0.0:
                raise TiltingDomainError(
                    f"OTTilting Gaussian fast path requires sigma_t > 0, got "
                    f"sigma_t={sigma_t!r} at eta={t!r} "
                    f"(sigma_post={sigma_a!r}, sigma_lik={sigma_b!r})."
                )
            return NormalDistribution(
                loc=(1.0 - t) * mu_a + t * mu_b,
                scale=sigma_t,
            )

        # Gaussian-likelihood quantile-mixture path (likelihood admits a
        # Distribution view directly).
        if isinstance(likelihood, GaussianLikelihood):
            q = NormalDistribution(loc=likelihood.D, scale=likelihood.sigma)
            return QuantileMixturePath(p=posterior, q=q, t=t)

        # Phase 3d generic numerical path. We need access to the model
        # and data to construct the likelihood-as-distribution; the
        # `tilt()` signature doesn't carry them, so we cannot compute
        # this path directly here. Surface a clear error pointing
        # callers to the data-aware entry point in `confidence_regions`.
        raise NotImplementedError(
            f"OTTilting.tilt() can't construct the generic numerical path "
            f"from a {type(likelihood).__name__!r} without knowing the model "
            f"and data — use `confidence_regions(alpha, data, model, prior, "
            f"statistic)` (which routes through the generic path internally) "
            f"or call `_generic_tilt_ot(posterior, likelihood, eta, model=, "
            f"data=, support=)` directly."
        )

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[Posterior]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, float(t))

    def is_identity(self, eta: float) -> bool:
        return float(eta) == self.param_space.eta_identity

    # ----- (TiltingScheme, TestStatistic) cross-product specialisations -----

    def tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float | NDArray[np.float64],
        model: Model,
        prior: NormalDistribution,
        eta: ArrayLike,
        statistic_name: str,
    ) -> float | jax.Array:
        """Tilted p-value evaluated against the W2-tilted Gaussian.

        Specialized for (ot, waldo) and (ot, wald) on Normal-Normal:

          (ot, wald): eta-independent two-sided Wald, 2 * Phi(-|D-theta|/sigma)
          (ot, waldo): closed form derived in docs/methods/ot.md, with the
            standard error s_t = (w + eta*(1-w))*sigma.

        Endpoint sanity: at eta=0 reduces to bare WALDO; at eta=1 reduces
        to bare Wald (s_t -> sigma, mu_t -> D, b -> 0, a -> |D-theta|/sigma).

        Returns ``jax.Array`` for bulk inputs (length >= 2 in any of
        theta/eta/D); returns Python float for the all-scalar fast
        path. Same shape-dispatch + return-type discipline as
        `PowerLawTilting.tilted_pvalue` — see that method's docstring
        for the rationale.
        """
        from ..models.normal_normal import NormalNormalModel

        if not is_normal_normal(model):
            raise NotImplementedError(
                "OTTilting.tilted_pvalue currently requires NormalNormalModel; "
                f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "OTTilting.tilted_pvalue currently requires a NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        # Validation runs in numpy (JAX can't raise mid-trace). The
        # closed-form WALDO p-value is parameterised by s_t = (w +
        # eta*(1-w))*sigma; it is well-defined whenever s_t > 0, i.e.
        # eta > -w/(1-w). Above that there is no upper bound from s_t.
        eta_np = np.asarray(eta, dtype=np.float64)
        eta_lower = -w / (1.0 - w)  # exclusive boundary
        invalid = ~(np.isfinite(eta_np) & (eta_np > eta_lower))
        if np.any(invalid):
            if eta_np.ndim == 0:
                raise TiltingDomainError(
                    f"OTTilting.tilted_pvalue requires eta > {eta_lower!r} "
                    f"(s_t > 0) and finite, got {float(eta_np)!r}."
                )
            bad = int(np.argmax(invalid))
            raise TiltingDomainError(
                f"OTTilting.tilted_pvalue requires eta > {eta_lower!r} and "
                f"finite; offending index {bad} eta={float(eta_np.flat[bad])!r}."
            )

        # Shape dispatch: scalar -> numpy fast path; bulk -> jit'd kernel.
        theta_np = np.asarray(theta, dtype=np.float64)
        D_np = np.asarray(D, dtype=np.float64)
        if theta_np.size == 1 and eta_np.size == 1 and D_np.size == 1:
            # Returns Python float, NOT jnp.asarray(float) — see the
            # corresponding note in PowerLawTilting.tilted_pvalue. The
            # signature's `float | jax.Array` union makes the relaxed
            # return type honest at the type level.
            return _ot_tilted_pvalue_numpy_scalar(
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
        return _ot_tilted_pvalue_kernel(theta_arr, eta_arr, D_arr, w, mu0, sigma, statistic_name)

    def tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: Model,
        prior: NormalDistribution,
        eta: float,
        statistic_name: str,
    ) -> tuple[float, float]:
        """Numerical CI inversion of `tilted_pvalue` via brentq_with_doubling."""
        from ..models.normal_normal import NormalNormalModel
        from ._solvers import brentq_with_doubling

        if not is_normal_normal(model):
            raise NotImplementedError(
                "OTTilting.tilted_confidence_interval currently requires " "NormalNormalModel."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "OTTilting.tilted_confidence_interval currently requires a "
                "NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)
        mu_n = w * D + (1.0 - w) * mu0

        mid = float(D) if statistic_name == "wald" else float((1.0 - eta) * mu_n + eta * D)  # waldo

        def f(theta_val: float) -> float:
            return (
                float(self.tilted_pvalue(float(theta_val), D, model, prior, eta, statistic_name))
                - alpha
            )

        half = 4.0 * sigma
        lo = brentq_with_doubling(f, midpoint=mid, initial_half_width=half, direction=-1)
        hi = brentq_with_doubling(f, midpoint=mid, initial_half_width=half, direction=+1)
        return (lo, hi)

    def dynamic_tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float,
        model: Model,
        prior: NormalDistribution,
        statistic_name: str,
        eta_at_theta: ArrayLike,
    ) -> NDArray[np.float64]:
        """p(theta) with eta varying per theta via a precomputed lookup."""
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        eta_arr = np.atleast_1d(np.asarray(eta_at_theta, dtype=np.float64))
        if theta_arr.shape != eta_arr.shape:
            raise ValueError(
                f"theta and eta_at_theta must have the same shape; got "
                f"{theta_arr.shape!r} and {eta_arr.shape!r}."
            )
        # eta validation now lives in `tilted_pvalue` (single source of
        # truth; was previously duplicated here for the Phase 1 skeptic
        # vector #5 contract). Removing the upfront check eliminates
        # ~5 µs of redundant work per scan and keeps error messages
        # consistent regardless of whether a caller hits this dynamic
        # entry-point or the inner `tilted_pvalue` directly (e.g. through
        # the vec_fn callback at `_dynamic.py:151`).
        # Vectorised path: `tilted_pvalue` now broadcasts over array eta,
        # so one bulk call replaces the scalar Python loop (Tier 1.3 N3).
        out = np.asarray(
            self.tilted_pvalue(theta_arr, D, model, prior, eta_arr, statistic_name),
            dtype=np.float64,
        )
        return out if out.size > 1 else np.asarray(float(out.item()))

    def dynamic_tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: Model,
        prior: NormalDistribution,
        statistic_name: str,
        eta_selector,
        n_grid: int = 401,
        coarse_n: int = 25,
        search_mult: float = 8.0,
    ) -> tuple[list[tuple[float, float]], float, int]:
        """Dynamic-eta CI: eta = eta*(|Delta(theta)|) per theta.

        Delegates to
        `frasian.tilting._dynamic.dynamic_ci_scan`; the scheme-specific
        bit is the `tilted_pvalue` closure.
        """
        from ..models.normal_normal import NormalNormalModel
        from ._dynamic import dynamic_ci_scan

        if not is_normal_normal(model):
            raise NotImplementedError(
                "OTTilting.dynamic_tilted_confidence_interval currently "
                "requires NormalNormalModel."
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

    def dynamic_tilted_confidence_interval_ot_generic(
        self,
        alpha: float,
        D: float,
        data: NDArray[np.float64],
        model: Model,
        prior: NormalDistribution,
        statistic_name: str,
        eta_selector,
        n_grid: int = 401,
        coarse_n: int = 25,
        search_mult: float = 8.0,
        n_mc: int = _GENERIC_TILTED_PVALUE_N_MC,
    ) -> tuple[list[tuple[float, float]], float, int]:
        """Dynamic-η CI inversion for OT using the GENERIC (MC) p-value.

        OT analog of
        ``power_law.dynamic_tilted_confidence_interval_generic``.
        Algorithm:
          1. Coarse-grid η*(θ) selection via ``eta_selector`` (closed-form
             internally).
          2. Interpolate η*(θ) onto the fine θ-grid.
          3. Vectorised generic-MC p-value via
             ``_generic_tilted_pvalue_ot_vec`` (currently a Python loop
             wrapper around the scalar — see helper docstring for the
             perf trade-off).
          4. Linear interpolation at α-crossings to build accept regions
             (same approach as power_law's vec dynamic).

        Cost: dominated by the n_grid × scalar-MC calls. At
        n_mc=2000, n_grid=401: ~30-40 s/CI under the loop wrapper. A
        true triple-batched OT vec helper (mirroring power_law's chunked
        D_3d / log_lik_3d layout) would bring this down to ~1-2 s/CI.

        Caveat: with low n_mc (~200), MC noise on the fine scan can
        produce spurious sub-α regions near the true CI boundaries —
        n_mc≥2000 is the safe regime at α=0.05.
        """
        from .._errors import BracketingFailed
        from ..models.normal_normal import NormalNormalModel
        from ._generic_pvalue import _stable_tilted_pvalue_seed
        from .eta_selectors import _NamedStatistic

        if not is_normal_normal(model):
            raise NotImplementedError(
                "dynamic_tilted_confidence_interval_ot_generic currently "
                "requires NormalNormalModel — the dynamic-η selector is "
                "NN-only by design (NumericalEtaSelector inside is closed-"
                "form NN); generalising to non-NN models requires a generic "
                "eta selector first."
            )
        sigma = float(model.sigma)

        derived_seed = _stable_tilted_pvalue_seed(
            np.atleast_1d(np.asarray(data, dtype=np.float64)),
            model, prior, 0.0, alpha, _GENERIC_TILTED_PVALUE_BASE_SEED,
        )

        for widen in (1.0, 2.0):
            search_half = float(widen * search_mult * sigma)
            theta_lo = D - search_half
            theta_hi = D + search_half
            theta_grid = np.linspace(theta_lo, theta_hi, int(n_grid))

            coarse_theta_grid = np.linspace(theta_lo, theta_hi, int(coarse_n))
            coarse_eta = eta_selector.select_grid(
                coarse_theta_grid, self,
                model=model, prior=prior, alpha=alpha,
                statistic=_NamedStatistic(statistic_name),
            )
            eta_at_theta = np.interp(theta_grid, coarse_theta_grid, coarse_eta)

            data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
            p_theta = _generic_tilted_pvalue_ot_vec(
                theta_grid, data_arr, model, prior, eta_at_theta, statistic_name,
                n_mc=int(n_mc), n_grid=_GENERIC_TILTED_PVALUE_N_GRID_MC,
                derived_seed=derived_seed, alpha=alpha,
                base_seed=_GENERIC_TILTED_PVALUE_BASE_SEED,
            )

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

            entries = list(crossings)
            hit_boundary = False
            if sgn[0] > 0:
                entries = [float(theta_lo)] + entries
                hit_boundary = True
            if sgn[-1] > 0:
                entries = entries + [float(theta_hi)]
                hit_boundary = True

            if not hit_boundary:
                break
            if widen == 2.0:
                raise BracketingFailed(
                    f"dynamic_tilted_confidence_interval_ot_generic: CI "
                    f"extends past search box (±{search_half}·σ around "
                    f"D={D!r}; sigma={sigma!r}). Increase search_mult."
                )

        if len(entries) % 2 != 0:
            raise BracketingFailed(
                f"dynamic_tilted_confidence_interval_ot_generic produced "
                f"odd-parity entries; got {entries!r}. Indicates a missed "
                f"tangential α-touch on the grid. Try a finer theta-grid."
            )

        regions: list[tuple[float, float]] = [
            (entries[i], entries[i + 1]) for i in range(0, len(entries), 2)
        ]
        total = float(sum(hi - lo for lo, hi in regions))
        return regions, total, len(regions)

    # ----- Uniform CI / regions / pvalue interface -----

    def _require_normal_sandbox(self, model: Model, prior: Prior) -> None:
        """Enforce Normal-Normal sandbox. Used by paths that have no
        generic fallback (currently the dynamic-η selector path).
        """
        from ..models.normal_normal import NormalNormalModel

        if not is_normal_normal(model):
            raise NotImplementedError(
                f"OTTilting requires NormalNormalModel for the uniform CI "
                f"interface; got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                f"OTTilting requires a NormalDistribution prior; " f"got {type(prior).__name__!r}."
            )

    @staticmethod
    def _is_normal_normal_pair(model: Model, prior: Prior) -> bool:
        """Predicate-only counterpart to `_require_normal_sandbox`."""
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

        ``config`` (optional, kw-only): when supplied, the dynamic-CI
        scan reads ``n_grid`` / ``coarse_n`` / ``search_mult`` from
        ``Config.dynamic_*``. When ``None`` (default), falls back to
        the selector's own attributes for backward compatibility.
        Skeptic Phase 5 vector #2.

        Phase 3d: non-Normal-Normal pairings supported via the generic
        numerical path with a STATIC selector. Dynamic selectors still
        require Normal-Normal (the dynamic scanner builds its θ-window
        from `D ± search_mult * sigma`, which is Normal-Normal-flavoured).
        """
        # `statistic.force_generic=True` collapses both branches onto
        # the generic MC path even on NN — same flag honored by
        # PowerLawTilting. Phase H: dynamic + force_generic IS now
        # supported on NN via `dynamic_tilted_confidence_interval_ot_generic`
        # (mirrors power_law's Phase F enhancement); the underlying
        # `_generic_tilted_pvalue_ot_vec` is currently a Python-loop
        # wrapper around the scalar so wall time is ~30-40 s/CI vs
        # power_law's 1-2 s/CI. Adequate for sanity checks; a true
        # triple-batched OT vec is a tractable follow-up.
        force_generic = bool(getattr(statistic, "force_generic", False))
        route_generic = force_generic or not self._is_normal_normal_pair(model, prior)
        if route_generic:
            if getattr(self.selector, "is_dynamic", False):
                if force_generic and self._is_normal_normal_pair(model, prior):
                    if config is not None:
                        n_grid = int(config.dynamic_n_grid)
                        coarse_n = int(config.dynamic_coarse_n)
                        search_mult = float(config.dynamic_search_mult)
                    else:
                        n_grid = int(getattr(self.selector, "n_grid", 401))
                        coarse_n = int(getattr(self.selector, "coarse_n", 25))
                        search_mult = float(getattr(self.selector, "search_mult", 8.0))
                    stat_n_mc = int(
                        getattr(statistic, "n_mc", _GENERIC_TILTED_PVALUE_N_MC)
                    )
                    D = _data_to_scalar_D(data)
                    regions, _, _ = self.dynamic_tilted_confidence_interval_ot_generic(
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
                            f"dynamic generic OT CI inversion produced no "
                            f"regions at D={D!r}"
                        )
                    return regions
                if force_generic:
                    raise NotImplementedError(
                        "OTTilting + dynamic-η selector + force_generic=True "
                        "currently only supported on NormalNormalModel + "
                        "NormalDistribution prior. Generalising to non-NN "
                        "models requires a generic eta-selector."
                    )
                raise NotImplementedError(
                    "OTTilting dynamic-η selector currently requires "
                    "NormalNormalModel + NormalDistribution prior. Use a "
                    "static selector (FixedEtaSelector) for non-Normal-Normal "
                    "pairings."
                )
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
            lo, hi = _generic_tilted_confidence_interval_ot(
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
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
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
        """Convex hull of `confidence_regions`; ``config`` forwarded
        to that method for the dynamic-CI scan resolution.
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
        # Phase 3d: dispatch generic for non-Normal-Normal pairings.
        # `statistic.force_generic=True` collapses both branches onto
        # the generic MC path even on NN.
        force_generic = bool(getattr(statistic, "force_generic", False))
        route_generic = force_generic or not self._is_normal_normal_pair(model, prior)
        if route_generic:
            if getattr(self.selector, "is_dynamic", False):
                if force_generic:
                    raise NotImplementedError(
                        "OTTilting.pvalue + dynamic-η selector + "
                        "force_generic=True is not supported. Use a static "
                        "selector."
                    )
                raise NotImplementedError(
                    "OTTilting.pvalue dynamic-η selector currently requires "
                    "NormalNormalModel + NormalDistribution prior."
                )
            from ..config import Config as _Config
            from ._generic_pvalue import _stable_tilted_pvalue_seed
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
            derived_seed = _stable_tilted_pvalue_seed(
                data_arr, model, prior, eta, alpha,
                _GENERIC_TILTED_PVALUE_BASE_SEED,
            )
            out = np.empty(theta_arr.shape, dtype=np.float64)
            for i, th in enumerate(theta_arr):
                out[i] = _generic_tilted_pvalue_ot(
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
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
            # Phase 3a-1: θ-space directly.
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            theta_lo = float(theta_arr.min())
            theta_hi = float(theta_arr.max())
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
