"""Differentiable JAX implementations of `tilted_pvalue` per scheme.

A registry `JAX_TILTED_PVALUE` keyed on `scheme.name` returning a
function that takes broadcastable JAX arrays and produces a `(N, n_theta)`
p-value tensor. Used inside the training loss; not used at inference
(production code uses the `tilted_pvalue` on each scheme, which has
its own JAX kernel + numpy scalar fast path; see `tilting/power_law.py`).

Each JAX implementation is a direct port of its numpy counterpart:
- power_law: `src/frasian/tilting/power_law.py`
- ot:        `src/frasian/tilting/ot.py`

Tested against the numpy reference to atol 1e-10 in
`tests/regression/test_jax_pvalue_matches_numpy.py`.

Why this is separate from `tilting/power_law.py::_tilted_pvalue_kernel`
-----------------------------------------------------------------------
The kernel in `tilting/` is the inference-time implementation: it
raises ``TiltingDomainError`` on invalid eta. The training-time
implementation here CLAMPS divisor-bearing intermediates so the loss
surface stays smooth and gradient-bearing for invalid eta — Head A's
width loss can descend toward valid eta even when EtaNet drifts
outside the admissible range. The validity helper (numpy path,
`validity.py`) labels Head B's BCE correctly regardless of what the
JAX surface returns. This is intentional and documented in the legacy
torch implementation (preserved verbatim in the port).
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active

_FORCE_X64 = _x64  # keep static-analysis from stripping the import

_SQRT2 = math.sqrt(2.0)


def _phi(x: jax.Array) -> jax.Array:
    """Standard-normal CDF via jax.scipy.stats.norm.cdf."""
    return jsp_stats.norm.cdf(x)


def power_law_tilted_pvalue_jax(
    theta: jax.Array,
    D: jax.Array,
    w: jax.Array,
    mu0: jax.Array,
    sigma: jax.Array,
    eta: jax.Array,
    statistic_name: str,
) -> jax.Array:
    """JAX port of `PowerLawTilting.tilted_pvalue` for (power_law, waldo|wald).

    All array inputs broadcast to the desired output shape. Typical use:
    ``theta`` is ``(B, N)``, ``D, w, mu0, eta`` are ``(B, 1)`` or ``(B, N)``,
    ``sigma`` is scalar.

    Two regimes:
      - Inside the admissible range (``eta < 1/(1-w)``): exact numpy
        behaviour up to float64 precision.
      - Outside: ``jnp.maximum(denom, 1e-6)`` keeps the algebra finite
        and produces a smooth surface that Head A's width loss can
        descend toward valid eta. The validity helper raises
        ``TiltingDomainError`` for invalid eta independently, so Head B's
        BCE labels are correct regardless.

    Returns an array broadcast-shaped from the inputs.
    """
    if statistic_name == "wald":
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - _phi(z))

    if statistic_name == "waldo":
        # denom = 1 - eta(1 - w); clamp to avoid divide-by-zero. The
        # clamped surface is smooth so Head A's width loss has a
        # gradient even when EtaNet predicts eta outside the admissible
        # range — letting the boundary penalty + width signal jointly
        # push eta back without masking the gradient out entirely.
        denom = jnp.maximum(1.0 - eta * (1.0 - w), 1e-6)
        mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * mu0) / denom
        norm_factor = w * sigma / denom
        a = jnp.abs(mu_eta - theta) / norm_factor
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / (denom * norm_factor)
        return _phi(b - a) + _phi(-a - b)

    raise NotImplementedError(
        f"power_law_tilted_pvalue_jax: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo')."
    )


def ot_tilted_pvalue_jax(
    theta: jax.Array,
    D: jax.Array,
    w: jax.Array,
    mu0: jax.Array,
    sigma: jax.Array,
    eta: jax.Array,
    statistic_name: str,
) -> jax.Array:
    """JAX port of `OTTilting.tilted_pvalue` for (ot, waldo|wald).

    Like ``power_law_tilted_pvalue_jax``, the JAX surface stays smooth
    and gradient-bearing for invalid eta via ``jnp.maximum(s_t, 1e-6)``
    rather than NaN-masking — Head A's width loss can descend toward
    valid eta even when EtaNet drifts outside [0, 1]. The numpy
    ``OTTilting.tilted_pvalue`` raises ``TiltingDomainError`` for
    invalid eta independently, so the validity helper (numpy-driven)
    labels Head B's BCE correctly regardless of what the JAX surface
    returns. An earlier round NaN-masked here too, which broke OT
    training entirely (every aux sample masked out of the boundary-
    penalty signal).

    See ``power_law_tilted_pvalue_jax`` for input/output shape conventions;
    signatures match for registry uniformity.
    """
    if statistic_name == "wald":
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - _phi(z))

    if statistic_name == "waldo":
        # mu_t = (1 - eta)*mu_n + eta*D, with mu_n = w*D + (1-w)*mu0.
        mu_n = w * D + (1.0 - w) * mu0
        mu_t = (1.0 - eta) * mu_n + eta * D
        # s_t = (w + eta*(1-w))*sigma; admissible iff > 0. We clamp to
        # keep the gradient alive even at slightly-invalid eta so Head A
        # can move out of the bad region under the joint width +
        # boundary signal. The validity helper (numpy path) raises
        # `TiltingDomainError` for eta outside [0, 1], so Head B's
        # labels remain correct.
        s_t = jnp.maximum((w + eta * (1.0 - w)) * sigma, 1e-6)
        a = jnp.abs(mu_t - theta) / s_t
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / s_t
        return _phi(b - a) + _phi(-a - b)

    raise NotImplementedError(
        f"ot_tilted_pvalue_jax: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo')."
    )


def _generic_grid_tilted_moments(
    eta: jax.Array,
    log_p_lik_grid: jax.Array,
    log_p_prior_grid: jax.Array,
    theta_grid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute (mu_tilted, var_tilted) of the tilted distribution on the grid.

    Shared kernel used by both `generic_grid_tilted_pvalue` and downstream
    moment-only consumers. Returns scalar (per-sample, per-θ_test) moments.

    Shapes:
    - `eta`: (B, G_test) — η per (sample, θ_test).
    - `log_p_lik_grid`: (B, N_grid) — log L(θ_grid; D_b).
    - `log_p_prior_grid`: (N_grid,) — log π(θ_grid), broadcast across batch.
    - `theta_grid`: (N_grid,).
    - Returns: ((B, G_test), (B, G_test)).

    The deriver verified analytically + numerically that
    `log q = log L + (1-η) log π` reduces to PowerLawTilting's Theorem 6
    (mu_eta, sigma_eta) on Normal-Normal at atol 1e-7 with N_grid=1024.
    """
    log_q = (
        log_p_lik_grid[:, None, :]
        + (1.0 - eta[:, :, None]) * log_p_prior_grid[None, None, :]
    )
    log_q_max = jnp.max(log_q, axis=-1, keepdims=True)
    pdf_unnorm = jnp.exp(log_q - log_q_max)
    Z = jnp.trapezoid(pdf_unnorm, theta_grid, axis=-1)
    pdf = pdf_unnorm / Z[..., None]
    mu = jnp.trapezoid(theta_grid * pdf, theta_grid, axis=-1)
    m2 = jnp.trapezoid(theta_grid * theta_grid * pdf, theta_grid, axis=-1)
    var = jnp.maximum(m2 - mu * mu, 1e-12)
    return mu, var


def generic_grid_tilted_pvalue(
    theta_test: jax.Array,
    eta: jax.Array,
    log_p_lik_grid: jax.Array,
    log_p_prior_grid: jax.Array,
    theta_grid: jax.Array,
    statistic_name: str,
) -> jax.Array:
    """Model-agnostic JAX-traceable tilted p-value via grid log-densities.

    Phase 4 generalisation of the per-scheme JAX kernels above. Works
    against any (model, prior) pair where `log L(θ_grid)` and
    `log π(θ_grid)` are JAX-traceable. The deriver verified
    (`/root/.claude/plans/.../`) that `log L + (1-η) log π` reduces
    to PowerLawTilting's Theorem 6 closed form on Normal-Normal at
    atol 1e-7 with N_grid=1024.

    Two-step structure mirroring the closed-form Theorem 8 path:
    1. **Data-side moments**: build the tilted log-density on
       `theta_grid` via `log q = log L + (1-η) log π`; max-subtract;
       trapezoidal Z-normalise; integrate to get
       `(μ_tilted, σ²_tilted)`. These are scalar (per-sample, per-θ_test).
    2. **Test-side**: evaluate the WALDO p-value via the *normal
       approximation* `t = (μ_tilted - θ_test)² / σ²_tilted` and
       `p = 2(1-Φ(√t))`. Production CI inversion uses MC over D' for
       the exact reference distribution under H_0; training only
       needs a smooth differentiable surrogate.

    Shapes (broadcast-friendly, mirrors the per-scheme kernels):
    - `theta_test`: `(B, G_test)` — points where p is evaluated.
    - `eta`: `(B, G_test)` or scalar — η at each (sample, θ_test). Closed
      under EtaNet's per-θ output.
    - `log_p_lik_grid`: `(B, N_grid)` — `log L(θ_grid; D_b)` per batch element.
    - `log_p_prior_grid`: `(N_grid,)` — `log π(θ_grid)`, broadcast across batch.
    - `theta_grid`: `(N_grid,)` — fixed support grid; same across batch.
    - Returns: `(B, G_test)` — p-value at each (sample, θ_test).

    The intermediate cube is `(B, G_test, N_grid)`. At B=8, G_test=401,
    N_grid=1024 this is ~26 MB float64 — fits in L3 / one allocator
    chunk, no streaming needed. JIT cache hit rate after epoch 0 is
    100% because shapes are constant across training iterations.

    Numerical stability: max-subtract before exp; `var` floored at
    1e-12 to prevent division-by-zero in the WALDO surface. Both
    safeguards are differentiable (max is `jax.lax.stop_gradient`-free;
    floor is `jnp.maximum`).

    Wald path: eta-independent (the kernel reduces to the per-scheme
    JAX implementations' Wald case, `2(1-Φ(|D-θ|/σ))`). For the grid
    path, "D" and "σ" aren't directly available — but the Wald
    statistic on the generic path uses the χ²₁ asymptotic via the
    posterior moments, NOT the closed-form Wald formula. For training
    purposes we delegate to a normal-approximation form using the
    likelihood's mode and Fisher info derived from the grid; this is
    a documented approximation. Callers that need exact Wald should
    use the per-scheme NN closed form via `power_law_tilted_pvalue_jax`.

    Important relationship to closed-form Theorem 8
    -----------------------------------------------
    The per-scheme `power_law_tilted_pvalue_jax` for "waldo" returns the
    asymmetric two-Φ form `Φ(b - a) + Φ(-a - b)` where `b = (1-η)(1-w)
    (μ₀ - θ)/(denom · norm_factor)` carries the prior-data conflict
    contribution. THIS kernel uses the simpler symmetric normal
    approximation `2(1 - Φ(|μ - θ|/σ))`. The two AGREE when `b = 0`
    (i.e. when `η = 1` drops the prior contribution, or when
    `μ₀ = θ`) but DIFFER by O(b) when the prior conflicts with the
    test point. The choice is intentional: training only needs a
    differentiable surrogate, and the symmetric form has cleaner
    gradients through η. The cross-check test pins moment-level
    agreement (μ_tilted, σ²_tilted match Theorem 6), NOT p-value
    agreement.
    """
    mu, var = _generic_grid_tilted_moments(
        eta, log_p_lik_grid, log_p_prior_grid, theta_grid
    )

    if statistic_name == "waldo":
        z = jnp.abs(mu - theta_test) / jnp.sqrt(var)
        return 2.0 * (1.0 - _phi(z))
    if statistic_name == "wald":
        # Wald is data-anchored (uses MLE not posterior). On the grid
        # path we approximate by treating the likelihood's mode as the
        # MLE and the inverse-Fisher-info as the variance scale; both
        # come from the log-likelihood's curvature at the grid maximum.
        # NOTE: this is a documented approximation; for exact Wald on
        # Normal-Normal use `power_law_tilted_pvalue_jax`.
        # Find the likelihood's mode on the grid.
        lik_max_idx = jnp.argmax(log_p_lik_grid, axis=-1)        # (B,)
        mle = theta_grid[lik_max_idx]                            # (B,)
        # Approximate sigma from the likelihood's curvature: use the
        # full-width at half-maximum / 2.355 heuristic on the
        # exponentiated likelihood (sufficient for Bernoulli-like
        # likelihoods). For Normal-Normal this would equal sigma_eff;
        # for Bernoulli it approximates sqrt(theta(1-theta)/n).
        lik_pdf = jnp.exp(log_p_lik_grid - jnp.max(log_p_lik_grid, axis=-1, keepdims=True))
        lik_Z = jnp.trapezoid(lik_pdf, theta_grid, axis=-1)
        lik_norm = lik_pdf / lik_Z[:, None]
        lik_mu = jnp.trapezoid(theta_grid * lik_norm, theta_grid, axis=-1)
        lik_m2 = jnp.trapezoid(theta_grid * theta_grid * lik_norm, theta_grid, axis=-1)
        lik_var = jnp.maximum(lik_m2 - lik_mu * lik_mu, 1e-12)
        sigma_eff = jnp.sqrt(lik_var)
        z_wald = jnp.abs(mle[:, None] - theta_test) / sigma_eff[:, None]
        return 2.0 * (1.0 - _phi(z_wald))
    raise NotImplementedError(
        f"generic_grid_tilted_pvalue: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo')."
    )


# Phase 4b: tuple-keyed registry on (scheme_name, model_kind).
# Keeps the per-scheme NN closed-form callables for the fast path
# (training byte-equality preserved); adds a "generic" key per scheme
# for the grid-based path which works on any model with JAX-traceable
# log_pdf / loglik.
JAX_TILTED_PVALUE: dict[tuple[str, str], Callable[..., jax.Array]] = {
    ("power_law", "normal_normal"): power_law_tilted_pvalue_jax,
    ("ot", "normal_normal"): ot_tilted_pvalue_jax,
    ("power_law", "generic"): generic_grid_tilted_pvalue,
    # ("ot", "generic") is deferred — QuantileMixturePath has no
    # closed-form log-density on a fixed grid; would need a separate
    # quantile-pushforward kernel.
}


def get_jax_tilted_pvalue(
    scheme_name: str,
    model_kind: str = "normal_normal",
) -> Callable[..., jax.Array]:
    """Look up the JAX tilted-p-value function for a (scheme, model) cell.

    Falls back to the per-scheme generic kernel (`(scheme, "generic")`)
    if the specific `(scheme, model_kind)` is not registered, so a
    novel model_kind automatically routes through the grid path when
    a generic kernel exists for the scheme.

    Raises `NotImplementedError` if neither the specific nor the
    generic key is registered for the scheme.

    `model_kind` defaults to `"normal_normal"` for backward compat
    with pre-Phase-4b callers; new callers should always pass it
    explicitly.
    """
    key = (scheme_name, model_kind)
    if key in JAX_TILTED_PVALUE:
        return JAX_TILTED_PVALUE[key]
    generic_key = (scheme_name, "generic")
    if generic_key in JAX_TILTED_PVALUE:
        return JAX_TILTED_PVALUE[generic_key]
    raise NotImplementedError(
        f"No JAX tilted_pvalue registered for cell "
        f"(scheme={scheme_name!r}, model={model_kind!r}). "
        f"Available cells: {sorted(JAX_TILTED_PVALUE)}. "
        f"To train against a new scheme/model, register a JAX p-value here."
    )
