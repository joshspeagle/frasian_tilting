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

    if statistic_name in ("waldo", "lrto", "scoreo"):
        # Trinity collapse on NN+Normal: q_η is a single Gaussian, so
        # τ_WALDO,η = τ_LRTO,η = τ_SCOREO,η identically and the three
        # statistics share the same p-value formula. See
        # ``docs/notes/2026-05-12-tilted-trinity-derivation.md``.
        #
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
        f"not supported (expected 'wald', 'waldo', 'lrto', or 'scoreo')."
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

    if statistic_name in ("waldo", "lrto", "scoreo"):
        # Trinity collapse on NN+Normal: q_η is a single Gaussian, so
        # τ_WALDO,η = τ_LRTO,η = τ_SCOREO,η identically and the three
        # statistics share the same p-value formula. See
        # ``docs/notes/2026-05-12-tilted-trinity-derivation.md``.
        #
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
        f"not supported (expected 'wald', 'waldo', 'lrto', or 'scoreo')."
    )


def fisher_rao_tilted_pvalue_jax(
    theta: jax.Array,
    D: jax.Array,
    w: jax.Array,
    mu0: jax.Array,
    sigma: jax.Array,
    eta: jax.Array,
    statistic_name: str,
) -> jax.Array:
    """JAX port of `FisherRaoTilting.tilted_pvalue` for (fisher_rao, waldo|wald).

    Thin wrapper around ``_fr_tilted_pvalue_kernel`` from
    ``frasian.tilting.fisher_rao`` — the kernel is already jit'd and
    vmap-friendly via per-X geodesic broadcasting. The wrapper just
    re-orders the args to match the registry convention shared with
    ``power_law_tilted_pvalue_jax`` / ``ot_tilted_pvalue_jax`` /
    ``mixture_tilted_pvalue_jax`` (``(theta, D, w, mu0, sigma, eta,
    statistic_name)``).

    Precision characteristics (per the FR kernel docstring): ~4e-4 at
    n_grid=8000 via trapezoidal quadrature over X. No closed form for
    interior eta because mu_FR(eta; X) is non-linear in X along the
    curved geodesic. This is sufficient gradient signal for training;
    audit-precision CIs use the adaptive brentq numpy-scalar path.

    Like the other JAX kernels, the surface stays smooth at the eta=[0,1]
    boundary so Head A's width loss can descend toward admissible eta
    even when EtaNet drifts. The validity helper (numpy-driven) labels
    Head B's BCE correctly regardless of what the JAX surface returns.

    See ``power_law_tilted_pvalue_jax`` for input/output shape conventions;
    signatures match for registry uniformity.
    """
    # Import inside the function to avoid a cycle: tilting/fisher_rao.py
    # already imports from statistics/, which imports back here via the
    # training package.
    from ...tilting.fisher_rao import _fr_tilted_pvalue_kernel

    # The underlying scalar kernel internally constructs an X-grid via
    # `jnp.linspace(theta - 8σ, theta + 8σ, n_grid=8000)`. For SCALAR
    # theta this yields shape `(n_grid,)` and the downstream
    # vmap+trapezoid chain returns a scalar. With non-scalar inputs,
    # the linspace and broadcast pattern break. PL/OT/MX use pure
    # closed-form scalar arithmetic and broadcast naturally; FR needs
    # an explicit broadcast wrapper because of the internal X-grid.
    #
    # The training loop's call pattern is heterogeneous:
    #   theta : shape (B, T)         — batch × theta-grid
    #   eta   : shape (B, T) or (B,) — per-θ from dynamic selector, OR per-batch
    #   D, w, mu0, sigma : shape (B,) — per-batch
    #
    # `jnp.vectorize` is the canonical JAX broadcast wrapper for this:
    # given a `signature='(),(),(),(),(),()->()'`, it auto-broadcasts
    # all inputs to a common shape and applies the scalar function
    # element-wise. Internally uses jax.vmap but handles the
    # broadcasting + nesting we'd otherwise hand-roll.
    #
    # Statistic name is static (Python str); close over it.
    def _scalar_call(th, et, d_, w_, m0, sg):
        return _fr_tilted_pvalue_kernel(th, et, d_, w_, m0, sg, statistic_name)

    vectorized = jnp.vectorize(
        _scalar_call, signature="(),(),(),(),(),()->()",
    )
    return vectorized(theta, eta, D, w, mu0, sigma)


_MIXTURE_QUADRATIC_LEADING_EPS = 1e-12
_MIXTURE_DISCRIMINANT_EPS = 1e-12


def mixture_tilted_pvalue_jax(
    theta: jax.Array,
    D: jax.Array,
    w: jax.Array,
    mu0: jax.Array,
    sigma: jax.Array,
    eta: jax.Array,
    statistic_name: str,
) -> jax.Array:
    """JAX port of `MixtureTilting.tilted_pvalue` for (mixture, waldo|wald).

    Mirror of ``_mixture_tilted_pvalue_numpy_scalar`` (see
    ``frasian/tilting/mixture.py`` for the derivation in
    ``docs/methods/mixture.md`` "Closed-form tilted-WALDO p-value").
    The accept set is the solution of a quadratic-in-X inequality
    Q(X) = L*X² + 2*M*X + N ≥ 0; the JAX implementation evaluates ALL
    branches and selects with ``jnp.where`` to stay vmap+jit-compatible.

    Endpoint shortcuts at eta=0 (bare WALDO) and eta=1 (bare 2-sided Wald)
    are applied via ``jnp.where`` for numerical robustness, mirroring
    the numpy reference; in the limit they would be reached by the
    quadratic formulation as well.

    Like ``power_law_tilted_pvalue_jax`` and ``ot_tilted_pvalue_jax``,
    this kernel does NOT raise on invalid eta — the surface stays
    smooth and gradient-bearing via ``jnp.maximum(..., eps)`` clamps on
    the variance terms. Validity is enforced by the numpy
    ``MixtureTilting.tilted_pvalue`` and the ``ValidityNet`` boundary
    penalty during training.
    """
    if statistic_name == "wald":
        # eta-independent: collapses to bare 2-sided Wald.
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - _phi(z))

    if statistic_name in ("lrto", "scoreo"):
        raise NotImplementedError(
            f"mixture_tilted_pvalue_jax: statistic={statistic_name!r} is out of "
            f"scope for the JAX training kernel. Mixture LRTO/SCOREO requires "
            f"autodiff through scipy.optimize for the mode, which JAX can't "
            f"trace. Use the numpy-scalar path in `tilting.mixture` for "
            f"inference; learned-η training against tilted-MX-LRTO/SCOREO "
            f"loss is out of scope. See "
            f"docs/notes/2026-05-12-tilted-trinity-derivation.md §6."
        )

    if statistic_name == "waldo":
        sigma_n_sq = w * sigma * sigma
        alpha_lin = w + eta * (1.0 - w)
        A = alpha_lin
        B = (1.0 - alpha_lin) * mu0 - theta
        C0 = (1.0 - eta) * sigma_n_sq + eta * sigma * sigma
        C1 = (1.0 - eta) * eta * (1.0 - w) * (1.0 - w)

        mu_til_D = alpha_lin * D + (1.0 - alpha_lin) * mu0
        var_til_D = jnp.maximum(
            C0 + C1 * (mu0 - D) * (mu0 - D),
            _MIXTURE_QUADRATIC_LEADING_EPS,
        )
        t_obs = (mu_til_D - theta) * (mu_til_D - theta) / var_til_D

        L = A * A - t_obs * C1
        M = A * B + t_obs * C1 * mu0
        N = B * B - t_obs * C0 - t_obs * C1 * mu0 * mu0
        disc_quarter = M * M - L * N

        # F(x) = Phi((x - theta) / sigma)
        def F(x: jax.Array) -> jax.Array:
            return _phi((x - theta) / sigma)

        # Quadratic branch (L > 0 or L < 0, disc ≥ 0):
        # roots clamped via L_safe to keep the gradient finite when L≈0.
        L_safe = jnp.where(
            jnp.abs(L) < _MIXTURE_QUADRATIC_LEADING_EPS,
            jnp.ones_like(L),
            L,
        )
        # Clamp BELOW zero (not at zero) so the gradient of sqrt stays
        # finite even when disc_quarter < 0. ``sqrt(max(x, 0))`` has a
        # 0×∞ gradient at x=0 that produces NaN; ``sqrt(max(x, eps))``
        # with eps > 0 gives finite gradient everywhere. The clamped
        # branch is gated out by the ``is_disc_neg`` mask below, so
        # the forward value is unchanged; only the gradient is sanitized.
        sqrt_disc = jnp.sqrt(jnp.maximum(disc_quarter, _MIXTURE_DISCRIMINANT_EPS))
        x_root_a = (-M - sqrt_disc) / L_safe
        x_root_b = (-M + sqrt_disc) / L_safe
        x_minus = jnp.minimum(x_root_a, x_root_b)
        x_plus = jnp.maximum(x_root_a, x_root_b)
        F_minus = F(x_minus)
        F_plus = F(x_plus)
        p_quad_L_pos = F_minus + (1.0 - F_plus)
        p_quad_L_neg = F_plus - F_minus

        # Discriminant ≤ 0 branch: Q has constant sign.
        p_disc_neg_L_pos = jnp.ones_like(L)
        p_disc_neg_L_neg = jnp.zeros_like(L)

        # Linear branch (|L| < eps): 2*M*X + N ≥ 0.
        M_safe = jnp.where(
            jnp.abs(M) < _MIXTURE_QUADRATIC_LEADING_EPS,
            jnp.ones_like(M),
            M,
        )
        x_lin = -N / (2.0 * M_safe)
        F_lin = F(x_lin)
        # If M > 0: X ≥ x_lin; p = 1 - F(x_lin). If M < 0: X ≤ x_lin; p = F(x_lin).
        p_linear_M_pos = 1.0 - F_lin
        p_linear_M_neg = F_lin
        p_constant = jnp.where(N >= 0.0, jnp.ones_like(N), jnp.zeros_like(N))

        # Compose:
        is_L_small = jnp.abs(L) <= _MIXTURE_QUADRATIC_LEADING_EPS
        is_M_small = jnp.abs(M) <= _MIXTURE_QUADRATIC_LEADING_EPS
        # Branch test uses true sign. The sqrt floor (line ~225) uses
        # ``_MIXTURE_DISCRIMINANT_EPS`` ONLY for gradient sanitization;
        # mixing the two constants conflates a numerical decision
        # (when is disc "essentially zero") with a gradient hygiene
        # decision (sqrt floor) and creates a tiny inconsistency zone
        # at ``0 < disc_quarter < eps`` where the branch picks the
        # quadratic-roots formula but the roots are computed with a
        # ``sqrt(eps)`` perturbation.
        is_disc_neg = disc_quarter < 0.0

        # Linear case selection (within is_L_small).
        p_linear = jnp.where(
            is_M_small,
            p_constant,
            jnp.where(M > 0.0, p_linear_M_pos, p_linear_M_neg),
        )
        # Discriminant-negative selection (L sign).
        p_disc_neg = jnp.where(L > 0.0, p_disc_neg_L_pos, p_disc_neg_L_neg)
        # Discriminant-non-negative selection (L sign).
        p_disc_pos = jnp.where(L > 0.0, p_quad_L_pos, p_quad_L_neg)
        # Compose disc branches.
        p_quadratic = jnp.where(is_disc_neg, p_disc_neg, p_disc_pos)
        # Compose linear vs quadratic.
        p = jnp.where(is_L_small, p_linear, p_quadratic)

        # Endpoint shortcuts (numerical robustness; mirror numpy reference).
        # eta = 0 → bare WALDO closed form.
        wsig = w * sigma
        p_eta_zero_mu_n = w * D + (1.0 - w) * mu0
        p_eta_zero_a = jnp.abs(p_eta_zero_mu_n - theta) / wsig
        p_eta_zero_b = (1.0 - w) * (mu0 - theta) / wsig
        p_eta_zero = _phi(p_eta_zero_b - p_eta_zero_a) + _phi(-p_eta_zero_a - p_eta_zero_b)
        # eta = 1 → bare 2-sided Wald.
        z_wald = jnp.abs(D - theta) / sigma
        p_eta_one = 2.0 * (1.0 - _phi(z_wald))

        p = jnp.where(eta == 0.0, p_eta_zero, p)
        p = jnp.where(eta == 1.0, p_eta_one, p)
        return p

    raise NotImplementedError(
        f"mixture_tilted_pvalue_jax: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo'; "
        f"'lrto'/'scoreo' are out of scope — see above)."
    )


def _mixture_grid_component_moments(
    log_p_lik_grid: jax.Array,
    log_p_prior_grid: jax.Array,
    theta_grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute (μ, σ²) of the posterior and (μ, σ²) of the
    likelihood-as-distribution on the θ-grid.

    The two pdf's are formed via:
      posterior(θ)             ∝ exp(log L(θ; D) + log π(θ))
      likelihood-as-dist(θ)    ∝ exp(log L(θ; D))

    Each is max-subtracted then trapezoidal-Z-normalized; moments come
    from straightforward trapezoidal integration of θ·pdf and θ²·pdf.

    Shapes:
    - `log_p_lik_grid`:   (B, N_grid)
    - `log_p_prior_grid`: (N_grid,)
    - `theta_grid`:       (N_grid,)
    - Returns: ((B,), (B,), (B,), (B,)) — (μ_post, σ²_post, μ_lik, σ²_lik).
    """
    # Posterior pdf on grid: log q_post = log L + log π.
    log_post = log_p_lik_grid + log_p_prior_grid[None, :]
    log_post_max = jnp.max(log_post, axis=-1, keepdims=True)
    post_pdf_un = jnp.exp(log_post - log_post_max)
    Z_post = jnp.trapezoid(post_pdf_un, theta_grid, axis=-1)
    post_pdf = post_pdf_un / Z_post[..., None]
    mu_post = jnp.trapezoid(theta_grid[None, :] * post_pdf, theta_grid, axis=-1)
    m2_post = jnp.trapezoid(
        theta_grid[None, :] * theta_grid[None, :] * post_pdf, theta_grid, axis=-1
    )
    var_post = jnp.maximum(m2_post - mu_post * mu_post, 1e-12)

    # Likelihood-as-distribution pdf on grid: log q_lik = log L.
    log_lik_max = jnp.max(log_p_lik_grid, axis=-1, keepdims=True)
    lik_pdf_un = jnp.exp(log_p_lik_grid - log_lik_max)
    Z_lik = jnp.trapezoid(lik_pdf_un, theta_grid, axis=-1)
    lik_pdf = lik_pdf_un / Z_lik[..., None]
    mu_lik = jnp.trapezoid(theta_grid[None, :] * lik_pdf, theta_grid, axis=-1)
    m2_lik = jnp.trapezoid(
        theta_grid[None, :] * theta_grid[None, :] * lik_pdf, theta_grid, axis=-1
    )
    var_lik = jnp.maximum(m2_lik - mu_lik * mu_lik, 1e-12)

    return mu_post, var_post, mu_lik, var_lik


def mixture_grid_tilted_pvalue(
    theta_test: jax.Array,
    eta: jax.Array,
    log_p_lik_grid: jax.Array,
    log_p_prior_grid: jax.Array,
    theta_grid: jax.Array,
    statistic_name: str,
) -> jax.Array:
    """Model-agnostic JAX tilted p-value for **MixtureTilting** (m-geodesic).

    Companion to `generic_grid_tilted_pvalue` (which is the **e-geodesic**
    for power_law). Where the e-geodesic uses
    `log q = log L + (1−η) log π` (log-space affine combination), the
    m-geodesic uses **density-space** mixture
    `q(θ) = (1−η)·posterior(θ) + η·likelihood-as-dist(θ)`.

    Implementation:
    1. Compute (μ_post, σ²_post) and (μ_lik, σ²_lik) once per batch
       element via trapezoid integration on the shared θ-grid (calls
       ``_mixture_grid_component_moments``).
    2. Combine to mixture moments at each (B, G_test) via the standard
       mixture formula:
         μ_mix    = (1−η)·μ_post + η·μ_lik
         E[θ²]_mix = (1−η)·(σ²_post + μ_post²) + η·(σ²_lik + μ_lik²)
         σ²_mix    = E[θ²]_mix − μ_mix²
    3. Apply the same WALDO normal-approximation surrogate as
       ``generic_grid_tilted_pvalue``: ``p = 2(1−Φ(|μ_mix−θ_test|/σ_mix))``.

    Same calibration caveat as the PL generic kernel: this is a moment-
    matching surrogate intended for differentiable training only.
    Inference-time generic on non-NN + mixture uses
    ``MixtureTilting._generic_tilted_pvalue`` (numpy MC) for exact
    calibration.

    Endpoint sanity:
    - η=0 → mixture moments collapse to (μ_post, σ²_post) → bare WALDO surrogate.
    - η=1 → mixture moments collapse to (μ_lik, σ²_lik) → likelihood-only surrogate.

    Shapes (mirror ``generic_grid_tilted_pvalue``):
    - `theta_test`: (B, G_test) — points where p is evaluated.
    - `eta`: (B, G_test) or scalar — η at each (sample, θ_test).
    - `log_p_lik_grid`: (B, N_grid).
    - `log_p_prior_grid`: (N_grid,).
    - `theta_grid`: (N_grid,).
    - Returns: (B, G_test).

    Memory: intermediate cube is the (B, N_grid) component-pdf grids
    (smaller than PL generic's (B, G_test, N_grid) cube — mixture
    component moments are scalar per-batch and broadcast against eta(B,
    G_test) only at the final combination step).
    """
    mu_post, var_post, mu_lik, var_lik = _mixture_grid_component_moments(
        log_p_lik_grid, log_p_prior_grid, theta_grid
    )
    # Broadcast component moments against (B, G_test) via the eta axis.
    one_minus_eta = 1.0 - eta
    mu_mix = one_minus_eta * mu_post[:, None] + eta * mu_lik[:, None]
    m2_post = var_post + mu_post * mu_post
    m2_lik = var_lik + mu_lik * mu_lik
    m2_mix = one_minus_eta * m2_post[:, None] + eta * m2_lik[:, None]
    var_mix = jnp.maximum(m2_mix - mu_mix * mu_mix, 1e-12)

    if statistic_name == "waldo":
        z = jnp.abs(mu_mix - theta_test) / jnp.sqrt(var_mix)
        return 2.0 * (1.0 - _phi(z))
    # Wald: same rationale as generic_grid_tilted_pvalue — Wald only
    # accepts identity tilting; this branch is unreachable through the
    # runner. Raise rather than ship a moment-of-likelihood approximation.
    raise NotImplementedError(
        f"mixture_grid_tilted_pvalue: statistic={statistic_name!r} "
        f"not supported (expected 'waldo'). Wald only accepts identity "
        f"tilting; use `statistics/wald.py::_generic_pvalue` directly."
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

    Important relationship to closed-form Theorem 8 (skeptic Phase 4 #1)
    -------------------------------------------------------------------
    The per-scheme `power_law_tilted_pvalue_jax` for "waldo" returns the
    asymmetric two-Φ form `Φ(b - a) + Φ(-a - b)` where `b = (1-η)(1-w)
    (μ₀ - θ)/(denom · norm_factor)` and `a = |μ_eta - θ|/(w·σ/denom)`.
    THIS kernel uses the simpler symmetric normal approximation
    `2(1 - Φ(|μ - θ|/σ_tilted))`. Two algebraic differences:

    1. **Different scale.** Theorem 8 uses `ν = w·σ/denom`, this uses
       `σ_tilted = σ·√(w/denom)`; ratio `ν/σ_tilted = √(w/denom)`.
    2. **No prior-conflict term `b`** in the symmetric form.

    Quantified bias on Normal-Normal at the conflict band (|Δ| ≥ 1.5):
    `argmin_η` of the surrogate drifts up to ~1.3 from `argmin_η` of
    Theorem 8 — a major bias if this kernel were used for NN training.

    **Why we ship the surrogate anyway.** Production NN training uses
    `power_law_tilted_pvalue_jax` (Theorem 8 exact) via the
    `("power_law", "normal_normal")` registry key — the surrogate is
    NEVER on the NN training path. The grid kernel is registered ONLY
    under `("power_law", "generic")` for future non-NN models, where
    computing `b` and `ν` exactly would require Monte Carlo over D'
    under H_0 — the inference-time `power_law._generic_tilted_pvalue`
    does this at n_mc=200, but it is too expensive for per-step
    training. The surrogate is the cheapest differentiable proxy that
    preserves moment-level agreement with Theorem 6.

    Inference-time calibration for future non-NN models would be
    verified against the MC reference path; the surrogate's bias is
    an η-target shift, not a calibration error.

    Bias is pinned by `tests/regression/test_grid_surrogate_vs_theorem8.py`
    as a regression that catches future widening.
    """
    mu, var = _generic_grid_tilted_moments(
        eta, log_p_lik_grid, log_p_prior_grid, theta_grid
    )

    if statistic_name == "waldo":
        z = jnp.abs(mu - theta_test) / jnp.sqrt(var)
        return 2.0 * (1.0 - _phi(z))
    # Wald is intentionally NOT supported on the grid path:
    # `WaldStatistic.accepts_tilting` is `identity`-only, so a
    # `(power_law | ot, "generic") + statistic_name="wald"` cell is
    # never built by the runner. Removing the dead branch closes
    # skeptic Phase 4 #2 (a moment-of-likelihood approximation that
    # was not pinned against closed-form Wald).
    raise NotImplementedError(
        f"generic_grid_tilted_pvalue: statistic={statistic_name!r} "
        f"not supported (expected 'waldo'). The grid path does not "
        f"implement Wald; Wald only accepts identity tilting and uses "
        f"`statistics/wald.py::_generic_pvalue` directly."
    )


# Phase 4b: tuple-keyed registry on (scheme_name, model_kind).
# Keeps the per-scheme NN closed-form callables for the fast path
# (training byte-equality preserved); adds a "generic" key per scheme
# for the grid-based path which works on any model with JAX-traceable
# log_pdf / loglik.
JAX_TILTED_PVALUE: dict[tuple[str, str], Callable[..., jax.Array]] = {
    ("power_law", "normal_normal"): power_law_tilted_pvalue_jax,
    ("ot", "normal_normal"): ot_tilted_pvalue_jax,
    ("mixture", "normal_normal"): mixture_tilted_pvalue_jax,
    ("fisher_rao", "normal_normal"): fisher_rao_tilted_pvalue_jax,
    ("power_law", "generic"): generic_grid_tilted_pvalue,
    ("mixture", "generic"): mixture_grid_tilted_pvalue,
    # ("ot", "generic") is deferred — QuantileMixturePath has no
    # closed-form log-density on a fixed grid; would need a separate
    # quantile-pushforward kernel.
    # ("fisher_rao", "generic") is deferred — FR currently requires
    # Gaussian endpoints (the half-plane closed form); a parametric-
    # family generic path is Stage B work.
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
