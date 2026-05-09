"""Differentiable training losses for the learned dynamic-η selector.

All losses operate on a `(B, N)` p-value array (one row per batch
element, one column per θ-grid point) and a `(N,)` or `(B, N)` θ-grid.
They return a scalar — the mean over *valid* batch samples — that is
end-to-end differentiable in the η_φ parameters that produced
`p_theta`.

The three losses (default → alternative → α-conditioned):

1. `integrated_pvalue_loss` (default): `E_{B} [∫ p_dyn(θ) dθ]`. By
   Fubini this equals `E_B [∫_α |C_α| dα]` with α uniform on (0, 1) —
   i.e., the CI width integrated over all α-levels. α-marginalised.

2. `cd_variance_loss` (alternative): `E_B [Var_{F_D}[θ]]` where the
   CD `F_D` is the Schweder-Hjort density built from p_dyn. Penalises
   long tails more aggressively. α-free.

3. `static_width_loss(alpha)` (α-conditioned): `E_B [|C_α(D)|]` at a
   fixed α, computed via a sigmoid-relaxed indicator. Used only when
   training in α-conditioned mode.

All three are end-to-end differentiable (no argmax, no brentq).

**Reactive NaN guard.** When the JAX p-value formula drifts numerically
(possible at extreme `(w, eta)` combinations as we extend support to
non-Normal-Normal models), individual samples can yield NaN/Inf
trapezoidal-integral values. Each loss masks those samples out of
the batch mean rather than letting one NaN contaminate the gradient.
The mask is jittable: `jnp.where(n_valid > 0, sum / n_valid, NaN)` —
the sentinel propagates without a host sync, and the caller (training
loop) is responsible for raising on a non-finite scalar loss. This
preserves the legacy "raise on all-bad batch" behaviour without
breaking the JAX trace.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .cd_jax import cd_density_jax

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _masked_mean(per_sample: jax.Array) -> jax.Array:
    """Mean over `per_sample` rows that are finite.

    Returns ``NaN`` when every entry is non-finite (the legacy torch
    path raised; the JAX path stays jittable so the caller is
    responsible for raising on a non-finite scalar loss). Non-finite
    entries are replaced with ``0`` before reduction so the gradient
    remains well-defined on the surviving samples and the masked
    division uses the live count.
    """
    valid = jnp.isfinite(per_sample)
    n_valid = valid.sum()
    safe = jnp.where(valid, per_sample, jnp.zeros_like(per_sample))
    total = safe.sum()
    return jnp.where(n_valid > 0, total / n_valid, jnp.asarray(jnp.nan, per_sample.dtype))


def integrated_pvalue_loss(
    p_theta: jax.Array,
    theta_grid: jax.Array,
) -> jax.Array:
    """L¹ norm of the dynamic p-value curve, averaged over the batch.

    `loss = mean_B[ ∫_θ p_dyn(θ; D, η_φ) dθ ]`

    By Fubini this equals `mean_B[ ∫_α |C_α| dα ]` with uniform α
    weighting on (0, 1). α-marginalised.

    `theta_grid` may be 1D `(N,)` (shared across the batch) or 2D
    `(B, N)` (per-sample grids; the training loop uses this).
    """
    if p_theta.ndim != 2:
        raise ValueError(f"p_theta must be (B, N); got shape {tuple(p_theta.shape)}")
    width_per_sample = jnp.trapezoid(p_theta, theta_grid, axis=-1)  # (B,)
    return _masked_mean(width_per_sample)


def cd_variance_loss(
    p_theta: jax.Array,
    theta_grid: jax.Array,
) -> jax.Array:
    """Variance of the Schweder-Hjort CD, averaged over the batch.

    `loss = mean_B[ Var_{F_D}[θ] ]`

    Builds the CD pdf via `cd_density_jax` (skips `signed_confidence`
    to stay differentiable), then computes `μ = ∫θ·pdf` and
    `Var = ∫(θ-μ)²·pdf`. α-free.

    `theta_grid` may be 1D `(N,)` or 2D `(B, N)`.
    """
    pdf = cd_density_jax(p_theta, theta_grid)  # (B, N)
    if theta_grid.ndim == 1:
        theta_b = jnp.broadcast_to(theta_grid, pdf.shape)
    else:
        theta_b = theta_grid
    mean_per_sample = jnp.trapezoid(pdf * theta_b, theta_grid, axis=-1)  # (B,)
    centred = theta_b - mean_per_sample[..., None]
    var_per_sample = jnp.trapezoid(pdf * centred * centred, theta_grid, axis=-1)
    return _masked_mean(var_per_sample)


def boundary_penalty_from_validity(
    validity_logits: jax.Array,
) -> jax.Array:
    """``-log P(valid | θ, η)`` averaged over the batch.

    Used as Head A's boundary penalty in the Phase E dual-head
    training loop. Caller passes raw logits from a parameter-
    detached ``ValidityNet`` (gradient flows back through the
    (θ, η) input into ``EtaNet`` but not into ``ValidityNet``'s
    weights — see ``_losses_compose.compose_boundary_penalty`` for
    the ``eqx.partition`` + ``stop_gradient`` pattern that
    replaces the legacy ``torch.func.functional_call``).

    No clamp on logits. ``jax.nn.log_sigmoid`` is numerically stable
    for any finite input (``-log_sigmoid(-100)`` evaluates to ≈100,
    no overflow), and its derivative is ``-sigmoid(-x) = -(1 -
    sigmoid(x))``, which saturates to ``-1`` (not zero) as logits go
    very negative — so the boundary signal stays alive on the wrong
    side of an overconfident classifier with bounded gradient. A
    ``jnp.clip`` would kill that gradient at the clamp boundary,
    defeating the purpose.

    Penalty value bounds:
    - logit → +∞: penalty → 0 (no gradient — η is well inside
      the valid region; nothing to push back).
    - logit → -∞: penalty → |logit| (linear), gradient → -1
      (constant pressure pushing η back toward the valid region).

    Parameters
    ----------
    validity_logits
        ``(N,)`` array of raw logits from ``ValidityNet``.

    Returns
    -------
    Scalar penalty.
    """
    if validity_logits.ndim != 1:
        raise ValueError(
            f"boundary_penalty_from_validity expects (N,) input; "
            f"got shape {tuple(validity_logits.shape)}."
        )
    return -jax.nn.log_sigmoid(validity_logits).mean()


def anti_wald_penalty(eta_pred: jax.Array) -> jax.Array:
    """Mean ``relu(η_pred)²`` — punishes positive η.

    Phase G diagnostic regularizer. The conditional EtaNet has a
    persistent failure mode: it converges to η ≈ +1 across the
    trained hyperparam range, which (for power_law on Normal-Normal)
    corresponds to *removing the prior entirely* — i.e. computing
    Wald CIs. The Wald solution is a wide stable plateau in the loss
    surface and SGD with smooth losses never escapes it.

    This penalty asymmetrically pushes η_pred toward the negative
    half-line where the optimal solutions live (oversharpened toward
    the prior). Combined with a decay schedule, it perturbs the
    optimizer out of the Wald basin during early training, then
    releases its bias so the underlying width loss owns convergence.

    See ``losses.eta_collapse_penalty`` for the value-agnostic
    variant that rewards η_pred variance across the batch instead.
    """
    if eta_pred.ndim != 1:
        raise ValueError(
            f"anti_wald_penalty expects (N,) eta_pred; got shape "
            f"{tuple(eta_pred.shape)}."
        )
    return (jax.nn.relu(eta_pred) ** 2).mean()


def eta_collapse_penalty(
    eta_pred: jax.Array,
    eps: float = 1e-3,
) -> jax.Array:
    """``1 / (Var_B[η_pred] + ε)`` — rewards η_pred spread across the batch.

    Phase G diagnostic regularizer. Value-agnostic counterpart to
    ``anti_wald_penalty``: punishes the model for predicting the
    same η across all (θ, prior_hp, lik_hp) batch elements regardless
    of *what* value it picks. Targets the 'collapsed to a constant'
    failure mode directly (η ≈ +1 *and* η ≈ +0 *and* any other
    constant land in this penalty's crosshairs).

    The ε floor (default 1e-3) caps the maximum penalty at ``1/ε``
    so an early all-zero-gradient batch doesn't dominate the loss.
    Decays alongside ``anti_wald_penalty`` via the same schedule
    so the optimizer can ultimately settle anywhere it likes.
    """
    if eta_pred.ndim != 1:
        raise ValueError(
            f"eta_collapse_penalty expects (N,) eta_pred; got shape "
            f"{tuple(eta_pred.shape)}."
        )
    var = jnp.var(eta_pred)
    return 1.0 / (var + eps)


def static_width_loss(
    p_theta: jax.Array,
    theta_grid: jax.Array,
    alpha: float,
    sharpness: float = 200.0,
) -> jax.Array:
    """α-specific static CI width, averaged over the batch.

    Uses a sigmoid-relaxed indicator to keep the width differentiable:

        |C_α| ≈ ∫_θ σ_β( p_dyn(θ) − α ) dθ,    β = `sharpness`

    where `σ_β(x) = 1 / (1 + exp(-β x))`. As `β → ∞` the relaxation
    converges to the true `1{p ≥ α}` indicator.

    Choosing `sharpness`. Empirically (deriver #4 in Phase C review):
    on a Wald p-curve over a wide θ-grid, the relative bias of the
    relaxed integral vs the true `|C_α|` is:
       β=50,  α=0.05 → +110 % bias  (relaxed integral catches
                                      tail mass where `σ_β(p-α) ≈ e^{-βα}`,
                                      heavily inflating the integrand
                                      far from the mode)
       β=200, α=0.05 → +0.4 % bias
       β=500, α=0.05 → +0.1 % bias
    So `β = 50` is **not** sharp enough at typical α. Default raised to
    `β = 200`. For very small α (≤ 0.01) prefer β ≥ 500.

    Used in α-conditioned training mode only. The trained MLP is
    valid only at the α it was trained for; the selector verifies
    this at load time. `theta_grid` may be 1D `(N,)` or 2D `(B, N)`.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    indicator = jax.nn.sigmoid(sharpness * (p_theta - alpha))  # (B, N)
    width_per_sample = jnp.trapezoid(indicator, theta_grid, axis=-1)
    return _masked_mean(width_per_sample)
