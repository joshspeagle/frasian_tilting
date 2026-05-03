"""Differentiable training losses for the learned dynamic-η selector.

All losses operate on a `(B, N)` p-value tensor (one row per batch
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

**Reactive NaN guard.** When the torch p-value formula drifts numerically
(possible at extreme `(w, eta)` combinations as we extend support to
non-Normal-Normal models), individual samples can yield NaN/Inf
trapezoidal-integral values. Each loss now masks those samples out of
the batch mean rather than letting one NaN contaminate the gradient.
If the entire batch is invalid, the loss raises `RuntimeError`.
"""

from __future__ import annotations

import torch

from .cd_torch import cd_density_torch


def _masked_mean(per_sample: torch.Tensor) -> torch.Tensor:
    """Mean over `per_sample` rows that are finite. Raises on all-bad batch."""
    valid = torch.isfinite(per_sample)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        raise RuntimeError(
            "loss: every sample in the batch produced a non-finite "
            "per-sample loss (NaN/Inf). Likely cause: η_φ ventured "
            "outside the admissible range; check the per-scheme "
            "transform / admissible_range registry."
        )
    if n_valid < per_sample.numel():
        # Replace NaN/Inf with 0 so gradients are well-defined; mask
        # via division by n_valid below.
        per_sample = torch.where(
            valid, per_sample, torch.zeros_like(per_sample)
        )
    return per_sample.sum() / n_valid


def integrated_pvalue_loss(
    p_theta: torch.Tensor,
    theta_grid: torch.Tensor,
) -> torch.Tensor:
    """L¹ norm of the dynamic p-value curve, averaged over the batch.

    `loss = mean_B[ ∫_θ p_dyn(θ; D, η_φ) dθ ]`

    By Fubini this equals `mean_B[ ∫_α |C_α| dα ]` with uniform α
    weighting on (0, 1). α-marginalised.

    `theta_grid` may be 1D `(N,)` (shared across the batch) or 2D
    `(B, N)` (per-sample grids; the training loop uses this).
    """
    if p_theta.dim() != 2:
        raise ValueError(
            f"p_theta must be (B, N); got shape {tuple(p_theta.shape)}"
        )
    width_per_sample = torch.trapezoid(p_theta, theta_grid, dim=-1)  # (B,)
    return _masked_mean(width_per_sample)


def cd_variance_loss(
    p_theta: torch.Tensor,
    theta_grid: torch.Tensor,
) -> torch.Tensor:
    """Variance of the Schweder-Hjort CD, averaged over the batch.

    `loss = mean_B[ Var_{F_D}[θ] ]`

    Builds the CD pdf via `cd_density_torch` (skips `signed_confidence`
    to stay differentiable), then computes `μ = ∫θ·pdf` and
    `Var = ∫(θ-μ)²·pdf`. α-free.

    `theta_grid` may be 1D `(N,)` or 2D `(B, N)`.
    """
    pdf = cd_density_torch(p_theta, theta_grid)              # (B, N)
    if theta_grid.dim() == 1:
        theta_b = theta_grid.unsqueeze(0).expand_as(pdf)
    else:
        theta_b = theta_grid
    mean_per_sample = torch.trapezoid(pdf * theta_b, theta_grid, dim=-1)  # (B,)
    centred = theta_b - mean_per_sample.unsqueeze(-1)
    var_per_sample = torch.trapezoid(pdf * centred * centred,
                                       theta_grid, dim=-1)
    return _masked_mean(var_per_sample)


def static_width_loss(
    p_theta: torch.Tensor,
    theta_grid: torch.Tensor,
    alpha: float,
    sharpness: float = 200.0,
) -> torch.Tensor:
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
    indicator = torch.sigmoid(sharpness * (p_theta - alpha))   # (B, N)
    width_per_sample = torch.trapezoid(indicator, theta_grid, dim=-1)
    return _masked_mean(width_per_sample)
