"""Differentiable training losses for the learned dynamic-η selector.

All losses operate on a `(B, N)` p-value tensor (one row per batch
element, one column per θ-grid point) and a `(N,)` θ-grid. They
return a scalar — the mean batch loss — that is end-to-end
differentiable in the η_φ parameters that produced `p_theta`.

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
Numerical safety: the p-value lives in `[0, 1]` exactly (the torch
tilted_pvalue formulas guarantee this), and the trapezoidal rule
`torch.trapezoid` is differentiable.
"""

from __future__ import annotations

import torch

from .cd_torch import cd_density_torch


def integrated_pvalue_loss(
    p_theta: torch.Tensor,
    theta_grid: torch.Tensor,
) -> torch.Tensor:
    """L¹ norm of the dynamic p-value curve, averaged over the batch.

    `loss = mean_B[ ∫_θ p_dyn(θ; D, η_φ) dθ ]`

    By Fubini this equals `mean_B[ ∫_α |C_α| dα ]` with uniform α
    weighting on (0, 1). α-marginalised.
    """
    if p_theta.dim() != 2:
        raise ValueError(
            f"p_theta must be (B, N); got shape {tuple(p_theta.shape)}"
        )
    width_per_sample = torch.trapezoid(p_theta, theta_grid, dim=-1)  # (B,)
    return width_per_sample.mean()


def cd_variance_loss(
    p_theta: torch.Tensor,
    theta_grid: torch.Tensor,
) -> torch.Tensor:
    """Variance of the Schweder-Hjort CD, averaged over the batch.

    `loss = mean_B[ Var_{F_D}[θ] ]`

    Builds the CD pdf via `cd_density_torch` (skips `signed_confidence`
    to stay differentiable), then computes `μ = ∫θ·pdf` and
    `Var = ∫(θ-μ)²·pdf`. α-free.
    """
    pdf = cd_density_torch(p_theta, theta_grid)             # (B, N)
    mean_per_sample = torch.trapezoid(
        pdf * theta_grid.unsqueeze(0), theta_grid, dim=-1,
    )                                                        # (B,)
    centred = theta_grid.unsqueeze(0) - mean_per_sample.unsqueeze(-1)
    var_per_sample = torch.trapezoid(
        pdf * centred * centred, theta_grid, dim=-1,
    )                                                        # (B,)
    return var_per_sample.mean()


def static_width_loss(
    p_theta: torch.Tensor,
    theta_grid: torch.Tensor,
    alpha: float,
    sharpness: float = 50.0,
) -> torch.Tensor:
    """α-specific static CI width, averaged over the batch.

    Uses a sigmoid-relaxed indicator to keep the width differentiable:

        |C_α| ≈ ∫_θ σ_β( p_dyn(θ) − α ) dθ,    β = `sharpness`

    where `σ_β(x) = 1 / (1 + exp(-β x))`. As `β → ∞` the relaxation
    converges to the true `1{p ≥ α}` indicator. Default `β = 50`
    keeps the gradient finite while staying close to the indicator
    on the relevant tails.

    Used in α-conditioned training mode only. The trained MLP is
    valid only at the α it was trained for; the selector verifies
    this at load time.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    indicator = torch.sigmoid(sharpness * (p_theta - alpha))   # (B, N)
    width_per_sample = torch.trapezoid(indicator, theta_grid, dim=-1)
    return width_per_sample.mean()
