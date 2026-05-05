"""Torch port of `frasian.cd.from_pvalue` — Schweder–Hjort CD pdf only.

Reference numpy implementation: `src/frasian/cd/from_pvalue.py:142-165`.

The full numpy `build_cd_from_pvalue` includes a `signed_confidence`
curve based on `np.argmax(pvalues)` (line 82). `argmax` is non-
differentiable — we **skip `signed_confidence` during training** and
keep only the `pdf` path.

Density formula (averaged-one-sided-difference, kink-robust):

    forward[i]  = |p[i+1] - p[i]| / (θ[i+1] - θ[i])     (i < N-1)
    backward[i] = forward[i-1]                          (i > 0)
    abs_dp = 0.5 (forward + backward)
    pdf_unnorm = 0.5 · abs_dp
    pdf = pdf_unnorm / Z,  Z = ∫ pdf_unnorm dθ

The averaged-one-sided-diff is what makes the density correct at
kink points (e.g. θ = D for Wald, θ = μ_n for WALDO) where central
differences cancel.
"""

from __future__ import annotations

import torch


def cd_density_torch(
    p_theta: torch.Tensor,
    theta_grid: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Schweder-Hjort CD pdf from a p-value curve.

    Accepts either a shared 1D `theta_grid` (`(N,)`, used for all batch
    rows) or a per-sample 2D grid (`(B, N)`, one row per batch element
    — the training pipeline uses this since each sample has its own
    `D ± search_mult·σ` window).

    Parameters
    ----------
    p_theta : (B, N) tensor
        p-value evaluated on `theta_grid` for each batch element.
    theta_grid : (N,) or (B, N) tensor
        Strictly-increasing θ grid. If 1D, broadcast to all rows; if
        2D, must match `p_theta.shape`.
    eps : float
        Floor for the normalisation constant `Z` to avoid div-by-zero
        when p is constant on the grid.

    Returns
    -------
    pdf : (B, N) tensor
        Normalised pdf integrating to 1 along `theta_grid` row-wise.
    """
    if p_theta.dim() != 2:
        raise ValueError(
            f"p_theta must be (B, N); got shape {tuple(p_theta.shape)}"
        )
    if theta_grid.dim() == 1:
        if theta_grid.shape[0] != p_theta.shape[1]:
            raise ValueError(
                f"theta_grid (1D) must have N={p_theta.shape[1]} elements; "
                f"got shape {tuple(theta_grid.shape)}"
            )
        # Forward differences along the shared grid.
        dtheta = (theta_grid[1:] - theta_grid[:-1]).unsqueeze(0)  # (1, N-1)
    elif theta_grid.dim() == 2:
        if theta_grid.shape != p_theta.shape:
            raise ValueError(
                f"theta_grid (2D) must match p_theta shape "
                f"{tuple(p_theta.shape)}; got {tuple(theta_grid.shape)}"
            )
        dtheta = theta_grid[..., 1:] - theta_grid[..., :-1]      # (B, N-1)
    else:
        raise ValueError(
            f"theta_grid must be 1D or 2D; got shape "
            f"{tuple(theta_grid.shape)}"
        )

    dp = torch.abs(p_theta[..., 1:] - p_theta[..., :-1])         # (B, N-1)
    forward_inner = dp / dtheta                                   # (B, N-1)
    forward = torch.cat([forward_inner, forward_inner[..., -1:]], dim=-1)
    backward = torch.cat([forward_inner[..., 0:1], forward_inner], dim=-1)
    abs_dp_dtheta = 0.5 * (forward + backward)
    pdf_unnorm = 0.5 * abs_dp_dtheta                              # (B, N)

    Z = torch.trapezoid(pdf_unnorm, theta_grid, dim=-1)           # (B,)
    Z = torch.clamp(Z, min=eps)
    return pdf_unnorm / Z.unsqueeze(-1)
