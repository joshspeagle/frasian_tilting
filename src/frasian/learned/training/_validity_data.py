"""Validity-data helpers for the Phase E dual-head training loop.

Extracted from ``train.py`` (Tier 1.2 §7 split). Three concerns:

1. Per-θ data sampling (``sample_data_per_theta``) with optional
   antithetic pairing (``2θ - D``) for variance reduction on
   Normal-Normal-symmetric loss components.
2. Building the (θ, η, valid) batch that feeds Head B's BCE step
   (``collect_validity_batch``).
3. The torch-tensor packaging helper ``validity_net_inputs``.

These helpers do not import torch except where the caller already has
torch tensors in hand (they accept tensor inputs and return tensors).
The numpy paths stay torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .sampling import ExperimentConfig
from .validity import compute_pvalues_per_sample, validity_mask

if TYPE_CHECKING:
    import torch


def antithetic_pair(theta: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Return the antithetic partner ``2θ − D`` of a Normal-Normal D draw.

    For ``D = θ + δ`` (likelihood is ``N(θ, σ²)``), the antithetic
    partner is ``D' = 2θ − D = θ − δ`` — exactly anti-correlated by
    the Normal-Normal symmetry, halving MC variance on even loss
    components.

    Parameters
    ----------
    theta
        ``(N,)`` array of θ values used to draw ``D``.
    D
        ``(N,)`` array of likelihood draws.

    Returns
    -------
    ``(N,)`` array of antithetic partners.
    """
    if theta.shape != D.shape:
        raise ValueError(
            f"antithetic_pair expects matching shapes; got θ={theta.shape}, D={D.shape}"
        )
    return 2.0 * theta - D


def sample_data_per_theta(
    model: Any,
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    antithetic: bool = False,
) -> np.ndarray:
    """For each θ, draw one ``D ~ likelihood(·|θ)``.

    Parameters
    ----------
    model
        Model with a ``sample_data(theta_scalar, rng, n=1)`` API.
    theta
        ``(N,)`` array of θ values.
    rng
        Per-consumer ``numpy.random.Generator``.
    antithetic
        If True, return a ``(2N,)`` array ``[D, 2θ − D]`` interleaved
        as ``[D₁, ..., D_N, 2θ₁ − D₁, ..., 2θ_N − D_N]`` — primary
        first, antithetic partner second. The caller is responsible
        for aggregating losses over the paired structure.

    Returns
    -------
    Array of shape ``(N,)`` (default) or ``(2N,)`` (antithetic).
    """
    out = np.empty(theta.shape, dtype=np.float64)
    for i, th in enumerate(theta):
        out[i] = float(model.sample_data(float(th), rng, n=1)[0])
    if not antithetic:
        return out
    paired = antithetic_pair(theta, out)
    return np.concatenate([out, paired])


def collect_validity_batch(
    *,
    eta_pred: torch.Tensor,
    theta_batch_np: np.ndarray,
    config: ExperimentConfig,
    scheme: Any,
    n_aux: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (theta_all, eta_all, valid_all) for Head B's BCE step.

    Mixes the main batch (with detached η_pred) and an auxiliary
    boundary-probing batch (i.i.d. θ from theta_distribution, η
    uniform on eta_explore_box). Returns torch tensors on the same
    device as ``eta_pred``.
    """
    import torch

    device = eta_pred.device
    dtype = eta_pred.dtype

    # Main: detach η_pred for label collection (gradient must not flow
    # through discrete labels).
    eta_main_np = eta_pred.detach().cpu().numpy().astype(np.float64)
    D_main_np = sample_data_per_theta(config.model, theta_batch_np, rng)

    # Aux: independent draws from the θ distribution + uniform η in box.
    theta_aux_np = config.theta_distribution.sample(n_aux, rng)
    eta_aux_np = rng.uniform(*config.eta_explore_box, size=n_aux).astype(np.float64)
    D_aux_np = sample_data_per_theta(config.model, theta_aux_np, rng)

    theta_all_np = np.concatenate([theta_batch_np, theta_aux_np])
    eta_all_np = np.concatenate([eta_main_np, eta_aux_np])
    D_all_np = np.concatenate([D_main_np, D_aux_np])

    p_all = compute_pvalues_per_sample(
        scheme,
        theta_all_np,
        D_all_np,
        config.model,
        config.prior,
        eta_all_np,
        config.statistic_name,
    )
    valid_all = validity_mask(p_all)  # bool (N+N_aux,)

    theta_all_t = torch.as_tensor(theta_all_np, dtype=dtype, device=device)
    eta_all_t = torch.as_tensor(eta_all_np, dtype=dtype, device=device)
    valid_all_t = torch.as_tensor(valid_all.astype(np.float32), device=device)
    return theta_all_t, eta_all_t, valid_all_t


def validity_net_inputs(theta: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Pack ``(θ, η)`` into ValidityNet's expected ``(N, theta_dim+1)``."""
    import torch

    if theta.dim() == 1:
        theta = theta.unsqueeze(-1)  # (N, 1)
    if eta.dim() == 1:
        eta = eta.unsqueeze(-1)  # (N, 1)
    return torch.cat([theta, eta], dim=-1)


def prepare_held_out_validity(
    *,
    scheme: Any,
    theta_held: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the frozen (η_held_aux, D_held, valid_held) triple.

    Used as the held-out set for Head B's accuracy diagnostic. Sampled
    once at training start and fixed across epochs.

    Returns
    -------
    Tuple ``(eta_held_aux, D_held, valid_held)`` of ``(M,)`` arrays.
    """
    eta_held_aux = rng.uniform(*config.eta_explore_box, size=len(theta_held)).astype(np.float64)
    D_held = sample_data_per_theta(config.model, theta_held, rng)
    p_held = compute_pvalues_per_sample(
        scheme,
        theta_held,
        D_held,
        config.model,
        config.prior,
        eta_held_aux,
        config.statistic_name,
    )
    valid_held = validity_mask(p_held)
    return eta_held_aux, D_held, valid_held
