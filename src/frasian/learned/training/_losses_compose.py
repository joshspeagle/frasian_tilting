"""Loss composition for the Phase E dual-head training loop.

Extracted from ``train.py`` (Tier 1.2 §7 split). Three concerns:

1. Width-loss adapter (``compose_width_loss``): wraps the
   torch tilted-pvalue port + the chosen functional in
   ``losses.py`` (``integrated_pvalue_loss`` / ``cd_variance_loss`` /
   ``static_width_loss``). Currently Normal-Normal only because
   the torch ports are; the wrapper is the single place that
   adapts ``(model, prior)`` → ``(w, mu0, sigma)``.

2. Boundary penalty (``compose_boundary_penalty``): forwards
   ``(θ, η)`` through a parameter-detached ``ValidityNet`` so
   gradient flows back into Head A but not Head B's weights.

3. λ schedule (``lambda_schedule``) and β schedule
   (``beta_schedule``) for the Head A loss components.

The β schedule is the Phase E annealing (1.4-S2 / 1.2-NN2):
``β(epoch) = 50 + 450 · min(1, epoch / (0.5·n_epochs))`` —
smooth gradient early, low bias late. Default endpoints
``50 → 500`` (the audit's recommended range).
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from ...models.distributions import NormalDistribution
from ...models.normal_normal import NormalNormalModel
from .architecture import EtaNet, ValidityNet
from .losses import (
    boundary_penalty_from_validity,
    cd_variance_loss,
    integrated_pvalue_loss,
    static_width_loss,
)
from .pvalue_torch import get_torch_tilted_pvalue
from .sampling import ExperimentConfig

# Default β-anneal endpoints. Per `audit/tier1/skeptic_learned_eta.md` S2
# and `audit/tier1/nn_training.md` §1, β=200 → +0.4% bias at α=0.05;
# β=500 reduces this to ~0.1%. Starting at β=50 gives smooth gradient
# early; the schedule ramps over the first half of training.
BETA_MIN: float = 50.0
BETA_MAX: float = 500.0
BETA_WARMUP_FRAC: float = 0.5


def extract_normal_normal_params(
    model: Any,
    prior: Any,
) -> tuple[float, float, float]:
    """Extract (w, mu0, sigma) for the existing torch tilted-pvalue ports.

    The torch ports in ``pvalue_torch.py`` were written before Phase E
    and take Normal-Normal coordinates ``(w, mu0, sigma)`` directly.
    We adapt by deriving them from ``(model, prior)`` exactly here —
    the rest of the training loop stays model-agnostic.

    Raises ``NotImplementedError`` for non-Normal-Normal experiments,
    consistent with the documented "model-agnostic in principle,
    Normal-Normal in practice" caveat.
    """
    if not isinstance(model, NormalNormalModel):
        raise NotImplementedError(
            "Phase E training currently requires a NormalNormalModel; "
            f"got {type(model).__name__}. Extending to non-Normal-Normal "
            f"requires registering a generic torch tilted_pvalue."
        )
    if not isinstance(prior, NormalDistribution):
        raise NotImplementedError(
            "Phase E training currently requires a NormalDistribution prior; "
            f"got {type(prior).__name__}."
        )
    sigma = float(model.sigma)
    sigma0 = float(prior.scale)
    mu0 = float(prior.loc)
    w = sigma0**2 / (sigma**2 + sigma0**2)
    # Skeptic block #4: w → 0 (delta prior) and w → 1 (improper) put
    # the WALDO admissible range into degenerate territory and the
    # torch-side `clamp(denom, min=1e-6)` silently distorts. Refuse.
    _W_EPS = 1e-3
    if not (_W_EPS < w < 1.0 - _W_EPS):
        raise ValueError(
            f"data weight w={w:.6f} is outside ({_W_EPS}, "
            f"{1.0 - _W_EPS}); prior.scale={sigma0} and model.sigma="
            f"{sigma} are too far apart for a stable learned-η "
            f"selector. Choose a less degenerate prior."
        )
    return w, mu0, sigma


def compose_width_loss(
    *,
    eta_net: EtaNet,
    theta_grid_t: torch.Tensor,
    D_batch_t: torch.Tensor,
    config: ExperimentConfig,
    loss_kind: str,
    alpha: float | None,
    beta: float = 200.0,
) -> torch.Tensor:
    """Integrated-p (or variant) width loss averaged over a D batch.

    Skeptic block #1: a single-D Monte Carlo estimator has too much
    variance for both training and validation signal — val_width
    oscillates with the per-step D draw. Vectorise over a (B,) D
    tensor and average; gives an unbiased estimator with variance
    ~1/B. The torch tilted-pvalue port broadcasts naturally so this
    is one tensor call, not a Python loop.

    The ``beta`` argument is forwarded to ``static_width_loss`` only
    (the other two losses are α-marginalised and don't use a
    sigmoid-relaxed indicator).
    """
    w, mu0, sigma = extract_normal_normal_params(config.model, config.prior)
    w_t = torch.tensor(w, dtype=theta_grid_t.dtype, device=theta_grid_t.device)
    mu0_t = torch.tensor(mu0, dtype=theta_grid_t.dtype, device=theta_grid_t.device)
    sigma_t = torch.tensor(sigma, dtype=theta_grid_t.dtype, device=theta_grid_t.device)

    if D_batch_t.dim() == 0:
        D_batch_t = D_batch_t.unsqueeze(0)  # (1,)
    if D_batch_t.dim() != 1:
        raise ValueError(f"D_batch_t must be 0D or 1D; got shape {tuple(D_batch_t.shape)}.")

    eta_grid = eta_net(theta_grid_t)  # (G,)
    tilted_pvalue = get_torch_tilted_pvalue(config.scheme_name)
    # Broadcast: theta=(1, G), D=(B, 1), eta=(1, G) → p=(B, G).
    G = theta_grid_t.shape[0]
    B = D_batch_t.shape[0]
    p_grid = tilted_pvalue(
        theta_grid_t.unsqueeze(0).expand(B, G),  # (B, G)
        D_batch_t.unsqueeze(-1).expand(B, G),  # (B, G)
        w_t,
        mu0_t,
        sigma_t,
        eta_grid.unsqueeze(0).expand(B, G),  # (B, G)
        config.statistic_name,
    )  # (B, G)
    # Skeptic caveat #12: float32 round-off in `_phi(b-a) + _phi(-a-b)`
    # can drift slightly outside [0, 1]. Warn (without breaking the
    # gradient) when drift exceeds tolerance — clamp would zero the
    # gradient at the boundary. Mirrors the legacy guard.
    if not torch.is_grad_enabled():
        out_of_range = ((p_grid < -1e-5) | (p_grid > 1.0 + 1e-5)).any().item()
        if out_of_range:
            warnings.warn(
                f"width-loss p-values drifted outside [0, 1] in "
                f"{config.scheme_name}/{config.statistic_name}; "
                f"consider float64 or tighter η bounds.",
                RuntimeWarning,
                stacklevel=2,
            )
    theta_grid_b = theta_grid_t.unsqueeze(0).expand(B, G)  # (B, G)

    if loss_kind == "integrated_p":
        return integrated_pvalue_loss(p_grid, theta_grid_b)
    if loss_kind == "cd_variance":
        return cd_variance_loss(p_grid, theta_grid_b)
    if loss_kind == "static_width":
        if alpha is None:
            raise ValueError("static_width loss requires alpha not None")
        return static_width_loss(p_grid, theta_grid_b, alpha=alpha, sharpness=beta)
    raise ValueError(f"Unknown loss_kind={loss_kind!r}")


def compose_boundary_penalty(
    *,
    val_net: ValidityNet,
    theta_batch_t: torch.Tensor,
    eta_pred: torch.Tensor,
) -> torch.Tensor:
    """``-log P(valid | θ_batch, η_pred)`` with ValidityNet detached.

    Uses ``torch.func.functional_call`` with detached params + buffers
    so gradient flows from η_pred through the (θ, η) input into
    EtaNet, but not into ValidityNet.
    """
    from ._validity_data import validity_net_inputs

    inputs = validity_net_inputs(theta_batch_t, eta_pred)
    v_p = {k: v.detach() for k, v in val_net.named_parameters()}
    v_b = {k: v.detach() for k, v in val_net.named_buffers()}
    logits = torch.func.functional_call(val_net, (v_p, v_b), (inputs,))
    return boundary_penalty_from_validity(logits)


def lambda_schedule(
    epoch: int,
    n_epochs: int,
    lambda_max: float,
    warmup_frac: float,
) -> float:
    """Linear ramp from 0 to ``lambda_max`` over the first
    ``warmup_frac`` of epochs, then constant.

    Special case: ``warmup_frac == 0`` returns ``lambda_max``
    immediately (no warmup); the user explicitly opted out.

    Otherwise: λ(0) = 0 (Head B trains in isolation for epoch 0;
    Head A's width loss has no boundary signal yet);
    λ(warmup_epochs) = λ_max; λ(epoch ≥ warmup_epochs) = λ_max.
    """
    if warmup_frac <= 0.0:
        return float(lambda_max)
    warmup_epochs = max(1, int(round(warmup_frac * n_epochs)))
    if epoch >= warmup_epochs:
        return float(lambda_max)
    return float(lambda_max) * epoch / warmup_epochs


def beta_schedule(
    epoch: int,
    n_epochs: int,
    *,
    beta_min: float = BETA_MIN,
    beta_max: float = BETA_MAX,
    warmup_frac: float = BETA_WARMUP_FRAC,
) -> float:
    """Linear anneal of the ``static_width_loss`` sharpness β.

    β(epoch) = beta_min + (beta_max − beta_min) · min(1, epoch / (warmup_frac · n_epochs))

    Defaults: ``50 → 500`` over the first 50 % of epochs, then
    constant. Used only when ``loss_kind == "static_width"``; the
    α-marginalised losses ignore β. Per `audit/tier1/skeptic_learned_eta.md`
    S2: β=200 → +0.4 % bias at α=0.05; β=500 → +0.1 %.

    Special case: ``warmup_frac <= 0`` returns ``beta_max`` (no anneal).
    """
    if warmup_frac <= 0.0:
        return float(beta_max)
    if n_epochs <= 0:
        return float(beta_max)
    span = max(1.0, warmup_frac * float(n_epochs))
    frac = min(1.0, float(epoch) / span)
    return float(beta_min) + (float(beta_max) - float(beta_min)) * frac
