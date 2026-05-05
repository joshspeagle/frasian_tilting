"""Phase E dual-head training loop for the learned-η selector.

Three-step training step (per minibatch of θ from the LHS pool):

1. Forward Head A (``EtaNet``) on the main batch. Sample auxiliary
   ``(θ_aux, η_aux)`` boundary-probing points so Head B sees both
   classes. Sample data ``D ~ likelihood(·|θ)`` for both batches.
   Run ``scheme.tilted_pvalue`` (numpy) per sample to label
   ``valid_all = is_pair_valid(p_all)``. Detach ``η_pred`` when
   feeding into the labelling step so Head A's gradient does not
   flow through the discrete labels.

2. Train Head B (``ValidityNet``) — BCE on the (θ, η, valid) triples
   accumulated this step. Aux samples ensure both classes are
   represented at every step.

3. Train Head A (``EtaNet``) — width loss + λ(epoch) · boundary
   penalty. The width loss is the integrated p-value across the
   canonical ``config.theta_grid`` for one drawn ``D`` (a marginal
   data sample); η = ``EtaNet(theta_grid_t)``. The boundary penalty
   is ``-log P(valid | θ_batch, η_pred)`` with ValidityNet's
   parameters detached via ``torch.func.functional_call`` — gradient
   flows through the (θ, η) input back into EtaNet.

The training loop never references Normal-Normal coordinates
(``w``, ``mu0``, ``sigma``, ``|Δ|``) or scheme-specific η
transforms. The only scheme-specific surface is the torch port in
``pvalue_torch.py`` (whose existing signature is Normal-Normal
specific; we adapt by extracting ``(w, mu0, sigma)`` from
``(model, prior)`` exactly once, inside a tiny helper, when calling
the torch port for the width loss).
"""

from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..._registry import registry as _registry
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
from .sampling import ExperimentConfig, lhs_1d
from .validity import compute_pvalues_per_sample, validity_mask

_LOSS_FNS = {
    "integrated_p": integrated_pvalue_loss,
    "cd_variance": cd_variance_loss,
}


# ---------------------------------------------------------------------------
# Width-loss adapter (model-aware, scheme-generic)
# ---------------------------------------------------------------------------


def _extract_normal_normal_params(
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


def _width_loss(
    *,
    eta_net: EtaNet,
    theta_grid_t: torch.Tensor,
    D_batch_t: torch.Tensor,
    config: ExperimentConfig,
    loss_kind: str,
    alpha: float | None,
) -> torch.Tensor:
    """Integrated-p (or variant) width loss averaged over a D batch.

    Skeptic block #1: a single-D Monte Carlo estimator has too much
    variance for both training and validation signal — val_width
    oscillates with the per-step D draw. Vectorise over a (B,) D
    tensor and average; gives an unbiased estimator with variance
    ~1/B. The torch tilted-pvalue port broadcasts naturally so this
    is one tensor call, not a Python loop.
    """
    w, mu0, sigma = _extract_normal_normal_params(config.model, config.prior)
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
        return static_width_loss(p_grid, theta_grid_b, alpha=alpha)
    raise ValueError(f"Unknown loss_kind={loss_kind!r}")


# ---------------------------------------------------------------------------
# Validity labelling (numpy) + Head B forward pass (torch)
# ---------------------------------------------------------------------------


def _sample_data_per_theta(
    model: Any,
    theta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """For each θ, draw one ``D ~ likelihood(·|θ)``."""
    out = np.empty(theta.shape, dtype=np.float64)
    for i, th in enumerate(theta):
        out[i] = float(model.sample_data(float(th), rng, n=1)[0])
    return out


def _collect_validity_batch(
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
    device = eta_pred.device
    dtype = eta_pred.dtype

    # Main: detach η_pred for label collection (gradient must not flow
    # through discrete labels).
    eta_main_np = eta_pred.detach().cpu().numpy().astype(np.float64)
    D_main_np = _sample_data_per_theta(config.model, theta_batch_np, rng)

    # Aux: independent draws from the θ distribution + uniform η in box.
    theta_aux_np = config.theta_distribution.sample(n_aux, rng)
    eta_aux_np = rng.uniform(*config.eta_explore_box, size=n_aux).astype(np.float64)
    D_aux_np = _sample_data_per_theta(config.model, theta_aux_np, rng)

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


def _validity_net_inputs(theta: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Pack ``(θ, η)`` into ValidityNet's expected ``(N, theta_dim+1)``."""
    if theta.dim() == 1:
        theta = theta.unsqueeze(-1)  # (N, 1)
    if eta.dim() == 1:
        eta = eta.unsqueeze(-1)  # (N, 1)
    return torch.cat([theta, eta], dim=-1)


def _boundary_penalty(
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
    inputs = _validity_net_inputs(theta_batch_t, eta_pred)
    v_p = {k: v.detach() for k, v in val_net.named_parameters()}
    v_b = {k: v.detach() for k, v in val_net.named_buffers()}
    logits = torch.func.functional_call(val_net, (v_p, v_b), (inputs,))
    return boundary_penalty_from_validity(logits)


# ---------------------------------------------------------------------------
# λ schedule
# ---------------------------------------------------------------------------


def _lambda_schedule(
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


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


@dataclass
class EtaTrainResult:
    artifact_path: Path
    train_losses: list[float]  # mean Head A total loss per epoch
    train_width_losses: list[float]  # mean width component per epoch
    train_penalty_losses: list[float]  # mean (unweighted) penalty per epoch
    val_losses: list[float]  # held-out width loss per epoch
    head_b_accuracy: list[float]  # held-out validity accuracy per epoch
    final_val_loss: float
    metadata: dict[str, Any]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _evaluate_head_b_accuracy(
    val_net: ValidityNet,
    theta_held: np.ndarray,
    eta_held: np.ndarray,
    valid_held: np.ndarray,
    device: str,
) -> float:
    """Held-out classification accuracy of Head B at threshold 0.5."""
    val_net.eval()
    with torch.no_grad():
        theta_t = torch.as_tensor(theta_held, dtype=torch.float32, device=device)
        eta_t = torch.as_tensor(eta_held, dtype=torch.float32, device=device)
        inputs = _validity_net_inputs(theta_t, eta_t)
        logits = val_net(inputs)
        pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
    return float((pred == valid_held).mean())


def fit_eta_artifact(
    *,
    config: ExperimentConfig,
    out_path: Path,
    loss_kind: str = "integrated_p",
    alpha: float | None = None,
    n_epochs: int = 30,
    batch_size: int = 256,
    n_aux: int | None = None,
    lr_a: float = 1e-3,
    lr_b: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_max: float = 10.0,
    lambda_warmup_frac: float = 0.3,
    patience: int = 8,
    min_delta: float = 1e-4,
    eta_hidden_sizes: tuple[int, ...] = (64, 64),
    validity_hidden_sizes: tuple[int, ...] = (64, 64),
    device: str = "auto",
    version: str = "v0",
    verbose: bool = True,
) -> EtaTrainResult:
    """Train an EtaNet + ValidityNet pair end-to-end on ``config``.

    Model-agnostic interface (drives off ``config``). The width-loss
    side currently requires a NormalNormalModel + NormalDistribution
    prior because the torch tilted_pvalue ports are Normal-Normal
    only — the training loop itself doesn't reference Normal-Normal
    coordinates anywhere else.

    Writes a checkpoint at ``out_path`` recording both nets' state,
    the experiment config (with fingerprints), the λ schedule, and a
    final calibration summary. Returns a ``EtaTrainResult``.
    """
    if loss_kind not in _LOSS_FNS and loss_kind != "static_width":
        raise ValueError(
            "loss_kind must be one of {integrated_p, cd_variance, "
            f"static_width}}; got {loss_kind!r}"
        )
    if loss_kind == "static_width":
        if alpha is None:
            raise ValueError("loss_kind=static_width requires alpha")
        if not (np.isfinite(alpha) and 0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be finite and in (0, 1); got {alpha!r}")
    elif alpha is not None:
        # Skeptic E.3 #7: an integrated_p / cd_variance run is
        # α-marginalised; recording a non-None alpha would lock the
        # checkpoint to that one α at inference, defeating the
        # marginalisation. Refuse loudly.
        raise ValueError(
            f"alpha={alpha} given but loss_kind={loss_kind!r} is "
            f"α-marginalised; pass alpha=None."
        )

    device_resolved = _resolve_device(device)
    # Skeptic block #11: sub-spawn independent RNGs per consumer so a
    # future re-ordering of training steps doesn't silently change the
    # random trajectory of any other consumer.
    base_rng = np.random.default_rng(config.seed)
    rng_train, rng_aux, rng_val_setup, rng_held = (
        (np.random.default_rng(s) for s in base_rng.spawn(4) if s is not None)
        if hasattr(base_rng, "spawn")
        else (np.random.default_rng(config.seed + i) for i in range(4))
    )
    torch.manual_seed(config.seed)
    if verbose:
        print(
            f"[fit_eta_artifact] device={device_resolved} "
            f"scheme={config.scheme_name} statistic={config.statistic_name}"
        )

    # Resolve scheme + statistic from registry (validates the names).
    scheme = _registry.tiltings[config.scheme_name]()
    _ = _registry.statistics[config.statistic_name]  # presence check

    theta_dim = int(config.model.param_dim)
    if n_aux is None:
        n_aux = batch_size

    # Build nets.
    eta_net = EtaNet(theta_dim=theta_dim, hidden_sizes=eta_hidden_sizes).to(device_resolved)
    val_net = ValidityNet(theta_dim=theta_dim, hidden_sizes=validity_hidden_sizes).to(
        device_resolved
    )
    optimizer_a = torch.optim.AdamW(eta_net.parameters(), lr=lr_a, weight_decay=weight_decay)
    optimizer_b = torch.optim.AdamW(val_net.parameters(), lr=lr_b, weight_decay=weight_decay)

    # LHS over θ once at training start; held-out subset for early stopping.
    theta_lhs = lhs_1d(config.theta_distribution, config.n_lhs, seed=config.seed)
    n_val = max(1, int(0.1 * config.n_lhs))
    perm = rng_train.permutation(config.n_lhs)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    theta_train = theta_lhs[train_idx]
    theta_held = theta_lhs[val_idx]

    n_train = len(theta_train)
    steps_per_epoch = max(1, n_train // batch_size)

    # Skeptic block #2: frozen validation set sampled ONCE at training
    # start, fixed across epochs. Was: one ad-hoc D per epoch, which
    # made early stopping pick the noisiest minimum. Now the val
    # signal is `mean over (θ_held, D_val) pairs` of the width loss.
    n_val_pairs = min(len(theta_held), 64)
    theta_val_np = theta_held[:n_val_pairs]
    D_val_np = _sample_data_per_theta(config.model, theta_val_np, rng_val_setup)
    # theta_val_np is consumed only by D_val sampling; the validation loss
    # below evaluates _width_loss over the canonical theta_grid_t and only
    # needs D_val_t for the data dimension.
    D_val_t = torch.as_tensor(D_val_np, dtype=torch.float32, device=device_resolved)

    # Pre-build held-out (θ, η, valid) for Head B accuracy diagnostic.
    eta_held_aux = rng_held.uniform(*config.eta_explore_box, size=len(theta_held)).astype(
        np.float64
    )
    D_held = _sample_data_per_theta(config.model, theta_held, rng_held)
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

    # Canonical inversion grid as torch tensor (training device + dtype).
    theta_grid_t = torch.as_tensor(config.theta_grid, dtype=torch.float32, device=device_resolved)

    train_losses: list[float] = []
    train_width_losses: list[float] = []
    train_penalty_losses: list[float] = []
    val_losses: list[float] = []
    head_b_accs: list[float] = []
    best_val = float("inf")
    best_epoch = -1
    best_state = {
        "eta": {k: v.detach().clone() for k, v in eta_net.state_dict().items()},
        "validity": {k: v.detach().clone() for k, v in val_net.state_dict().items()},
    }
    epochs_since_best = 0
    epochs_run = 0
    stopped_early = False

    # Number of D draws per training step for the width-loss MC average
    # (skeptic block #1). Reuses the `n_mc` knob from the legacy loop.
    n_mc_train = 8

    for epoch in range(n_epochs):
        epochs_run = epoch + 1
        lam = _lambda_schedule(epoch, n_epochs, lambda_max, lambda_warmup_frac)
        eta_net.train()
        val_net.train()
        ep_perm = rng_train.permutation(n_train)
        epoch_train_loss = 0.0
        epoch_width_loss = 0.0
        epoch_penalty_loss = 0.0
        epoch_steps_taken = 0
        epoch_aux_valid_rate_sum = 0.0
        for step in range(steps_per_epoch):
            idx = ep_perm[step * batch_size : (step + 1) * batch_size]
            if idx.size == 0:
                continue
            theta_batch_np = theta_train[idx]
            theta_batch_t = torch.as_tensor(
                theta_batch_np, dtype=torch.float32, device=device_resolved
            )

            # Step (1): forward + label.
            eta_pred = eta_net(theta_batch_t)
            theta_all_t, eta_all_t, valid_all_t = _collect_validity_batch(
                eta_pred=eta_pred,
                theta_batch_np=theta_batch_np,
                config=config,
                scheme=scheme,
                n_aux=n_aux,
                rng=rng_aux,
            )
            # Track aux validity rate (sanity: should be in (0.05, 0.95)).
            aux_valid_rate = float(valid_all_t[len(idx) :].mean().item())
            epoch_aux_valid_rate_sum += aux_valid_rate

            # Step (2): train Head B (BCE on (θ, η, valid)).
            optimizer_b.zero_grad()
            inputs = _validity_net_inputs(theta_all_t, eta_all_t)
            logits = val_net(inputs)
            loss_b = F.binary_cross_entropy_with_logits(logits, valid_all_t)
            loss_b.backward()
            optimizer_b.step()

            # Step (3): train Head A (width + λ · boundary).
            optimizer_a.zero_grad()
            # n_mc_train D draws per minibatch. θ-pairing: cycle through
            # the first n_mc_train θ-batch entries (which are themselves a
            # random permutation of the LHS pool, so marginal). The width
            # loss averages over these n_mc D's, reducing per-step
            # variance ~1/n_mc compared to a single-D estimator.
            theta_for_D = theta_batch_np[: min(n_mc_train, len(idx))]
            D_batch_np = _sample_data_per_theta(
                config.model,
                theta_for_D,
                rng_train,
            )
            D_batch_t = torch.as_tensor(D_batch_np, dtype=torch.float32, device=device_resolved)
            try:
                loss_width = _width_loss(
                    eta_net=eta_net,
                    theta_grid_t=theta_grid_t,
                    D_batch_t=D_batch_t,
                    config=config,
                    loss_kind=loss_kind,
                    alpha=alpha,
                )
            except RuntimeError as e:
                # Width loss raises only if every D's integrand is
                # non-finite — rare, skip the step.
                if verbose:
                    warnings.warn(
                        f"[width loss] step {step} skipped: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            penalty = _boundary_penalty(
                val_net=val_net,
                theta_batch_t=theta_batch_t,
                eta_pred=eta_net(theta_batch_t),
            )
            loss_a = loss_width + lam * penalty
            loss_a.backward()
            optimizer_a.step()
            epoch_train_loss += float(loss_a.item())
            epoch_width_loss += float(loss_width.item())
            epoch_penalty_loss += float(penalty.item())
            epoch_steps_taken += 1

        denom = max(epoch_steps_taken, 1)
        train_losses.append(epoch_train_loss / denom)
        train_width_losses.append(epoch_width_loss / denom)
        train_penalty_losses.append(epoch_penalty_loss / denom)

        # Skeptic block #5: warn if Head B's BCE batch is class-degenerate.
        mean_aux_rate = epoch_aux_valid_rate_sum / denom
        if mean_aux_rate < 0.05 or mean_aux_rate > 0.95:
            warnings.warn(
                f"[head B] epoch {epoch + 1} aux validity rate = "
                f"{mean_aux_rate:.3f} (outside (0.05, 0.95)). Head B's "
                f"BCE is class-degenerate; widen `eta_explore_box` for "
                f"this scheme.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Validation — Head A's loss on the frozen (θ_val, D_val) set.
        # Same set across all epochs, so the val signal is a deterministic
        # function of (eta_net params, D_val draws) — no per-epoch RNG noise.
        eta_net.eval()
        val_net.eval()
        with torch.no_grad():
            try:
                v_loss_t = _width_loss(
                    eta_net=eta_net,
                    theta_grid_t=theta_grid_t,
                    D_batch_t=D_val_t,
                    config=config,
                    loss_kind=loss_kind,
                    alpha=alpha,
                )
                v_loss = float(v_loss_t.item())
            except RuntimeError:
                v_loss = float("inf")
            val_losses.append(v_loss)

            head_b_acc = _evaluate_head_b_accuracy(
                val_net,
                theta_held,
                eta_held_aux,
                valid_held,
                device_resolved,
            )
            head_b_accs.append(head_b_acc)

        if v_loss < best_val - min_delta:
            best_val = v_loss
            best_epoch = epoch
            best_state = {
                "eta": {k: v.detach().clone() for k, v in eta_net.state_dict().items()},
                "validity": {k: v.detach().clone() for k, v in val_net.state_dict().items()},
            }
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if verbose and ((epoch + 1) % max(n_epochs // 10, 1) == 0 or epoch == 0):
            mean_aux_rate = epoch_aux_valid_rate_sum / max(steps_per_epoch, 1)
            print(
                f"[epoch {epoch + 1}/{n_epochs}] "
                f"loss_a={train_losses[-1]:.4f} "
                f"(width={train_width_losses[-1]:.4f}, "
                f"pen={train_penalty_losses[-1]:.4f}) "
                f"val_width={v_loss:.4f} "
                f"best={best_val:.4f} (ep {best_epoch + 1}) "
                f"head_b_acc={head_b_acc:.3f} "
                f"aux_valid={mean_aux_rate:.2f} "
                f"λ={lam:.2f}"
            )

        if epochs_since_best >= patience:
            stopped_early = True
            if verbose:
                print(
                    f"[early stop] no improvement for {patience} epochs "
                    f"at epoch {epoch + 1}; best val={best_val:.4f}."
                )
            break

    # Roll back to best checkpoint.
    eta_net.load_state_dict(best_state["eta"])
    val_net.load_state_dict(best_state["validity"])

    # Final Head B accuracy.
    final_head_b_acc = _evaluate_head_b_accuracy(
        val_net,
        theta_held,
        eta_held_aux,
        valid_held,
        device_resolved,
    )
    # Final Head A empirical validity rate on held-out θ (compute η_pred,
    # then call scheme.tilted_pvalue per sample).
    with torch.no_grad():
        eta_pred_held_t = eta_net(
            torch.as_tensor(theta_held, dtype=torch.float32, device=device_resolved)
        )
    eta_pred_held = eta_pred_held_t.cpu().numpy().astype(np.float64)
    p_pred = compute_pvalues_per_sample(
        scheme,
        theta_held,
        D_held,
        config.model,
        config.prior,
        eta_pred_held,
        config.statistic_name,
    )
    final_eta_pred_valid_rate = float(validity_mask(p_pred).mean())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Skeptic block #8: atomic write — torch.save to a tmp path then
    # os.replace, so a crash mid-write never produces a corrupt .pt.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    state = {
        "checkpoint_format_version": 2,  # E.2 bump
        "architecture": "EtaNet+ValidityNet",
        "eta_architecture_kwargs": eta_net.architecture_kwargs(),
        "validity_architecture_kwargs": val_net.architecture_kwargs(),
        "eta_state_dict": eta_net.state_dict(),
        "validity_state_dict": val_net.state_dict(),
        "experiment_config": config.to_dict(),
        "loss_kind": loss_kind,
        "alpha": alpha,
        "lambda_max": lambda_max,
        "lambda_warmup_frac": lambda_warmup_frac,
        "n_aux": n_aux,
        "lr_a": lr_a,
        "lr_b": lr_b,
        "weight_decay": weight_decay,
        "n_epochs": n_epochs,
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch + 1,
        "patience": patience,
        "min_delta": min_delta,
        "batch_size": batch_size,
        "seed": config.seed,
        "version": version,
        "training_finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "train_losses": train_losses,
        "train_width_losses": train_width_losses,
        "train_penalty_losses": train_penalty_losses,
        "val_losses": val_losses,
        "head_b_accuracies": head_b_accs,
        "final_val_loss": best_val,
        "final_head_b_accuracy": final_head_b_acc,
        "final_eta_pred_valid_rate": final_eta_pred_valid_rate,
    }
    torch.save(state, str(tmp_path))
    import os as _os

    _os.replace(tmp_path, out_path)  # atomic
    if verbose:
        print(
            f"[fit_eta_artifact] wrote {out_path}; "
            f"head_b_acc={final_head_b_acc:.3f}, "
            f"η_pred_valid_rate={final_eta_pred_valid_rate:.3f}"
        )

    return EtaTrainResult(
        artifact_path=out_path,
        train_losses=train_losses,
        train_width_losses=train_width_losses,
        train_penalty_losses=train_penalty_losses,
        val_losses=val_losses,
        head_b_accuracy=head_b_accs,
        final_val_loss=best_val,
        metadata={
            k: v for k, v in state.items() if k not in ("eta_state_dict", "validity_state_dict")
        },
    )
