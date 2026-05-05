"""Per-epoch training loop for the Phase E dual-head selector.

Extracted from ``train.py`` (Tier 1.2 §7 split). The orchestrator
builds nets / RNGs / optimisers / held-out sets; ``run_epoch_loop``
here owns the minibatch / optimiser / per-step pattern.

Training step (per minibatch):
1. Forward Head A + collect validity labels (mix aux batch so Head B
   sees both classes every step).
2. Train Head B (BCE on (θ, η, valid)).
3. Train Head A (width loss + λ · boundary penalty).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ._losses_compose import (
    beta_schedule,
    compose_boundary_penalty,
    compose_width_loss,
    lambda_schedule,
)
from ._validity_data import (
    collect_validity_batch,
    sample_data_per_theta,
    validity_net_inputs,
)
from .architecture import EtaNet, ValidityNet
from .sampling import ExperimentConfig

# Number of D draws per training step for the width-loss MC average
# (skeptic block #1). Reuses the `n_mc` knob from the legacy loop.
N_MC_TRAIN: int = 8


@dataclass
class EpochLoopOutputs:
    """Metrics + best state returned by ``run_epoch_loop``."""

    train_losses: list[float] = field(default_factory=list)
    train_width_losses: list[float] = field(default_factory=list)
    train_penalty_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    head_b_accuracies: list[float] = field(default_factory=list)
    best_val: float = float("inf")
    best_epoch: int = -1
    best_state: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    epochs_run: int = 0
    stopped_early: bool = False


@dataclass
class _EpochAggregates:
    """Accumulators reset at the top of every epoch."""

    train_loss: float = 0.0
    width_loss: float = 0.0
    penalty_loss: float = 0.0
    steps_taken: int = 0
    aux_valid_rate_sum: float = 0.0


@dataclass
class LoopArgs:
    """All inputs to ``run_epoch_loop`` — bundled to keep helper
    signatures small (each helper takes ``LoopArgs`` instead of 15+
    kw-only args, which would balloon the file past the 350-line cap).
    """

    eta_net: EtaNet
    val_net: ValidityNet
    optimizer_a: torch.optim.Optimizer
    optimizer_b: torch.optim.Optimizer
    theta_train: np.ndarray
    theta_held: np.ndarray
    eta_held_aux: np.ndarray
    valid_held: np.ndarray
    D_val_t: torch.Tensor
    theta_grid_t: torch.Tensor
    config: ExperimentConfig
    scheme: Any
    n_aux: int
    rng_train: np.random.Generator
    rng_aux: np.random.Generator
    n_epochs: int
    batch_size: int
    loss_kind: str
    alpha: float | None
    lambda_max: float
    lambda_warmup_frac: float
    patience: int
    min_delta: float
    antithetic: bool
    device: str
    verbose: bool


def evaluate_head_b_accuracy(
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
        inputs = validity_net_inputs(theta_t, eta_t)
        logits = val_net(inputs)
        pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
    return float((pred == valid_held).mean())


def compute_final_eta_pred_valid_rate(
    *,
    eta_net: EtaNet,
    scheme: Any,
    theta_held: np.ndarray,
    D_held: np.ndarray,
    config: ExperimentConfig,
    device: str,
) -> float:
    """Empirical validity rate of Head A's predicted η on held-out θ.

    For each held-out θ, predict η = EtaNet(θ); pair with the
    pre-sampled D_held; ask the scheme's tilted_pvalue whether the
    (θ, η, D) triple yields a finite p-value in [0, 1]. Mean of that
    indicator is the rate. Used as a final calibration diagnostic.
    """
    from .validity import compute_pvalues_per_sample, validity_mask

    with torch.no_grad():
        eta_pred_held_t = eta_net(
            torch.as_tensor(theta_held, dtype=torch.float32, device=device)
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
    return float(validity_mask(p_pred).mean())


def _training_step(
    args: LoopArgs, *, theta_batch_np: np.ndarray, beta: float | None, lam: float, step_idx: int
) -> tuple[float, float, float, float] | None:
    """Run one minibatch step. Returns metrics or None if the step is skipped."""
    theta_batch_t = torch.as_tensor(theta_batch_np, dtype=torch.float32, device=args.device)

    # Step (1): forward + label.
    eta_pred = args.eta_net(theta_batch_t)
    theta_all_t, eta_all_t, valid_all_t = collect_validity_batch(
        eta_pred=eta_pred,
        theta_batch_np=theta_batch_np,
        config=args.config,
        scheme=args.scheme,
        n_aux=args.n_aux,
        rng=args.rng_aux,
    )
    aux_valid_rate = float(valid_all_t[len(theta_batch_np) :].mean().item())

    # Step (2): train Head B (BCE on (θ, η, valid)).
    args.optimizer_b.zero_grad()
    logits = args.val_net(validity_net_inputs(theta_all_t, eta_all_t))
    loss_b = F.binary_cross_entropy_with_logits(logits, valid_all_t)
    loss_b.backward()
    args.optimizer_b.step()

    # Step (3): train Head A (width + λ · boundary). With antithetic on,
    # each MC draw is paired with its 2θ−D partner (2·n_mc rows).
    args.optimizer_a.zero_grad()
    n_mc = min(N_MC_TRAIN, len(theta_batch_np))
    D_batch_np = sample_data_per_theta(
        args.config.model, theta_batch_np[:n_mc], args.rng_train, antithetic=args.antithetic
    )
    D_batch_t = torch.as_tensor(D_batch_np, dtype=torch.float32, device=args.device)
    try:
        loss_width = compose_width_loss(
            eta_net=args.eta_net,
            theta_grid_t=args.theta_grid_t,
            D_batch_t=D_batch_t,
            config=args.config,
            loss_kind=args.loss_kind,
            alpha=args.alpha,
            beta=beta,
        )
    except RuntimeError as e:
        if args.verbose:
            warnings.warn(
                f"[width loss] step {step_idx} skipped: {e}", RuntimeWarning, stacklevel=2
            )
        return None
    penalty = compose_boundary_penalty(
        val_net=args.val_net,
        theta_batch_t=theta_batch_t,
        eta_pred=args.eta_net(theta_batch_t),
    )
    loss_a = loss_width + lam * penalty
    loss_a.backward()
    args.optimizer_a.step()
    return float(loss_a.item()), float(loss_width.item()), float(penalty.item()), aux_valid_rate


def _run_epoch_steps(
    args: LoopArgs, *, n_train: int, steps_per_epoch: int, beta: float | None, lam: float
) -> _EpochAggregates:
    """Inner per-step loop. Returns one epoch's aggregates."""
    args.eta_net.train()
    args.val_net.train()
    ep_perm = args.rng_train.permutation(n_train)
    agg = _EpochAggregates()
    for step in range(steps_per_epoch):
        idx = ep_perm[step * args.batch_size : (step + 1) * args.batch_size]
        if idx.size == 0:
            continue
        metrics = _training_step(
            args,
            theta_batch_np=args.theta_train[idx],
            beta=beta,
            lam=lam,
            step_idx=step,
        )
        if metrics is None:
            continue
        l_a, l_w, l_p, aux_rate = metrics
        agg.train_loss += l_a
        agg.width_loss += l_w
        agg.penalty_loss += l_p
        agg.aux_valid_rate_sum += aux_rate
        agg.steps_taken += 1
    return agg


def _evaluate_epoch(
    args: LoopArgs, beta: float | None
) -> tuple[float, float]:
    """Compute (val_loss, head_b_accuracy) on the frozen held-out set."""
    args.eta_net.eval()
    args.val_net.eval()
    with torch.no_grad():
        try:
            v_loss = float(
                compose_width_loss(
                    eta_net=args.eta_net,
                    theta_grid_t=args.theta_grid_t,
                    D_batch_t=args.D_val_t,
                    config=args.config,
                    loss_kind=args.loss_kind,
                    alpha=args.alpha,
                    beta=beta,
                ).item()
            )
        except RuntimeError:
            v_loss = float("inf")
        head_b_acc = evaluate_head_b_accuracy(
            args.val_net, args.theta_held, args.eta_held_aux, args.valid_held, args.device
        )
    return v_loss, head_b_acc


def _maybe_warn_class_degenerate(epoch: int, mean_aux_rate: float) -> None:
    """Skeptic block #5: warn on class-degenerate Head-B BCE batch."""
    if 0.05 <= mean_aux_rate <= 0.95:
        return
    warnings.warn(
        f"[head B] epoch {epoch + 1} aux validity rate = "
        f"{mean_aux_rate:.3f} (outside (0.05, 0.95)). Head B's BCE is "
        f"class-degenerate; widen `eta_explore_box` for this scheme.",
        RuntimeWarning,
        stacklevel=2,
    )


def _maybe_log_epoch(
    *, epoch: int, n_epochs: int, out: EpochLoopOutputs,
    v_loss: float, head_b_acc: float, mean_aux_rate: float,
    lam: float, beta: float | None, verbose: bool,
) -> None:
    """Print one verbose-mode line every ~10 % of epochs (and on epoch 0)."""
    if not verbose:
        return
    if not ((epoch + 1) % max(n_epochs // 10, 1) == 0 or epoch == 0):
        return
    beta_str = "n/a" if beta is None else f"{beta:.0f}"
    print(
        f"[epoch {epoch + 1}/{n_epochs}] "
        f"loss_a={out.train_losses[-1]:.4f} "
        f"(width={out.train_width_losses[-1]:.4f}, "
        f"pen={out.train_penalty_losses[-1]:.4f}) "
        f"val_width={v_loss:.4f} "
        f"best={out.best_val:.4f} (ep {out.best_epoch + 1}) "
        f"head_b_acc={head_b_acc:.3f} "
        f"aux_valid={mean_aux_rate:.2f} "
        f"λ={lam:.2f} β={beta_str}"
    )


def _snapshot_state(
    eta_net: EtaNet, val_net: ValidityNet
) -> dict[str, dict[str, torch.Tensor]]:
    """Detach + clone both nets' state dicts."""
    return {
        "eta": {k: v.detach().clone() for k, v in eta_net.state_dict().items()},
        "validity": {k: v.detach().clone() for k, v in val_net.state_dict().items()},
    }


def _epoch_iteration(
    args: LoopArgs, out: EpochLoopOutputs,
    *, epoch: int, n_train: int, steps_per_epoch: int,
) -> bool:
    """Run one full epoch: train steps + validation + best-state update.

    Returns ``improved`` so the outer loop can update its early-stop counter.
    """
    out.epochs_run = epoch + 1
    lam = lambda_schedule(epoch, args.n_epochs, args.lambda_max, args.lambda_warmup_frac)
    # β is the sigmoid-sharpness for ``static_width_loss`` only; the
    # α-marginalised losses don't use a relaxed indicator. Pass None
    # so ``compose_width_loss`` would raise if anyone wires β through
    # for the wrong loss kind by accident.
    beta = beta_schedule(epoch, args.n_epochs) if args.loss_kind == "static_width" else None

    agg = _run_epoch_steps(
        args, n_train=n_train, steps_per_epoch=steps_per_epoch, beta=beta, lam=lam
    )
    denom = max(agg.steps_taken, 1)
    out.train_losses.append(agg.train_loss / denom)
    out.train_width_losses.append(agg.width_loss / denom)
    out.train_penalty_losses.append(agg.penalty_loss / denom)
    mean_aux_rate = agg.aux_valid_rate_sum / denom
    _maybe_warn_class_degenerate(epoch, mean_aux_rate)

    v_loss, head_b_acc = _evaluate_epoch(args, beta)
    out.val_losses.append(v_loss)
    out.head_b_accuracies.append(head_b_acc)

    improved = v_loss < out.best_val - args.min_delta
    if improved:
        out.best_val = v_loss
        out.best_epoch = epoch
        out.best_state = _snapshot_state(args.eta_net, args.val_net)

    _maybe_log_epoch(
        epoch=epoch, n_epochs=args.n_epochs, out=out,
        v_loss=v_loss, head_b_acc=head_b_acc, mean_aux_rate=mean_aux_rate,
        lam=lam, beta=beta, verbose=args.verbose,
    )
    return improved


def run_epoch_loop(args: LoopArgs) -> EpochLoopOutputs:
    """Run all epochs; track best val state; early-stop on patience."""
    n_train = len(args.theta_train)
    steps_per_epoch = max(1, n_train // args.batch_size)
    out = EpochLoopOutputs(best_state=_snapshot_state(args.eta_net, args.val_net))
    epochs_since_best = 0
    for epoch in range(args.n_epochs):
        improved = _epoch_iteration(
            args, out, epoch=epoch, n_train=n_train, steps_per_epoch=steps_per_epoch
        )
        epochs_since_best = 0 if improved else epochs_since_best + 1
        if epochs_since_best >= args.patience:
            out.stopped_early = True
            if args.verbose:
                print(
                    f"[early stop] no improvement for {args.patience} epochs "
                    f"at epoch {epoch + 1}; best val={out.best_val:.4f}."
                )
            break
    return out
