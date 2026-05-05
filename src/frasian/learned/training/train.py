"""Phase E dual-head training entry point — orchestrator after Tier 1.2 §7 split.

This module keeps only the public entry ``fit_eta_artifact``; the
heavy lifting lives in:

- ``_train_loop.run_epoch_loop`` — minibatch / optimiser / early-stop
- ``_validity_data`` — LHS sampling, validity batch building, antithetic
- ``_losses_compose`` — width loss, boundary penalty, λ + β schedules
- ``_checkpoint`` — atomic save + torch_version / arch_sha metadata

The training step is documented in ``_train_loop.py``; this file
just wires the pieces together.

**Determinism (1.2-NN4).** When torch is available, the orchestrator
sets ``torch.use_deterministic_algorithms(True)`` and the cudnn flags
as the first call. Combined with ``torch.manual_seed(seed)`` and
``np.random.seed(seed)`` this gives byte-reproducible runs on the
same torch + GPU build. CUBLAS deterministic mode requires the env
var ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (set by the orchestrator).
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..._registry import registry as _registry
from ._checkpoint import save_checkpoint
from ._losses_compose import (
    compose_width_loss as _width_loss,  # noqa: F401  (test back-compat)
)
from ._losses_compose import (
    extract_normal_normal_params as _extract_normal_normal_params,  # noqa: F401
)
from ._losses_compose import (
    lambda_schedule as _lambda_schedule,  # noqa: F401  (test back-compat)
)
from ._train_loop import LoopArgs, evaluate_head_b_accuracy, run_epoch_loop
from ._validity_data import prepare_held_out_validity, sample_data_per_theta
from .architecture import EtaNet, ValidityNet
from .sampling import ExperimentConfig, lhs_1d
from .validity import compute_pvalues_per_sample, validity_mask

_LOSS_KINDS = ("integrated_p", "cd_variance", "static_width")


@dataclass
class EtaTrainResult:
    artifact_path: Path
    train_losses: list[float]
    train_width_losses: list[float]
    train_penalty_losses: list[float]
    val_losses: list[float]
    head_b_accuracy: list[float]
    final_val_loss: float
    metadata: dict[str, Any]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _enable_determinism(seed: int) -> None:
    """Enable deterministic torch + numpy paths.

    Called at the top of ``fit_eta_artifact`` before any torch-side
    state is created. CUBLAS deterministic mode requires the
    workspace config env var; we set it lazily so users who don't
    care about CUDA are unaffected.
    """
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def _validate_loss_kind(loss_kind: str, alpha: float | None) -> None:
    if loss_kind not in _LOSS_KINDS:
        raise ValueError(f"loss_kind must be one of {_LOSS_KINDS}; got {loss_kind!r}")
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


def _spawn_rngs(
    seed: int,
) -> tuple[
    np.random.Generator, np.random.Generator, np.random.Generator, np.random.Generator
]:
    """Sub-spawn 4 independent RNGs (skeptic block #11)."""
    base_rng = np.random.default_rng(seed)
    if hasattr(base_rng, "spawn"):
        rngs = [np.random.default_rng(s) for s in base_rng.spawn(4) if s is not None]
    else:
        rngs = [np.random.default_rng(seed + i) for i in range(4)]
    return rngs[0], rngs[1], rngs[2], rngs[3]


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
    antithetic: bool = True,
    verbose: bool = True,
) -> EtaTrainResult:
    """Train an EtaNet + ValidityNet pair end-to-end on ``config``.

    Model-agnostic interface (drives off ``config``). The width-loss
    side currently requires a NormalNormalModel + NormalDistribution
    prior because the torch tilted_pvalue ports are Normal-Normal
    only — the training loop itself doesn't reference Normal-Normal
    coordinates anywhere else.

    Writes a checkpoint at ``out_path`` recording both nets' state,
    the experiment config (with fingerprints), the λ + β schedules,
    ``torch_version``, ``arch_sha`` (for cross-environment compat
    diagnostics), and a final calibration summary. Returns a
    ``EtaTrainResult``.

    Parameters of note
    ------------------
    antithetic
        If True (default), each Monte-Carlo D draw is paired with
        its ``2θ − D`` antithetic partner, halving variance on
        even loss components for Normal-Normal symmetry. Set False
        to reproduce the legacy estimator.

    Notes
    -----
    Determinism: ``torch.use_deterministic_algorithms(True, warn_only=True)``
    + ``torch.manual_seed(seed)`` + ``np.random.seed(seed)`` are
    set at the top of the function. Set
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` for CUDA determinism (auto-set
    if not present).
    """
    _validate_loss_kind(loss_kind, alpha)
    _enable_determinism(config.seed)

    device_resolved = _resolve_device(device)
    rng_train, rng_aux, rng_val_setup, rng_held = _spawn_rngs(config.seed)
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

    # Skeptic block #2: frozen validation set sampled ONCE at training
    # start, fixed across epochs.
    n_val_pairs = min(len(theta_held), 64)
    theta_val_np = theta_held[:n_val_pairs]
    D_val_np = sample_data_per_theta(config.model, theta_val_np, rng_val_setup)
    D_val_t = torch.as_tensor(D_val_np, dtype=torch.float32, device=device_resolved)

    eta_held_aux, D_held, valid_held = prepare_held_out_validity(
        scheme=scheme,
        theta_held=theta_held,
        config=config,
        rng=rng_held,
    )

    theta_grid_t = torch.as_tensor(config.theta_grid, dtype=torch.float32, device=device_resolved)

    out = run_epoch_loop(
        LoopArgs(
            eta_net=eta_net,
            val_net=val_net,
            optimizer_a=optimizer_a,
            optimizer_b=optimizer_b,
            theta_train=theta_train,
            theta_held=theta_held,
            eta_held_aux=eta_held_aux,
            valid_held=valid_held,
            D_val_t=D_val_t,
            theta_grid_t=theta_grid_t,
            config=config,
            scheme=scheme,
            n_aux=n_aux,
            rng_train=rng_train,
            rng_aux=rng_aux,
            n_epochs=n_epochs,
            batch_size=batch_size,
            loss_kind=loss_kind,
            alpha=alpha,
            lambda_max=lambda_max,
            lambda_warmup_frac=lambda_warmup_frac,
            patience=patience,
            min_delta=min_delta,
            antithetic=antithetic,
            device=device_resolved,
            verbose=verbose,
        )
    )

    # Roll back to best checkpoint.
    eta_net.load_state_dict(out.best_state["eta"])
    val_net.load_state_dict(out.best_state["validity"])

    # Final Head B accuracy + Head A empirical validity rate on held-out θ.
    final_head_b_acc = evaluate_head_b_accuracy(
        val_net,
        theta_held,
        eta_held_aux,
        valid_held,
        device_resolved,
    )
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
    metadata = save_checkpoint(
        out_path=out_path,
        eta_net=eta_net,
        val_net=val_net,
        config=config,
        loss_kind=loss_kind,
        alpha=alpha,
        lambda_max=lambda_max,
        lambda_warmup_frac=lambda_warmup_frac,
        n_aux=n_aux,
        lr_a=lr_a,
        lr_b=lr_b,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        epochs_run=out.epochs_run,
        stopped_early=out.stopped_early,
        best_epoch=out.best_epoch + 1,
        patience=patience,
        min_delta=min_delta,
        batch_size=batch_size,
        seed=config.seed,
        version=version,
        train_losses=out.train_losses,
        train_width_losses=out.train_width_losses,
        train_penalty_losses=out.train_penalty_losses,
        val_losses=out.val_losses,
        head_b_accuracies=out.head_b_accuracies,
        final_val_loss=out.best_val,
        final_head_b_accuracy=final_head_b_acc,
        final_eta_pred_valid_rate=final_eta_pred_valid_rate,
        antithetic=antithetic,
    )
    if verbose:
        print(
            f"[fit_eta_artifact] wrote {out_path}; "
            f"head_b_acc={final_head_b_acc:.3f}, "
            f"η_pred_valid_rate={final_eta_pred_valid_rate:.3f}"
        )

    return EtaTrainResult(
        artifact_path=out_path,
        train_losses=out.train_losses,
        train_width_losses=out.train_width_losses,
        train_penalty_losses=out.train_penalty_losses,
        val_losses=out.val_losses,
        head_b_accuracy=out.head_b_accuracies,
        final_val_loss=out.best_val,
        metadata=metadata,
    )
