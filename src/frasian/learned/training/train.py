"""Phase E dual-head training entry point — orchestrator after Tier 1.2 §7 split.

This module keeps only the public entry ``fit_eta_artifact``; the
heavy lifting lives in:

- ``_setup`` — pre-flight (device / determinism / loss-kind / RNGs)
- ``_train_loop.run_epoch_loop`` — minibatch / optimiser / early-stop
- ``_validity_data`` — LHS sampling, validity batch building, antithetic
- ``_losses_compose`` — width loss, boundary penalty, λ + β schedules
- ``_checkpoint`` — atomic save + torch_version / arch_sha metadata

The training step is documented in ``_train_loop.py``; this file
just wires the pieces together. Phase 4 skeptic §8 audit-line target
is ~150; we land at ~310, with the remaining residual being the
17-kwarg ``fit_eta_artifact`` signature (most are tunables that
must surface to the public API), the ~40-line ``LoopArgs``
construction (one kwarg per LoopArgs field — collapsing this
into a builder would just relocate the ceremony), and the
~25-line ``save_checkpoint`` call (one kwarg per metadata field —
same trade-off). Further factoring would add indirection
without reducing the underlying parameter-passing complexity.

**Determinism (1.2-NN4).** When torch is available, the orchestrator
sets ``torch.use_deterministic_algorithms(True)`` and the cudnn flags
as the first call. Combined with ``torch.manual_seed(seed)`` and
``np.random.seed(seed)`` this gives byte-reproducible runs on the
same torch + GPU build. CUBLAS deterministic mode requires the env
var ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (set by the orchestrator).
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..._registry import registry as _registry
from ._checkpoint import save_checkpoint

# === Back-compat aliases for tests/properties/test_dual_head_invariants.py ===
# These three names are imported by the property tests at lines
# 645, 730, 852 of that file. Do NOT remove without first migrating
# the test imports to the canonical names in
# _losses_compose.{extract_normal_normal_params, compose_width_loss,
# lambda_schedule}.
from ._losses_compose import (
    compose_width_loss as _width_loss,  # noqa: F401  (test back-compat)
)
from ._losses_compose import (
    extract_normal_normal_params as _extract_normal_normal_params,  # noqa: F401
)
from ._losses_compose import (
    lambda_schedule as _lambda_schedule,  # noqa: F401  (test back-compat)
)

# Pre-flight helpers (resolve device / determinism / loss-kind / RNGs)
# live in ``_setup.py`` per Phase 4 skeptic §8. The ``_enable_determinism``
# alias is kept for ``tests/regression/test_torch_determinism.py``.
from ._setup import (
    enable_determinism as _enable_determinism,
)
from ._setup import (
    resolve_device as _resolve_device,
)
from ._setup import (
    spawn_rngs as _spawn_rngs,
)
from ._setup import (
    validate_loss_kind as _validate_loss_kind,
)
from ._train_loop import (
    LoopArgs,
    compute_final_eta_pred_valid_rate,
    evaluate_head_b_accuracy,
    run_epoch_loop,
)
from ._validity_data import prepare_held_out_validity, sample_data_per_theta
from .architecture import EtaNet, ValidityNet
from .sampling import ExperimentConfig, lhs_1d


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
        Effective only when ``loss_kind == "static_width"``: each
        Monte-Carlo D draw is paired with its ``2θ − D`` partner,
        reducing variance on the sigmoid-relaxed indicator (which
        has odd Taylor structure in ``D − θ``). For
        ``integrated_p`` / ``cd_variance`` the integrand is even in
        ``D − θ`` on the θ-symmetric grid, so the paired and IID
        estimators are algebraically identical — passing
        ``antithetic=True`` with those losses emits a
        ``UserWarning`` and proceeds with antithetic=False.

    Notes
    -----
    Determinism: ``torch.use_deterministic_algorithms(True, warn_only=True)``
    + ``torch.manual_seed(seed)`` + ``np.random.seed(seed)`` are
    set at the top of the function. Set
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` for CUDA determinism (auto-set
    if not present).
    """
    _validate_loss_kind(loss_kind, alpha)
    # Phase 4 skeptic §1: antithetic pairing is a no-op for the
    # α-marginalised losses (their integrand is even in D − θ over a
    # θ-symmetric grid, so the paired and IID estimators coincide).
    # Honour the flag only for ``static_width`` and warn otherwise.
    effective_antithetic = bool(antithetic) and loss_kind == "static_width"
    if antithetic and not effective_antithetic:
        _warnings.warn(
            f"antithetic=True is a no-op for loss_kind={loss_kind!r}: the "
            f"integrated_p / cd_variance losses are even in D − θ on the "
            f"θ-symmetric grid, so paired and IID estimators are "
            f"algebraically identical. Proceeding with antithetic=False. "
            f"Set antithetic=True only with loss_kind='static_width'.",
            UserWarning,
            stacklevel=2,
        )
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
            antithetic=effective_antithetic,
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
    final_eta_pred_valid_rate = compute_final_eta_pred_valid_rate(
        eta_net=eta_net,
        scheme=scheme,
        theta_held=theta_held,
        D_held=D_held,
        config=config,
        device=device_resolved,
    )

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
        antithetic=effective_antithetic,
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
