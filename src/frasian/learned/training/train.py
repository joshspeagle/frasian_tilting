"""Phase E dual-head training entry point — JAX/Equinox/Optax orchestrator.

This module keeps only the public entry ``fit_eta_artifact``; the
heavy lifting lives in:

- ``_setup`` — pre-flight (device / determinism / loss-kind / RNGs)
- ``_train_loop.run_epoch_loop`` — minibatch / optimiser / early-stop
- ``_validity_data`` — LHS sampling, validity batch building, antithetic
- ``_losses_compose`` — width loss, boundary penalty, λ + β schedules
- ``_checkpoint`` — atomic save + equinox / arch_sha metadata

The training step is documented in ``_train_loop.py``; this file
just wires the pieces together.

**Determinism.** JAX is bit-deterministic on CPU at a fixed
``jax.random.PRNGKey``. The orchestrator derives the root key from
``config.seed`` at the top, splits it for net init / data sampling,
and seeds numpy globally for any side-channel randomness.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active
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

_FORCE_X64 = _x64


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
    prior because the JAX tilted_pvalue ports are Normal-Normal only.

    Writes a checkpoint at ``out_path`` recording both nets' state,
    the experiment config (with fingerprints), the λ + β schedules,
    ``equinox_version`` + ``jax_version``, ``arch_sha`` (for cross-
    environment compat diagnostics), and a final calibration summary.
    Returns a ``EtaTrainResult``.

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
        ``UserWarning`` and proceeds with ``antithetic=False``.
    """
    _validate_loss_kind(loss_kind, alpha)
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
    root_key = _enable_determinism(config.seed)

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

    eta_init_key, val_init_key = jax.random.split(root_key)
    eta_net = EtaNet(theta_dim=theta_dim, hidden_sizes=eta_hidden_sizes, key=eta_init_key)
    val_net = ValidityNet(
        theta_dim=theta_dim, hidden_sizes=validity_hidden_sizes, key=val_init_key
    )

    optimizer_a = optax.adamw(learning_rate=lr_a, weight_decay=weight_decay)
    optimizer_b = optax.adamw(learning_rate=lr_b, weight_decay=weight_decay)
    # Optax expects only the trainable (array) leaves of an Equinox
    # module. ``eqx.filter`` selects exactly those.
    import equinox as eqx

    opt_state_a = optimizer_a.init(eqx.filter(eta_net, eqx.is_array))
    opt_state_b = optimizer_b.init(eqx.filter(val_net, eqx.is_array))

    # LHS over θ once at training start; held-out subset for early stopping.
    theta_lhs = lhs_1d(config.theta_distribution, config.n_lhs, seed=config.seed)
    n_val = max(1, int(0.1 * config.n_lhs))
    perm = rng_train.permutation(config.n_lhs)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    theta_train = theta_lhs[train_idx]
    theta_held = theta_lhs[val_idx]

    # Frozen validation set sampled ONCE at training start.
    n_val_pairs = min(len(theta_held), 64)
    theta_val_np = theta_held[:n_val_pairs]
    D_val_np = sample_data_per_theta(
        config.model, theta_val_np, rng_val_setup, n_data=config.n_data
    )
    D_val_t = jnp.asarray(D_val_np)

    eta_held_aux, D_held, valid_held = prepare_held_out_validity(
        scheme=scheme,
        theta_held=theta_held,
        config=config,
        rng=rng_held,
    )

    theta_grid_t = jnp.asarray(config.theta_grid)

    # For non-Normal-Normal experiments, precompute the integration
    # grid + log-prior grid once + the per-validation-batch
    # log-likelihood grid (frozen across epochs).
    support_theta_grid_np: np.ndarray | None = None
    log_p_lik_val_t: jax.Array | None = None
    if config.model.fingerprint()[0] != "normal_normal":
        from ._losses_compose import (
            compute_log_p_lik_grid_np,
            precompute_generic_grids,
        )

        support_theta_grid_t, _ = precompute_generic_grids(
            config.model, config.prior
        )
        support_theta_grid_np = np.asarray(support_theta_grid_t)
        log_p_lik_val_np = compute_log_p_lik_grid_np(
            config.model, D_val_np, support_theta_grid_np
        )
        log_p_lik_val_t = jnp.asarray(log_p_lik_val_np)

    args = LoopArgs(
        eta_net=eta_net,
        val_net=val_net,
        optimizer_a=optimizer_a,
        optimizer_b=optimizer_b,
        opt_state_a=opt_state_a,
        opt_state_b=opt_state_b,
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
        support_theta_grid_np=support_theta_grid_np,
        log_p_lik_val_t=log_p_lik_val_t,
    )
    out = run_epoch_loop(args)

    # Roll back to best checkpoint via reference swap (Equinox modules
    # are immutable PyTrees).
    eta_net = out.best_eta_net if out.best_eta_net is not None else args.eta_net
    val_net = out.best_val_net if out.best_val_net is not None else args.val_net

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
