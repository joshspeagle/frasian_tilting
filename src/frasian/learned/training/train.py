"""Phase G dual-head training entry point — JAX/Equinox/Optax orchestrator.

This module keeps only the public entry ``fit_eta_artifact``; the
heavy lifting lives in:

- ``_setup`` — pre-flight (device / determinism / loss-kind / RNGs)
- ``_train_loop.run_epoch_loop`` — minibatch / optimiser / early-stop
- ``_validity_data`` — held-out sampling, validity batch building
- ``_losses_compose`` — width loss, boundary penalty, λ + β schedules
- ``_checkpoint`` — atomic save + equinox / arch_sha metadata
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
from ._losses_compose import (
    compose_width_loss as _width_loss,  # noqa: F401  (test back-compat)
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
from ._validity_data import prepare_held_out_validity
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
    n_aux: int = 64,
    lr_a: float = 1e-3,
    lr_b: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_max: float = 10.0,
    lambda_warmup_frac: float = 0.3,
    patience: int = 8,
    min_delta: float = 1e-4,
    eta_hidden_sizes: tuple[int, ...] = (128, 128, 128),
    validity_hidden_sizes: tuple[int, ...] = (128, 128, 128),
    device: str = "auto",
    version: str = "v0",
    antithetic: bool = False,
    verbose: bool = True,
) -> EtaTrainResult:
    """Train an EtaNet + ValidityNet pair end-to-end on ``config``.

    Phase G conditional architecture: per-batch (prior_hp, lik_hp)
    sampled from ``config.hyperparam_distribution``.
    """
    _validate_loss_kind(loss_kind, alpha)
    if antithetic:
        _warnings.warn(
            "antithetic=True is not supported in the Phase G conditional "
            "training loop; ignored.",
            UserWarning, stacklevel=2,
        )
    root_key = _enable_determinism(config.seed)

    device_resolved = _resolve_device(device)
    rng_train, rng_aux, rng_val_setup, rng_held = _spawn_rngs(config.seed)
    if verbose:
        print(
            f"[fit_eta_artifact] device={device_resolved} "
            f"scheme={config.scheme_name} statistic={config.statistic_name} "
            f"prior_class={config.prior_cls.__name__} "
            f"model_class={config.model_cls.__name__}"
        )

    scheme = _registry.tiltings[config.scheme_name]()
    _ = _registry.statistics[config.statistic_name]  # presence check

    theta_dim = 1
    prior_dim = config.prior_cls.hyperparam_dim
    lik_dim = config.model_cls.hyperparam_dim

    eta_init_key, val_init_key = jax.random.split(root_key)
    eta_net = EtaNet(
        theta_dim=theta_dim, prior_dim=prior_dim, lik_dim=lik_dim,
        hidden_sizes=eta_hidden_sizes, key=eta_init_key,
    )
    val_net = ValidityNet(
        theta_dim=theta_dim, prior_dim=prior_dim, lik_dim=lik_dim,
        hidden_sizes=validity_hidden_sizes, key=val_init_key,
    )

    optimizer_a = optax.adamw(learning_rate=lr_a, weight_decay=weight_decay)
    optimizer_b = optax.adamw(learning_rate=lr_b, weight_decay=weight_decay)
    import equinox as eqx
    opt_state_a = optimizer_a.init(eqx.filter(eta_net, eqx.is_array))
    opt_state_b = optimizer_b.init(eqx.filter(val_net, eqx.is_array))

    theta_lhs = lhs_1d(config.theta_distribution, config.n_lhs, seed=config.seed)
    n_val = max(1, int(0.1 * config.n_lhs))
    perm = rng_train.permutation(config.n_lhs)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    theta_train = theta_lhs[train_idx]
    theta_held = theta_lhs[val_idx]

    n_val_pairs = min(len(theta_held), 64)
    theta_val_np = theta_held[:n_val_pairs]
    prior_names = config.prior_cls.hyperparam_names()
    lik_names = config.model_cls.hyperparam_names()
    prior_hp_val_np, lik_hp_val_np = config.hyperparam_distribution.sample(
        n_val_pairs, rng_val_setup,
        prior_names=prior_names, lik_names=lik_names,
    )
    D_val_np = config.model_cls.sample_data_batch_with_hp(
        theta_val_np, lik_hp_val_np, rng_val_setup, n_data=config.n_data,
    )
    if D_val_np.ndim == 2 and D_val_np.shape[1] == 1:
        D_val_np = D_val_np[:, 0]
    D_val_t = jnp.asarray(D_val_np)
    prior_hp_val_t = jnp.asarray(prior_hp_val_np)
    lik_hp_val_t = jnp.asarray(lik_hp_val_np)

    eta_held_aux, D_held, prior_hp_held, lik_hp_held, valid_held = (
        prepare_held_out_validity(
            scheme=scheme,
            theta_held=theta_held,
            config=config,
            rng=rng_held,
        )
    )

    theta_grid_t = jnp.asarray(config.theta_grid)

    support_theta_grid_np: np.ndarray | None = None
    log_p_lik_val_t: jax.Array | None = None
    if config.model_cls.__name__ != "NormalNormalModel":
        from ._losses_compose import (
            compute_log_p_lik_grid_np,
            precompute_generic_grids,
        )
        # Representative model + prior at midpoint of hp ranges.
        prior_midpoint = {
            n: 0.5 * (s.low + s.high)
            for n, s in config.hyperparam_distribution.prior_specs.items()
        }
        lik_midpoint = {
            n: 0.5 * (s.low + s.high)
            for n, s in config.hyperparam_distribution.lik_specs.items()
        }
        rep_prior_hp = np.array(
            [prior_midpoint[n] for n in prior_names], dtype=np.float64
        )
        rep_lik_hp = np.array(
            [lik_midpoint[n] for n in lik_names], dtype=np.float64
        )
        rep_prior = config.prior_cls.from_hyperparams(rep_prior_hp)
        rep_model = config.model_cls.from_hyperparams(rep_lik_hp)
        support_theta_grid_t, _ = precompute_generic_grids(rep_model, rep_prior)
        support_theta_grid_np = np.asarray(support_theta_grid_t)
        log_p_lik_val_np = compute_log_p_lik_grid_np(
            rep_model, D_val_np, support_theta_grid_np
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
        prior_hp_held=prior_hp_held,
        lik_hp_held=lik_hp_held,
        valid_held=valid_held,
        D_val_t=D_val_t,
        prior_hp_val_t=prior_hp_val_t,
        lik_hp_val_t=lik_hp_val_t,
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
        device=device_resolved,
        verbose=verbose,
        support_theta_grid_np=support_theta_grid_np,
        log_p_lik_val_t=log_p_lik_val_t,
    )
    out = run_epoch_loop(args)

    eta_net = out.best_eta_net if out.best_eta_net is not None else args.eta_net
    val_net = out.best_val_net if out.best_val_net is not None else args.val_net

    final_head_b_acc = evaluate_head_b_accuracy(
        val_net, theta_held, eta_held_aux, prior_hp_held, lik_hp_held,
        valid_held, device_resolved,
    )
    final_eta_pred_valid_rate = compute_final_eta_pred_valid_rate(
        eta_net=eta_net, scheme=scheme,
        theta_held=theta_held, D_held=D_held,
        prior_hp_held=prior_hp_held, lik_hp_held=lik_hp_held,
        config=config, device=device_resolved,
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
        antithetic=False,
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
