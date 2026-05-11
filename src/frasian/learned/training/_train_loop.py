"""Per-epoch training loop for the Phase G conditional dual-head selector.

Per-batch hyperparam threading:
- Each minibatch step samples (prior_hp, lik_hp) per element from
  ``config.hyperparam_distribution``.
- The conditional EtaNet/ValidityNet receive the per-batch hyperparams
  as additional input blocks.

Training step (per minibatch):
1. Sample per-batch (prior_hp, lik_hp).
2. Forward Head A + collect per-element validity labels (numpy-side).
3. Train Head B BCE on (θ, prior_hp, lik_hp, η, valid).
4. Train Head A width loss + λ · boundary penalty.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from ._losses_compose import (
    beta_schedule,
    compose_boundary_penalty,
    compose_width_loss,
    compute_log_p_lik_grid_np,
    decay_schedule,
    lambda_schedule,
    precompute_generic_grids,
)
from ._validity_data import collect_validity_batch
from .architecture import EtaNet, ValidityNet
from .losses import anti_wald_penalty, eta_collapse_penalty
from .sampling import ExperimentConfig, anchor_theta_to_prior

if TYPE_CHECKING:
    from .diagnostics import ProbeBatch

_FORCE_X64 = _x64


def _resolve_n_mc_train() -> int:
    import os

    raw = os.environ.get("FRASIAN_N_MC_TRAIN", "8")
    try:
        v = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"FRASIAN_N_MC_TRAIN must be a positive int; got {raw!r}."
        ) from exc
    if v < 1:
        raise ValueError(f"FRASIAN_N_MC_TRAIN must be >= 1; got {v}.")
    return v


N_MC_TRAIN: int = _resolve_n_mc_train()


@eqx.filter_jit
def _eta_forward_jit(
    net: Any, theta: jax.Array, prior_hp: jax.Array, lik_hp: jax.Array,
) -> jax.Array:
    return net(theta, prior_hp, lik_hp)


@eqx.filter_jit
def _val_forward_jit(
    net: Any, theta: jax.Array, prior_hp: jax.Array, lik_hp: jax.Array, eta: jax.Array,
) -> jax.Array:
    return net(theta, prior_hp, lik_hp, eta)


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
    best_eta_net: Any = None
    best_val_net: Any = None
    epochs_run: int = 0
    stopped_early: bool = False
    # Diagnostic fields (populated when probe_batch is passed in):
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _EpochAggregates:
    train_loss: float = 0.0
    width_loss: float = 0.0
    penalty_loss: float = 0.0
    steps_taken: int = 0
    aux_valid_rate_sum: float = 0.0


@dataclass
class LoopArgs:
    """All inputs to ``run_epoch_loop``."""

    eta_net: EtaNet
    val_net: ValidityNet
    optimizer_a: optax.GradientTransformation
    optimizer_b: optax.GradientTransformation
    opt_state_a: Any
    opt_state_b: Any
    theta_train: np.ndarray
    theta_held: np.ndarray
    eta_held_aux: np.ndarray
    prior_hp_held: np.ndarray
    lik_hp_held: np.ndarray
    valid_held: np.ndarray
    D_val_t: jax.Array
    prior_hp_val_t: jax.Array
    lik_hp_val_t: jax.Array
    theta_grid_t: jax.Array
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
    device: str
    verbose: bool
    support_theta_grid_np: np.ndarray | None = None
    log_p_lik_val_t: jax.Array | None = None
    # Phase G diagnostic regularizers (opt-in; default 0 = off)
    anti_wald_max: float = 0.0
    anti_collapse_max: float = 0.0
    anti_decay_frac: float = 0.5
    # Optional held-out probe batch for per-epoch D1-D4 diagnostics.
    # When non-None, _epoch_iteration computes diagnostics after each
    # epoch's update and appends them to ``out.diagnostics``.
    probe_batch: "ProbeBatch | None" = None


def evaluate_head_b_accuracy(
    val_net: ValidityNet,
    theta_held: np.ndarray,
    eta_held: np.ndarray,
    prior_hp_held: np.ndarray,
    lik_hp_held: np.ndarray,
    valid_held: np.ndarray,
    device: str,
) -> float:
    """Held-out classification accuracy of Head B at threshold 0.5."""
    logits = _val_forward_jit(
        val_net,
        jnp.asarray(theta_held),
        jnp.asarray(prior_hp_held),
        jnp.asarray(lik_hp_held),
        jnp.asarray(eta_held),
    )
    pred = (jax.nn.sigmoid(logits) >= 0.5)
    pred_np = np.asarray(pred)
    return float((pred_np == valid_held).mean())


def compute_final_eta_pred_valid_rate(
    *,
    eta_net: EtaNet,
    scheme: Any,
    theta_held: np.ndarray,
    D_held: np.ndarray,
    prior_hp_held: np.ndarray,
    lik_hp_held: np.ndarray,
    config: ExperimentConfig,
    device: str,
) -> float:
    """Empirical validity rate of Head A's predicted η on held-out θ."""
    from .validity import compute_pvalues_per_sample_with_hp, validity_mask

    eta_pred_held_t = _eta_forward_jit(
        eta_net,
        jnp.asarray(theta_held),
        jnp.asarray(prior_hp_held),
        jnp.asarray(lik_hp_held),
    )
    eta_pred_held = np.asarray(eta_pred_held_t, dtype=np.float64)
    p_pred = compute_pvalues_per_sample_with_hp(
        scheme,
        theta_held,
        D_held,
        config.prior_cls,
        config.model_cls,
        prior_hp_held,
        lik_hp_held,
        eta_pred_held,
        config.statistic_name,
    )
    return float(validity_mask(p_pred).mean())


def _make_step_fns(
    config: ExperimentConfig,
    loss_kind: str,
    alpha: float | None,
    use_beta: bool,
) -> tuple[Callable, Callable]:
    """Build the (head_b_grad, head_a_grad) jit-compiled step functions."""

    @eqx.filter_jit
    def head_b_step(
        val_net: ValidityNet,
        theta_all_t: jax.Array,
        prior_hp_all_t: jax.Array,
        lik_hp_all_t: jax.Array,
        eta_all_t: jax.Array,
        valid_all_t: jax.Array,
    ) -> tuple[jax.Array, ValidityNet]:
        def loss_fn(vn: ValidityNet) -> jax.Array:
            logits = vn(theta_all_t, prior_hp_all_t, lik_hp_all_t, eta_all_t)
            return optax.sigmoid_binary_cross_entropy(logits, valid_all_t).mean()
        return eqx.filter_value_and_grad(loss_fn)(val_net)

    # Generic-grid path: representative (rep_model, rep_prior) used for
    # support + log_p_prior. For NN, these are unused.
    is_generic = config.model_cls.__name__ != "NormalNormalModel"
    if is_generic:
        # Build a representative (model, prior) at the midpoint of the
        # hyperparam ranges. The support is constant for models with
        # bounded fixed support; the log_p_prior is approximated at the
        # midpoint.
        prior_midpoint = {
            n: 0.5 * (s.low + s.high)
            for n, s in config.hyperparam_distribution.prior_specs.items()
        }
        lik_midpoint = {
            n: 0.5 * (s.low + s.high)
            for n, s in config.hyperparam_distribution.lik_specs.items()
        }
        rep_prior_hp = np.array(
            [prior_midpoint[n] for n in config.prior_cls.hyperparam_names()],
            dtype=np.float64,
        )
        rep_lik_hp = np.array(
            [lik_midpoint[n] for n in config.model_cls.hyperparam_names()],
            dtype=np.float64,
        )
        rep_prior = config.prior_cls.from_hyperparams(rep_prior_hp)
        rep_model = config.model_cls.from_hyperparams(rep_lik_hp)
        support_theta_grid_t, log_p_prior_grid_t = precompute_generic_grids(
            rep_model, rep_prior,
        )
    else:
        support_theta_grid_t: jax.Array | None = None
        log_p_prior_grid_t: jax.Array | None = None

    @eqx.filter_jit
    def head_a_step(
        eta_net: EtaNet,
        val_net: ValidityNet,
        theta_grid_t: jax.Array,
        D_batch_t: jax.Array,
        theta_batch_t: jax.Array,
        prior_hp_batch_t: jax.Array,
        lik_hp_batch_t: jax.Array,
        lam: jax.Array,
        beta: jax.Array,
        lam_anti_wald: jax.Array,
        lam_anti_collapse: jax.Array,
        log_p_lik_grid_t: jax.Array | None = None,
    ) -> tuple[tuple[jax.Array, tuple[jax.Array, jax.Array]], EtaNet]:
        def loss_fn(en: EtaNet) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            beta_pass = beta if use_beta else None
            loss_width = compose_width_loss(
                eta_net=en,
                theta_grid_t=theta_grid_t,
                D_batch_t=D_batch_t,
                config=config,
                prior_hp_batch_t=prior_hp_batch_t,
                lik_hp_batch_t=lik_hp_batch_t,
                loss_kind=loss_kind,
                alpha=alpha,
                beta=beta_pass,
                log_p_lik_grid_t=log_p_lik_grid_t,
                support_theta_grid_t=support_theta_grid_t,
                log_p_prior_grid_t=log_p_prior_grid_t,
                val_net=val_net,
            )
            eta_pred = en(theta_batch_t, prior_hp_batch_t, lik_hp_batch_t)
            penalty = compose_boundary_penalty(
                val_net=val_net,
                theta_batch_t=theta_batch_t,
                prior_hp_batch_t=prior_hp_batch_t,
                lik_hp_batch_t=lik_hp_batch_t,
                eta_pred=eta_pred,
            )
            penalty_aw = anti_wald_penalty(eta_pred)
            penalty_ac = eta_collapse_penalty(eta_pred)
            total = (
                loss_width
                + lam * penalty
                + lam_anti_wald * penalty_aw
                + lam_anti_collapse * penalty_ac
            )
            return total, (loss_width, penalty)
        return eqx.filter_value_and_grad(loss_fn, has_aux=True)(eta_net)

    return head_b_step, head_a_step


def _make_eval_fn(
    config: ExperimentConfig,
    loss_kind: str,
    alpha: float | None,
    use_beta: bool,
) -> Callable:
    is_generic = config.model_cls.__name__ != "NormalNormalModel"
    if is_generic:
        prior_midpoint = {
            n: 0.5 * (s.low + s.high)
            for n, s in config.hyperparam_distribution.prior_specs.items()
        }
        lik_midpoint = {
            n: 0.5 * (s.low + s.high)
            for n, s in config.hyperparam_distribution.lik_specs.items()
        }
        rep_prior_hp = np.array(
            [prior_midpoint[n] for n in config.prior_cls.hyperparam_names()],
            dtype=np.float64,
        )
        rep_lik_hp = np.array(
            [lik_midpoint[n] for n in config.model_cls.hyperparam_names()],
            dtype=np.float64,
        )
        rep_prior = config.prior_cls.from_hyperparams(rep_prior_hp)
        rep_model = config.model_cls.from_hyperparams(rep_lik_hp)
        support_theta_grid_t, log_p_prior_grid_t = precompute_generic_grids(
            rep_model, rep_prior,
        )
    else:
        support_theta_grid_t: jax.Array | None = None
        log_p_prior_grid_t: jax.Array | None = None

    @eqx.filter_jit
    def eval_loss(
        eta_net: EtaNet,
        val_net: ValidityNet,
        theta_grid_t: jax.Array,
        D_val_t: jax.Array,
        prior_hp_val_t: jax.Array,
        lik_hp_val_t: jax.Array,
        beta: jax.Array,
        log_p_lik_grid_t: jax.Array | None = None,
    ) -> jax.Array:
        beta_pass = beta if use_beta else None
        return compose_width_loss(
            eta_net=eta_net,
            theta_grid_t=theta_grid_t,
            D_batch_t=D_val_t,
            config=config,
            prior_hp_batch_t=prior_hp_val_t,
            lik_hp_batch_t=lik_hp_val_t,
            loss_kind=loss_kind,
            alpha=alpha,
            beta=beta_pass,
            log_p_lik_grid_t=log_p_lik_grid_t,
            support_theta_grid_t=support_theta_grid_t,
            log_p_prior_grid_t=log_p_prior_grid_t,
            val_net=val_net,
        )

    return eval_loss


def _training_step(
    args: LoopArgs,
    *,
    head_b_step: Callable,
    head_a_step: Callable,
    theta_batch_np: np.ndarray,
    beta: jax.Array,
    lam: jax.Array,
    lam_anti_wald: jax.Array,
    lam_anti_collapse: jax.Array,
    step_idx: int,
) -> tuple[float, float, float, float] | None:
    """Run one minibatch step (Phase G conditional)."""
    config = args.config
    prior_names = config.prior_cls.hyperparam_names()
    lik_names = config.model_cls.hyperparam_names()

    prior_hp_batch_np, lik_hp_batch_np = config.hyperparam_distribution.sample(
        len(theta_batch_np), args.rng_train,
        prior_names=prior_names, lik_names=lik_names,
    )
    # Convert relative θ → absolute when theta_distribution is anchored.
    theta_batch_np = anchor_theta_to_prior(
        theta_batch_np, prior_hp_batch_np, prior_names, config.theta_distribution,
    )

    theta_batch_t = jnp.asarray(theta_batch_np)
    prior_hp_batch_t = jnp.asarray(prior_hp_batch_np)
    lik_hp_batch_t = jnp.asarray(lik_hp_batch_np)

    eta_pred = _eta_forward_jit(
        args.eta_net, theta_batch_t, prior_hp_batch_t, lik_hp_batch_t,
    )
    theta_all_t, prior_hp_all_t, lik_hp_all_t, eta_all_t, valid_all_t = (
        collect_validity_batch(
            eta_pred=eta_pred,
            theta_batch_np=theta_batch_np,
            prior_hp_batch_np=prior_hp_batch_np,
            lik_hp_batch_np=lik_hp_batch_np,
            config=config,
            scheme=args.scheme,
            n_aux=args.n_aux,
            rng=args.rng_aux,
        )
    )
    aux_valid_rate = float(np.asarray(valid_all_t)[len(theta_batch_np):].mean())

    loss_b, grads_b = head_b_step(
        args.val_net, theta_all_t, prior_hp_all_t, lik_hp_all_t,
        eta_all_t, valid_all_t,
    )
    updates_b, args.opt_state_b = args.optimizer_b.update(
        grads_b, args.opt_state_b, args.val_net,
    )
    args.val_net = eqx.apply_updates(args.val_net, updates_b)

    n_mc = min(N_MC_TRAIN, len(theta_batch_np))
    D_batch_np = config.model_cls.sample_data_batch_with_hp(
        theta_batch_np[:n_mc], lik_hp_batch_np[:n_mc], args.rng_train,
        n_data=config.n_data,
    )
    if D_batch_np.ndim == 2 and D_batch_np.shape[1] == 1:
        D_batch_np = D_batch_np[:, 0]
    D_batch_t = jnp.asarray(D_batch_np)

    log_p_lik_grid_t: jax.Array | None = None
    if config.model_cls.__name__ != "NormalNormalModel":
        if args.support_theta_grid_np is None:
            raise RuntimeError(
                "non-NN training expects args.support_theta_grid_np precomputed."
            )
        # Per-element model varies; use representative (first element) since
        # batched_loglik_grid uses sufficient stats independent of individual
        # model state for the generic + power_law case.
        rep_model = config.model_cls.from_hyperparams(lik_hp_batch_np[0])
        log_p_lik_grid_np = compute_log_p_lik_grid_np(
            rep_model, D_batch_np, args.support_theta_grid_np,
        )
        log_p_lik_grid_t = jnp.asarray(log_p_lik_grid_np)

    try:
        prior_hp_for_loss_t = jnp.asarray(prior_hp_batch_np[:n_mc])
        lik_hp_for_loss_t = jnp.asarray(lik_hp_batch_np[:n_mc])
        (loss_a, (loss_width, penalty)), grads_a = head_a_step(
            args.eta_net,
            args.val_net,
            args.theta_grid_t,
            D_batch_t,
            theta_batch_t[:n_mc],
            prior_hp_for_loss_t,
            lik_hp_for_loss_t,
            lam,
            beta,
            lam_anti_wald,
            lam_anti_collapse,
            log_p_lik_grid_t,
        )
    except (ValueError, RuntimeError, ArithmeticError) as e:
        if args.verbose:
            warnings.warn(
                f"[width loss] step {step_idx} skipped: {e}",
                RuntimeWarning, stacklevel=2,
            )
        return None

    if not bool(jnp.isfinite(loss_a)):
        if args.verbose:
            warnings.warn(
                f"[width loss] step {step_idx} produced non-finite loss; skipping",
                RuntimeWarning, stacklevel=2,
            )
        return None

    updates_a, args.opt_state_a = args.optimizer_a.update(
        grads_a, args.opt_state_a, args.eta_net,
    )
    args.eta_net = eqx.apply_updates(args.eta_net, updates_a)

    return float(loss_a), float(loss_width), float(penalty), aux_valid_rate


def _run_epoch_steps(
    args: LoopArgs,
    *,
    head_b_step: Callable,
    head_a_step: Callable,
    n_train: int,
    steps_per_epoch: int,
    beta: jax.Array,
    lam: jax.Array,
    lam_anti_wald: jax.Array,
    lam_anti_collapse: jax.Array,
) -> _EpochAggregates:
    ep_perm = args.rng_train.permutation(n_train)
    agg = _EpochAggregates()
    for step in range(steps_per_epoch):
        idx = ep_perm[step * args.batch_size : (step + 1) * args.batch_size]
        if idx.size == 0:
            continue
        metrics = _training_step(
            args,
            head_b_step=head_b_step,
            head_a_step=head_a_step,
            theta_batch_np=args.theta_train[idx],
            beta=beta,
            lam=lam,
            lam_anti_wald=lam_anti_wald,
            lam_anti_collapse=lam_anti_collapse,
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
    args: LoopArgs, eval_fn: Callable, beta: jax.Array
) -> tuple[float, float]:
    try:
        v_loss = float(
            eval_fn(
                args.eta_net,
                args.val_net,
                args.theta_grid_t,
                args.D_val_t,
                args.prior_hp_val_t,
                args.lik_hp_val_t,
                beta,
                args.log_p_lik_val_t,
            )
        )
    except (ValueError, RuntimeError, ArithmeticError):
        v_loss = float("inf")
    if not np.isfinite(v_loss):
        v_loss = float("inf")
    head_b_acc = evaluate_head_b_accuracy(
        args.val_net, args.theta_held, args.eta_held_aux,
        args.prior_hp_held, args.lik_hp_held, args.valid_held, args.device,
    )
    return v_loss, head_b_acc


def _maybe_warn_class_degenerate(
    epoch: int, mean_aux_rate: float, model_kind: str = "normal_normal"
) -> None:
    if model_kind != "normal_normal":
        return
    if 0.05 <= mean_aux_rate <= 0.95:
        return
    warnings.warn(
        f"[head B] epoch {epoch + 1} aux validity rate = "
        f"{mean_aux_rate:.3f} (outside (0.05, 0.95)). Head B's BCE is "
        f"class-degenerate.",
        RuntimeWarning,
        stacklevel=2,
    )


def _maybe_log_epoch(
    *, epoch: int, n_epochs: int, out: EpochLoopOutputs,
    v_loss: float, head_b_acc: float, mean_aux_rate: float,
    lam: float, beta_val: float | None, verbose: bool,
) -> None:
    if not verbose:
        return
    if not ((epoch + 1) % max(n_epochs // 10, 1) == 0 or epoch == 0):
        return
    beta_str = "n/a" if beta_val is None else f"{beta_val:.0f}"
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


def _epoch_iteration(
    args: LoopArgs,
    out: EpochLoopOutputs,
    *,
    head_b_step: Callable,
    head_a_step: Callable,
    eval_fn: Callable,
    epoch: int,
    n_train: int,
    steps_per_epoch: int,
) -> bool:
    out.epochs_run = epoch + 1
    lam_val = lambda_schedule(epoch, args.n_epochs, args.lambda_max, args.lambda_warmup_frac)
    lam = jnp.asarray(lam_val)
    lam_aw_val = decay_schedule(
        epoch, args.n_epochs, args.anti_wald_max, args.anti_decay_frac,
    )
    lam_ac_val = decay_schedule(
        epoch, args.n_epochs, args.anti_collapse_max, args.anti_decay_frac,
    )
    lam_anti_wald = jnp.asarray(lam_aw_val)
    lam_anti_collapse = jnp.asarray(lam_ac_val)
    if args.loss_kind == "static_width":
        beta_val = beta_schedule(epoch, args.n_epochs)
        beta = jnp.asarray(beta_val)
    else:
        beta_val = None
        beta = jnp.asarray(0.0)

    agg = _run_epoch_steps(
        args,
        head_b_step=head_b_step,
        head_a_step=head_a_step,
        n_train=n_train,
        steps_per_epoch=steps_per_epoch,
        beta=beta,
        lam=lam,
        lam_anti_wald=lam_anti_wald,
        lam_anti_collapse=lam_anti_collapse,
    )
    denom = max(agg.steps_taken, 1)
    out.train_losses.append(agg.train_loss / denom)
    out.train_width_losses.append(agg.width_loss / denom)
    out.train_penalty_losses.append(agg.penalty_loss / denom)
    mean_aux_rate = agg.aux_valid_rate_sum / denom
    model_kind = (
        "normal_normal"
        if args.config.model_cls.__name__ == "NormalNormalModel"
        else "generic"
    )
    _maybe_warn_class_degenerate(epoch, mean_aux_rate, model_kind=model_kind)

    v_loss, head_b_acc = _evaluate_epoch(args, eval_fn, beta)
    out.val_losses.append(v_loss)
    out.head_b_accuracies.append(head_b_acc)

    # Diagnostic computation (only when probe_batch supplied).
    # args.eta_net here reflects the POST-update state for this epoch:
    # _training_step assigns args.eta_net = eqx.apply_updates(...) after
    # each minibatch.
    if args.probe_batch is not None:
        from .diagnostics import compute_epoch_diagnostics
        out.diagnostics.append(compute_epoch_diagnostics(
            args.eta_net, args.probe_batch,
            scheme_name=args.config.scheme_name,
            statistic_name=args.config.statistic_name,
            epoch=epoch + 1,
            train_loss=out.train_losses[-1],
            val_loss=v_loss,
        ))

    improved = v_loss < out.best_val - args.min_delta
    if improved:
        out.best_val = v_loss
        out.best_epoch = epoch
        out.best_eta_net = jax.tree.map(
            lambda x: jnp.copy(x) if isinstance(x, jax.Array) else x, args.eta_net
        )
        out.best_val_net = jax.tree.map(
            lambda x: jnp.copy(x) if isinstance(x, jax.Array) else x, args.val_net
        )

    _maybe_log_epoch(
        epoch=epoch, n_epochs=args.n_epochs, out=out,
        v_loss=v_loss, head_b_acc=head_b_acc, mean_aux_rate=mean_aux_rate,
        lam=lam_val, beta_val=beta_val, verbose=args.verbose,
    )
    return improved


def run_epoch_loop(args: LoopArgs) -> EpochLoopOutputs:
    """Run all epochs; track best val state; early-stop on patience."""
    is_generic = args.config.model_cls.__name__ != "NormalNormalModel"
    if is_generic and args.support_theta_grid_np is None:
        raise ValueError(
            "run_epoch_loop: non-NN config requires "
            "args.support_theta_grid_np precomputed."
        )
    if not is_generic and args.support_theta_grid_np is not None:
        raise ValueError(
            "run_epoch_loop: NN config does not use args.support_theta_grid_np."
        )

    n_train = len(args.theta_train)
    steps_per_epoch = max(1, n_train // args.batch_size)
    use_beta = args.loss_kind == "static_width"
    head_b_step, head_a_step = _make_step_fns(
        args.config, args.loss_kind, args.alpha, use_beta=use_beta
    )
    eval_fn = _make_eval_fn(args.config, args.loss_kind, args.alpha, use_beta=use_beta)

    out = EpochLoopOutputs(
        best_eta_net=jax.tree.map(
            lambda x: jnp.copy(x) if isinstance(x, jax.Array) else x, args.eta_net
        ),
        best_val_net=jax.tree.map(
            lambda x: jnp.copy(x) if isinstance(x, jax.Array) else x, args.val_net
        ),
    )
    epochs_since_best = 0
    for epoch in range(args.n_epochs):
        improved = _epoch_iteration(
            args, out,
            head_b_step=head_b_step,
            head_a_step=head_a_step,
            eval_fn=eval_fn,
            epoch=epoch, n_train=n_train, steps_per_epoch=steps_per_epoch,
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
