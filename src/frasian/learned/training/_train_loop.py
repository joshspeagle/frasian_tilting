"""Per-epoch training loop for the Phase E dual-head selector (JAX/Equinox port).

The orchestrator builds nets / RNGs / optimisers / held-out sets;
``run_epoch_loop`` here owns the minibatch / optimiser / per-step
pattern.

Training step (per minibatch):
1. Forward Head A + collect validity labels (mix aux batch so Head B
   sees both classes every step). Validity labels are computed in
   numpy via ``scheme.tilted_pvalue``; only the JAX boundary needs
   the per-step ``jnp.asarray``.
2. Train Head B (BCE on (θ, η, valid)).
3. Train Head A (width loss + λ · boundary penalty).

Each gradient computation is wrapped in ``@eqx.filter_jit`` for
fused-kernel speed; the numpy-side validity collection and the
optax update step run eagerly between jit-compiled forwards.
``compose_width_loss`` carries a string ``loss_kind`` and a Python
``alpha`` plus the ``ExperimentConfig`` dataclass — none of those
are valid JAX-array leaves, so we close over them via a per-epoch
factory rather than passing them through the jit boundary.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

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
    lambda_schedule,
    precompute_generic_grids,
)
from ._validity_data import (
    collect_validity_batch,
    sample_data_per_theta,
    validity_net_inputs,
)
from .architecture import EtaNet, ValidityNet
from .sampling import ExperimentConfig

_FORCE_X64 = _x64

# Number of D draws per training step for the width-loss MC average
# (skeptic block #1).
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
    best_eta_net: Any = None
    best_val_net: Any = None
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
    signatures small.
    """

    eta_net: EtaNet
    val_net: ValidityNet
    optimizer_a: optax.GradientTransformation
    optimizer_b: optax.GradientTransformation
    opt_state_a: Any
    opt_state_b: Any
    theta_train: np.ndarray
    theta_held: np.ndarray
    eta_held_aux: np.ndarray
    valid_held: np.ndarray
    D_val_t: jax.Array
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
    antithetic: bool
    device: str
    verbose: bool
    # Pre-computed integration grid for the Phase 4 generic tilted-pvalue
    # path (Bernoulli + any non-Normal-Normal model). None for Normal-
    # Normal experiments — those use the closed-form fast path that
    # ignores the grids.
    support_theta_grid_np: np.ndarray | None = None
    log_p_lik_val_t: jax.Array | None = None


def evaluate_head_b_accuracy(
    val_net: ValidityNet,
    theta_held: np.ndarray,
    eta_held: np.ndarray,
    valid_held: np.ndarray,
    device: str,
) -> float:
    """Held-out classification accuracy of Head B at threshold 0.5."""
    inputs = validity_net_inputs(jnp.asarray(theta_held), jnp.asarray(eta_held))
    logits = val_net(inputs)
    pred = (jax.nn.sigmoid(logits) >= 0.5)
    pred_np = np.asarray(pred)
    return float((pred_np == valid_held).mean())


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

    eta_pred_held_t = eta_net(jnp.asarray(theta_held))
    eta_pred_held = np.asarray(eta_pred_held_t, dtype=np.float64)
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


# --------------------------------------------------------------------------
# Per-epoch jit-step factory
# --------------------------------------------------------------------------


def _make_step_fns(
    config: ExperimentConfig,
    loss_kind: str,
    alpha: float | None,
    use_beta: bool,
) -> tuple[Callable, Callable]:
    """Build the (head_b_grad, head_a_grad) jit-compiled step functions
    for the current ``(config, loss_kind, alpha)`` combination.

    Closing over the static config / loss_kind / alpha avoids passing
    them through the jit boundary as PyTree leaves (they are not
    JAX arrays and cannot be hashed cheaply for jit's static_argnums).

    Phase 4 skeptic #4 (closure-stale-config defense):
    ``ExperimentConfig`` is ``frozen=True`` so direct mutation raises
    ``FrozenInstanceError``. The closure here captures ``config`` by
    reference; since the same ``config`` is used to build ``LoopArgs``
    in ``fit_eta_artifact`` (lockstep construction), there is no path
    in the orchestrator where the closure and ``args.config`` can
    desync. ``run_epoch_loop`` asserts the lockstep at entry as a
    defensive guard against external callers that build step fns +
    LoopArgs separately.
    """

    @eqx.filter_jit
    def head_b_step(
        val_net: ValidityNet,
        theta_all_t: jax.Array,
        eta_all_t: jax.Array,
        valid_all_t: jax.Array,
    ) -> tuple[jax.Array, ValidityNet]:
        """Compute (loss_b, grad_b) for the BCE step."""
        def loss_fn(vn: ValidityNet) -> jax.Array:
            logits = vn(validity_net_inputs(theta_all_t, eta_all_t))
            return optax.sigmoid_binary_cross_entropy(logits, valid_all_t).mean()

        return eqx.filter_value_and_grad(loss_fn)(val_net)

    # For non-Normal-Normal experiments, precompute the integration
    # grid + log-prior grid once per training run. JAX cannot trace
    # through `model.support()` / `prior.logpdf` (Python state), so
    # they live in the closure of the jit'd kernel. Per-step
    # log_p_lik_grid is computed eagerly in `_training_step` and
    # passed through as a jax.Array.
    model_kind = config.model.fingerprint()[0]
    if model_kind == "normal_normal":
        support_theta_grid_t: jax.Array | None = None
        log_p_prior_grid_t: jax.Array | None = None
    else:
        support_theta_grid_t, log_p_prior_grid_t = precompute_generic_grids(
            config.model, config.prior
        )

    @eqx.filter_jit
    def head_a_step(
        eta_net: EtaNet,
        val_net: ValidityNet,
        theta_grid_t: jax.Array,
        D_batch_t: jax.Array,
        theta_batch_t: jax.Array,
        lam: jax.Array,
        beta: jax.Array,
        log_p_lik_grid_t: jax.Array | None = None,
    ) -> tuple[tuple[jax.Array, tuple[jax.Array, jax.Array]], EtaNet]:
        """Compute (loss_a, (loss_width, penalty)), grad_a."""
        def loss_fn(en: EtaNet) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            beta_pass = beta if use_beta else None
            loss_width = compose_width_loss(
                eta_net=en,
                theta_grid_t=theta_grid_t,
                D_batch_t=D_batch_t,
                config=config,
                loss_kind=loss_kind,
                alpha=alpha,
                beta=beta_pass,
                log_p_lik_grid_t=log_p_lik_grid_t,
                support_theta_grid_t=support_theta_grid_t,
                log_p_prior_grid_t=log_p_prior_grid_t,
            )
            eta_pred = en(theta_batch_t)
            penalty = compose_boundary_penalty(
                val_net=val_net,
                theta_batch_t=theta_batch_t,
                eta_pred=eta_pred,
            )
            return loss_width + lam * penalty, (loss_width, penalty)

        return eqx.filter_value_and_grad(loss_fn, has_aux=True)(eta_net)

    return head_b_step, head_a_step


def _make_eval_fn(
    config: ExperimentConfig,
    loss_kind: str,
    alpha: float | None,
    use_beta: bool,
) -> Callable:
    """Jit'd held-out width-loss evaluator (closure over static fields)."""

    model_kind = config.model.fingerprint()[0]
    if model_kind == "normal_normal":
        support_theta_grid_t: jax.Array | None = None
        log_p_prior_grid_t: jax.Array | None = None
    else:
        support_theta_grid_t, log_p_prior_grid_t = precompute_generic_grids(
            config.model, config.prior
        )

    @eqx.filter_jit
    def eval_loss(
        eta_net: EtaNet,
        theta_grid_t: jax.Array,
        D_val_t: jax.Array,
        beta: jax.Array,
        log_p_lik_grid_t: jax.Array | None = None,
    ) -> jax.Array:
        beta_pass = beta if use_beta else None
        return compose_width_loss(
            eta_net=eta_net,
            theta_grid_t=theta_grid_t,
            D_batch_t=D_val_t,
            config=config,
            loss_kind=loss_kind,
            alpha=alpha,
            beta=beta_pass,
            log_p_lik_grid_t=log_p_lik_grid_t,
            support_theta_grid_t=support_theta_grid_t,
            log_p_prior_grid_t=log_p_prior_grid_t,
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
    step_idx: int,
) -> tuple[float, float, float, float] | None:
    """Run one minibatch step. Returns metrics or None if the step is skipped."""
    theta_batch_t = jnp.asarray(theta_batch_np)

    # Step (1): forward + collect validity batch (numpy-driven labels).
    eta_pred = args.eta_net(theta_batch_t)
    theta_all_t, eta_all_t, valid_all_t = collect_validity_batch(
        eta_pred=eta_pred,
        theta_batch_np=theta_batch_np,
        config=args.config,
        scheme=args.scheme,
        n_aux=args.n_aux,
        rng=args.rng_aux,
    )
    aux_valid_rate = float(np.asarray(valid_all_t)[len(theta_batch_np):].mean())

    # Step (2): Head B BCE step.
    loss_b, grads_b = head_b_step(args.val_net, theta_all_t, eta_all_t, valid_all_t)
    updates_b, args.opt_state_b = args.optimizer_b.update(
        grads_b, args.opt_state_b, args.val_net
    )
    args.val_net = eqx.apply_updates(args.val_net, updates_b)

    # Step (3): Head A width + boundary step.
    n_mc = min(N_MC_TRAIN, len(theta_batch_np))
    D_batch_np = sample_data_per_theta(
        args.config.model, theta_batch_np[:n_mc], args.rng_train,
        antithetic=args.antithetic, n_data=args.config.n_data,
    )
    D_batch_t = jnp.asarray(D_batch_np)

    # For non-NN models, compute the per-step log-likelihood grid
    # numpy-side. (model.likelihood is a Python factory; JAX cannot
    # trace through `BernoulliLikelihood(int(arr.sum()), ...)`.)
    log_p_lik_grid_t: jax.Array | None = None
    if args.config.model.fingerprint()[0] != "normal_normal":
        if args.support_theta_grid_np is None:
            raise RuntimeError(
                "non-Normal-Normal training expects "
                "args.support_theta_grid_np to be precomputed."
            )
        log_p_lik_grid_np = compute_log_p_lik_grid_np(
            args.config.model, D_batch_np, args.support_theta_grid_np
        )
        log_p_lik_grid_t = jnp.asarray(log_p_lik_grid_np)

    try:
        (loss_a, (loss_width, penalty)), grads_a = head_a_step(
            args.eta_net,
            args.val_net,
            args.theta_grid_t,
            D_batch_t,
            theta_batch_t,
            lam,
            beta,
            log_p_lik_grid_t,
        )
    except (ValueError, RuntimeError) as e:
        if args.verbose:
            warnings.warn(
                f"[width loss] step {step_idx} skipped: {e}", RuntimeWarning, stacklevel=2
            )
        return None

    if not bool(jnp.isfinite(loss_a)):
        if args.verbose:
            warnings.warn(
                f"[width loss] step {step_idx} produced non-finite loss; skipping",
                RuntimeWarning,
                stacklevel=2,
            )
        return None

    updates_a, args.opt_state_a = args.optimizer_a.update(
        grads_a, args.opt_state_a, args.eta_net
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
) -> _EpochAggregates:
    """Inner per-step loop. Returns one epoch's aggregates."""
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
    """Compute (val_loss, head_b_accuracy) on the frozen held-out set."""
    try:
        v_loss = float(
            eval_fn(
                args.eta_net,
                args.theta_grid_t,
                args.D_val_t,
                beta,
                args.log_p_lik_val_t,
            )
        )
    except (ValueError, RuntimeError):
        v_loss = float("inf")
    if not np.isfinite(v_loss):
        v_loss = float("inf")
    head_b_acc = evaluate_head_b_accuracy(
        args.val_net, args.theta_held, args.eta_held_aux, args.valid_held, args.device
    )
    return v_loss, head_b_acc


def _maybe_warn_class_degenerate(
    epoch: int, mean_aux_rate: float, model_kind: str = "normal_normal"
) -> None:
    """Warn on class-degenerate Head-B BCE batch.

    Bernoulli + power_law (and other non-NN models with the generic
    grid path) have no closed-form admissibility boundary — virtually
    every (θ, η) in ``eta_explore_box`` produces a finite p-value in
    [0, 1] under MC labelling. ValidityNet has nothing to learn, and
    the boundary penalty becomes a near-no-op. That is the design,
    not a bug — the dual-head architecture is over-specified for
    these models. Skip the warning.
    """
    if model_kind != "normal_normal":
        return
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
    lam: float, beta_val: float | None, verbose: bool,
) -> None:
    """Print one verbose-mode line every ~10 % of epochs (and on epoch 0)."""
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
    """Run one full epoch: train steps + validation + best-state update.

    Returns ``improved`` so the outer loop can update its early-stop counter.
    """
    out.epochs_run = epoch + 1
    lam_val = lambda_schedule(epoch, args.n_epochs, args.lambda_max, args.lambda_warmup_frac)
    lam = jnp.asarray(lam_val)
    if args.loss_kind == "static_width":
        beta_val = beta_schedule(epoch, args.n_epochs)
        beta = jnp.asarray(beta_val)
    else:
        # Pass a tracer-compatible scalar even when unused; the closed-
        # over ``use_beta`` flag in the step fns guards.
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
    )
    denom = max(agg.steps_taken, 1)
    out.train_losses.append(agg.train_loss / denom)
    out.train_width_losses.append(agg.width_loss / denom)
    out.train_penalty_losses.append(agg.penalty_loss / denom)
    mean_aux_rate = agg.aux_valid_rate_sum / denom
    _maybe_warn_class_degenerate(
        epoch, mean_aux_rate, model_kind=args.config.model.fingerprint()[0]
    )

    v_loss, head_b_acc = _evaluate_epoch(args, eval_fn, beta)
    out.val_losses.append(v_loss)
    out.head_b_accuracies.append(head_b_acc)

    improved = v_loss < out.best_val - args.min_delta
    if improved:
        out.best_val = v_loss
        out.best_epoch = epoch
        # Equinox modules are immutable PyTrees; deep-copy array leaves
        # so a later in-place update of args.eta_net cannot mutate the
        # snapshot.
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
    # Phase 4 skeptic #4 (closure-stale-config defense): when
    # ``args.config`` is non-NN the orchestrator (``fit_eta_artifact``)
    # populates ``args.support_theta_grid_np`` from the same config; an
    # external caller building ``LoopArgs`` manually could pass a
    # mismatched precomputed grid. Assert presence consistency at
    # entry — mismatched fingerprints would surface as either a NaN
    # log-prior grid or a dimension mismatch in the per-step
    # ``compute_log_p_lik_grid_np``. The cheap None / shape consistency
    # check below catches the most likely misuse pattern.
    is_generic = args.config.model.fingerprint()[0] != "normal_normal"
    if is_generic and args.support_theta_grid_np is None:
        raise ValueError(
            "run_epoch_loop: non-Normal-Normal config requires "
            "args.support_theta_grid_np precomputed (via "
            "_losses_compose.precompute_generic_grids); got None."
        )
    if not is_generic and args.support_theta_grid_np is not None:
        raise ValueError(
            "run_epoch_loop: Normal-Normal config does not use "
            "args.support_theta_grid_np; got a non-None grid which "
            "suggests the LoopArgs was built from a different config."
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
