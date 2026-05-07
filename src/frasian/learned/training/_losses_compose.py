"""Loss composition for the Phase E dual-head training loop.

Extracted from ``train.py`` (Tier 1.2 §7 split). Three concerns:

1. Width-loss adapter (``compose_width_loss``): wraps the
   JAX tilted-pvalue port + the chosen functional in
   ``losses.py`` (``integrated_pvalue_loss`` / ``cd_variance_loss`` /
   ``static_width_loss``). Currently Normal-Normal only because
   the JAX ports are; the wrapper is the single place that
   adapts ``(model, prior)`` → ``(w, mu0, sigma)``.

2. Boundary penalty (``compose_boundary_penalty``): forwards
   ``(θ, η)`` through a parameter-detached ``ValidityNet`` so
   gradient flows back into Head A but not Head B's weights.
   Implemented via ``eqx.partition`` + ``jax.lax.stop_gradient`` +
   ``eqx.combine`` (the Equinox idiom that replaces the legacy
   ``torch.func.functional_call(detached_params)``).

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

import equinox as eqx
import jax
import jax.numpy as jnp

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from ...models.distributions import NormalDistribution
from ...models.normal_normal import NormalNormalModel
from .architecture import EtaNet, ValidityNet
from .losses import (
    boundary_penalty_from_validity,
    cd_variance_loss,
    integrated_pvalue_loss,
    static_width_loss,
)
from .pvalue_jax import get_jax_tilted_pvalue
from .sampling import ExperimentConfig

_FORCE_X64 = _x64  # keep static-analysis from stripping the import

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
    """Extract (w, mu0, sigma) for the JAX tilted-pvalue ports.

    The JAX ports in ``pvalue_jax.py`` are direct equivalents of the
    legacy torch ports and take Normal-Normal coordinates
    ``(w, mu0, sigma)`` directly. We adapt by deriving them from
    ``(model, prior)`` exactly here — the rest of the training loop
    stays model-agnostic.

    Raises ``NotImplementedError`` for non-Normal-Normal experiments,
    consistent with the documented "model-agnostic in principle,
    Normal-Normal in practice" caveat.
    """
    if not isinstance(model, NormalNormalModel):
        raise NotImplementedError(
            "Phase E training currently requires a NormalNormalModel; "
            f"got {type(model).__name__}. Extending to non-Normal-Normal "
            f"requires registering a generic JAX tilted_pvalue."
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
    # JAX-side `jnp.maximum(denom, 1e-6)` silently distorts. Refuse.
    _W_EPS = 1e-3
    if not (_W_EPS < w < 1.0 - _W_EPS):
        raise ValueError(
            f"data weight w={w:.6f} is outside ({_W_EPS}, "
            f"{1.0 - _W_EPS}); prior.scale={sigma0} and model.sigma="
            f"{sigma} are too far apart for a stable learned-η "
            f"selector. Choose a less degenerate prior."
        )
    return w, mu0, sigma


# Phase 4b: model-kind dispatch for the width-loss adapter.

# Number of grid points used by the generic (grid-based) tilted pvalue
# during TRAINING. Lower than inference (1024) since MC noise dominates
# over grid-discretisation noise at training-time, and per-step cost is
# the dominant wall-time of training. Bernoulli v0_smoke target is ~30 min
# at this setting.
_N_GRID_GENERIC_TRAINING: int = 512


def _call_normal_normal_pvalue(
    *,
    eta_net: EtaNet,
    theta_grid_t: jax.Array,
    D_batch_t: jax.Array,
    model: Any,
    prior: Any,
    statistic_name: str,
    scheme_name: str,
) -> jax.Array:
    """Adapter: extract Normal-Normal (w, mu0, sigma) and call the
    per-scheme JAX kernel (closed-form fast path).

    Output is byte-identical to the pre-Phase-4b path on Normal-Normal —
    same callable, same broadcasting, same JIT cache key. Pinned by
    `test_fit_eta_artifact_byte_equality.py`.
    """
    w, mu0, sigma = extract_normal_normal_params(model, prior)
    w_t = jnp.asarray(w)
    mu0_t = jnp.asarray(mu0)
    sigma_t = jnp.asarray(sigma)

    if D_batch_t.ndim != 1:
        raise NotImplementedError(
            "Normal-Normal closed-form width-loss adapter expects scalar "
            f"D per batch element (D.ndim == 1); got shape "
            f"{tuple(D_batch_t.shape)}. n_data > 1 with a NormalNormalModel "
            "must route through the generic grid path; route via "
            "WIDTH_LOSS_DISPATCH[('<scheme>', 'generic')] instead."
        )

    eta_grid = eta_net(theta_grid_t)  # (G,)
    tilted_pvalue = get_jax_tilted_pvalue(scheme_name, "normal_normal")
    G = theta_grid_t.shape[0]
    B = D_batch_t.shape[0]
    return tilted_pvalue(
        jnp.broadcast_to(theta_grid_t[None, :], (B, G)),
        jnp.broadcast_to(D_batch_t[:, None], (B, G)),
        w_t,
        mu0_t,
        sigma_t,
        jnp.broadcast_to(eta_grid[None, :], (B, G)),
        statistic_name,
    )


def _call_generic_grid_pvalue(
    *,
    eta_net: EtaNet,
    theta_grid_t: jax.Array,
    D_batch_t: jax.Array,
    model: Any,
    prior: Any,
    statistic_name: str,
    scheme_name: str,
) -> jax.Array:
    """Adapter: build grid log-densities from (model, prior, data) and
    call `generic_grid_tilted_pvalue` (Phase 4 generic path).

    The integration grid (`support_theta_grid`) lives on the model's
    parameter support. For bounded supports (Bernoulli's [0, 1]) we
    use the support window directly with an interior padding to avoid
    log(0) at the boundary. For unbounded supports we fall back to
    a posterior-mean ± 6σ heuristic on the first batch element (all
    batch elements share the same support window since the model is
    fixed; only D varies, which only affects log_lik on the grid).
    """
    from .pvalue_jax import generic_grid_tilted_pvalue

    if not hasattr(model, "support"):
        raise NotImplementedError(
            f"_call_generic_grid_pvalue requires `model.support()`; "
            f"got {type(model).__name__!r} without it."
        )
    support_lo, support_hi = model.support()
    n_grid = _N_GRID_GENERIC_TRAINING
    if jnp.isfinite(support_lo) and jnp.isfinite(support_hi):
        # Bounded support — pad by 1% inward to avoid log-density
        # divergence at the boundary on Bernoulli (log(0) = -inf at θ ∈ {0, 1}).
        pad = 0.01 * (float(support_hi) - float(support_lo))
        support_theta_grid = jnp.linspace(
            float(support_lo) + pad, float(support_hi) - pad, n_grid
        )
    else:
        raise NotImplementedError(
            "_call_generic_grid_pvalue: unbounded support fallback not yet "
            "wired (no current Phase 4 consumer needs it)."
        )

    log_p_prior_grid = jnp.asarray(prior.logpdf(support_theta_grid))  # (N_grid,)
    # Vectorise log-likelihood over batch via numpy loop + stack
    # (model.likelihood is a Python factory; jax.vmap can't trace through
    # it without protocol changes). The loop is on B (typically 4-8),
    # negligible relative to the kernel's O(B*G*N_grid) cost.
    log_lik_per_b = []
    for d_b in D_batch_t:
        likelihood_b = model.likelihood(jnp.atleast_1d(d_b))
        log_lik_per_b.append(jnp.asarray(likelihood_b.loglik(support_theta_grid)))
    log_p_lik_grid = jnp.stack(log_lik_per_b, axis=0)  # (B, N_grid)

    eta_grid = eta_net(theta_grid_t)  # (G_test,)
    G = theta_grid_t.shape[0]
    B = D_batch_t.shape[0]
    return generic_grid_tilted_pvalue(
        jnp.broadcast_to(theta_grid_t[None, :], (B, G)),  # (B, G_test)
        jnp.broadcast_to(eta_grid[None, :], (B, G)),  # (B, G_test)
        log_p_lik_grid,                                    # (B, N_grid)
        log_p_prior_grid,                                  # (N_grid,)
        support_theta_grid,                                # (N_grid,)
        statistic_name,
    )


# Width-loss dispatch keyed on (scheme_name, model_kind). Mirrors
# JAX_TILTED_PVALUE but at the adapter level (extractor + kernel call).
# Falls back to ("scheme", "generic") if specific (scheme, model_kind)
# isn't registered — matches `get_jax_tilted_pvalue`'s pattern.
WIDTH_LOSS_DISPATCH: dict[tuple[str, str], Any] = {
    ("power_law", "normal_normal"): _call_normal_normal_pvalue,
    ("ot", "normal_normal"): _call_normal_normal_pvalue,
    ("power_law", "generic"): _call_generic_grid_pvalue,
    # ("ot", "generic") deferred — see JAX_TILTED_PVALUE comment.
}


def _resolve_width_loss_adapter(
    scheme_name: str, model_kind: str
) -> Any:
    """Look up the (scheme, model_kind) adapter, with `(scheme, "generic")`
    fallback. Mirrors `get_jax_tilted_pvalue`."""
    key = (scheme_name, model_kind)
    if key in WIDTH_LOSS_DISPATCH:
        return WIDTH_LOSS_DISPATCH[key]
    generic_key = (scheme_name, "generic")
    if generic_key in WIDTH_LOSS_DISPATCH:
        return WIDTH_LOSS_DISPATCH[generic_key]
    raise NotImplementedError(
        f"Phase E training doesn't support cell (scheme={scheme_name!r}, "
        f"model={model_kind!r}). Available cells: {sorted(WIDTH_LOSS_DISPATCH)}."
    )


def compose_width_loss(
    *,
    eta_net: EtaNet,
    theta_grid_t: jax.Array,
    D_batch_t: jax.Array,
    config: ExperimentConfig,
    loss_kind: str,
    alpha: float | None,
    beta: float | None = None,
) -> jax.Array:
    """Integrated-p (or variant) width loss averaged over a D batch.

    Phase 4b: dispatches to a per-(scheme, model_kind) adapter that
    extracts model-specific kwargs and calls the matching JAX kernel.
    On Normal-Normal this routes through the closed-form fast path
    (byte-identical to pre-Phase-4b output); on Bernoulli + power_law
    it routes through the grid-based generic path.

    Skeptic block #1: a single-D Monte Carlo estimator has too much
    variance for both training and validation signal. Vectorise over
    a (B,) D array and average; gives an unbiased estimator with
    variance ~1/B. The JAX tilted-pvalue port broadcasts naturally so
    this is one tensor call, not a Python loop.

    The ``beta`` argument is forwarded to ``static_width_loss`` only.
    """
    if D_batch_t.ndim == 0:
        D_batch_t = D_batch_t[None]
    if D_batch_t.ndim not in (1, 2):
        raise ValueError(
            f"D_batch_t must be 0D, 1D, or 2D; got shape {tuple(D_batch_t.shape)}."
        )

    model_kind = config.model.fingerprint()[0]
    adapter = _resolve_width_loss_adapter(config.scheme_name, model_kind)
    p_grid = adapter(
        eta_net=eta_net,
        theta_grid_t=theta_grid_t,
        D_batch_t=D_batch_t,
        model=config.model,
        prior=config.prior,
        statistic_name=config.statistic_name,
        scheme_name=config.scheme_name,
    )
    G = theta_grid_t.shape[0]
    B = D_batch_t.shape[0]
    # Skeptic caveat #12: float64 round-off in `_phi(b-a) + _phi(-a-b)`
    # can drift slightly outside [0, 1]. Warn (without breaking the
    # gradient) when drift exceeds tolerance — clamp would zero the
    # gradient at the boundary. The legacy guard keyed on
    # ``torch.is_grad_enabled``; in JAX we issue the check
    # eagerly if the input is concrete (i.e., not tracer-backed),
    # and skip the diagnostic during tracing.
    if not isinstance(p_grid, jax.core.Tracer):
        out_of_range = bool(((p_grid < -1e-5) | (p_grid > 1.0 + 1e-5)).any())
        if out_of_range:
            warnings.warn(
                f"width-loss p-values drifted outside [0, 1] in "
                f"{config.scheme_name}/{config.statistic_name}; "
                f"consider tighter η bounds.",
                RuntimeWarning,
                stacklevel=2,
            )
    theta_grid_b = jnp.broadcast_to(theta_grid_t[None, :], (B, G))  # (B, G)

    if loss_kind == "integrated_p":
        return integrated_pvalue_loss(p_grid, theta_grid_b)
    if loss_kind == "cd_variance":
        return cd_variance_loss(p_grid, theta_grid_b)
    if loss_kind == "static_width":
        if alpha is None:
            raise ValueError("static_width loss requires alpha not None")
        if beta is None:
            raise ValueError(
                "compose_width_loss(loss_kind='static_width') requires "
                "beta to be passed explicitly (the sigmoid-relaxed "
                "indicator's sharpness). Use beta_schedule(epoch, n_epochs) "
                "or a fixed value."
            )
        return static_width_loss(p_grid, theta_grid_b, alpha=alpha, sharpness=beta)
    raise ValueError(f"Unknown loss_kind={loss_kind!r}")


def compose_boundary_penalty(
    *,
    val_net: ValidityNet,
    theta_batch_t: jax.Array,
    eta_pred: jax.Array,
) -> jax.Array:
    """``-log P(valid | θ_batch, η_pred)`` with ValidityNet detached.

    The legacy torch idiom ``torch.func.functional_call(val_net,
    detached_params, inputs)`` is replaced by Equinox's
    ``eqx.partition`` + ``jax.lax.stop_gradient`` + ``eqx.combine``
    pattern. The detached module forwards the same logits but blocks
    gradient flow into ValidityNet's parameters; gradient still flows
    back through the (θ, η) input into EtaNet via ``eta_pred``.
    """
    from ._validity_data import validity_net_inputs

    inputs = validity_net_inputs(theta_batch_t, eta_pred)
    # Split the module into its array leaves vs static metadata, stop
    # gradients on the array leaves, and recombine. The recombined
    # module is a forward-equivalent ValidityNet whose parameters do
    # not contribute to the autodiff graph.
    params, static = eqx.partition(val_net, eqx.is_array)
    params_detached = jax.tree.map(jax.lax.stop_gradient, params)
    val_net_detached = eqx.combine(params_detached, static)
    logits = val_net_detached(inputs)
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
