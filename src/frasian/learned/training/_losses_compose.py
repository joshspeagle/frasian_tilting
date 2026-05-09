"""Loss composition for the Phase G conditional dual-head training loop.

Three concerns:

1. Width-loss adapter (``compose_width_loss``): wraps the JAX
   tilted-pvalue port + the chosen functional in ``losses.py``. Phase G:
   per-batch (prior_hp, lik_hp) extracted from the input tensors and
   broadcast across the θ-grid axis.

2. Boundary penalty (``compose_boundary_penalty``): forwards
   ``(θ, prior_hp, lik_hp, η)`` through a parameter-detached
   ``ValidityNet`` so gradient flows back into Head A but not Head B's
   weights.

3. λ schedule (``lambda_schedule``) and β schedule (``beta_schedule``)
   for the Head A loss components.
"""

from __future__ import annotations

import warnings
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .architecture import EtaNet, ValidityNet
from .losses import (
    boundary_penalty_from_validity,
    cd_variance_loss,
    integrated_pvalue_loss,
    static_width_loss,
)
from .pvalue_jax import get_jax_tilted_pvalue
from .sampling import ExperimentConfig

_FORCE_X64 = _x64

BETA_MIN: float = 50.0
BETA_MAX: float = 500.0
BETA_WARMUP_FRAC: float = 0.5


def _resolve_n_grid_generic_training() -> int:
    import os

    raw = os.environ.get("FRASIAN_N_GRID_GENERIC_TRAINING", "512")
    try:
        v = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"FRASIAN_N_GRID_GENERIC_TRAINING must be int; got {raw!r}."
        ) from exc
    if v < 16:
        raise ValueError(
            f"FRASIAN_N_GRID_GENERIC_TRAINING must be >= 16; got {v}."
        )
    return v


_N_GRID_GENERIC_TRAINING: int = _resolve_n_grid_generic_training()


def _call_normal_normal_pvalue(
    *,
    eta_net: EtaNet,
    theta_grid_t: jax.Array,
    D_batch_t: jax.Array,
    prior_hp_batch_t: jax.Array,    # (B, 2) — [loc, scale]
    lik_hp_batch_t: jax.Array,      # (B, 1) — [sigma]
    statistic_name: str,
    scheme_name: str,
) -> jax.Array:
    """Adapter (Phase G): per-batch (w, mu0, sigma) from hyperparams.

    Returns the (B, G) p-value grid. The eta_net is called per (b, g)
    via flatten/reshape since it takes a 1D batch axis.
    """
    mu0_b = prior_hp_batch_t[:, 0]
    sigma0_b = prior_hp_batch_t[:, 1]
    sigma_b = lik_hp_batch_t[:, 0]
    w_b = sigma0_b**2 / (sigma_b**2 + sigma0_b**2)

    G = theta_grid_t.shape[0]
    B = prior_hp_batch_t.shape[0]
    theta_bg_2d = jnp.broadcast_to(theta_grid_t[None, :], (B, G))
    theta_bg_flat = theta_bg_2d.reshape(B * G)
    prior_hp_bg = jnp.broadcast_to(
        prior_hp_batch_t[:, None, :], (B, G, prior_hp_batch_t.shape[1]),
    ).reshape(B * G, prior_hp_batch_t.shape[1])
    lik_hp_bg = jnp.broadcast_to(
        lik_hp_batch_t[:, None, :], (B, G, lik_hp_batch_t.shape[1]),
    ).reshape(B * G, lik_hp_batch_t.shape[1])
    eta_flat = eta_net(theta_bg_flat, prior_hp_bg, lik_hp_bg)
    eta_bg = eta_flat.reshape(B, G)

    tilted_pvalue = get_jax_tilted_pvalue(scheme_name, "normal_normal")
    return tilted_pvalue(
        theta_bg_2d,
        jnp.broadcast_to(D_batch_t[:, None], (B, G)),
        w_b[:, None],
        mu0_b[:, None],
        sigma_b[:, None],
        eta_bg,
        statistic_name,
    )


def compute_log_p_lik_grid_np(
    model: Any, D_batch_np: np.ndarray, support_theta_grid_np: np.ndarray
) -> np.ndarray:
    """Build ``(B, N_grid)`` log-likelihood grid per batch element."""
    from ...models.base import batch_loglik_grid as _batch_loglik_grid

    arr = np.asarray(D_batch_np, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.asarray(
        _batch_loglik_grid(model, arr, np.asarray(support_theta_grid_np, dtype=np.float64)),
        dtype=np.float64,
    )


def precompute_generic_grids(
    model: Any, prior: Any, n_grid: int = _N_GRID_GENERIC_TRAINING
) -> tuple[jax.Array, jax.Array]:
    """Precompute ``(support_theta_grid, log_p_prior_grid)`` once per
    representative (model, prior).

    Phase G note: both model and prior now vary per batch element; this
    helper is kept for the case where a representative (rep_model,
    rep_prior) is used (e.g. Bernoulli where support is constant).
    Caller is responsible for warming this up per-step if hyperparams
    differ across the batch.
    """
    if not hasattr(model, "support"):
        raise NotImplementedError(
            f"precompute_generic_grids requires `model.support()`; "
            f"got {type(model).__name__!r} without it."
        )
    support_lo, support_hi = model.support()
    support_lo_f = float(support_lo)
    support_hi_f = float(support_hi)
    if not (np.isfinite(support_lo_f) and np.isfinite(support_hi_f)):
        raise NotImplementedError(
            "precompute_generic_grids: unbounded support fallback not yet wired."
        )
    width = support_hi_f - support_lo_f
    pad = 1e-6 * width
    pad_cap = 0.05 * width
    lp_lo = lp_hi = float("-inf")
    while pad <= pad_cap:
        boundary_lo = support_lo_f + pad
        boundary_hi = support_hi_f - pad
        lp_lo = float(np.asarray(prior.logpdf(jnp.asarray([boundary_lo]))).item())
        lp_hi = float(np.asarray(prior.logpdf(jnp.asarray([boundary_hi]))).item())
        if np.isfinite(lp_lo) and np.isfinite(lp_hi):
            break
        pad *= 2.0
    if not (np.isfinite(lp_lo) and np.isfinite(lp_hi)):
        raise ValueError(
            f"precompute_generic_grids: prior log-pdf non-finite at both "
            f"grid endpoints even after padding to {pad_cap:.4f}."
        )
    support_theta_grid = jnp.linspace(
        support_lo_f + pad, support_hi_f - pad, n_grid
    )
    log_p_prior_grid = jnp.asarray(prior.logpdf(support_theta_grid))
    if not bool(jnp.all(jnp.isfinite(log_p_prior_grid))):
        raise ValueError(
            f"precompute_generic_grids: prior log-pdf is non-finite at "
            f"interior grid points (pad={pad:.4g})."
        )
    return support_theta_grid, log_p_prior_grid


def _call_generic_grid_pvalue(
    *,
    eta_net: EtaNet,
    theta_grid_t: jax.Array,
    D_batch_t: jax.Array,
    prior_hp_batch_t: jax.Array,
    lik_hp_batch_t: jax.Array,
    statistic_name: str,
    scheme_name: str,
    log_p_lik_grid_t: jax.Array | None = None,
    support_theta_grid_t: jax.Array | None = None,
    log_p_prior_grid_t: jax.Array | None = None,
) -> jax.Array:
    """Adapter: call ``generic_grid_tilted_pvalue`` (Phase 4 generic path).

    Phase G: representative (rep_model, rep_prior) prior_grid + support
    are supplied by the caller. This is a known approximation when
    hyperparams vary per-batch — see plan Step 6.4 caveat.
    """
    from .pvalue_jax import generic_grid_tilted_pvalue

    del D_batch_t, scheme_name  # unused in the kernel call
    if (
        log_p_lik_grid_t is None
        or support_theta_grid_t is None
        or log_p_prior_grid_t is None
    ):
        raise ValueError(
            "_call_generic_grid_pvalue requires pre-computed grids."
        )

    G = theta_grid_t.shape[0]
    B = log_p_lik_grid_t.shape[0]
    theta_bg_2d = jnp.broadcast_to(theta_grid_t[None, :], (B, G))
    theta_bg_flat = theta_bg_2d.reshape(B * G)
    prior_hp_bg = jnp.broadcast_to(
        prior_hp_batch_t[:, None, :], (B, G, prior_hp_batch_t.shape[1]),
    ).reshape(B * G, prior_hp_batch_t.shape[1])
    lik_hp_bg = jnp.broadcast_to(
        lik_hp_batch_t[:, None, :], (B, G, lik_hp_batch_t.shape[1]),
    ).reshape(B * G, lik_hp_batch_t.shape[1])
    eta_flat = eta_net(theta_bg_flat, prior_hp_bg, lik_hp_bg)
    eta_bg = eta_flat.reshape(B, G)
    return generic_grid_tilted_pvalue(
        theta_bg_2d,
        eta_bg,
        log_p_lik_grid_t,
        log_p_prior_grid_t,
        support_theta_grid_t,
        statistic_name,
    )


WIDTH_LOSS_DISPATCH: dict[tuple[str, str], Any] = {
    ("power_law", "normal_normal"): _call_normal_normal_pvalue,
    ("ot", "normal_normal"): _call_normal_normal_pvalue,
    ("power_law", "generic"): _call_generic_grid_pvalue,
}


def _resolve_width_loss_adapter(
    scheme_name: str, model_kind: str
) -> Any:
    key = (scheme_name, model_kind)
    if key in WIDTH_LOSS_DISPATCH:
        return WIDTH_LOSS_DISPATCH[key]
    generic_key = (scheme_name, "generic")
    if generic_key in WIDTH_LOSS_DISPATCH:
        return WIDTH_LOSS_DISPATCH[generic_key]
    raise NotImplementedError(
        f"Phase G training doesn't support cell (scheme={scheme_name!r}, "
        f"model={model_kind!r}). Available cells: {sorted(WIDTH_LOSS_DISPATCH)}."
    )


def compose_width_loss(
    *,
    eta_net: EtaNet,
    theta_grid_t: jax.Array,
    D_batch_t: jax.Array,
    config: ExperimentConfig,
    prior_hp_batch_t: jax.Array,
    lik_hp_batch_t: jax.Array,
    loss_kind: str,
    alpha: float | None,
    beta: float | None = None,
    log_p_lik_grid_t: jax.Array | None = None,
    support_theta_grid_t: jax.Array | None = None,
    log_p_prior_grid_t: jax.Array | None = None,
) -> jax.Array:
    """Conditional width loss (Phase G). Per-batch (prior_hp, lik_hp)."""
    if D_batch_t.ndim == 0:
        D_batch_t = D_batch_t[None]
    if D_batch_t.ndim not in (1, 2):
        raise ValueError(
            f"D_batch_t must be 0D, 1D, or 2D; got shape {tuple(D_batch_t.shape)}."
        )

    model_kind_key = (
        "normal_normal"
        if config.model_cls.__name__ == "NormalNormalModel"
        else "generic"
    )
    adapter = _resolve_width_loss_adapter(config.scheme_name, model_kind_key)

    adapter_kwargs: dict[str, Any] = dict(
        eta_net=eta_net,
        theta_grid_t=theta_grid_t,
        D_batch_t=D_batch_t,
        prior_hp_batch_t=prior_hp_batch_t,
        lik_hp_batch_t=lik_hp_batch_t,
        statistic_name=config.statistic_name,
        scheme_name=config.scheme_name,
    )
    if adapter is _call_generic_grid_pvalue:
        adapter_kwargs.update(
            log_p_lik_grid_t=log_p_lik_grid_t,
            support_theta_grid_t=support_theta_grid_t,
            log_p_prior_grid_t=log_p_prior_grid_t,
        )
    p_grid = adapter(**adapter_kwargs)
    G = theta_grid_t.shape[0]
    B = D_batch_t.shape[0]
    if not isinstance(p_grid, jax.core.Tracer):
        out_of_range = bool(((p_grid < -1e-5) | (p_grid > 1.0 + 1e-5)).any())
        if out_of_range:
            warnings.warn(
                f"width-loss p-values drifted outside [0, 1] in "
                f"{config.scheme_name}/{config.statistic_name}.",
                RuntimeWarning,
                stacklevel=2,
            )
    theta_grid_b = jnp.broadcast_to(theta_grid_t[None, :], (B, G))

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
                "beta to be passed explicitly."
            )
        return static_width_loss(p_grid, theta_grid_b, alpha=alpha, sharpness=beta)
    raise ValueError(f"Unknown loss_kind={loss_kind!r}")


def compose_boundary_penalty(
    *,
    val_net: ValidityNet,
    theta_batch_t: jax.Array,
    prior_hp_batch_t: jax.Array,
    lik_hp_batch_t: jax.Array,
    eta_pred: jax.Array,
) -> jax.Array:
    """``-log P(valid | θ_batch, prior_hp, lik_hp, η_pred)`` with
    ValidityNet detached.
    """
    params, static = eqx.partition(val_net, eqx.is_array)
    params_detached = jax.tree.map(jax.lax.stop_gradient, params)
    val_net_detached = eqx.combine(params_detached, static)
    logits = val_net_detached(
        theta_batch_t, prior_hp_batch_t, lik_hp_batch_t, eta_pred,
    )
    return boundary_penalty_from_validity(logits)


def lambda_schedule(
    epoch: int,
    n_epochs: int,
    lambda_max: float,
    warmup_frac: float,
) -> float:
    if warmup_frac <= 0.0:
        return float(lambda_max)
    warmup_epochs = max(1, int(round(warmup_frac * n_epochs)))
    if epoch >= warmup_epochs:
        return float(lambda_max)
    return float(lambda_max) * epoch / warmup_epochs


def decay_schedule(
    epoch: int,
    n_epochs: int,
    lambda_max: float,
    decay_frac: float,
) -> float:
    """Linear decay from ``lambda_max`` at epoch 0 to 0 at
    ``decay_frac * n_epochs``.

    Mirror of ``lambda_schedule`` but inverted in time. Used by the
    Phase G anti-Wald / anti-collapse regularizers
    (``losses.anti_wald_penalty``, ``losses.eta_collapse_penalty``)
    to perturb the optimizer out of the η ≈ 1 (Wald) basin during
    early training, then release the bias so the underlying width
    loss owns final convergence. Set ``lambda_max=0`` to disable
    entirely (the framework default — these regularizers are
    opt-in diagnostics, not required components of the loss).
    """
    if lambda_max <= 0.0:
        return 0.0
    if decay_frac <= 0.0:
        return 0.0
    decay_epochs = max(1, int(round(decay_frac * n_epochs)))
    if epoch >= decay_epochs:
        return 0.0
    return float(lambda_max) * (1.0 - epoch / decay_epochs)


def beta_schedule(
    epoch: int,
    n_epochs: int,
    *,
    beta_min: float = BETA_MIN,
    beta_max: float = BETA_MAX,
    warmup_frac: float = BETA_WARMUP_FRAC,
) -> float:
    if warmup_frac <= 0.0:
        return float(beta_max)
    if n_epochs <= 0:
        return float(beta_max)
    span = max(1.0, warmup_frac * float(n_epochs))
    frac = min(1.0, float(epoch) / span)
    return float(beta_min) + (float(beta_max) - float(beta_min)) * frac
