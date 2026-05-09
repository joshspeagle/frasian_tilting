"""Validity-data helpers for the Phase G dual-head training loop.

Per-batch hyperparam threading:
- ``collect_validity_batch`` builds (theta_all, prior_hp_all, lik_hp_all,
  eta_all, valid_all) for Head B's BCE step. Each batch element has its
  own (prior_hp, lik_hp); validity labels come from per-element
  scheme.tilted_pvalue calls.
- ``prepare_held_out_validity`` does the same for the frozen held-out set.

The numpy core (sampling, validity labelling) stays numpy because the
validity helpers in ``validity.py`` operate on numpy arrays. Only the
boundary-side packaging — the call into Head B and the input
concatenation — uses JAX.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .sampling import ExperimentConfig
from .validity import compute_pvalues_per_sample_with_hp, validity_mask

_FORCE_X64 = _x64

# Aux η-explore range. Default mirrors the v3 ``eta_explore_box`` default.
# Conservative wide range — Head B learns the admissible boundary from
# observed (θ, η, valid) triples, so the box just needs to bracket the
# true admissible region for the trained hyperparam ranges.
_ETA_EXPLORE_LO: float = -5.0
_ETA_EXPLORE_HI: float = 5.0


def collect_validity_batch(
    *,
    eta_pred: jax.Array,
    theta_batch_np: np.ndarray,
    prior_hp_batch_np: np.ndarray,
    lik_hp_batch_np: np.ndarray,
    config: ExperimentConfig,
    scheme: Any,
    n_aux: int,
    rng: np.random.Generator,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build (theta_all, prior_hp_all, lik_hp_all, eta_all, valid_all)
    for the conditional Head B BCE step.

    Mixes the main batch (with detached η_pred + per-element prior/model)
    and an auxiliary boundary-probing batch (i.i.d. (θ, prior_hp, lik_hp)
    + η uniform on the explore range).
    """
    eta_main_np = np.asarray(jax.lax.stop_gradient(eta_pred), dtype=np.float64)
    D_main_np = config.model_cls.sample_data_batch_with_hp(
        theta_batch_np, lik_hp_batch_np, rng, n_data=config.n_data,
    )
    if D_main_np.ndim == 2 and D_main_np.shape[1] == 1:
        D_main_np = D_main_np[:, 0]

    prior_names = config.prior_cls.hyperparam_names()
    lik_names = config.model_cls.hyperparam_names()
    prior_hp_aux, lik_hp_aux = config.hyperparam_distribution.sample(
        n_aux, rng, prior_names=prior_names, lik_names=lik_names,
    )
    theta_aux_np = config.theta_distribution.sample(n_aux, rng)
    eta_aux_np = rng.uniform(_ETA_EXPLORE_LO, _ETA_EXPLORE_HI,
                              size=n_aux).astype(np.float64)
    D_aux_np = config.model_cls.sample_data_batch_with_hp(
        theta_aux_np, lik_hp_aux, rng, n_data=config.n_data,
    )
    if D_aux_np.ndim == 2 and D_aux_np.shape[1] == 1:
        D_aux_np = D_aux_np[:, 0]

    theta_all_np = np.concatenate([theta_batch_np, theta_aux_np])
    prior_hp_all_np = np.concatenate([prior_hp_batch_np, prior_hp_aux], axis=0)
    lik_hp_all_np = np.concatenate([lik_hp_batch_np, lik_hp_aux], axis=0)
    eta_all_np = np.concatenate([eta_main_np, eta_aux_np])
    D_all_np = np.concatenate([D_main_np, D_aux_np], axis=0)

    p_all = compute_pvalues_per_sample_with_hp(
        scheme, theta_all_np, D_all_np,
        config.prior_cls, config.model_cls,
        prior_hp_all_np, lik_hp_all_np,
        eta_all_np, config.statistic_name,
    )
    valid_all = validity_mask(p_all)

    return (
        jnp.asarray(theta_all_np),
        jnp.asarray(prior_hp_all_np),
        jnp.asarray(lik_hp_all_np),
        jnp.asarray(eta_all_np),
        jnp.asarray(valid_all.astype(np.float64)),
    )


def prepare_held_out_validity(
    *,
    scheme: Any,
    theta_held: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Build the frozen (eta_held_aux, D_held, prior_hp_held, lik_hp_held,
    valid_held) tuple for Head B accuracy diagnostic.

    Sampled once at training start and fixed across epochs. Each held-out
    θ gets its own (prior_hp, lik_hp) draw.
    """
    n_held = len(theta_held)
    prior_names = config.prior_cls.hyperparam_names()
    lik_names = config.model_cls.hyperparam_names()
    prior_hp_held, lik_hp_held = config.hyperparam_distribution.sample(
        n_held, rng, prior_names=prior_names, lik_names=lik_names,
    )
    eta_held_aux = rng.uniform(_ETA_EXPLORE_LO, _ETA_EXPLORE_HI,
                                size=n_held).astype(np.float64)
    D_held = config.model_cls.sample_data_batch_with_hp(
        theta_held, lik_hp_held, rng, n_data=config.n_data,
    )
    if D_held.ndim == 2 and D_held.shape[1] == 1:
        D_held = D_held[:, 0]
    p_held = compute_pvalues_per_sample_with_hp(
        scheme, theta_held, D_held,
        config.prior_cls, config.model_cls,
        prior_hp_held, lik_hp_held,
        eta_held_aux, config.statistic_name,
    )
    valid_held = validity_mask(p_held)
    return eta_held_aux, D_held, prior_hp_held, lik_hp_held, valid_held
