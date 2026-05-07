"""Validity-data helpers for the Phase E dual-head training loop.

Extracted from ``train.py`` (Tier 1.2 §7 split). Three concerns:

1. Per-θ data sampling (``sample_data_per_theta``) with optional
   antithetic pairing (``2θ - D``) for variance reduction on
   Normal-Normal-symmetric loss components.
2. Building the (θ, η, valid) batch that feeds Head B's BCE step
   (``collect_validity_batch``).
3. The JAX-array packaging helper ``validity_net_inputs``.

The numpy core (sampling, validity labelling) stays numpy because the
validity helpers in ``validity.py`` operate on numpy arrays and call
into ``scheme.tilted_pvalue`` (a numpy API). Only the boundary-side
packaging — the call into Head B and the input concatenation — uses
JAX. After the Phase F port the data path is **numpy → jax.Array** at
the input to the loss kernels, exactly mirroring the legacy
**numpy → torch.Tensor** flow.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .sampling import ExperimentConfig
from .validity import compute_pvalues_per_sample, validity_mask

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def antithetic_pair(theta: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Return the antithetic partner ``2θ − D`` of a Normal-Normal D draw.

    For ``D = θ + δ`` (likelihood is ``N(θ, σ²)``), the antithetic
    partner is ``D' = 2θ − D = θ − δ`` — exactly anti-correlated by
    the Normal-Normal symmetry, halving MC variance on even loss
    components.

    Parameters
    ----------
    theta
        ``(N,)`` array of θ values used to draw ``D``.
    D
        ``(N,)`` array of likelihood draws.

    Returns
    -------
    ``(N,)`` array of antithetic partners.
    """
    if theta.shape != D.shape:
        raise ValueError(
            f"antithetic_pair expects matching shapes; got θ={theta.shape}, D={D.shape}"
        )
    return 2.0 * theta - D


def sample_data_per_theta(
    model: Any,
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    antithetic: bool = False,
    n_data: int = 1,
) -> np.ndarray:
    """For each θ, draw ``n_data`` independent observations
    ``D ~ likelihood(·|θ)``.

    Shape contract: ``n_data == 1`` returns ``(N,)`` (preserving
    byte-equality with the pre-Phase-4c Normal-Normal pipeline);
    ``n_data > 1`` returns ``(N, n_data)``. Downstream consumers
    (``compute_pvalues_per_sample``) detect the rank and broadcast.

    Parameters
    ----------
    model
        Model with a ``sample_data(theta_scalar, rng, n)`` API.
    theta
        ``(N,)`` array of θ values.
    rng
        Per-consumer ``numpy.random.Generator``.
    antithetic
        If True, return a ``(2N,)`` array ``[D, 2θ − D]`` interleaved
        as ``[D₁, ..., D_N, 2θ₁ − D₁, ..., 2θ_N − D_N]`` — primary
        first, antithetic partner second. Only valid when
        ``n_data == 1`` (the antithetic partner is a Normal-Normal
        construction; for n_data > 1 raise rather than silently
        producing meaningless reflected pairs).
    n_data
        Number of independent likelihood observations per θ. Default 1.

    Returns
    -------
    Array of shape ``(N,)`` (default), ``(2N,)`` (antithetic) or
    ``(N, n_data)`` (n_data > 1).
    """
    if n_data < 1:
        raise ValueError(f"n_data must be >= 1; got {n_data}")
    if antithetic and n_data != 1:
        raise ValueError(
            f"antithetic=True requires n_data == 1 (the antithetic "
            f"partner 2θ − D is a Normal-Normal scalar construction); "
            f"got n_data={n_data}."
        )
    if n_data == 1:
        out = np.empty(theta.shape, dtype=np.float64)
        for i, th in enumerate(theta):
            out[i] = float(model.sample_data(float(th), rng, n=1)[0])
        if not antithetic:
            return out
        paired = antithetic_pair(theta, out)
        return np.concatenate([out, paired])
    out2d = np.empty((theta.shape[0], n_data), dtype=np.float64)
    for i, th in enumerate(theta):
        out2d[i] = np.asarray(
            model.sample_data(float(th), rng, n=n_data), dtype=np.float64
        )
    return out2d


def collect_validity_batch(
    *,
    eta_pred: jax.Array,
    theta_batch_np: np.ndarray,
    config: ExperimentConfig,
    scheme: Any,
    n_aux: int,
    rng: np.random.Generator,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build (theta_all, eta_all, valid_all) for Head B's BCE step.

    Mixes the main batch (with detached η_pred) and an auxiliary
    boundary-probing batch (i.i.d. θ from theta_distribution, η
    uniform on eta_explore_box). Returns JAX arrays for downstream
    consumption by the equinox ``ValidityNet``.
    """
    # Main: detach η_pred for label collection (gradient must not flow
    # through discrete labels). ``stop_gradient`` is a no-op in numpy
    # space, but the pattern stays explicit so the boundary between
    # JAX-traced and numpy-side code is clear.
    eta_main_np = np.asarray(jax.lax.stop_gradient(eta_pred), dtype=np.float64)
    n_data = config.n_data
    D_main_np = sample_data_per_theta(
        config.model, theta_batch_np, rng, n_data=n_data
    )

    # Aux: independent draws from the θ distribution + uniform η in box.
    theta_aux_np = config.theta_distribution.sample(n_aux, rng)
    eta_aux_np = rng.uniform(*config.eta_explore_box, size=n_aux).astype(np.float64)
    D_aux_np = sample_data_per_theta(config.model, theta_aux_np, rng, n_data=n_data)

    theta_all_np = np.concatenate([theta_batch_np, theta_aux_np])
    eta_all_np = np.concatenate([eta_main_np, eta_aux_np])
    # axis=0 covers both 1D (n_data == 1) and 2D ((N, n_data)) shapes.
    D_all_np = np.concatenate([D_main_np, D_aux_np], axis=0)

    p_all = compute_pvalues_per_sample(
        scheme,
        theta_all_np,
        D_all_np,
        config.model,
        config.prior,
        eta_all_np,
        config.statistic_name,
    )
    valid_all = validity_mask(p_all)  # bool (N+N_aux,)

    theta_all_t = jnp.asarray(theta_all_np)
    eta_all_t = jnp.asarray(eta_all_np)
    valid_all_t = jnp.asarray(valid_all.astype(np.float64))
    return theta_all_t, eta_all_t, valid_all_t


def validity_net_inputs(theta: jax.Array, eta: jax.Array) -> jax.Array:
    """Pack ``(θ, η)`` into ValidityNet's expected ``(N, theta_dim+1)``."""
    if theta.ndim == 1:
        theta = theta[..., None]  # (N, 1)
    if eta.ndim == 1:
        eta = eta[..., None]  # (N, 1)
    return jnp.concatenate([theta, eta], axis=-1)


def prepare_held_out_validity(
    *,
    scheme: Any,
    theta_held: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the frozen (η_held_aux, D_held, valid_held) triple.

    Used as the held-out set for Head B's accuracy diagnostic. Sampled
    once at training start and fixed across epochs.

    Returns
    -------
    Tuple ``(eta_held_aux, D_held, valid_held)`` of ``(M,)`` arrays.
    """
    eta_held_aux = rng.uniform(*config.eta_explore_box, size=len(theta_held)).astype(np.float64)
    D_held = sample_data_per_theta(config.model, theta_held, rng, n_data=config.n_data)
    p_held = compute_pvalues_per_sample(
        scheme,
        theta_held,
        D_held,
        config.model,
        config.prior,
        eta_held_aux,
        config.statistic_name,
    )
    valid_held = validity_mask(p_held)
    return eta_held_aux, D_held, valid_held
