"""Per-epoch diagnostics for learned-η training.

Helpers to compute four diagnostic families during/after each
training epoch (full design in
``docs/superpowers/specs/2026-05-09-learned-eta-diagnostic-instrumentation-design.md``):

  D1 -- output statistics on a held-out probe batch
  D2 -- gradient norms by EtaNet layer + by w-bin subgroup
  D3 -- penultimate-layer activation statistics
  D4 -- training-loss decomposition by w-bin

The probe batch is FIXED for the lifetime of a training run: 64
(theta, D, prior_hp, lik_hp) tuples sampled from the v4 hyperparam
distribution, with offline-computed per-slice constant-eta argmin
values (using `integrated_pvalue_loss`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .losses import integrated_pvalue_loss
from .pvalue_jax import get_jax_tilted_pvalue


@dataclass(frozen=True)
class ProbeBatch:
    """Held-out probe batch with offline argmin η values.

    Used by per-epoch D1/D2/D3 diagnostics. Constructed ONCE before
    training starts; passed read-only to the training loop.
    """

    theta: NDArray[np.float64]       # (n,)
    D: NDArray[np.float64]           # (n,)
    prior_hp: NDArray[np.float64]    # (n, prior_dim)
    lik_hp: NDArray[np.float64]      # (n, lik_dim)
    argmin_eta: NDArray[np.float64]  # (n,) — per-slice constant-η argmin
    w: NDArray[np.float64]           # (n,) — sigma_0^2 / (sigma^2 + sigma_0^2)


_W_BIN_LO = 0.33
_W_BIN_HI = 0.67


def w_bin(w: float) -> str:
    """Map w to one of three bins: lowW / midW / highW."""
    if w <= _W_BIN_LO:
        return "lowW"
    if w <= _W_BIN_HI:
        return "midW"
    return "highW"


def _compute_argmin_constant_eta(
    scheme_name: str,
    D: float,
    mu0: float,
    sigma0: float,
    sigma: float,
    eta_grid: NDArray[np.float64] = np.linspace(-1.5, 1.5, 121),
    K: float = 5.0,
) -> float:
    """Per-slice constant-η argmin of `integrated_pvalue_loss`.

    For each candidate eta_const, build the tilted p-curve on a
    σ-anchored θ-grid, integrate, find the η that minimizes.
    """
    pvalue_fn = get_jax_tilted_pvalue(scheme_name, "normal_normal")
    theta_grid = np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401)
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
    losses = np.empty(eta_grid.size, dtype=np.float64)
    for i, eta in enumerate(eta_grid):
        eta_arr = jnp.full(theta_grid.shape, float(eta))
        p = pvalue_fn(
            theta=jnp.asarray(theta_grid), D=jnp.asarray(D),
            w=jnp.asarray(w), mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
            eta=eta_arr, statistic_name="waldo",
        )
        losses[i] = float(integrated_pvalue_loss(
            jnp.asarray(p)[None, :], jnp.asarray(theta_grid)[None, :]
        ))
    valid = np.isfinite(losses)
    if not valid.any():
        return float("nan")
    idx = int(np.argmin(np.where(valid, losses, np.inf)))
    return float(eta_grid[idx])


def build_probe_batch(
    scheme_name: str,
    n: int,
    rng: np.random.Generator,
    *,
    K: float = 5.0,
) -> ProbeBatch:
    """Sample n (theta, D, prior_hp, lik_hp) tuples spanning the v4
    hyperparam range, compute per-slice argmin eta offline.

    Sampling matches the v4 YAML hyperparam_distribution exactly:
    mu0 ~ U(-2, 2), sigma_0 ~ Loguniform(0.2, 5), sigma ~ Loguniform(0.5, 2).
    theta ~ U(mu0 - K*sigma_0, mu0 + K*sigma_0). D ~ Normal(theta, sigma).
    """
    mu0 = rng.uniform(-2.0, 2.0, size=n)
    sigma0 = np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=n))
    sigma = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=n))
    theta = rng.uniform(mu0 - K * sigma0, mu0 + K * sigma0)
    D = rng.normal(loc=theta, scale=sigma)
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

    argmin_eta = np.empty(n, dtype=np.float64)
    for i in range(n):
        argmin_eta[i] = _compute_argmin_constant_eta(
            scheme_name, float(D[i]),
            float(mu0[i]), float(sigma0[i]), float(sigma[i]),
            K=K,
        )

    prior_hp = np.stack([mu0, sigma0], axis=-1)  # (n, 2)
    lik_hp = sigma[:, None]                       # (n, 1)
    return ProbeBatch(
        theta=theta, D=D, prior_hp=prior_hp, lik_hp=lik_hp,
        argmin_eta=argmin_eta, w=w,
    )
