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
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .losses import integrated_pvalue_loss
from .pvalue_jax import get_jax_tilted_pvalue

if TYPE_CHECKING:
    from .architecture import EtaNet


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


def compute_d1_output_stats(eta_net: "EtaNet", probe: ProbeBatch) -> dict[str, float]:
    """D1: output statistics on the held-out probe batch.

    Calls EtaNet on the probe batch, returns:
      - eta_mean, eta_std, eta_range: distribution of trained η.
      - corr_with_argmin: Pearson correlation between trained η and
        per-slice argmin η.
      - residual_mean: mean(trained_η - argmin_η).
    """
    eta_pred = np.asarray(eta_net(
        jnp.asarray(probe.theta),
        jnp.asarray(probe.prior_hp),
        jnp.asarray(probe.lik_hp),
    ), dtype=np.float64)
    eta_mean = float(np.mean(eta_pred))
    eta_std = float(np.std(eta_pred))
    eta_range = float(np.ptp(eta_pred))
    if np.std(probe.argmin_eta) < 1e-9 or np.std(eta_pred) < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(probe.argmin_eta, eta_pred)[0, 1])
    residual_mean = float(np.mean(eta_pred - probe.argmin_eta))
    return {
        "eta_mean": eta_mean, "eta_std": eta_std, "eta_range": eta_range,
        "corr_with_argmin": corr, "residual_mean": residual_mean,
    }


_DEAD_NEURON_THRESHOLD = 1e-3


def compute_d3_activation_stats(
    eta_net: "EtaNet", probe: ProbeBatch,
) -> dict[str, float | int]:
    """D3: penultimate-layer activation statistics on the probe batch.

    Computes the EtaNet's penultimate-layer activations (post-activation
    output of the last hidden layer, i.e. just before the final linear
    head) for each probe sample, returns per-neuron std across the
    batch, the min std, and a count of "dead" neurons (std < threshold).

    A penultimate layer where most neurons have low std means the
    network has collapsed to roughly constant output regardless of
    input — the dead-input-pathway hypothesis.
    """
    # Replicate EtaNet.__call__'s input pipeline (concat + log/zscore
    # normalization) so we hit the MLP with the same vector it sees
    # in training. Then forward through all-but-the-final Linear,
    # applying the MLP's activation between layers (eqx.nn.MLP stores
    # only Linear layers; activations are applied by __call__).
    theta_arr = jnp.asarray(probe.theta)
    if eta_net.theta_dim == 1 and theta_arr.ndim == 1:
        theta_2d = theta_arr[:, None]
    else:
        theta_2d = theta_arr
    x = jnp.concatenate([
        theta_2d,
        jnp.asarray(probe.prior_hp),
        jnp.asarray(probe.lik_hp),
    ], axis=-1)
    loc = jnp.asarray(eta_net.feature_loc)
    scale = jnp.asarray(eta_net.feature_scale)
    log_mask = jnp.asarray(eta_net.feature_log)
    x_log = jnp.log(jnp.maximum(x, 1e-12))
    x = jnp.where(log_mask, x_log, x)
    x = (x - loc) / scale  # shape (n, in_features)

    # Forward through the MLP up to (but not including) the final
    # Linear head. eqx.nn.MLP.layers is a tuple of Linear-only layers;
    # the canonical __call__ does (Linear -> activation) for each
    # layer in layers[:-1], then a bare Linear for layers[-1]. The
    # penultimate-layer activations we want are the post-activation
    # output after the last layer in layers[:-1] runs.
    activation = eta_net.mlp.activation

    def forward_to_penult(xi: jax.Array) -> jax.Array:
        h = xi
        for layer in eta_net.mlp.layers[:-1]:
            h = layer(h)
            h = activation(h)
        return h

    h = jax.vmap(forward_to_penult)(x)
    h_np = np.asarray(h, dtype=np.float64)  # (n, last_hidden_size)
    per_neuron_std = h_np.std(axis=0)
    return {
        "penult_std_mean": float(np.mean(per_neuron_std)),
        "penult_std_min": float(np.min(per_neuron_std)),
        "n_dead_neurons": int(np.sum(per_neuron_std < _DEAD_NEURON_THRESHOLD)),
    }
