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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .losses import integrated_pvalue_loss
from .pvalue_jax import get_jax_tilted_pvalue

if TYPE_CHECKING:
    from .architecture import EtaNet
    from .hyperparam_distribution import HyperparamDistribution


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
    eta_grid: NDArray[np.float64] | None = None,
    K: float = 5.0,
) -> float:
    """Per-slice constant-η argmin of `integrated_pvalue_loss`.

    For each candidate eta_const, build the tilted p-curve on a
    σ-anchored θ-grid, integrate, find the η that minimizes.
    """
    if eta_grid is None:
        eta_grid = np.linspace(-1.5, 1.5, 121)
    pvalue_fn = get_jax_tilted_pvalue(scheme_name, "normal_normal")
    # n_grid=401 here is hardcoded by design: the offline argmin is a
    # 1D smooth integration over a σ-anchored window, so the grid
    # only needs to be fine enough to resolve the integrand — it does
    # NOT have to match the training-loop integration grid (which is
    # the whole point of the offline reference target).
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
    hyperparam_distribution: "HyperparamDistribution",
    *,
    prior_names: tuple[str, ...] = ("loc", "scale"),
    lik_names: tuple[str, ...] = ("sigma",),
    K: float = 5.0,
) -> ProbeBatch:
    """Sample n (theta, D, prior_hp, lik_hp) tuples spanning the
    training hyperparam_distribution, compute per-slice argmin eta
    offline.

    Hyperparams (μ₀, σ₀, σ) are drawn from
    ``hyperparam_distribution.sample(...)`` so the probe matches
    whatever ranges the training run actually used (rather than
    silently hardcoding the v4 ranges). For the Normal-Normal case the
    expected ``prior_names`` / ``lik_names`` are
    ``("loc", "scale")`` / ``("sigma",)``; this function only supports
    that schema (the offline argmin is Normal-Normal-specific).

    θ is then σ-anchored: θ ~ U(μ₀ - K·σ₀, μ₀ + K·σ₀). D ~ N(θ, σ).
    """
    if "loc" not in prior_names or "scale" not in prior_names:
        raise ValueError(
            "build_probe_batch currently requires Normal-Normal prior schema "
            f"with names ('loc', 'scale'); got {prior_names!r}."
        )
    if "sigma" not in lik_names:
        raise ValueError(
            "build_probe_batch currently requires Normal-Normal lik schema "
            f"with names ('sigma',); got {lik_names!r}."
        )
    prior_hp_b, lik_hp_b = hyperparam_distribution.sample(
        n, rng, prior_names=prior_names, lik_names=lik_names,
    )
    loc_idx = prior_names.index("loc")
    scale_idx = prior_names.index("scale")
    sigma_idx = lik_names.index("sigma")
    mu0 = prior_hp_b[:, loc_idx]
    sigma0 = prior_hp_b[:, scale_idx]
    sigma = lik_hp_b[:, sigma_idx]

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

    return ProbeBatch(
        theta=theta, D=D, prior_hp=prior_hp_b, lik_hp=lik_hp_b,
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


def _per_sample_loss_on_probe(
    eta_net: "EtaNet",
    probe: ProbeBatch,
    scheme_name: str,
    statistic_name: str,
    K: float = 5.0,
) -> jax.Array:
    """Per-sample integrated-p loss on the probe batch.

    Returns shape (n,) -- one loss value per probe sample.
    """
    pvalue_fn = get_jax_tilted_pvalue(scheme_name, "normal_normal")
    n = probe.theta.size
    losses = []
    for i in range(n):
        mu0_i = float(probe.prior_hp[i, 0])
        sigma0_i = float(probe.prior_hp[i, 1])
        sigma_i = float(probe.lik_hp[i, 0])
        D_i = float(probe.D[i])
        w_i = sigma0_i ** 2 / (sigma_i ** 2 + sigma0_i ** 2)
        theta_grid = jnp.asarray(np.linspace(
            mu0_i - K * sigma0_i, mu0_i + K * sigma0_i, 401,
        ))
        prior_hp_b = jnp.broadcast_to(jnp.asarray(probe.prior_hp[i]), (401, 2))
        lik_hp_b = jnp.broadcast_to(jnp.asarray(probe.lik_hp[i]), (401, 1))
        eta_arr = eta_net(theta_grid, prior_hp_b, lik_hp_b)  # (401,)
        p = pvalue_fn(
            theta=theta_grid, D=jnp.asarray(D_i), w=jnp.asarray(w_i),
            mu0=jnp.asarray(mu0_i), sigma=jnp.asarray(sigma_i),
            eta=eta_arr, statistic_name=statistic_name,
        )
        loss_i = jnp.trapezoid(p, theta_grid)
        losses.append(loss_i)
    return jnp.stack(losses)  # (n,)


def compute_d2_gradient_norms(
    eta_net: "EtaNet",
    probe: ProbeBatch,
    *,
    scheme_name: str,
    statistic_name: str,
) -> dict[str, float]:
    """D2: gradient norms by layer + by w-bin subgroup.

    Computes the gradient of mean per-sample loss on the probe batch
    with respect to EtaNet parameters. Returns:
      - per-layer gradient norms (input_w, output_w, output_b).
      - per-w-bin gradient norms.
    """
    def loss_per_sample_sum(en):
        per = _per_sample_loss_on_probe(en, probe, scheme_name, statistic_name)
        return jnp.mean(per)

    # NOTE: this function calls eqx.filter_grad 4 times per invocation
    # (1 full + 3 per-bin subsets). Acceptable since D2 is a once-per-epoch
    # diagnostic on a 64-sample probe; not a hot loop. If profiling later
    # shows this matters, batch via jax.jacrev + per-sample mask.
    grad_tree = eqx.filter_grad(loss_per_sample_sum)(eta_net)

    # Walk grad_tree to extract input-layer weights, output weights/bias.
    # eqx.nn.MLP layout: layers is a tuple of eqx.nn.Linear; first is
    # input layer, last is output layer. Each Linear has .weight and
    # .bias.
    mlp_grad = grad_tree.mlp
    input_layer = mlp_grad.layers[0]
    output_layer = mlp_grad.layers[-1]

    def _norm(t):
        if t is None:
            return 0.0
        arr = np.asarray(t).flatten()
        return float(np.sqrt(np.sum(arr * arr)))

    norms = {
        "grad_norm_input_w": _norm(getattr(input_layer, "weight", None)),
        "grad_norm_output_w": _norm(getattr(output_layer, "weight", None)),
        "grad_norm_output_b": _norm(getattr(output_layer, "bias", None)),
    }

    # Per-w-bin gradients: re-compute gradient on each subset.
    for bin_name in ("lowW", "midW", "highW"):
        mask = np.array([w_bin(float(w)) == bin_name for w in probe.w])
        if not mask.any():
            norms[f"grad_norm_{bin_name}"] = 0.0
            continue
        # Subset probe to this bin
        sub = ProbeBatch(
            theta=probe.theta[mask], D=probe.D[mask],
            prior_hp=probe.prior_hp[mask], lik_hp=probe.lik_hp[mask],
            argmin_eta=probe.argmin_eta[mask], w=probe.w[mask],
        )
        def sub_loss(en, _sub=sub):
            per = _per_sample_loss_on_probe(en, _sub, scheme_name, statistic_name)
            return jnp.mean(per)
        sub_grad = eqx.filter_grad(sub_loss)(eta_net)
        # Total grad norm across the EtaNet's params:
        leaves = jax.tree.leaves(sub_grad)
        total = 0.0
        for leaf in leaves:
            if leaf is None:
                continue
            arr = np.asarray(leaf).flatten()
            total += float(np.sum(arr * arr))
        norms[f"grad_norm_{bin_name}"] = float(np.sqrt(total))
    return norms


def compute_d4_loss_by_bin(
    eta_net: "EtaNet",
    probe: ProbeBatch,
    *,
    scheme_name: str,
    statistic_name: str,
) -> dict[str, float]:
    """D4: integrated-p loss values broken down by w-bin.

    Useful for spotting gradient-magnitude asymmetry: if low-w slices
    have 10x the loss of high-w (or vice versa), the optimizer is
    biased.
    """
    losses = np.asarray(
        _per_sample_loss_on_probe(eta_net, probe, scheme_name, statistic_name),
        dtype=np.float64,
    )
    bins = np.array([w_bin(float(w)) for w in probe.w])
    out = {}
    for bin_name in ("lowW", "midW", "highW"):
        mask = bins == bin_name
        if mask.any():
            out[f"loss_{bin_name}"] = float(np.mean(losses[mask]))
        else:
            out[f"loss_{bin_name}"] = float("nan")
    return out
