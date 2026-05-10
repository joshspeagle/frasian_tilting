"""Phase 2 teacher-forcing test (post-Phase-1 stability investigation).

Plan:
  1. Build a teacher dataset: K probe samples with offline-computed
     per-sample argmin_eta (post-selection oracle, since calibrated
     oracle is too slow).
  2. Pre-train EtaNet to predict argmin_eta from (theta, prior_hp, lik_hp)
     via MSE-only loss. Many epochs until MSE converges.
  3. Save pre-trained checkpoint.
  4. Train with width-loss-only (default config + lambda_max=0) starting
     from the pre-trained checkpoint. Track whether the network's
     output drifts back to Basin A or holds the structured solution.

If the network drifts to Basin A from the pre-trained checkpoint, the
loss surface is fundamentally misaligned with the calibration objective.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from pathlib import Path

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.training.architecture import EtaNet, _build_mlp
from frasian.learned.training.diagnostics import build_probe_batch
from frasian.learned.training.hyperparam_distribution import (
    HyperparamDistribution, ScalarDist,
)


def _v4_hp() -> HyperparamDistribution:
    return HyperparamDistribution(
        prior_specs={
            "loc":   ScalarDist(kind="uniform", low=-2.0, high=2.0),
            "scale": ScalarDist(kind="loguniform", low=0.2, high=5.0),
        },
        lik_specs={
            "sigma": ScalarDist(kind="loguniform", low=0.5, high=2.0),
        },
    )


def main():
    # 1. Build teacher dataset (large probe batch).
    print("Building teacher dataset (n=512 probe samples; ~minutes)...")
    rng = np.random.default_rng(0xBEEF)
    hp = _v4_hp()
    probe = build_probe_batch(
        scheme_name="power_law", n=512, rng=rng,
        hyperparam_distribution=hp,
        prior_names=["loc", "scale"], lik_names=["sigma"],
    )
    print(f"  argmin_eta range: [{probe.argmin_eta.min():+.2f}, "
          f"{probe.argmin_eta.max():+.2f}]")
    print(f"  argmin_eta mean:  {probe.argmin_eta.mean():+.3f}")

    # Build training-pair tensors. EtaNet input: (theta, prior_hp, lik_hp);
    # target: argmin_eta scalar.
    theta_t = jnp.asarray(probe.theta, dtype=jnp.float64)  # (n,)
    prior_hp_t = jnp.asarray(probe.prior_hp, dtype=jnp.float64)  # (n, 2)
    lik_hp_t = jnp.asarray(probe.lik_hp, dtype=jnp.float64)  # (n, 1)
    target_eta = jnp.asarray(probe.argmin_eta, dtype=jnp.float64)  # (n,)

    # 2. Build EtaNet matching the v4 architecture.
    hidden_sizes = (128, 128, 128)
    # Get feature_loc/scale/log to match v4 normalization.
    K = 5.0
    sigma0_low, sigma0_high = 0.2, 5.0
    sigma_low, sigma_high = 0.5, 2.0
    # Match v4 normalization stats.
    mu0_loc = 0.0
    mu0_scale = (2.0 - (-2.0)) / float(np.sqrt(12.0))
    sigma0_loc = 0.5 * (np.log(sigma0_low) + np.log(sigma0_high))
    sigma0_scale = (np.log(sigma0_high) - np.log(sigma0_low)) / float(np.sqrt(12.0))
    sigma_loc = 0.5 * (np.log(sigma_low) + np.log(sigma_high))
    sigma_scale = (np.log(sigma_high) - np.log(sigma_low)) / float(np.sqrt(12.0))
    # theta normalization: use absolute integration grid bounds [-30, 30].
    theta_loc = 0.0
    theta_scale = (30.0 - (-30.0)) / float(np.sqrt(12.0))
    feat_loc = (theta_loc, mu0_loc, sigma0_loc, sigma_loc)
    feat_scale = (theta_scale, mu0_scale, sigma0_scale, sigma_scale)
    feat_log = (False, False, True, True)  # sigma0 and sigma are loguniform

    eta_net = EtaNet(
        theta_dim=1, prior_dim=2, lik_dim=1,
        hidden_sizes=hidden_sizes,
        feature_loc=feat_loc, feature_scale=feat_scale, feature_log=feat_log,
        key=jax.random.PRNGKey(42),
    )

    # 3. Pre-train MSE.
    print("\nPre-training EtaNet on teacher dataset (MSE only)...")
    opt = optax.adamw(learning_rate=3e-3, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(eta_net, eqx.is_array))

    @eqx.filter_jit
    def step(net, opt_state, theta, prior, lik, target):
        def loss_fn(net):
            pred = net(theta, prior, lik)  # (n,)
            return jnp.mean((pred - target) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
        updates, new_opt_state = opt.update(
            grads, opt_state, eqx.filter(net, eqx.is_array),
        )
        new_net = eqx.apply_updates(net, updates)
        return new_net, new_opt_state, loss

    n_pretrain_epochs = 200
    for epoch in range(n_pretrain_epochs):
        eta_net, opt_state, mse = step(
            eta_net, opt_state, theta_t, prior_hp_t, lik_hp_t, target_eta,
        )
        if (epoch + 1) % 25 == 0 or epoch < 5:
            preds = np.asarray(eta_net(theta_t, prior_hp_t, lik_hp_t))
            corr = float(np.corrcoef(preds, np.asarray(target_eta))[0, 1])
            print(f"  ep {epoch+1:3d}: MSE = {float(mse):.4f}  "
                  f"corr(pred, target) = {corr:+.3f}  "
                  f"pred mean = {preds.mean():+.3f}")

    # 4. Save the pre-trained checkpoint via the framework's mechanism.
    # We'll save the EtaNet in a format compatible with EtaArtifact.
    # Use eqx.tree_serialise_leaves directly.
    print("\nSaving pre-trained EtaNet checkpoint...")
    out_path = Path("artifacts/probe_v4_phase2_pretrained.eqx")
    # Cheat: build a full EtaArtifact-compatible wrapper.
    # The simplest path: save as raw eqx checkpoint, then load+wrap.
    # Use eqx.tree_serialise_leaves
    eqx.tree_serialise_leaves(str(out_path), eta_net)
    print(f"  wrote {out_path}")

    # 5. Verify the pre-trained network on the probe batch.
    print("\n=== Pre-training verification ===")
    preds_full = np.asarray(eta_net(theta_t, prior_hp_t, lik_hp_t))
    final_corr = float(np.corrcoef(preds_full, np.asarray(target_eta))[0, 1])
    final_mse = float(np.mean((preds_full - np.asarray(target_eta)) ** 2))
    print(f"Final MSE on teacher set = {final_mse:.4f}")
    print(f"Final corr(pred, target) = {final_corr:+.3f}")
    print(f"Pred range: [{preds_full.min():+.3f}, {preds_full.max():+.3f}]")
    print(f"Pred mean : {preds_full.mean():+.3f}")
    print(f"Target range: [{float(target_eta.min()):+.3f}, "
          f"{float(target_eta.max()):+.3f}]")


if __name__ == "__main__":
    main()
