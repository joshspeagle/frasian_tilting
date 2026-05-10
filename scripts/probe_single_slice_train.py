"""Single-slice training test.

Fix (mu0, sigma0, sigma) at a single slice; train EtaNet (v4 architecture,
constant conditioning) with integrated_p loss against (theta_true, D)
samples from the sigma0-anchored (prior-anchored) training distribution at that slice.

Compare to:
  - wald baseline (eta=0)
  - fn-min target (function-constrained D-marginalized min from
    probe_function_constrained_min.py)

Question: does training reach fn-min, or get stuck near Wald?

If single-slice training reaches fn-min → the JOINT (conditional)
training with hyperparam mixing is the bottleneck.

If single-slice training also gets stuck → optimization itself is
broken; not just a conditional-architecture issue.
"""
from __future__ import annotations
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.training.architecture import EtaNet
from frasian.learned.training.losses import integrated_pvalue_loss
from frasian.learned.training.pvalue_jax import power_law_tilted_pvalue_jax


def train_single_slice(mu0, sigma0, sigma,
                        n_train=4096, batch_size=128,
                        n_epochs=100, n_grid=51, K=5.0,
                        lr=1e-3, weight_decay=1e-4,
                        seed=0, target=None, label=""):
    rng = np.random.default_rng(seed)
    w = float(sigma0**2 / (sigma**2 + sigma0**2))
    theta_grid_np = np.linspace(mu0 - K*sigma0, mu0 + K*sigma0, n_grid)
    theta_grid = jnp.asarray(theta_grid_np, dtype=jnp.float64)

    theta_true_train = rng.uniform(mu0 - K*sigma0, mu0 + K*sigma0, n_train)
    D_train = rng.normal(theta_true_train, sigma)

    n_val = 2048
    rng_v = np.random.default_rng(seed + 1000)
    theta_true_val = rng_v.uniform(mu0 - K*sigma0, mu0 + K*sigma0, n_val)
    D_val = jnp.asarray(rng_v.normal(theta_true_val, sigma), dtype=jnp.float64)

    sigma0_low, sigma0_high = 0.2, 5.0
    sigma_low, sigma_high = 0.5, 2.0
    feat_loc = (0.0, 0.0,
                0.5 * (np.log(sigma0_low) + np.log(sigma0_high)),
                0.5 * (np.log(sigma_low) + np.log(sigma_high)))
    feat_scale = (60.0 / np.sqrt(12.0),
                  4.0 / np.sqrt(12.0),
                  (np.log(sigma0_high) - np.log(sigma0_low)) / np.sqrt(12.0),
                  (np.log(sigma_high) - np.log(sigma_low)) / np.sqrt(12.0))
    feat_log = (False, False, True, True)
    eta_net = EtaNet(
        theta_dim=1, prior_dim=2, lik_dim=1,
        hidden_sizes=(128, 128, 128),
        feature_loc=feat_loc, feature_scale=feat_scale, feature_log=feat_log,
        key=jax.random.PRNGKey(42),
    )

    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = opt.init(eqx.filter(eta_net, eqx.is_array))

    w_j = jnp.asarray(w, dtype=jnp.float64)
    mu0_j = jnp.asarray(mu0, dtype=jnp.float64)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float64)
    prior_hp_grid = jnp.asarray(
        np.broadcast_to(np.asarray([mu0, sigma0]), (n_grid, 2)).copy(),
        dtype=jnp.float64)
    lik_hp_grid = jnp.asarray(
        np.broadcast_to(np.asarray([sigma]), (n_grid, 1)).copy(),
        dtype=jnp.float64)

    def per_sample_loss(net, D):
        eta = net(theta_grid, prior_hp_grid, lik_hp_grid)
        p = power_law_tilted_pvalue_jax(
            theta=theta_grid, D=D, w=w_j, mu0=mu0_j, sigma=sigma_j,
            eta=eta, statistic_name="waldo",
        )
        return integrated_pvalue_loss(p[None, :], theta_grid[None, :])

    def batch_loss(net, D_batch):
        return jnp.mean(jax.vmap(per_sample_loss, in_axes=(None, 0))(net, D_batch))

    @eqx.filter_jit
    def step(net, opt_state, D_batch):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(net, D_batch)
        updates, new_state = opt.update(
            grads, opt_state, eqx.filter(net, eqx.is_array))
        new_net = eqx.apply_updates(net, updates)
        return new_net, new_state, loss

    @eqx.filter_jit
    def eval_loss(net, D_batch):
        return batch_loss(net, D_batch)

    val_loss_init = float(eval_loss(eta_net, D_val))
    print(f"  [{label}] init val_loss = {val_loss_init:.4f}")

    n_batches = n_train // batch_size
    rng_shuf = np.random.default_rng(seed + 2000)
    val_curve = [val_loss_init]

    t0 = time.time()
    for epoch in range(n_epochs):
        idx = rng_shuf.permutation(n_train)
        D_train_shuf = D_train[idx]
        for b in range(n_batches):
            D_b = jnp.asarray(D_train_shuf[b*batch_size:(b+1)*batch_size],
                               dtype=jnp.float64)
            eta_net, opt_state, _ = step(eta_net, opt_state, D_b)
        val_loss = float(eval_loss(eta_net, D_val))
        val_curve.append(val_loss)
        if (epoch + 1) % 10 == 0:
            extra = ""
            if target is not None:
                extra = f"  (target={target:.4f}; gap={val_loss - target:+.4f})"
            print(f"  [{label}] ep {epoch+1:>3d}  val_loss = {val_loss:.4f}{extra}")

    elapsed = time.time() - t0
    eta_curve = np.asarray(eta_net(theta_grid, prior_hp_grid, lik_hp_grid))
    return {
        'final_val_loss': val_curve[-1],
        'val_curve': val_curve,
        'eta_curve': eta_curve,
        'theta_grid': theta_grid_np,
        'elapsed_sec': elapsed,
    }


def main():
    # From probe_function_constrained_min: mid-w (largest headroom)
    # Also include high-w for cross-check
    slices = [
        # (mu0, sigma0, sigma, label, wald_target, fn_min_target)
        (0.0, 1.0, 1.0, "mid-w",  1.900, 1.390),
        (0.0, 2.0, 1.0, "mid-hi", 2.056, 1.469),
    ]

    for mu0, sigma0, sigma, label, wald, fn_min in slices:
        w = sigma0**2 / (sigma**2 + sigma0**2)
        print(f"\n=== {label}: mu0={mu0}, sigma0={sigma0}, sigma={sigma} (w={w:.3f}) ===")
        print(f"  Wald baseline: {wald:.4f}")
        print(f"  fn-min target: {fn_min:.4f}  "
              f"(headroom = {100*(wald-fn_min)/wald:.1f}% over Wald)")

        result = train_single_slice(
            mu0, sigma0, sigma, target=fn_min, label=label,
        )

        captured = (wald - result['final_val_loss']) / (wald - fn_min)
        eta_c = result['eta_curve']
        print(f"\n  RESULT [{label}]:")
        print(f"    elapsed         = {result['elapsed_sec']:.1f}s")
        print(f"    final val_loss  = {result['final_val_loss']:.4f}")
        print(f"    headroom captured = {100*captured:+.1f}% of (wald → fn-min)")
        print(f"    eta(theta) range  = [{eta_c.min():+.3f}, {eta_c.max():+.3f}]  "
              f"mean = {eta_c.mean():+.3f}")

        out_dir = Path("artifacts")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"single_slice_{label}.npz"
        np.savez(out_path,
                 theta_grid=result['theta_grid'],
                 eta_curve=result['eta_curve'],
                 val_curve=np.asarray(result['val_curve']),
                 wald_target=wald, fn_min_target=fn_min)
        print(f"    saved to {out_path}")


if __name__ == "__main__":
    main()
