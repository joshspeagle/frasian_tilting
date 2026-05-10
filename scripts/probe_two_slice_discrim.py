"""2-slice discrimination test.

Train EtaNet on a 50/50 mix of two slices that prefer different η(θ)
curves. Evaluate per-slice loss and η(θ) curves at the end.

If conditional EtaNet works → both slices reach their per-slice fn-min.
If conditional pathway fails → network outputs a compromise curve;
both slices stay above fn-min, especially the one whose optimum is
farther from the marginal.

Slices chosen (both have substantial headroom, but different optimal η):
  A: mid-w   (σ₀=1, σ=1, w=0.5)   — fn-min 1.39, η range [-0.30, +1.27]
  B: high-w  (σ₀=4, σ=1, w=0.94)  — fn-min 1.51, η range [-0.13, +5.69]

Comparisons:
  1. Single-slice training on A alone   (baseline ceiling for A)
  2. Single-slice training on B alone   (baseline ceiling for B)
  3. 2-slice mixed training, eval per-slice
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


def make_etanet(seed=42):
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
    return EtaNet(
        theta_dim=1, prior_dim=2, lik_dim=1,
        hidden_sizes=(128, 128, 128),
        feature_loc=feat_loc, feature_scale=feat_scale, feature_log=feat_log,
        key=jax.random.PRNGKey(seed),
    )


def build_dataset(slices, n_per_slice, n_grid, K, rng):
    """Build (theta_grid, D, mu0, sigma0, sigma, slice_id) per-sample arrays."""
    n_total = len(slices) * n_per_slice
    theta_grids = np.empty((n_total, n_grid), dtype=np.float64)
    D_arr = np.empty(n_total, dtype=np.float64)
    mu0_arr = np.empty(n_total, dtype=np.float64)
    sigma0_arr = np.empty(n_total, dtype=np.float64)
    sigma_arr = np.empty(n_total, dtype=np.float64)
    slice_idx = np.empty(n_total, dtype=np.int32)
    for i, (mu0, sigma0, sigma) in enumerate(slices):
        s = slice(i*n_per_slice, (i+1)*n_per_slice)
        theta_true = rng.uniform(mu0 - K*sigma0, mu0 + K*sigma0, n_per_slice)
        D_arr[s] = rng.normal(theta_true, sigma)
        theta_grids[s] = np.linspace(mu0 - K*sigma0, mu0 + K*sigma0, n_grid)
        mu0_arr[s] = mu0
        sigma0_arr[s] = sigma0
        sigma_arr[s] = sigma
        slice_idx[s] = i
    return theta_grids, D_arr, mu0_arr, sigma0_arr, sigma_arr, slice_idx


def per_sample_loss(net, theta_grid_i, D_i, mu0_i, sigma0_i, sigma_i):
    prior_hp = jnp.stack([
        jnp.full(theta_grid_i.shape, mu0_i),
        jnp.full(theta_grid_i.shape, sigma0_i),
    ], axis=-1)
    lik_hp = jnp.full(theta_grid_i.shape + (1,), sigma_i)
    eta = net(theta_grid_i, prior_hp, lik_hp)
    w = sigma0_i**2 / (sigma_i**2 + sigma0_i**2)
    p = power_law_tilted_pvalue_jax(
        theta=theta_grid_i, D=D_i, w=w, mu0=mu0_i, sigma=sigma_i,
        eta=eta, statistic_name="waldo",
    )
    return integrated_pvalue_loss(p[None, :], theta_grid_i[None, :])


def batch_loss(net, theta_grids, D_batch, mu0_batch, sigma0_batch, sigma_batch):
    return jnp.mean(jax.vmap(per_sample_loss, in_axes=(None, 0, 0, 0, 0, 0))(
        net, theta_grids, D_batch, mu0_batch, sigma0_batch, sigma_batch
    ))


def train(slices, n_epochs=100, batch_size=128, n_per_slice=4096,
           n_grid=51, K=5.0, lr=1e-3, weight_decay=1e-4,
           seed=42, label=""):
    """Train EtaNet on the union of `slices` (mix of all samples).

    Returns:
        final_eta_net, history (dict).
    """
    rng = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 1000)
    n_val_per_slice = 1024

    train_data = build_dataset(slices, n_per_slice, n_grid, K, rng)
    val_data = build_dataset(slices, n_val_per_slice, n_grid, K, rng_val)
    (theta_grids, D_arr, mu0_arr, sigma0_arr, sigma_arr, slice_idx) = train_data
    (val_thetag, val_D, val_mu0, val_sigma0, val_sigma, val_idx) = val_data

    n_total = D_arr.shape[0]
    n_val = val_D.shape[0]

    eta_net = make_etanet(seed=seed)
    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = opt.init(eqx.filter(eta_net, eqx.is_array))

    @eqx.filter_jit
    def step(net, opt_state, theta_g, D, mu0, sigma0, sigma):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(
            net, theta_g, D, mu0, sigma0, sigma)
        updates, new_state = opt.update(
            grads, opt_state, eqx.filter(net, eqx.is_array))
        new_net = eqx.apply_updates(net, updates)
        return new_net, new_state, loss

    @eqx.filter_jit
    def eval_loss(net, theta_g, D, mu0, sigma0, sigma):
        return batch_loss(net, theta_g, D, mu0, sigma0, sigma)

    # Per-slice initial val loss
    n_slices = len(slices)
    init_val_per_slice = np.empty(n_slices)
    for i in range(n_slices):
        m = (val_idx == i)
        init_val_per_slice[i] = float(eval_loss(
            eta_net,
            jnp.asarray(val_thetag[m]), jnp.asarray(val_D[m]),
            jnp.asarray(val_mu0[m]), jnp.asarray(val_sigma0[m]),
            jnp.asarray(val_sigma[m]),
        ))
    print(f"  [{label}] init val: " + ", ".join(
        f"slice{i}={v:.4f}" for i, v in enumerate(init_val_per_slice)))

    n_batches = n_total // batch_size
    rng_shuf = np.random.default_rng(seed + 2000)
    val_curves = [list(init_val_per_slice)]

    t0 = time.time()
    for epoch in range(n_epochs):
        idx = rng_shuf.permutation(n_total)
        for b in range(n_batches):
            sl = idx[b*batch_size:(b+1)*batch_size]
            eta_net, opt_state, _ = step(
                eta_net, opt_state,
                jnp.asarray(theta_grids[sl]), jnp.asarray(D_arr[sl]),
                jnp.asarray(mu0_arr[sl]), jnp.asarray(sigma0_arr[sl]),
                jnp.asarray(sigma_arr[sl]),
            )
        per_slice = np.empty(n_slices)
        for i in range(n_slices):
            m = (val_idx == i)
            per_slice[i] = float(eval_loss(
                eta_net,
                jnp.asarray(val_thetag[m]), jnp.asarray(val_D[m]),
                jnp.asarray(val_mu0[m]), jnp.asarray(val_sigma0[m]),
                jnp.asarray(val_sigma[m]),
            ))
        val_curves.append(list(per_slice))
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            print(f"  [{label}] ep {epoch+1:>3d}: " + ", ".join(
                f"slice{i}={v:.4f}" for i, v in enumerate(per_slice)))

    elapsed = time.time() - t0

    # Final eta curves at each slice
    eta_curves = []
    for i, (mu0, sigma0, sigma) in enumerate(slices):
        tg = jnp.asarray(np.linspace(mu0 - K*sigma0, mu0 + K*sigma0, n_grid),
                          dtype=jnp.float64)
        prior_hp = jnp.stack([
            jnp.full(tg.shape, mu0), jnp.full(tg.shape, sigma0),
        ], axis=-1)
        lik_hp = jnp.full(tg.shape + (1,), sigma)
        eta_curves.append(np.asarray(eta_net(tg, prior_hp, lik_hp)))

    return {
        'final_per_slice': val_curves[-1],
        'val_curves': np.asarray(val_curves),
        'eta_curves': eta_curves,
        'elapsed_sec': elapsed,
    }


def main():
    # (mu0, sigma0, sigma)
    slice_A = (0.0, 1.0, 1.0)  # mid-w
    slice_B = (0.0, 4.0, 1.0)  # high-w
    label_A = "mid-w (σ₀=1)"
    label_B = "high-w (σ₀=4)"

    # Targets from probe_function_constrained_min
    wald_A, fnmin_A = 1.900, 1.390
    wald_B, fnmin_B = 1.820, 1.510

    print("\n=== STEP 1: single-slice baseline on A ===")
    res_A = train([slice_A], n_epochs=80, label="A-only")
    print(f"  elapsed: {res_A['elapsed_sec']:.1f}s")

    print("\n=== STEP 2: single-slice baseline on B ===")
    res_B = train([slice_B], n_epochs=80, label="B-only")
    print(f"  elapsed: {res_B['elapsed_sec']:.1f}s")

    print("\n=== STEP 3: 2-slice mixed training ===")
    res_mix = train([slice_A, slice_B], n_epochs=80, label="mix")
    print(f"  elapsed: {res_mix['elapsed_sec']:.1f}s")

    # Comparison
    A_solo = res_A['final_per_slice'][0]
    B_solo = res_B['final_per_slice'][0]
    A_mix, B_mix = res_mix['final_per_slice'][0], res_mix['final_per_slice'][1]

    def pct(loss, wald, fnmin):
        return 100.0 * (wald - loss) / (wald - fnmin)

    print("\n" + "="*78)
    print("Comparison: per-slice val loss in single vs. mixed training")
    print("="*78)
    print(f"{'slice':<22} {'wald':>7}  {'fn-min':>7}  {'solo':>7}  "
          f"{'mix':>7}  {'solo cap':>9}  {'mix cap':>9}")
    print(f"{'A: '+label_A:<22} {wald_A:>7.4f}  {fnmin_A:>7.4f}  "
          f"{A_solo:>7.4f}  {A_mix:>7.4f}  "
          f"{pct(A_solo, wald_A, fnmin_A):>+8.1f}%  "
          f"{pct(A_mix, wald_A, fnmin_A):>+8.1f}%")
    print(f"{'B: '+label_B:<22} {wald_B:>7.4f}  {fnmin_B:>7.4f}  "
          f"{B_solo:>7.4f}  {B_mix:>7.4f}  "
          f"{pct(B_solo, wald_B, fnmin_B):>+8.1f}%  "
          f"{pct(B_mix, wald_B, fnmin_B):>+8.1f}%")

    # Eta curves
    print("\nFinal η(θ) summary:")
    print(f"  A solo:  range [{res_A['eta_curves'][0].min():+.3f}, "
          f"{res_A['eta_curves'][0].max():+.3f}]  mean {res_A['eta_curves'][0].mean():+.3f}")
    print(f"  B solo:  range [{res_B['eta_curves'][0].min():+.3f}, "
          f"{res_B['eta_curves'][0].max():+.3f}]  mean {res_B['eta_curves'][0].mean():+.3f}")
    print(f"  A mix:   range [{res_mix['eta_curves'][0].min():+.3f}, "
          f"{res_mix['eta_curves'][0].max():+.3f}]  mean {res_mix['eta_curves'][0].mean():+.3f}")
    print(f"  B mix:   range [{res_mix['eta_curves'][1].min():+.3f}, "
          f"{res_mix['eta_curves'][1].max():+.3f}]  mean {res_mix['eta_curves'][1].mean():+.3f}")

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "two_slice_discrim.npz"
    # Save K from train() default = 5.0 to keep theta_grids reproducible
    K = 5.0
    np.savez(
        out_path,
        slice_A=np.asarray(slice_A), slice_B=np.asarray(slice_B),
        wald_A=wald_A, fnmin_A=fnmin_A, wald_B=wald_B, fnmin_B=fnmin_B,
        eta_A_solo=res_A['eta_curves'][0],
        eta_B_solo=res_B['eta_curves'][0],
        eta_A_mix=res_mix['eta_curves'][0],
        eta_B_mix=res_mix['eta_curves'][1],
        theta_grid_A=np.linspace(slice_A[0]-K*slice_A[1], slice_A[0]+K*slice_A[1], 51),
        theta_grid_B=np.linspace(slice_B[0]-K*slice_B[1], slice_B[0]+K*slice_B[1], 51),
        val_curves_A_solo=res_A['val_curves'],
        val_curves_B_solo=res_B['val_curves'],
        val_curves_mix=res_mix['val_curves'],
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
