"""Continuous-σ₀ training with K-D-MC per (θ_true).

Reproduces v4's continuous hyperparam distribution but varies the
number of D samples averaged per (θ_true, slice) tuple. Directly tests
whether per-slice gradient sparsity (1 D realization per slice) is
the v4 failure mode.

Train two configs:
  - K=16: each (θ_true, slice) gets 16 D realizations; loss is averaged.
  - K=1:  matches v4 sparsity (one D per (θ_true, slice)).

Evaluate both on:
  1. Three fixed slices from probe_function_constrained_min:
       mid-w (σ₀=1), mid-hi (σ₀=2), high-w (σ₀=4).
  2. A continuous holdout drawn from the v4 distribution.

Compare to fn-min targets and to v4's L_wald=1.363 / L_oracle_3=1.175
references.
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


def sample_v4_hp(rng, B):
    """Sample B hyperparam tuples from the v4 distribution.

    Returns: mu0 (B,), sigma0 (B,), sigma (B,)
    """
    mu0 = rng.uniform(-2.0, 2.0, B)
    log_sigma0 = rng.uniform(np.log(0.2), np.log(5.0), B)
    sigma0 = np.exp(log_sigma0)
    log_sigma = rng.uniform(np.log(0.5), np.log(2.0), B)
    sigma = np.exp(log_sigma)
    return mu0, sigma0, sigma


def sample_batch(rng, B, K, n_grid=51, K_anchor=5.0):
    """Sample a fresh batch.

    Returns:
      theta_grids: (B, n_grid)  — sigma0-anchored (prior-anchored) per sample
      D_arrs:     (B, K)         — K D realizations per (θ_true, slice)
      mu0_b:      (B,)
      sigma0_b:   (B,)
      sigma_b:    (B,)
    """
    mu0_b, sigma0_b, sigma_b = sample_v4_hp(rng, B)
    theta_true = rng.uniform(
        mu0_b - K_anchor*sigma0_b, mu0_b + K_anchor*sigma0_b)
    # Sigma-anchored integration grids
    theta_grids = np.empty((B, n_grid), dtype=np.float64)
    for i in range(B):
        theta_grids[i] = np.linspace(
            mu0_b[i] - K_anchor*sigma0_b[i],
            mu0_b[i] + K_anchor*sigma0_b[i],
            n_grid,
        )
    # K D draws per sample, all centered at θ_true_i with std σ_i
    # Each D_arrs[i, k] ~ N(theta_true[i], sigma_b[i])
    D_arrs = rng.normal(
        loc=theta_true[:, None],
        scale=sigma_b[:, None],
        size=(B, K),
    )
    return theta_grids, D_arrs, mu0_b, sigma0_b, sigma_b


def per_sample_loss_kd(net, theta_grid_i, D_arr_i, mu0_i, sigma0_i, sigma_i):
    """Mean integrated_p loss over K D realizations for one (θ_true, slice)."""
    prior_hp = jnp.stack([
        jnp.full(theta_grid_i.shape, mu0_i),
        jnp.full(theta_grid_i.shape, sigma0_i),
    ], axis=-1)
    lik_hp = jnp.full(theta_grid_i.shape + (1,), sigma_i)
    eta = net(theta_grid_i, prior_hp, lik_hp)
    w = sigma0_i**2 / (sigma_i**2 + sigma0_i**2)

    def loss_for_one_d(D):
        p = power_law_tilted_pvalue_jax(
            theta=theta_grid_i, D=D, w=w, mu0=mu0_i, sigma=sigma_i,
            eta=eta, statistic_name="waldo",
        )
        return integrated_pvalue_loss(p[None, :], theta_grid_i[None, :])
    return jnp.mean(jax.vmap(loss_for_one_d)(D_arr_i))


def batch_loss_kd(net, theta_grids, D_arrs, mu0_b, sigma0_b, sigma_b):
    return jnp.mean(jax.vmap(per_sample_loss_kd, in_axes=(None, 0, 0, 0, 0, 0))(
        net, theta_grids, D_arrs, mu0_b, sigma0_b, sigma_b
    ))


def train_continuous(K, n_steps=600, B=128, n_grid=51, lr=1e-3,
                      weight_decay=1e-4, seed=42, eval_every=100,
                      eval_fixed_slices=None, label=""):
    rng_data = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 9999)

    eta_net = make_etanet(seed=seed)
    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = opt.init(eqx.filter(eta_net, eqx.is_array))

    @eqx.filter_jit
    def step(net, opt_state, theta_g, D, mu0, sigma0, sigma):
        loss, grads = eqx.filter_value_and_grad(batch_loss_kd)(
            net, theta_g, D, mu0, sigma0, sigma)
        updates, new_state = opt.update(
            grads, opt_state, eqx.filter(net, eqx.is_array))
        new_net = eqx.apply_updates(net, updates)
        return new_net, new_state, loss

    @eqx.filter_jit
    def eval_loss(net, theta_g, D, mu0, sigma0, sigma):
        return batch_loss_kd(net, theta_g, D, mu0, sigma0, sigma)

    # Pre-build a continuous-distribution holdout for monitoring
    holdout = sample_batch(rng_eval, B=512, K=8, n_grid=n_grid)
    (h_tg, h_D, h_mu0, h_sg0, h_sg) = holdout
    h_tg = jnp.asarray(h_tg); h_D = jnp.asarray(h_D)
    h_mu0 = jnp.asarray(h_mu0); h_sg0 = jnp.asarray(h_sg0); h_sg = jnp.asarray(h_sg)

    history = {'step': [], 'train_loss': [], 'continuous_holdout': []}
    for slice_label, _ in (eval_fixed_slices or []):
        history[f'fixed_{slice_label}'] = []

    t0 = time.time()
    for s in range(n_steps):
        theta_g, D, mu0, sg0, sg = sample_batch(rng_data, B, K, n_grid)
        eta_net, opt_state, train_loss = step(
            eta_net, opt_state,
            jnp.asarray(theta_g), jnp.asarray(D),
            jnp.asarray(mu0), jnp.asarray(sg0), jnp.asarray(sg),
        )
        if (s + 1) % eval_every == 0 or s == 0:
            cont_loss = float(eval_loss(eta_net, h_tg, h_D, h_mu0, h_sg0, h_sg))
            history['step'].append(s + 1)
            history['train_loss'].append(float(train_loss))
            history['continuous_holdout'].append(cont_loss)
            extra = ""
            for slice_label, slice_data in (eval_fixed_slices or []):
                (mu0_v, sg0_v, sg_v, n_pts) = slice_data
                rng_fix = np.random.default_rng(hash(slice_label) % (2**31))
                theta_true_f = rng_fix.uniform(
                    mu0_v - 5*sg0_v, mu0_v + 5*sg0_v, n_pts)
                D_f = rng_fix.normal(theta_true_f, sg_v)
                tg_f = np.linspace(mu0_v - 5*sg0_v, mu0_v + 5*sg0_v, n_grid)
                tg_b = np.broadcast_to(tg_f, (n_pts, n_grid)).copy()
                D_b = D_f[:, None]  # K=1
                fix_loss = float(eval_loss(
                    eta_net,
                    jnp.asarray(tg_b), jnp.asarray(D_b),
                    jnp.asarray(np.full(n_pts, mu0_v)),
                    jnp.asarray(np.full(n_pts, sg0_v)),
                    jnp.asarray(np.full(n_pts, sg_v)),
                ))
                history[f'fixed_{slice_label}'].append(fix_loss)
                extra += f" {slice_label}={fix_loss:.3f}"
            print(f"  [{label}] step {s+1:>4d}: train={float(train_loss):.4f}  "
                  f"holdout={cont_loss:.4f}{extra}")

    elapsed = time.time() - t0
    return eta_net, history, elapsed


def main():
    fixed_slices = [
        ("midw",  (0.0, 1.0, 1.0, 512)),
        ("midhi", (0.0, 2.0, 1.0, 512)),
        ("highw", (0.0, 4.0, 1.0, 512)),
    ]
    print("\nReference targets per fixed slice (from probe_function_constrained_min):")
    print("  mid-w  (σ₀=1): wald 1.900, fn-min 1.390  (27% headroom)")
    print("  mid-hi (σ₀=2): wald 2.056, fn-min 1.469  (29% headroom)")
    print("  high-w (σ₀=4): wald 1.820, fn-min 1.510  (17% headroom)")
    print("\nReference for continuous holdout (from synthesis note):")
    print("  L_wald = 1.363, L_oracle_3 = 1.175")

    print("\n=== K=16 D-MC per (θ_true) ===")
    net16, hist16, t16 = train_continuous(
        K=16, n_steps=600, eval_every=100,
        eval_fixed_slices=fixed_slices, label="K=16",
    )
    print(f"  elapsed: {t16:.1f}s")

    print("\n=== K=1 D-MC per (θ_true) [matches v4 sparsity] ===")
    net1, hist1, t1 = train_continuous(
        K=1, n_steps=600, eval_every=100,
        eval_fixed_slices=fixed_slices, label="K=1",
    )
    print(f"  elapsed: {t1:.1f}s")

    print("\n" + "="*78)
    print("Final per-evaluation comparison")
    print("="*78)
    metrics = ['continuous_holdout', 'fixed_midw', 'fixed_midhi', 'fixed_highw']
    refs = {
        'continuous_holdout': ('continuous holdout', 1.363, 1.175),
        'fixed_midw':         ('mid-w  fixed slice', 1.900, 1.390),
        'fixed_midhi':        ('mid-hi fixed slice', 2.056, 1.469),
        'fixed_highw':        ('high-w fixed slice', 1.820, 1.510),
    }
    print(f"{'metric':<24} {'wald':>7} {'fn-min':>7} {'K=16':>7} {'K=1':>7}  "
          f"{'K=16 cap':>9} {'K=1 cap':>9}")
    for m in metrics:
        label, wald, fn_min = refs[m]
        v16 = hist16[m][-1]
        v1 = hist1[m][-1]
        cap16 = 100*(wald - v16)/(wald - fn_min)
        cap1 = 100*(wald - v1)/(wald - fn_min)
        print(f"{label:<24} {wald:>7.4f} {fn_min:>7.4f} "
              f"{v16:>7.4f} {v1:>7.4f}  "
              f"{cap16:>+8.1f}% {cap1:>+8.1f}%")

    # η(θ) at each fixed slice (final)
    print("\nFinal η(θ) range at each fixed slice:")
    for (slice_label, (mu0_v, sg0_v, sg_v, _)) in fixed_slices:
        tg = jnp.asarray(np.linspace(mu0_v - 5*sg0_v, mu0_v + 5*sg0_v, 51),
                          dtype=jnp.float64)
        prior_hp = jnp.stack([
            jnp.full(tg.shape, mu0_v), jnp.full(tg.shape, sg0_v),
        ], axis=-1)
        lik_hp = jnp.full(tg.shape + (1,), sg_v)
        eta16 = np.asarray(net16(tg, prior_hp, lik_hp))
        eta1 = np.asarray(net1(tg, prior_hp, lik_hp))
        print(f"  {slice_label}:  K=16 [{eta16.min():+.3f}, {eta16.max():+.3f}] mean {eta16.mean():+.3f}  "
              f"|  K=1 [{eta1.min():+.3f}, {eta1.max():+.3f}] mean {eta1.mean():+.3f}")

    # Save
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "continuous_kd.npz"
    np.savez(out_path,
             hist16_step=hist16['step'],
             hist16_holdout=hist16['continuous_holdout'],
             hist1_step=hist1['step'],
             hist1_holdout=hist1['continuous_holdout'],
             hist16_midw=hist16['fixed_midw'],
             hist1_midw=hist1['fixed_midw'],
             hist16_highw=hist16['fixed_highw'],
             hist1_highw=hist1['fixed_highw'])
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
