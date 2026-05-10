"""Function-constrained D-marginalized integrated_p minimum.

For each fixed (mu0, sigma0, sigma) slice, ask: what is the lowest
integrated_p loss achievable by ANY function eta(theta) (parameterized
on a grid, jointly optimized via gradient descent over E_{theta_true,D}
[integrated_p])?

Compare to:
  - Wald baseline (eta=0 everywhere).
  - Per-theta_true argmin sum (the "L_oracle_3" benchmark style):
    each theta_true gets its own *constant* eta minimizing its
    E_D[integrated_p]. This is a lower bound — generally NOT achievable
    by a function eta(theta) because each theta_true would want a
    different constant.

If function_constrained_min ≈ per_test_argmin → the benchmark is
achievable as a function; gradient descent leaving headroom = problem.

If function_constrained_min >> per_test_argmin → the benchmark is
unachievable; Basin A may already be near the function-optimum.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.training.losses import integrated_pvalue_loss
from frasian.learned.training.pvalue_jax import power_law_tilted_pvalue_jax


def _per_sample_loss(theta_grid, eta_at_grid, D, w, mu0, sigma):
    """integrated_p loss for ONE D realization with eta(theta_grid)."""
    p = power_law_tilted_pvalue_jax(
        theta=theta_grid, D=D, w=w, mu0=mu0, sigma=sigma,
        eta=eta_at_grid, statistic_name="waldo",
    )
    return integrated_pvalue_loss(p[None, :], theta_grid[None, :])


def function_constrained_min(mu0, sigma0, sigma, n_mc=512, n_grid=51,
                              K=5.0, n_steps=2000, lr=1e-2, seed=0):
    """Optimize eta(theta_grid) to minimize E_{theta_true, D}[integrated_p].

    LIKELIHOOD-ANCHORED: integration domain and theta_true sampling both
    scale with sigma (likelihood scale), not sigma0 (prior scale). This
    makes the loss σ₀-independent for prior-free statistics like Wald.
    """
    rng = np.random.default_rng(seed)
    w = float(sigma0**2 / (sigma**2 + sigma0**2))
    theta_grid = jnp.asarray(
        np.linspace(mu0 - K * sigma, mu0 + K * sigma, n_grid),
        dtype=jnp.float64,
    )
    theta_true_samples = rng.uniform(mu0 - K*sigma, mu0 + K*sigma, n_mc)
    D_samples = jnp.asarray(rng.normal(theta_true_samples, sigma),
                             dtype=jnp.float64)

    eta_knots = jnp.zeros(n_grid, dtype=jnp.float64)

    w_j = jnp.asarray(w, dtype=jnp.float64)
    mu0_j = jnp.asarray(mu0, dtype=jnp.float64)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float64)

    def loss_fn(eta_knots, D_batch):
        per_sample = jax.vmap(
            lambda D: _per_sample_loss(theta_grid, eta_knots, D, w_j, mu0_j, sigma_j)
        )(D_batch)
        return jnp.mean(per_sample)

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eta_knots)

    @jax.jit
    def step(eta_knots, opt_state, D_batch):
        loss, grads = jax.value_and_grad(loss_fn)(eta_knots, D_batch)
        updates, new_state = opt.update(grads, opt_state, eta_knots)
        new_eta = optax.apply_updates(eta_knots, updates)
        return new_eta, new_state, loss

    losses = []
    for _ in range(n_steps):
        eta_knots, opt_state, loss = step(eta_knots, opt_state, D_samples)
        losses.append(float(loss))

    final_loss = float(loss_fn(eta_knots, D_samples))
    return {
        'final_loss': final_loss,
        'eta_knots': np.asarray(eta_knots),
        'theta_grid': np.asarray(theta_grid),
        'training_curve': losses,
    }


def per_theta_true_argmin(mu0, sigma0, sigma, n_theta_true=64, n_d_inner=32,
                          n_grid=51, eta_n=21, K=5.0, seed=1):
    """For each theta_true_i, find the constant eta_i* minimizing
    E_D[integrated_p | theta_true_i]. Average that minimum over theta_true_i.

    This is a lower bound on what any function eta(theta) can achieve at
    this slice, because the per-i optimum is uncoupled across i.
    """
    rng = np.random.default_rng(seed)
    w = float(sigma0**2 / (sigma**2 + sigma0**2))
    theta_grid = jnp.asarray(
        np.linspace(mu0 - K * sigma, mu0 + K * sigma, n_grid),
        dtype=jnp.float64,
    )
    eta_const_grid = jnp.asarray(np.linspace(-1.5, 1.5, eta_n), dtype=jnp.float64)
    w_j = jnp.asarray(w, dtype=jnp.float64)
    mu0_j = jnp.asarray(mu0, dtype=jnp.float64)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float64)

    @jax.jit
    def loss_grid(D_batch):
        # D_batch: (n_d,). Return loss for each (D, eta_const) combo.
        def for_eta(eta_c):
            eta_arr = jnp.full(theta_grid.shape, eta_c)
            return jax.vmap(
                lambda D: _per_sample_loss(theta_grid, eta_arr, D, w_j, mu0_j, sigma_j)
            )(D_batch)
        return jax.vmap(for_eta)(eta_const_grid)  # (n_eta, n_d)

    theta_true_samples = rng.uniform(mu0 - K*sigma, mu0 + K*sigma, n_theta_true)
    per_test_min = []
    per_test_argmin_eta = []
    for theta_t in theta_true_samples:
        D_inner = rng.normal(theta_t, sigma, n_d_inner)
        D_inner = jnp.asarray(D_inner, dtype=jnp.float64)
        losses = loss_grid(D_inner)  # (n_eta, n_d)
        avg_per_eta = jnp.mean(losses, axis=1)  # (n_eta,)
        best_idx = int(jnp.argmin(avg_per_eta))
        per_test_min.append(float(avg_per_eta[best_idx]))
        per_test_argmin_eta.append(float(eta_const_grid[best_idx]))

    return {
        'mean_loss': float(np.mean(per_test_min)),
        'per_test_argmin_eta_range': (
            float(np.min(per_test_argmin_eta)),
            float(np.max(per_test_argmin_eta)),
        ),
        'theta_true_samples': theta_true_samples,
        'per_test_argmin_eta': np.asarray(per_test_argmin_eta),
    }


def constant_eta_baseline(mu0, sigma0, sigma, eta_value, n_mc=512, n_grid=51,
                           K=5.0, seed=2):
    """Loss with a CONSTANT eta_value across the integration grid.

    eta_value=0.0 → WALDO (full posterior + prior offset)
    eta_value=1.0 → Wald (data-only; collapses prior dependence)
    """
    rng = np.random.default_rng(seed)
    w = float(sigma0**2 / (sigma**2 + sigma0**2))
    theta_grid = jnp.asarray(
        np.linspace(mu0 - K * sigma, mu0 + K * sigma, n_grid),
        dtype=jnp.float64,
    )
    theta_true_samples = rng.uniform(mu0 - K*sigma, mu0 + K*sigma, n_mc)
    D_samples = jnp.asarray(rng.normal(theta_true_samples, sigma),
                             dtype=jnp.float64)
    eta_const = jnp.full(theta_grid.shape, eta_value, dtype=jnp.float64)
    w_j = jnp.asarray(w, dtype=jnp.float64)
    mu0_j = jnp.asarray(mu0, dtype=jnp.float64)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float64)

    @jax.jit
    def compute(D_batch):
        return jnp.mean(jax.vmap(
            lambda D: _per_sample_loss(theta_grid, eta_const, D, w_j, mu0_j, sigma_j)
        )(D_batch))

    return float(compute(D_samples))


def main():
    slices = [
        # (mu0, sigma0, sigma, label)
        (0.0, 0.3, 1.0, "low-w  (σ₀=0.3)"),
        (0.0, 1.0, 1.0, "mid-w  (σ₀=1.0)"),
        (0.0, 2.0, 1.0, "mid-hi (σ₀=2.0)"),
        (0.0, 4.0, 1.0, "high-w (σ₀=4.0)"),
    ]
    print(f"\n{'slice':<22}  {'w':>6}  {'WALDO':>8}  {'Wald':>8}  "
          f"{'fn-min':>8}  {'per-test':>9}  eta(θ) range")
    print(f"{' ':<22}  {' ':>6}  {'(η=0)':>8}  {'(η=1)':>8}  "
          f"{' ':>8}  {' ':>9}")
    print("-" * 100)

    results = []
    for mu0, sigma0, sigma, label in slices:
        waldo = constant_eta_baseline(mu0, sigma0, sigma, eta_value=0.0)
        wald = constant_eta_baseline(mu0, sigma0, sigma, eta_value=1.0)
        fn = function_constrained_min(mu0, sigma0, sigma, n_steps=1500)
        per = per_theta_true_argmin(mu0, sigma0, sigma)
        eta_lo = float(fn['eta_knots'].min())
        eta_hi = float(fn['eta_knots'].max())
        w = float(sigma0**2 / (sigma**2 + sigma0**2))
        print(f"{label:<22}  {w:>6.3f}  {waldo:>8.4f}  {wald:>8.4f}  "
              f"{fn['final_loss']:>8.4f}  {per['mean_loss']:>9.4f}  "
              f"[{eta_lo:+.2f}, {eta_hi:+.2f}]")
        results.append({
            'label': label, 'w': w, 'waldo': waldo, 'wald': wald,
            'fn_min': fn['final_loss'], 'per_test': per['mean_loss'],
            'eta_knots': fn['eta_knots'], 'theta_grid': fn['theta_grid'],
        })

    print("\nWald (η=1, data-only) is the natural reference: it ignores the")
    print("prior, so any improvement from using prior info must beat this.")
    print("\nfn-min vs Wald (η=1) per slice — TRUE achievable improvement:")
    for r in results:
        improvement = r['wald'] - r['fn_min']
        improvement_pct = 100.0 * improvement / r['wald']
        verdict = "headroom" if improvement > 0 else "fn-min WORSE than Wald"
        print(f"  {r['label']:<22}  Δ = {improvement:+.4f}  "
              f"({improvement_pct:+.1f}%)  [{verdict}]")

    print("\nfn-min vs WALDO (η=0, full prior) per slice — for reference:")
    for r in results:
        improvement = r['waldo'] - r['fn_min']
        improvement_pct = 100.0 * improvement / r['waldo']
        print(f"  {r['label']:<22}  Δ = {improvement:+.4f}  ({improvement_pct:+.1f}%)")


if __name__ == "__main__":
    main()
