"""Calibrated benchmarks for learned-eta.

Three benchmarks computed:
  #1 (post-selection oracle) — per-sample argmin (uses D[i]). Already in
     probe_loss_distance.py; not calibrated.
  #2 (best constant eta) — single eta applied to all probe samples;
     pick eta minimizing mean loss. Calibrated (no D-dependence).
  #3 (best calibrated eta function) — for each (theta, prior_hp, lik_hp)
     point on the probe, find argmin_eta E_D[p(theta; D, eta)] where
     D ~ likelihood(theta, sigma). Then evaluate the integrated_p loss
     using this calibrated eta function. Approximates the ceiling for
     a calibrated learner.

If #3 << Wald loss, there's real calibrated headroom.
If #3 ≈ Wald loss, no calibrated learner can beat constant-Wald and we
  need to rethink the framework's design assumptions.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.training.diagnostics import build_probe_batch
from frasian.learned.training.hyperparam_distribution import (
    HyperparamDistribution, ScalarDist,
)
from frasian.learned.training.losses import integrated_pvalue_loss
from frasian.learned.training.pvalue_jax import power_law_tilted_pvalue_jax


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


def per_sample_loss_const_eta(probe, eta_const, K=5.0):
    """Per-sample integrated_p with a CONSTANT eta. (Same as in
    probe_loss_distance.py.)"""
    n = probe.theta.size
    losses = np.empty(n, dtype=np.float64)
    for i in range(n):
        mu0 = float(probe.prior_hp[i, 0])
        sigma0 = float(probe.prior_hp[i, 1])
        sigma = float(probe.lik_hp[i, 0])
        D = float(probe.D[i])
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        theta_grid = jnp.asarray(np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401))
        eta_arr = jnp.full(theta_grid.shape, float(eta_const))
        p = power_law_tilted_pvalue_jax(
            theta=theta_grid, D=jnp.asarray(D), w=jnp.asarray(w),
            mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
            eta=eta_arr, statistic_name="waldo",
        )
        losses[i] = float(integrated_pvalue_loss(
            jnp.asarray(p)[None, :], jnp.asarray(theta_grid)[None, :]
        ))
    return losses


def calibrated_eta_function_oracle(probe, n_d_mc=200, K=5.0,
                                    eta_grid=np.linspace(-1.5, 1.5, 41)):
    """Benchmark #3: per-(theta, prior_hp, lik_hp) point oracle that
    integrates over D (preserving calibration).

    For each probe sample (theta_i, mu0_i, sigma0_i, sigma_i):
      For each eta in eta_grid:
        Sample n_d_mc draws D ~ N(theta_i, sigma_i).
        Compute mean_D[p(theta_i; D, eta)] -- but actually we need
        E_D[integrated_p loss], so we need to integrate over theta on
        a grid, then over D.
      Pick the eta minimizing E_D[loss].
      Compute the loss at this eta with the OBSERVED D[i].

    Total cost: n × |eta_grid| × n_d_mc × n_theta_grid pvalue calls.
    With n=64, |eta|=41, n_d_mc=200, n_theta=401: ~210M pvalue calls.
    Too slow.

    Simplification: for each probe sample, find argmin_eta of
    E_D[loss], computed by a small MC over D. Then evaluate the
    integrated_p loss at that calibrated eta with the original D[i].

    Reduced cost: n × |eta_grid| × n_d_mc × n_theta = ~210M -- still
    too slow. Use n_d_mc=40 and |eta_grid|=21 -> ~22M calls; ~30 min on
    CPU. Not feasible.

    Practical version: use the (theta_test, prior_hp, lik_hp) of each
    probe sample but compute argmin over D-marginalized loss using a
    small MC. Then report the loss at this eta against the original D.
    """
    n = probe.theta.size
    rng = np.random.default_rng(0xCAFE)
    calibrated_etas = np.empty(n, dtype=np.float64)
    losses_at_calibrated_eta = np.empty(n, dtype=np.float64)

    for i in range(n):
        theta_i = float(probe.theta[i])
        mu0 = float(probe.prior_hp[i, 0])
        sigma0 = float(probe.prior_hp[i, 1])
        sigma = float(probe.lik_hp[i, 0])
        D_obs = float(probe.D[i])
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        theta_grid_np = np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401)
        theta_grid = jnp.asarray(theta_grid_np)

        # Sample D values around theta_i for MC over D.
        D_samples = rng.normal(loc=theta_i, scale=sigma, size=n_d_mc)

        # For each eta in the search grid, compute the AVERAGE integrated-p
        # loss over D samples.
        avg_loss_per_eta = np.empty(eta_grid.size, dtype=np.float64)
        for j, eta in enumerate(eta_grid):
            eta_arr = jnp.full(theta_grid.shape, float(eta))
            losses_per_d = []
            for D_mc in D_samples:
                p = power_law_tilted_pvalue_jax(
                    theta=theta_grid, D=jnp.asarray(float(D_mc)),
                    w=jnp.asarray(w), mu0=jnp.asarray(mu0),
                    sigma=jnp.asarray(sigma), eta=eta_arr,
                    statistic_name="waldo",
                )
                losses_per_d.append(float(integrated_pvalue_loss(
                    jnp.asarray(p)[None, :], jnp.asarray(theta_grid)[None, :]
                )))
            avg_loss_per_eta[j] = float(np.mean(losses_per_d))

        # Pick eta minimizing E_D[loss].
        best_idx = int(np.argmin(avg_loss_per_eta))
        calibrated_etas[i] = float(eta_grid[best_idx])

        # Evaluate loss at this eta against the OBSERVED D[i].
        eta_cal = float(eta_grid[best_idx])
        eta_arr_cal = jnp.full(theta_grid.shape, eta_cal)
        p_obs = power_law_tilted_pvalue_jax(
            theta=theta_grid, D=jnp.asarray(D_obs),
            w=jnp.asarray(w), mu0=jnp.asarray(mu0),
            sigma=jnp.asarray(sigma), eta=eta_arr_cal,
            statistic_name="waldo",
        )
        losses_at_calibrated_eta[i] = float(integrated_pvalue_loss(
            jnp.asarray(p_obs)[None, :], jnp.asarray(theta_grid)[None, :]
        ))

    return calibrated_etas, losses_at_calibrated_eta


def main():
    rng = np.random.default_rng(0xCAFE)
    hp = _v4_hp()
    probe = build_probe_batch(
        scheme_name="power_law", n=32, rng=rng,  # use 32 for speed
        hyperparam_distribution=hp,
        prior_names=["loc", "scale"], lik_names=["sigma"],
    )
    print(f"\nProbe batch: n={probe.theta.size}")

    # Benchmark #1: per-sample post-selection oracle (already in
    # probe_loss_distance.py; quick recompute here)
    print("\n#1 Per-sample post-selection oracle (uses D[i]):")
    eta_grid_post = np.linspace(-1.5, 1.5, 121)
    losses_post = []
    for i in range(probe.theta.size):
        per_eta = []
        for e in eta_grid_post:
            losses_e = per_sample_loss_const_eta(
                # Build a tiny "probe of one" inline
                _make_one_sample_probe(probe, i), float(e),
            )
            per_eta.append(float(losses_e[0]))
        losses_post.append(min(per_eta))
    mean_post = float(np.mean(losses_post))
    print(f"  loss = {mean_post:.4f}")

    # Benchmark #2: best constant eta across the WHOLE batch
    print("\n#2 Best constant eta (calibrated, simplest learner):")
    eta_grid_const = np.linspace(-1.5, 1.5, 41)
    avg_per_eta = np.empty(eta_grid_const.size, dtype=np.float64)
    for j, e in enumerate(eta_grid_const):
        losses_e = per_sample_loss_const_eta(probe, float(e))
        avg_per_eta[j] = float(np.mean(losses_e))
    best_idx = int(np.argmin(avg_per_eta))
    best_const_eta = float(eta_grid_const[best_idx])
    best_const_loss = float(avg_per_eta[best_idx])
    print(f"  argmin_const_eta = {best_const_eta:+.3f}")
    print(f"  loss             = {best_const_loss:.4f}")

    # Reference: WALDO (eta=0) and Wald (eta=1)
    waldo_loss = float(np.mean(per_sample_loss_const_eta(probe, 0.0)))
    wald_loss = float(np.mean(per_sample_loss_const_eta(probe, 1.0)))
    print(f"\nReference:")
    print(f"  WALDO (eta=0)  loss = {waldo_loss:.4f}")
    print(f"  Wald  (eta=1)  loss = {wald_loss:.4f}")

    # Benchmark #3: per-(theta, prior, lik) calibrated oracle
    print("\n#3 Per-(theta, prior, lik) calibrated oracle (D-marginalized):")
    print("    (this is slow — n_d_mc=40 over n=32 probe samples)")
    cal_etas, cal_losses = calibrated_eta_function_oracle(
        probe, n_d_mc=40, eta_grid=np.linspace(-1.5, 1.5, 21),
    )
    cal_loss_mean = float(np.mean(cal_losses))
    print(f"  calibrated etas range: [{cal_etas.min():+.2f}, {cal_etas.max():+.2f}]")
    print(f"  calibrated etas mean:  {cal_etas.mean():+.3f}")
    print(f"  loss (D-marginalized argmin, observed D):  {cal_loss_mean:.4f}")

    # Summary
    print("\n" + "="*60)
    print(f"{'benchmark':<50} {'loss':>8}")
    print("-"*60)
    print(f"{'#1 per-sample post-selection oracle':<50} {mean_post:>8.4f}")
    print(f"{'#2 best constant eta (calibrated)':<50} {best_const_loss:>8.4f}")
    print(f"{'#3 per-(theta,prior,lik) calibrated oracle':<50} {cal_loss_mean:>8.4f}")
    print(f"{'WALDO (eta=0)':<50} {waldo_loss:>8.4f}")
    print(f"{'Wald (eta=1)':<50} {wald_loss:>8.4f}")
    print()
    if cal_loss_mean < wald_loss:
        margin = (wald_loss - cal_loss_mean) / wald_loss * 100
        print(f"#3 IS LOWER than Wald by {margin:.1f}% → calibrated headroom EXISTS")
    else:
        margin = (cal_loss_mean - wald_loss) / wald_loss * 100
        print(f"#3 is {margin:.1f}% HIGHER than Wald → calibrated learner cannot beat Wald")


def _make_one_sample_probe(probe, i):
    """Make a probe-batch of size 1 from probe_batch sample i."""
    from frasian.learned.training.diagnostics import ProbeBatch
    return ProbeBatch(
        theta=probe.theta[i:i+1],
        D=probe.D[i:i+1],
        prior_hp=probe.prior_hp[i:i+1],
        lik_hp=probe.lik_hp[i:i+1],
        argmin_eta=probe.argmin_eta[i:i+1],
        w=probe.w[i:i+1],
    )


if __name__ == "__main__":
    main()
