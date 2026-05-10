"""Thread 5: how far from global-optimum loss does each trained fixture sit?

For each of the 5 trained probes + the original v4 baseline:
  1. Compute the trained network's integrated_p loss averaged over a
     held-out probe batch (using the network's predicted eta(theta)).
  2. Compute the GLOBAL constant-eta optimum loss on the same batch:
     for each probe sample, find the constant eta that minimizes its
     integrated_p loss, take that loss value, average over the batch.
  3. Compute the WALDO baseline loss (constant eta=0, the bare WALDO).
  4. Compute the Wald baseline loss (constant eta=1).
  5. Report each fixture's loss as both:
     - absolute value
     - ratio to the global per-slice optimum (1.0 = matches optimum)
     - rank in the [optimum, Wald] envelope (0 = optimal, 1 = Wald-equivalent)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.eta_artifact import EtaArtifact
from frasian.learned.training.diagnostics import build_probe_batch
from frasian.learned.training.hyperparam_distribution import (
    HyperparamDistribution,
    ScalarDist,
)
from frasian.learned.training.losses import integrated_pvalue_loss
from frasian.learned.training.pvalue_jax import power_law_tilted_pvalue_jax


def _v4_hp_distribution() -> HyperparamDistribution:
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
    """Per-sample integrated_p at a CONSTANT eta value across all theta."""
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


def per_sample_loss_argmin(probe, eta_grid=np.linspace(-1.5, 1.5, 121), K=5.0):
    """Per-sample minimum integrated_p across a constant-eta grid."""
    n = probe.theta.size
    losses_argmin = np.empty(n, dtype=np.float64)
    for i in range(n):
        mu0 = float(probe.prior_hp[i, 0])
        sigma0 = float(probe.prior_hp[i, 1])
        sigma = float(probe.lik_hp[i, 0])
        D = float(probe.D[i])
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        theta_grid = jnp.asarray(np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401))
        per_eta = []
        for eta in eta_grid:
            eta_arr = jnp.full(theta_grid.shape, float(eta))
            p = power_law_tilted_pvalue_jax(
                theta=theta_grid, D=jnp.asarray(D), w=jnp.asarray(w),
                mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
                eta=eta_arr, statistic_name="waldo",
            )
            per_eta.append(float(integrated_pvalue_loss(
                jnp.asarray(p)[None, :], jnp.asarray(theta_grid)[None, :]
            )))
        losses_argmin[i] = min(per_eta)
    return losses_argmin


def per_sample_loss_trained(probe, art: EtaArtifact, K=5.0):
    """Per-sample integrated_p using the trained network's eta(theta)."""
    n = probe.theta.size
    losses = np.empty(n, dtype=np.float64)
    for i in range(n):
        mu0 = float(probe.prior_hp[i, 0])
        sigma0 = float(probe.prior_hp[i, 1])
        sigma = float(probe.lik_hp[i, 0])
        D = float(probe.D[i])
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        theta_grid_np = np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401)
        eta_curve = art.predict_eta(
            theta=theta_grid_np,
            prior_hp=np.asarray([mu0, sigma0]),
            lik_hp=np.asarray([sigma]),
        )
        theta_grid = jnp.asarray(theta_grid_np)
        eta_arr = jnp.asarray(eta_curve)
        p = power_law_tilted_pvalue_jax(
            theta=theta_grid, D=jnp.asarray(D), w=jnp.asarray(w),
            mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
            eta=eta_arr, statistic_name="waldo",
        )
        losses[i] = float(integrated_pvalue_loss(
            jnp.asarray(p)[None, :], jnp.asarray(theta_grid)[None, :]
        ))
    return losses


def main():
    rng = np.random.default_rng(0xCAFE)
    hp_dist = _v4_hp_distribution()
    probe = build_probe_batch(
        scheme_name="power_law", n=64, rng=rng,
        hyperparam_distribution=hp_dist,
        prior_names=["loc", "scale"], lik_names=["sigma"],
    )

    print(f"\nProbe batch: n={probe.theta.size}")
    print(f"argmin_eta range: [{probe.argmin_eta.min():+.2f}, {probe.argmin_eta.max():+.2f}]")
    print(f"argmin_eta mean:  {probe.argmin_eta.mean():+.3f}\n")

    # Reference points: optimum, WALDO (eta=0), Wald (eta=1)
    print("Computing reference losses...")
    losses_optimum = per_sample_loss_argmin(probe)
    losses_waldo = per_sample_loss_const_eta(probe, 0.0)
    losses_wald = per_sample_loss_const_eta(probe, 1.0)
    mean_optimum = float(losses_optimum.mean())
    mean_waldo = float(losses_waldo.mean())
    mean_wald = float(losses_wald.mean())
    print(f"  loss(optimum) = {mean_optimum:.4f}")
    print(f"  loss(WALDO)   = {mean_waldo:.4f}  (eta=0)")
    print(f"  loss(Wald)    = {mean_wald:.4f}  (eta=1)\n")

    # Trained fixtures
    fixtures = [
        ("baseline",     "artifacts/probe_v4_baseline.eqx"),
        ("no_boundary",  "artifacts/probe_v4_no_boundary.eqx"),
        ("no_norm",      "artifacts/probe_v4_no_norm.eqx"),
        ("anti_wald_10", "artifacts/probe_v4_anti_wald_10.eqx"),
        ("stratified",   "artifacts/probe_v4_stratified.eqx"),
    ]

    print(f"{'fixture':>15} {'loss':>10} {'ratio_opt':>12} {'envelope':>10}")
    print(f"{'-'*15} {'-'*10} {'-'*12} {'-'*10}")
    print(f"{'OPTIMUM':>15} {mean_optimum:10.4f} {1.000:>12.3f} {0.000:>10.3f}")
    print(f"{'WALDO (eta=0)':>15} {mean_waldo:10.4f} {mean_waldo/mean_optimum:>12.3f} "
          f"{(mean_waldo-mean_optimum)/(mean_wald-mean_optimum):>10.3f}")

    for name, path in fixtures:
        if not Path(path).exists():
            print(f"{name:>15} (missing)")
            continue
        art = EtaArtifact(
            artifact_path=Path(path), name="probe", version=f"probe_v4_{name}",
        )
        art.load()
        losses_trained = per_sample_loss_trained(probe, art)
        mean_trained = float(losses_trained.mean())
        ratio_opt = mean_trained / mean_optimum
        envelope_pos = (mean_trained - mean_optimum) / (mean_wald - mean_optimum)
        print(f"{name:>15} {mean_trained:10.4f} {ratio_opt:>12.3f} "
              f"{envelope_pos:>10.3f}")

    print(f"{'Wald (eta=1)':>15} {mean_wald:10.4f} {mean_wald/mean_optimum:>12.3f} "
          f"{1.000:>10.3f}")
    print()
    print("Legend:")
    print("  ratio_opt = loss / loss(optimum). 1.0 = matches global optimum.")
    print("  envelope  = (loss - loss_opt) / (loss_Wald - loss_opt).")
    print("              0.0 = at optimum; 1.0 = same as constant Wald.")


if __name__ == "__main__":
    main()
