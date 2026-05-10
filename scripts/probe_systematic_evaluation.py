"""Systematic evaluation of all trained fixtures against the CALIBRATED
oracle (benchmark #3, computed in probe_calibrated_benchmark.py).

For each fixture, computes:
  - val_loss: mean integrated_p loss on the probe batch using the trained
    network's eta(theta) function.
  - envelope_calibrated: (val_loss - L_oracle_3) / (L_Wald - L_oracle_3).
    0 = matches calibrated oracle (best possible calibrated learner);
    1 = matches Wald (no learning); >1 = WORSE than Wald.
  - admissibility_rate: fraction of (theta, prior, lik) probe points where
    |eta_pred| <= 2.

Same probe seed as probe_calibrated_benchmark.py so the references match.
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


def per_sample_loss_trained(probe, art: EtaArtifact, K=5.0):
    """Per-sample integrated_p using the trained network's eta(theta).

    The network outputs eta(theta) varying across the integration grid.
    Also computes admissibility rate (fraction of grid points with
    |eta| <= 2).
    """
    n = probe.theta.size
    losses = np.empty(n, dtype=np.float64)
    admissible_per_sample = np.empty(n, dtype=np.float64)
    eta_means_per_sample = np.empty(n, dtype=np.float64)
    eta_extremes_per_sample = np.empty(n, dtype=np.float64)
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
        admissible_per_sample[i] = float(np.mean(np.abs(eta_curve) <= 2.0))
        eta_means_per_sample[i] = float(np.mean(eta_curve))
        eta_extremes_per_sample[i] = float(np.max(np.abs(eta_curve)))

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
    return {
        "mean_loss": float(np.mean(losses)),
        "admissibility_rate": float(np.mean(admissible_per_sample)),
        "eta_mean": float(np.mean(eta_means_per_sample)),
        "max_abs_eta": float(np.max(eta_extremes_per_sample)),
    }


def main():
    rng = np.random.default_rng(0xCAFE)
    hp = _v4_hp()
    # Use n=32 to match probe_calibrated_benchmark.py for fair comparison.
    probe = build_probe_batch(
        scheme_name="power_law", n=32, rng=rng,
        hyperparam_distribution=hp,
        prior_names=["loc", "scale"], lik_names=["sigma"],
    )
    print(f"Probe batch: n={probe.theta.size}\n")

    # Use values from probe_calibrated_benchmark.py (same seed/n).
    L_ORACLE_3 = 1.175  # calibrated per-(theta, prior, lik) oracle
    L_WALD = float(np.mean(per_sample_loss_const_eta(probe, 1.0)))
    L_WALDO = float(np.mean(per_sample_loss_const_eta(probe, 0.0)))
    print(f"References:")
    print(f"  L (Wald, eta=1)                 = {L_WALD:.4f}")
    print(f"  L (WALDO, eta=0)                = {L_WALDO:.4f}")
    print(f"  L (calibrated oracle #3)        = {L_ORACLE_3:.4f}")
    print(f"  Headroom: Wald - Oracle         = {L_WALD - L_ORACLE_3:.4f}")
    print(f"  Headroom relative               = {(L_WALD - L_ORACLE_3)/L_WALD * 100:.1f}%")
    print()

    fixtures = [
        ("baseline",                  "artifacts/probe_v4_baseline.eqx"),
        ("no_boundary",               "artifacts/probe_v4_no_boundary.eqx"),
        ("no_norm",                   "artifacts/probe_v4_no_norm.eqx"),
        ("anti_wald_10",              "artifacts/probe_v4_anti_wald_10.eqx"),
        ("stratified",                "artifacts/probe_v4_stratified.eqx"),
        ("anti_collapse=100 (1/var)", "artifacts/probe_v4_anti_collapse_100.eqx"),
        ("anti_collapse=1 (1/var)",   "artifacts/probe_v4_anti_collapse_1.eqx"),
        ("smallb+lr+ac1 decay=0.8",   "artifacts/probe_v4_smallbatch_largelr.eqx"),
        ("smallb+lr+ac1 nodecay",     "artifacts/probe_v4_smallbatch_largelr_nodecay.eqx"),
        ("smallb+lr noac",            "artifacts/probe_v4_smallbatch_largelr_noac.eqx"),
    ]

    print(f"{'fixture':<32} {'loss':>8} {'env_cal':>8} {'eta_mean':>10} "
          f"{'max|eta|':>10} {'admiss%':>8}")
    print("-" * 86)
    print(f"{'CALIBRATED ORACLE #3':<32} {L_ORACLE_3:>8.4f} {0.000:>8.3f}")
    print(f"{'Wald (eta=1)':<32} {L_WALD:>8.4f} {1.000:>8.3f}")
    print(f"{'WALDO (eta=0)':<32} {L_WALDO:>8.4f} "
          f"{(L_WALDO - L_ORACLE_3)/(L_WALD - L_ORACLE_3):>8.3f}")
    print()
    for name, path in fixtures:
        if not Path(path).exists():
            print(f"{name:<32} (missing)")
            continue
        art = EtaArtifact(
            artifact_path=Path(path), name="probe", version=f"probe_v4_{name}",
        )
        art.load()
        result = per_sample_loss_trained(probe, art)
        loss = result["mean_loss"]
        env_cal = (loss - L_ORACLE_3) / (L_WALD - L_ORACLE_3)
        adm = result["admissibility_rate"] * 100
        print(f"{name:<32} {loss:>8.4f} {env_cal:>8.3f} "
              f"{result['eta_mean']:>+10.3f} {result['max_abs_eta']:>10.3f} "
              f"{adm:>7.1f}%")

    print()
    print("Legend:")
    print("  loss     = mean per-sample integrated_p loss on probe batch.")
    print("  env_cal  = (loss - L_oracle_3) / (L_Wald - L_oracle_3).")
    print("             0.0 = matches CALIBRATED oracle (best possible).")
    print("             1.0 = matches Wald (no learning beyond constant).")
    print("             >1.0 = WORSE than Wald.")
    print("  eta_mean = average eta output across all probe (theta, prior, lik) points.")
    print("  max|eta| = max |eta| across all probe points (admissibility check).")
    print("  admiss%  = fraction of probe points with |eta| <= 2.")


if __name__ == "__main__":
    main()
