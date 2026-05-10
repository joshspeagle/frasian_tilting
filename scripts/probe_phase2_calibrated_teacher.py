"""Phase 2 with CALIBRATED (D-marginalized) oracle teacher.

Differs from probe_phase2_teacher.py: instead of using probe.argmin_eta
(per-sample post-selection), compute a per-(theta, prior, lik)
D-marginalized argmin via MC over D ~ N(theta, sigma). This gives a
teacher whose held-out loss should be LOW (close to L_oracle_3 = 1.175),
unlike the post-selection oracle (loss 1.625).

If the network pre-trained on this teacher then drifts to Basin A
under width-loss-only training, that's unambiguous evidence the
calibrated oracle is NOT a local minimum of the integrated_p training
loss.
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

from frasian.learned.training.architecture import EtaNet
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


def calibrated_argmin_d_marginalized(theta_test, mu0, sigma0, sigma,
                                      n_d_mc=40, K=5.0,
                                      eta_grid=np.linspace(-1.5, 1.5, 21),
                                      rng=None):
    """For one (theta_test, prior, lik) point, compute the D-marginalized
    argmin of integrated_p loss.

    For each eta candidate, MC-estimate E_D[integrated_p(theta_grid; D, eta)]
    where D ~ N(theta_test, sigma). Pick eta minimizing this expectation.
    """
    if rng is None:
        rng = np.random.default_rng(0xCAFE)
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
    theta_grid = jnp.asarray(np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401))
    D_samples = rng.normal(loc=theta_test, scale=sigma, size=n_d_mc)

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
    best_idx = int(np.argmin(avg_loss_per_eta))
    return float(eta_grid[best_idx])


def main():
    print("Building teacher dataset (n=128) with D-marginalized argmin...")
    print("  (n_d_mc=20 per teacher point; ~5-10 minutes)")
    rng = np.random.default_rng(0xBEEF)
    hp = _v4_hp()
    probe = build_probe_batch(
        scheme_name="power_law", n=128, rng=rng,
        hyperparam_distribution=hp,
        prior_names=["loc", "scale"], lik_names=["sigma"],
    )

    # Compute D-marginalized argmin for each teacher point.
    rng_mc = np.random.default_rng(0xDEAD)
    calibrated_targets = np.empty(128, dtype=np.float64)
    for i in range(128):
        calibrated_targets[i] = calibrated_argmin_d_marginalized(
            theta_test=float(probe.theta[i]),
            mu0=float(probe.prior_hp[i, 0]),
            sigma0=float(probe.prior_hp[i, 1]),
            sigma=float(probe.lik_hp[i, 0]),
            n_d_mc=20,  # cheaper; trades MC precision for speed
            rng=rng_mc,
        )
        if (i + 1) % 16 == 0:
            print(f"  computed {i+1}/128 teacher targets")

    print(f"\nCalibrated teacher targets:")
    print(f"  range: [{calibrated_targets.min():+.2f}, {calibrated_targets.max():+.2f}]")
    print(f"  mean:  {calibrated_targets.mean():+.3f}")
    print(f"  vs post-selection argmin range: [{probe.argmin_eta.min():+.2f}, "
          f"{probe.argmin_eta.max():+.2f}]")

    # Pre-train EtaNet on calibrated targets.
    print("\nPre-training EtaNet (MSE to calibrated targets)...")
    theta_t = jnp.asarray(probe.theta, dtype=jnp.float64)
    prior_hp_t = jnp.asarray(probe.prior_hp, dtype=jnp.float64)
    lik_hp_t = jnp.asarray(probe.lik_hp, dtype=jnp.float64)
    target_eta = jnp.asarray(calibrated_targets, dtype=jnp.float64)

    # Match v4 normalization stats.
    sigma0_low, sigma0_high = 0.2, 5.0
    sigma_low, sigma_high = 0.5, 2.0
    mu0_loc = 0.0
    mu0_scale = (2.0 - (-2.0)) / float(np.sqrt(12.0))
    sigma0_loc = 0.5 * (np.log(sigma0_low) + np.log(sigma0_high))
    sigma0_scale = (np.log(sigma0_high) - np.log(sigma0_low)) / float(np.sqrt(12.0))
    sigma_loc = 0.5 * (np.log(sigma_low) + np.log(sigma_high))
    sigma_scale = (np.log(sigma_high) - np.log(sigma_low)) / float(np.sqrt(12.0))
    theta_loc = 0.0
    theta_scale = 60.0 / float(np.sqrt(12.0))
    feat_loc = (theta_loc, mu0_loc, sigma0_loc, sigma_loc)
    feat_scale = (theta_scale, mu0_scale, sigma0_scale, sigma_scale)
    feat_log = (False, False, True, True)
    eta_net = EtaNet(
        theta_dim=1, prior_dim=2, lik_dim=1,
        hidden_sizes=(128, 128, 128),
        feature_loc=feat_loc, feature_scale=feat_scale, feature_log=feat_log,
        key=jax.random.PRNGKey(42),
    )

    opt = optax.adamw(learning_rate=3e-3, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(eta_net, eqx.is_array))

    @eqx.filter_jit
    def step(net, opt_state, theta, prior, lik, target):
        def loss_fn(net):
            pred = net(theta, prior, lik)
            return jnp.mean((pred - target) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
        updates, new_opt_state = opt.update(
            grads, opt_state, eqx.filter(net, eqx.is_array),
        )
        new_net = eqx.apply_updates(net, updates)
        return new_net, new_opt_state, loss

    for epoch in range(300):
        eta_net, opt_state, mse = step(
            eta_net, opt_state, theta_t, prior_hp_t, lik_hp_t, target_eta,
        )
        if (epoch + 1) % 50 == 0:
            preds = np.asarray(eta_net(theta_t, prior_hp_t, lik_hp_t))
            corr = float(np.corrcoef(preds, np.asarray(target_eta))[0, 1])
            print(f"  ep {epoch+1:3d}: MSE = {float(mse):.4f}  "
                  f"corr = {corr:+.3f}  pred mean = {preds.mean():+.3f}")

    out_path = Path("artifacts/probe_v4_phase2cal_pretrained.eqx")
    eqx.tree_serialise_leaves(str(out_path), eta_net)
    print(f"\nWrote {out_path}")

    # Verify on a held-out probe of the SAME hp distribution.
    rng_holdout = np.random.default_rng(0xCAFE)  # different seed
    holdout = build_probe_batch(
        scheme_name="power_law", n=32, rng=rng_holdout,
        hyperparam_distribution=hp,
        prior_names=["loc", "scale"], lik_names=["sigma"],
    )
    K = 5.0
    losses = []
    for i in range(holdout.theta.size):
        mu0 = float(holdout.prior_hp[i, 0])
        sigma0 = float(holdout.prior_hp[i, 1])
        sigma = float(holdout.lik_hp[i, 0])
        D = float(holdout.D[i])
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        tg = np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401)
        prior_hp_b = np.broadcast_to(np.asarray([mu0, sigma0]), (401, 2))
        lik_hp_b = np.broadcast_to(np.asarray([sigma]), (401, 1))
        eta = np.asarray(eta_net(jnp.asarray(tg), jnp.asarray(prior_hp_b),
                                  jnp.asarray(lik_hp_b)))
        p = power_law_tilted_pvalue_jax(
            theta=jnp.asarray(tg), D=jnp.asarray(D), w=jnp.asarray(w),
            mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
            eta=jnp.asarray(eta), statistic_name="waldo",
        )
        losses.append(float(integrated_pvalue_loss(
            jnp.asarray(p)[None, :], jnp.asarray(tg)[None, :]
        )))
    L_ORACLE_3, L_WALD = 1.175, 1.363
    held_loss = float(np.mean(losses))
    held_env = (held_loss - L_ORACLE_3) / (L_WALD - L_ORACLE_3)
    print(f"\nHeld-out probe (n=32): loss = {held_loss:.4f}, env = {held_env:.3f}")
    print(f"  reference oracle env = 0.000, Wald env = 1.000")
    print(f"  if env < 1.0, the calibrated teacher beats Wald on held-out probe.")


if __name__ == "__main__":
    main()
