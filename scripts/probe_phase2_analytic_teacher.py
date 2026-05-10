"""Phase 2 with the analytic DynamicNumericalEtaSelector as teacher.

Uses the framework's existing calibrated η-selector (deterministic,
fast, no MC noise) to generate teacher targets for MSE pre-training.

The DynamicNumericalEtaSelector:
- For each θ, calls NumericalEtaSelector.select(data=[θ], ...).
- The selector minimizes the analytic Wald-like CI width at "data=θ".
- Calibrated by construction (η depends only on θ, not on observed D).
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
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
from frasian.tilting.power_law import PowerLawTilting


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
    print("Generating teacher dataset (n=512) with analytic DynamicNumericalEtaSelector...")
    rng = np.random.default_rng(0xBEEF)
    hp = _v4_hp()
    probe = build_probe_batch(
        scheme_name="power_law", n=512, rng=rng,
        hyperparam_distribution=hp,
        prior_names=["loc", "scale"], lik_names=["sigma"],
    )

    selector = DynamicNumericalEtaSelector()
    scheme = PowerLawTilting()
    statistic = WaldoStatistic()

    # Compute analytic targets per teacher point.
    targets = np.empty(512, dtype=np.float64)
    for i in range(512):
        mu0 = float(probe.prior_hp[i, 0])
        sigma0 = float(probe.prior_hp[i, 1])
        sigma = float(probe.lik_hp[i, 0])
        theta = float(probe.theta[i])
        prior_obj = NormalDistribution(loc=mu0, scale=sigma0)
        model_obj = NormalNormalModel(sigma=sigma)
        eta_arr = selector.select_grid(
            np.asarray([theta]), scheme,
            model=model_obj, prior=prior_obj, alpha=0.05,
            statistic=statistic,
        )
        targets[i] = float(eta_arr[0])
        if (i + 1) % 64 == 0:
            print(f"  computed {i+1}/512 teacher targets")

    print(f"\nAnalytic teacher targets:")
    print(f"  range: [{targets.min():+.2f}, {targets.max():+.2f}]")
    print(f"  mean:  {targets.mean():+.3f}")

    # Pre-train EtaNet.
    print("\nPre-training EtaNet (MSE to analytic targets)...")
    theta_t = jnp.asarray(probe.theta, dtype=jnp.float64)
    prior_hp_t = jnp.asarray(probe.prior_hp, dtype=jnp.float64)
    lik_hp_t = jnp.asarray(probe.lik_hp, dtype=jnp.float64)
    target_eta = jnp.asarray(targets, dtype=jnp.float64)

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

    # Train with mini-batches to avoid one-batch overfitting.
    batch_size = 64
    n_epochs = 200
    rng_shuf = np.random.default_rng(0xDEAD)
    n = target_eta.shape[0]
    for epoch in range(n_epochs):
        idx = rng_shuf.permutation(n)
        epoch_mse = []
        for start in range(0, n, batch_size):
            sl = idx[start:start + batch_size]
            eta_net, opt_state, mse = step(
                eta_net, opt_state, theta_t[sl], prior_hp_t[sl], lik_hp_t[sl],
                target_eta[sl],
            )
            epoch_mse.append(float(mse))
        if (epoch + 1) % 25 == 0:
            preds = np.asarray(eta_net(theta_t, prior_hp_t, lik_hp_t))
            corr = float(np.corrcoef(preds, np.asarray(target_eta))[0, 1])
            print(f"  ep {epoch+1:3d}: avg MSE = {float(np.mean(epoch_mse)):.4f}  "
                  f"corr = {corr:+.3f}  pred mean = {preds.mean():+.3f}")

    out_path = Path("artifacts/probe_v4_phase2ana_pretrained.eqx")
    eqx.tree_serialise_leaves(str(out_path), eta_net)
    print(f"\nWrote {out_path}")

    # Held-out validation.
    rng_holdout = np.random.default_rng(0xCAFE)
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
    print(f"  reference: oracle env = 0.000, Wald env = 1.000, no_boundary env = 0.851")


if __name__ == "__main__":
    main()
