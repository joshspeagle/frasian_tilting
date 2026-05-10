"""Evaluate v4 fixtures on per-slice fn-min capture metrics.

Loads each provided EtaArtifact and computes:
  1. integrated_p loss at three fixed slices (mid-w, mid-hi, high-w)
  2. integrated_p loss on a continuous-holdout drawn from the v4 distribution
  3. The η(θ) range at each fixed slice

Compares against fn-min targets from probe_function_constrained_min and
the continuous-holdout reference (L_wald=1.363, L_oracle_3=1.175).

If a v4 fixture captures 90%+ on per-slice mid-w but ~0% on continuous
holdout, the v4 "training failure" is essentially the same constant-η
compromise we observed in probe_continuous_kd — not a separate bug.
"""
from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.eta_artifact import EtaArtifact
from frasian.learned.training.losses import integrated_pvalue_loss
from frasian.learned.training.pvalue_jax import power_law_tilted_pvalue_jax


# Fixtures to evaluate (paths relative to repo root)
FIXTURES = [
    ("v4_default",         "artifacts/learned_eta_canonical_normal_normal_powerlaw_v4.eqx"),
    ("v4_no_boundary",     "artifacts/probe_v4_no_boundary.eqx"),
    ("v4_baseline",        "artifacts/probe_v4_baseline.eqx"),
    ("v4_basinB_init",     "artifacts/probe_v4_basinB_init.eqx"),
    ("v4_no_norm",         "artifacts/probe_v4_no_norm.eqx"),
    ("pl_no_boundary_v4",  "artifacts/probe_pl_no_boundary_v4.eqx"),
]

FIXED_SLICES = [
    # (label, mu0, sigma0, sigma, wald, fn_min)
    # low-w wald/fn-min from probe_function_constrained_min:
    #   slice (σ₀=0.3, σ=1): wald=0.9773, fn-min=0.9565
    ("low-w  (σ₀=0.3)", 0.0, 0.3, 1.0, 0.9773, 0.9565),
    ("mid-w  (σ₀=1)",   0.0, 1.0, 1.0, 1.9000, 1.3900),
    ("mid-hi (σ₀=2)",   0.0, 2.0, 1.0, 2.0560, 1.4690),
    ("high-w (σ₀=4)",   0.0, 4.0, 1.0, 1.8200, 1.5100),
]


def integrated_p_at_slice(eta_curve, theta_grid, D_arr, mu0, sigma0, sigma):
    """Mean integrated_p loss over the D_arr realizations for one slice."""
    w = sigma0**2 / (sigma**2 + sigma0**2)
    losses = []
    for D in D_arr:
        p = power_law_tilted_pvalue_jax(
            theta=jnp.asarray(theta_grid), D=jnp.asarray(D),
            w=jnp.asarray(w), mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
            eta=jnp.asarray(eta_curve), statistic_name="waldo",
        )
        losses.append(float(integrated_pvalue_loss(
            p[None, :], jnp.asarray(theta_grid)[None, :])))
    return float(np.mean(losses))


def integrated_p_const_eta_at_slice(eta_value, theta_grid, D_arr, mu0,
                                     sigma0, sigma):
    """Loss with a CONSTANT eta_value across the integration grid.

    eta_value=0.0 → WALDO; eta_value=1.0 → Wald.
    """
    eta_const = np.full_like(theta_grid, eta_value)
    return integrated_p_at_slice(eta_const, theta_grid, D_arr, mu0, sigma0, sigma)


def eval_fixed_slice(artifact, mu0, sigma0, sigma, n_eval=512, n_grid=51, K=5.0,
                     seed=0, apply_ood_clamp=True, training_K=5.0):
    """Eval at a fixed slice — LIKELIHOOD-ANCHORED.

    Integration domain and theta_true sampling both scale with sigma
    (likelihood scale), not sigma0 (prior scale).

    If `apply_ood_clamp` (default True), η is overridden to 1.0 (Wald)
    for θ_test outside the training-distribution box [μ₀ ± training_K·σ₀],
    matching `LearnedDynamicEtaSelector.clamp_outside_training`.

    Returns (trained_loss, waldo_loss, wald_loss, eta_curve).
    """
    rng = np.random.default_rng(seed)
    theta_grid = np.linspace(mu0 - K*sigma, mu0 + K*sigma, n_grid)
    theta_true = rng.uniform(mu0 - K*sigma, mu0 + K*sigma, n_eval)
    D_arr = rng.normal(theta_true, sigma)
    eta_curve = np.asarray(artifact.predict_eta(
        theta_grid, np.asarray([mu0, sigma0]), np.asarray([sigma])),
        dtype=np.float64)
    if apply_ood_clamp:
        train_lo = mu0 - training_K * sigma0
        train_hi = mu0 + training_K * sigma0
        out_of_box = (theta_grid < train_lo) | (theta_grid > train_hi)
        if out_of_box.any():
            eta_curve = np.where(out_of_box, 1.0, eta_curve)
    trained_loss = integrated_p_at_slice(eta_curve, theta_grid, D_arr, mu0, sigma0, sigma)
    waldo_loss = integrated_p_const_eta_at_slice(0.0, theta_grid, D_arr, mu0, sigma0, sigma)
    wald_loss = integrated_p_const_eta_at_slice(1.0, theta_grid, D_arr, mu0, sigma0, sigma)
    return trained_loss, waldo_loss, wald_loss, eta_curve


def eval_continuous_holdout(artifact, n_holdout=512, n_grid=51, K=5.0, seed=99):
    """Eval on continuous-distribution holdout — LIKELIHOOD-ANCHORED.

    Integration domain and theta_true sampling both scale with sigma
    (likelihood scale).

    Returns (trained_loss, waldo_loss, wald_loss, per_w_bin_dict).
    """
    rng = np.random.default_rng(seed)
    mu0_b = rng.uniform(-2.0, 2.0, n_holdout)
    sigma0_b = np.exp(rng.uniform(np.log(0.2), np.log(5.0), n_holdout))
    sigma_b = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n_holdout))
    theta_true = rng.uniform(mu0_b - K*sigma_b, mu0_b + K*sigma_b)
    D_b = rng.normal(theta_true, sigma_b)

    losses_trained = np.empty(n_holdout)
    losses_waldo = np.empty(n_holdout)  # eta=0 (WALDO with full prior)
    losses_wald = np.empty(n_holdout)   # eta=1 (Wald data-only)
    eta_means = np.empty(n_holdout)
    w_arr = np.empty(n_holdout)
    training_K = 5.0  # σ₀-anchored training: box = [μ₀ ± training_K·σ₀]
    for i in range(n_holdout):
        theta_grid = np.linspace(
            mu0_b[i] - K*sigma_b[i], mu0_b[i] + K*sigma_b[i], n_grid)
        eta_curve = np.asarray(artifact.predict_eta(
            theta_grid,
            np.asarray([mu0_b[i], sigma0_b[i]]),
            np.asarray([sigma_b[i]])), dtype=np.float64)
        # OOD-θ clamp: outside training box [μ₀_i ± training_K·σ₀_i] → η=1.
        train_lo = mu0_b[i] - training_K * sigma0_b[i]
        train_hi = mu0_b[i] + training_K * sigma0_b[i]
        out_of_box = (theta_grid < train_lo) | (theta_grid > train_hi)
        if out_of_box.any():
            eta_curve = np.where(out_of_box, 1.0, eta_curve)
        w = sigma0_b[i]**2 / (sigma_b[i]**2 + sigma0_b[i]**2)
        w_arr[i] = w
        eta_means[i] = float(eta_curve.mean())
        common = dict(
            theta=jnp.asarray(theta_grid),
            D=jnp.asarray(float(D_b[i])),
            w=jnp.asarray(w),
            mu0=jnp.asarray(mu0_b[i]),
            sigma=jnp.asarray(sigma_b[i]),
            statistic_name="waldo",
        )
        p_trained = power_law_tilted_pvalue_jax(eta=jnp.asarray(eta_curve), **common)
        p_waldo = power_law_tilted_pvalue_jax(
            eta=jnp.zeros_like(jnp.asarray(eta_curve)), **common)
        p_wald = power_law_tilted_pvalue_jax(
            eta=jnp.ones_like(jnp.asarray(eta_curve)), **common)
        losses_trained[i] = float(integrated_pvalue_loss(
            p_trained[None, :], jnp.asarray(theta_grid)[None, :]))
        losses_waldo[i] = float(integrated_pvalue_loss(
            p_waldo[None, :], jnp.asarray(theta_grid)[None, :]))
        losses_wald[i] = float(integrated_pvalue_loss(
            p_wald[None, :], jnp.asarray(theta_grid)[None, :]))

    # Bin by w
    w_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    per_bin = {}
    for lo, hi in w_bins:
        mask = (w_arr >= lo) & (w_arr < hi)
        if mask.sum() > 5:
            per_bin[(lo, hi)] = {
                'n': int(mask.sum()),
                'trained': float(losses_trained[mask].mean()),
                'waldo': float(losses_waldo[mask].mean()),
                'wald': float(losses_wald[mask].mean()),
                'eta_mean': float(eta_means[mask].mean()),
            }
    return (float(losses_trained.mean()),
            float(losses_waldo.mean()),
            float(losses_wald.mean()),
            per_bin)


def main():
    print("\n" + "="*100)
    print("Per-slice fn-min capture for v4 fixtures")
    print("="*100)
    print("References: continuous wald=1.363, oracle_3=1.175")
    for label, _mu, _s0, _s, wald, fn_min in FIXED_SLICES:
        print(f"           {label}: wald={wald:.4f}, fn-min={fn_min:.4f}")

    rows = []
    for fix_label, fix_path in FIXTURES:
        path = Path(fix_path)
        if not path.exists():
            print(f"\n[SKIP] {fix_label}: {fix_path} not found")
            continue
        print(f"\n--- Loading {fix_label} from {fix_path} ---")
        try:
            artifact = EtaArtifact(artifact_path=path, name=fix_label)
            artifact.load()
        except Exception as e:
            print(f"  [ERROR] could not load: {e}")
            continue

        # Per-slice — recompute Wald (η=1) and WALDO (η=0) on the same sample.
        slice_results = []
        for slice_label, mu0, sg0, sg, _wald_ref, fn_min in FIXED_SLICES:
            try:
                loss, waldo_loss, wald_loss, eta_curve = eval_fixed_slice(
                    artifact, mu0, sg0, sg)
            except Exception as e:
                print(f"  [{slice_label}] eval failed: {e}")
                continue
            # Compare to Wald (η=1) — the natural data-only baseline.
            delta_vs_wald = loss - wald_loss
            slice_results.append((slice_label, loss, waldo_loss, wald_loss,
                                   delta_vs_wald, eta_curve))
            print(f"  {slice_label}: trained={loss:.4f}  WALDO(η=0)={waldo_loss:.4f}  "
                  f"Wald(η=1)={wald_loss:.4f}  Δvsw={delta_vs_wald:+.4f}  "
                  f"η range=[{eta_curve.min():+.3f}, {eta_curve.max():+.3f}]  "
                  f"mean={eta_curve.mean():+.3f}")

        # Continuous holdout (with on-the-fly WALDO + Wald baselines)
        try:
            cont_trained, cont_waldo, cont_wald, per_bin = eval_continuous_holdout(
                artifact, n_holdout=512)
            cont_delta_wald = cont_trained - cont_wald  # negative = better than Wald
            cont_delta_waldo = cont_trained - cont_waldo
            print(f"  continuous holdout (n=512): trained={cont_trained:.4f}  "
                  f"WALDO={cont_waldo:.4f}  Wald={cont_wald:.4f}")
            print(f"    Δ vs Wald = {cont_delta_wald:+.4f}  "
                  f"({'better' if cont_delta_wald < 0 else 'WORSE'} than Wald)")
            print(f"    Δ vs WALDO = {cont_delta_waldo:+.4f}  "
                  f"({'better' if cont_delta_waldo < 0 else 'worse'} than WALDO)")
            print("    per-w bin:")
            for (lo, hi), b in sorted(per_bin.items()):
                d_wald = b['trained'] - b['wald']
                d_waldo = b['trained'] - b['waldo']
                print(f"      w∈[{lo:.1f},{hi:.1f}) (n={b['n']:>3d}): "
                      f"trained={b['trained']:.4f}  Wald={b['wald']:.4f}  "
                      f"Δw={d_wald:+.4f}  Δwo={d_waldo:+.4f}  "
                      f"η_mean={b['eta_mean']:+.3f}")
        except Exception as e:
            print(f"  [continuous] eval failed: {e}")
            cont_trained = cont_waldo = cont_wald = cont_delta_wald = np.nan

        rows.append((fix_label, slice_results, cont_trained, cont_waldo,
                     cont_wald, cont_delta_wald))

    # Summary table — Δ vs Wald (η=1) per slice and continuous.
    print("\n" + "="*100)
    print("Δ-vs-Wald (η=1) per slice (negative = better than Wald):")
    print(f"{'fixture':<20}  {'low-w':>10}  {'mid-w':>10}  {'mid-hi':>10}  "
          f"{'high-w':>10}  {'continuous':>11}")
    print("-"*100)
    for fix_label, slice_results, cont_trained, cont_waldo, cont_wald, \
            cont_delta_wald in rows:
        if len(slice_results) >= 4:
            d_lw = slice_results[0][4]
            d_mw = slice_results[1][4]
            d_mh = slice_results[2][4]
            d_hw = slice_results[3][4]
        else:
            d_lw = d_mw = d_mh = d_hw = float('nan')
        print(f"{fix_label:<20}  {d_lw:>+10.4f}  {d_mw:>+10.4f}  {d_mh:>+10.4f}  "
              f"{d_hw:>+10.4f}  {cont_delta_wald:>+11.4f}")


if __name__ == "__main__":
    main()
