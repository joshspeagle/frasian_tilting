"""Hyperparam-grid audit: does v4 capture the achievable headroom
across the full v4 (μ₀, σ₀, σ) range?

Sweeps over a grid of (σ₀, σ) values covering the v4 loguniform ranges
(σ₀ ∈ [0.2, 5.0], σ ∈ [0.5, 2.0]) at μ₀=0. For each slice computes
on a σ₀-anchored grid (matching training):

  - Per-θ_test argmin η (oracle)
  - v4's η curve
  - 95% CI widths for Wald, optimal η*, v4
  - Coverage rates for each (target 0.95)
  - Correlation between v4's η curve and the per-θ_test optimum

Acceptance criteria:
  1. v4 CI width ≤ Wald width at every slice (≥ no degradation)
  2. v4 coverage ∈ [0.93, 0.97] at every slice (calibrated within MC noise)
  3. v4 captures most of the per-θ_test-optimum headroom over Wald
     (≥ 50% of (Wald - opt) gap is captured)

Any slice failing one of these is flagged as a regression.
"""
from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.eta_artifact import EtaArtifact
from frasian.learned.training.pvalue_jax import power_law_tilted_pvalue_jax


# v4 hyperparam ranges (from canonical_normal_normal_powerlaw_v4.yaml):
# σ₀ ~ Loguniform(0.2, 5.0); σ ~ Loguniform(0.5, 2.0); μ₀ ~ Uniform(-2, 2)
SIGMA0_GRID = [0.2, 0.5, 1.0, 2.0, 5.0]
SIGMA_GRID = [0.5, 1.0, 2.0]
MU0 = 0.0


def _G_at_theta_test(theta_test, mu0, sigma0, sigma, eta_value, D_arr):
    w = sigma0**2 / (sigma**2 + sigma0**2)
    D_j = jnp.asarray(D_arr, dtype=jnp.float64)
    theta_j = jnp.full(D_j.shape, theta_test, dtype=jnp.float64)
    eta_j = jnp.full(D_j.shape, eta_value, dtype=jnp.float64)
    w_j = jnp.full(D_j.shape, w, dtype=jnp.float64)
    mu0_j = jnp.full(D_j.shape, mu0, dtype=jnp.float64)
    sigma_j = jnp.full(D_j.shape, sigma, dtype=jnp.float64)
    p = power_law_tilted_pvalue_jax(
        theta=theta_j, D=D_j, w=w_j, mu0=mu0_j, sigma=sigma_j,
        eta=eta_j, statistic_name="waldo",
    )
    return float(jnp.mean(p))


def per_theta_test_argmin(mu0, sigma0, sigma, theta_grid, D_arr,
                          eta_grid=np.linspace(-1.5, 2.5, 41)):
    optimal_eta = np.empty(theta_grid.size)
    for i, t_test in enumerate(theta_grid):
        G_per_eta = np.empty(eta_grid.size)
        for j, eta in enumerate(eta_grid):
            G_per_eta[j] = _G_at_theta_test(
                t_test, mu0, sigma0, sigma, float(eta), D_arr)
        optimal_eta[i] = float(eta_grid[int(np.argmin(G_per_eta))])
    return optimal_eta


def integrated_p_with_eta_curve(eta_curve, theta_grid, D_arr,
                                 mu0, sigma0, sigma):
    """Mean over D of ∫ p(θ; D, η(θ)) dθ."""
    w = sigma0**2 / (sigma**2 + sigma0**2)
    eta_j = jnp.asarray(eta_curve, dtype=jnp.float64)
    theta_j = jnp.asarray(theta_grid, dtype=jnp.float64)
    w_j = jnp.asarray(w, dtype=jnp.float64)
    mu0_j = jnp.asarray(mu0, dtype=jnp.float64)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float64)

    @jax.jit
    def integrand_at_D(D_scalar):
        p = power_law_tilted_pvalue_jax(
            theta=theta_j, D=D_scalar, w=w_j, mu0=mu0_j, sigma=sigma_j,
            eta=eta_j, statistic_name="waldo",
        )
        return jnp.trapezoid(p, theta_j)

    losses = []
    for D_v in D_arr:
        losses.append(float(integrand_at_D(jnp.asarray(D_v, dtype=jnp.float64))))
    return float(np.mean(losses))


def ci_width_and_coverage(eta_curve_fine, theta_grid_fine,
                           theta_true_arr, D_arr, mu0, sigma0, sigma,
                           alpha=0.05):
    w = sigma0**2 / (sigma**2 + sigma0**2)
    eta_j = jnp.asarray(eta_curve_fine, dtype=jnp.float64)
    theta_j = jnp.asarray(theta_grid_fine, dtype=jnp.float64)
    w_j = jnp.asarray(w, dtype=jnp.float64)
    mu0_j = jnp.asarray(mu0, dtype=jnp.float64)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float64)

    @jax.jit
    def p_at_grid(D_scalar):
        return power_law_tilted_pvalue_jax(
            theta=theta_j, D=D_scalar, w=w_j, mu0=mu0_j, sigma=sigma_j,
            eta=eta_j, statistic_name="waldo",
        )

    widths = []
    covered = []
    for i in range(D_arr.size):
        p_curve = np.asarray(p_at_grid(jnp.asarray(D_arr[i], dtype=jnp.float64)))
        in_ci = (p_curve >= alpha).astype(np.float64)
        widths.append(float(np.trapezoid(in_ci, theta_grid_fine)))
        p_at_true = float(np.interp(theta_true_arr[i], theta_grid_fine, p_curve))
        covered.append(p_at_true >= alpha)
    return float(np.mean(widths)), float(np.mean(covered))


def query_v4_eta(artifact, mu0, sigma0, sigma, theta_grid, training_K=5.0,
                  apply_clamp=True):
    """Get v4's η curve at theta_grid, with optional OOD clamp."""
    eta = np.asarray(artifact.predict_eta(
        theta_grid, np.asarray([mu0, sigma0]), np.asarray([sigma])),
        dtype=np.float64)
    if apply_clamp:
        train_lo = mu0 - training_K * sigma0
        train_hi = mu0 + training_K * sigma0
        out_of_box = (theta_grid < train_lo) | (theta_grid > train_hi)
        if out_of_box.any():
            eta = np.where(out_of_box, 1.0, eta)
    return eta


def audit_slice(mu0, sigma0, sigma, artifact, K=5.0, n_grid=51,
                 n_grid_fine=401, n_d_mc=512, seed=0):
    """Run the audit at one (μ₀, σ₀, σ) slice."""
    rng = np.random.default_rng(seed)
    # σ₀-anchored everything (matches training)
    theta_grid = np.linspace(mu0 - K*sigma0, mu0 + K*sigma0, n_grid)
    theta_grid_fine = np.linspace(mu0 - K*sigma0, mu0 + K*sigma0, n_grid_fine)
    theta_true = rng.uniform(mu0 - K*sigma0, mu0 + K*sigma0, n_d_mc)
    D_arr = rng.normal(theta_true, sigma)

    # Per-θ_test argmin η (oracle)
    opt_eta_coarse = per_theta_test_argmin(mu0, sigma0, sigma, theta_grid, D_arr)
    opt_eta_fine = np.interp(theta_grid_fine, theta_grid, opt_eta_coarse)

    # v4 η (with clamp; mostly no-op since grid IS σ₀-anchored)
    v4_eta_coarse = query_v4_eta(artifact, mu0, sigma0, sigma, theta_grid)
    v4_eta_fine = query_v4_eta(artifact, mu0, sigma0, sigma, theta_grid_fine)

    wald_eta_fine = np.ones_like(theta_grid_fine)

    # Integrated G (training-loss equivalent)
    G_wald = integrated_p_with_eta_curve(
        np.ones_like(theta_grid), theta_grid, D_arr, mu0, sigma0, sigma)
    G_opt = integrated_p_with_eta_curve(
        opt_eta_coarse, theta_grid, D_arr, mu0, sigma0, sigma)
    G_v4 = integrated_p_with_eta_curve(
        v4_eta_coarse, theta_grid, D_arr, mu0, sigma0, sigma)

    # 95% CI widths + coverage
    w_wald, cov_wald = ci_width_and_coverage(
        wald_eta_fine, theta_grid_fine, theta_true, D_arr, mu0, sigma0, sigma)
    w_opt, cov_opt = ci_width_and_coverage(
        opt_eta_fine, theta_grid_fine, theta_true, D_arr, mu0, sigma0, sigma)
    w_v4, cov_v4 = ci_width_and_coverage(
        v4_eta_fine, theta_grid_fine, theta_true, D_arr, mu0, sigma0, sigma)

    # Headroom capture metric: fraction of (Wald - opt) recovered by v4
    if abs(G_wald - G_opt) > 1e-6:
        capture_G = (G_wald - G_v4) / (G_wald - G_opt)
    else:
        capture_G = 1.0  # no headroom available; v4 just needs to ≈ Wald
    if abs(w_wald - w_opt) > 1e-6:
        capture_W = (w_wald - w_v4) / (w_wald - w_opt)
    else:
        capture_W = 1.0

    return {
        'mu0': mu0, 'sigma0': sigma0, 'sigma': sigma,
        'w': sigma0**2 / (sigma**2 + sigma0**2),
        'G_wald': G_wald, 'G_opt': G_opt, 'G_v4': G_v4,
        'capture_G': capture_G,
        'w_wald': w_wald, 'w_opt': w_opt, 'w_v4': w_v4,
        'capture_W': capture_W,
        'cov_wald': cov_wald, 'cov_opt': cov_opt, 'cov_v4': cov_v4,
        'eta_corr': float(np.corrcoef(v4_eta_coarse, opt_eta_coarse)[0,1])
            if np.std(v4_eta_coarse) > 0 and np.std(opt_eta_coarse) > 0 else float('nan'),
        'v4_eta_mean': float(v4_eta_coarse.mean()),
        'opt_eta_mean': float(opt_eta_coarse.mean()),
    }


def main():
    fixture_path = Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v4.eqx")
    if not fixture_path.exists():
        print(f"[ERROR] fixture not found: {fixture_path}")
        return
    art = EtaArtifact(artifact_path=fixture_path, name="v4_default")
    art.load()

    print(f"\nHyperparam-grid audit of v4 (`{fixture_path.name}`)")
    print(f"σ₀ × σ grid: {len(SIGMA0_GRID)} × {len(SIGMA_GRID)} = "
          f"{len(SIGMA0_GRID)*len(SIGMA_GRID)} slices, μ₀={MU0}")
    print(f"σ₀-anchored evaluation (matches training distribution).")
    print()

    rows = []
    for sigma0 in SIGMA0_GRID:
        for sigma in SIGMA_GRID:
            r = audit_slice(MU0, sigma0, sigma, art)
            rows.append(r)
            print(f"  σ₀={sigma0:>4.1f} σ={sigma:>4.1f} (w={r['w']:.3f}): "
                  f"Wald W={r['w_wald']:.3f} cov={r['cov_wald']:.3f}  "
                  f"v4 W={r['w_v4']:.3f} cov={r['cov_v4']:.3f}  "
                  f"opt W={r['w_opt']:.3f}  "
                  f"capture(W)={100*r['capture_W']:>+5.1f}%  "
                  f"capture(G)={100*r['capture_G']:>+5.1f}%  "
                  f"corr={r['eta_corr']:+.2f}")

    # Acceptance gates
    print()
    print("="*100)
    print("Acceptance check:")
    print("="*100)
    fails = []
    for r in rows:
        slice_label = f"σ₀={r['sigma0']:.1f} σ={r['sigma']:.1f}"
        # Gate 1: v4 width ≤ Wald width
        if r['w_v4'] > r['w_wald'] + 1e-3:
            fails.append(f"  [{slice_label}] FAIL gate 1: v4 width {r['w_v4']:.4f} > Wald {r['w_wald']:.4f}")
        # Gate 2: coverage in [0.93, 0.97]
        if not (0.93 <= r['cov_v4'] <= 0.97):
            fails.append(f"  [{slice_label}] FAIL gate 2: v4 coverage {r['cov_v4']:.3f} outside [0.93, 0.97]")
        # Gate 3: capture ≥ 50% (only if there's headroom to capture)
        if r['G_wald'] - r['G_opt'] > 1e-3 and r['capture_G'] < 0.5:
            fails.append(f"  [{slice_label}] FAIL gate 3: v4 captured {100*r['capture_G']:.1f}% of headroom (target ≥ 50%)")

    if not fails:
        print("✓ All slices pass all three gates (width, coverage, capture).")
    else:
        print(f"✗ {len(fails)} gate failures:")
        for f in fails:
            print(f)

    # Save
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v4_hyperparam_audit.npz"
    np.savez(
        out_path,
        sigma0_grid=np.array(SIGMA0_GRID),
        sigma_grid=np.array(SIGMA_GRID),
        rows=np.array([(r['sigma0'], r['sigma'], r['w'],
                         r['G_wald'], r['G_opt'], r['G_v4'],
                         r['w_wald'], r['w_opt'], r['w_v4'],
                         r['cov_wald'], r['cov_opt'], r['cov_v4'],
                         r['eta_corr'])
                        for r in rows], dtype=np.float64),
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
