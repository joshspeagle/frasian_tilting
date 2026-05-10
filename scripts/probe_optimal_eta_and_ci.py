"""Per-θ_test argmin η + 95% CI widths — σ₀-anchored to match training.

Corrects the previous probe's data marginal: this one uses σ₀-anchored
θ_true sampling (the framework's training distribution) and σ₀-anchored
integration domain. So the per-θ_test argmin η here IS what the v4
training loss is supposed to converge to.

Also computes the **95% CI width** (concrete inferential metric) for:
  - Wald (η=1, ignores prior)
  - WALDO (η=0, full posterior + prior offset)
  - Optimal η(θ_test) (oracle from per-θ_test argmin)
  - v4 fixtures' η(θ_test) (queried from loaded artifacts)

All evaluated on the same (θ_true, D) MC sample so widths are
directly comparable. We also report coverage (fraction of (θ_true, D)
samples for which θ_true ∈ CI) so we can verify calibration.
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


SLICES = [
    # (label, mu0, sigma0, sigma)
    ("low-w  σ₀=0.3", 0.0, 0.3, 1.0),
    ("mid-w  σ₀=1.0", 0.0, 1.0, 1.0),
    ("mid-hi σ₀=2.0", 0.0, 2.0, 1.0),
    ("high-w σ₀=4.0", 0.0, 4.0, 1.0),
]

FIXTURES = [
    ("v4_default",     "artifacts/learned_eta_canonical_normal_normal_powerlaw_v4.eqx"),
    ("v4_baseline",    "artifacts/probe_v4_baseline.eqx"),
]


def _G_at_theta_test(theta_test, mu0, sigma0, sigma, eta_value, D_arr):
    """E_D[p(θ_test; D, η)] over a vector of D realizations."""
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


def per_theta_test_argmin(mu0, sigma0, sigma, theta_test_grid,
                          D_arr, eta_grid=np.linspace(-1.5, 2.5, 81)):
    """For each θ_test, find argmin_η G(θ_test, η)."""
    optimal_eta = np.empty(theta_test_grid.size)
    optimal_G = np.empty(theta_test_grid.size)
    G_waldo = np.empty(theta_test_grid.size)
    G_wald = np.empty(theta_test_grid.size)
    for i, t_test in enumerate(theta_test_grid):
        G_per_eta = np.empty(eta_grid.size)
        for j, eta in enumerate(eta_grid):
            G_per_eta[j] = _G_at_theta_test(
                t_test, mu0, sigma0, sigma, float(eta), D_arr)
        best = int(np.argmin(G_per_eta))
        optimal_eta[i] = float(eta_grid[best])
        optimal_G[i] = float(G_per_eta[best])
        G_waldo[i] = _G_at_theta_test(t_test, mu0, sigma0, sigma, 0.0, D_arr)
        G_wald[i] = _G_at_theta_test(t_test, mu0, sigma0, sigma, 1.0, D_arr)
    return optimal_eta, optimal_G, G_waldo, G_wald


def compute_ci_width_and_coverage(eta_curve_at_grid, theta_grid_fine,
                                    theta_true_arr, D_arr,
                                    mu0, sigma0, sigma, alpha=0.05):
    """For each (θ_true, D) sample, compute 95% CI width and whether
    it covers θ_true.

    eta_curve_at_grid: (n_grid_fine,) η values at theta_grid_fine.
    Returns (mean_width, coverage_rate).
    """
    w = sigma0**2 / (sigma**2 + sigma0**2)
    widths = []
    covered = []
    n = D_arr.size
    eta_j = jnp.asarray(eta_curve_at_grid, dtype=jnp.float64)
    theta_grid_j = jnp.asarray(theta_grid_fine, dtype=jnp.float64)
    w_j = jnp.asarray(w, dtype=jnp.float64)
    mu0_j = jnp.asarray(mu0, dtype=jnp.float64)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float64)

    @jax.jit
    def p_at_grid(D_scalar):
        return power_law_tilted_pvalue_jax(
            theta=theta_grid_j, D=D_scalar, w=w_j, mu0=mu0_j, sigma=sigma_j,
            eta=eta_j, statistic_name="waldo",
        )

    for i in range(n):
        D_v = jnp.asarray(D_arr[i], dtype=jnp.float64)
        p_curve = np.asarray(p_at_grid(D_v))
        in_ci = (p_curve >= alpha).astype(np.float64)
        # Width: integral of indicator
        width = float(np.trapezoid(in_ci, theta_grid_fine))
        widths.append(width)
        # Coverage: is θ_true_i in CI?
        # Linear interp of p at θ_true_i; cover if p ≥ α
        p_at_true = float(np.interp(theta_true_arr[i], theta_grid_fine, p_curve))
        covered.append(p_at_true >= alpha)
    return float(np.mean(widths)), float(np.mean(covered))


def main():
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    K = 5.0

    # Load fixtures
    artifacts = {}
    for name, path in FIXTURES:
        p = Path(path)
        if not p.exists():
            print(f"[skip fixture] {name}: not found")
            continue
        try:
            a = EtaArtifact(artifact_path=p, name=name)
            a.load()
            artifacts[name] = a
        except Exception as e:
            print(f"[fixture load failed] {name}: {e}")

    print("\n" + "="*100)
    print("σ₀-ANCHORED probe (matches training distribution)")
    print("="*100)

    output = {}

    for label, mu0, sigma0, sigma in SLICES:
        print(f"\n=== {label} (μ₀={mu0}, σ₀={sigma0}, σ={sigma}) ===")
        # σ₀-anchored grid (matches training integration domain)
        theta_test_grid = np.linspace(mu0 - K*sigma0, mu0 + K*sigma0, 51)
        # Fine grid for CI inversion
        theta_grid_fine = np.linspace(mu0 - K*sigma0, mu0 + K*sigma0, 401)

        # σ₀-anchored θ_true sampling (matches training)
        rng = np.random.default_rng(42)
        n_d_mc = 1024
        theta_true = rng.uniform(mu0 - K*sigma0, mu0 + K*sigma0, n_d_mc)
        D_arr = rng.normal(theta_true, sigma)

        # Per-θ_test argmin η
        opt_eta_coarse, opt_G, G_waldo_grid, G_wald_grid = per_theta_test_argmin(
            mu0, sigma0, sigma, theta_test_grid, D_arr)

        # Print key η values
        print(f"  Per-θ_test argmin η:")
        for i in [0, 12, 25, 38, 50]:  # 5 points along grid
            t = theta_test_grid[i]
            print(f"    θ_test = {t:>+7.3f}: η* = {opt_eta_coarse[i]:>+7.3f}  "
                  f"G* = {opt_G[i]:.4f}  G_waldo = {G_waldo_grid[i]:.4f}  "
                  f"G_wald = {G_wald_grid[i]:.4f}")

        loss_opt = float(np.trapezoid(opt_G, theta_test_grid))
        loss_waldo = float(np.trapezoid(G_waldo_grid, theta_test_grid))
        loss_wald = float(np.trapezoid(G_wald_grid, theta_test_grid))
        print(f"  Integrated G: opt={loss_opt:.4f}  WALDO={loss_waldo:.4f}  "
              f"Wald={loss_wald:.4f}")
        print(f"  opt vs Wald: Δ={loss_opt - loss_wald:+.4f}  "
              f"({100*(loss_wald-loss_opt)/loss_wald:+.1f}% improvement over Wald)")

        # Interpolate optimal η to fine grid for CI computation
        opt_eta_fine = np.interp(theta_grid_fine, theta_test_grid, opt_eta_coarse)
        wald_eta_fine = np.ones_like(theta_grid_fine)
        waldo_eta_fine = np.zeros_like(theta_grid_fine)

        # 95% CI widths + coverage
        print(f"\n  95% CI widths (mean) + coverage:")
        for label2, eta_fine in [
            ("Wald  (η=1)", wald_eta_fine),
            ("WALDO (η=0)", waldo_eta_fine),
            ("optimal η*", opt_eta_fine),
        ]:
            w_, cov_ = compute_ci_width_and_coverage(
                eta_fine, theta_grid_fine, theta_true, D_arr,
                mu0, sigma0, sigma, alpha=0.05)
            print(f"    {label2:<14}: width = {w_:.4f}  coverage = {cov_:.3f} (target 0.95)")

        # v4 fixtures
        for name, art in artifacts.items():
            try:
                eta_v4_fine = art.predict_eta(
                    theta_grid_fine,
                    np.asarray([mu0, sigma0]),
                    np.asarray([sigma]))
                w_v4, cov_v4 = compute_ci_width_and_coverage(
                    eta_v4_fine, theta_grid_fine, theta_true, D_arr,
                    mu0, sigma0, sigma, alpha=0.05)
                # Compute corr with optimum
                eta_v4_coarse = art.predict_eta(
                    theta_test_grid,
                    np.asarray([mu0, sigma0]),
                    np.asarray([sigma]))
                corr = float(np.corrcoef(eta_v4_coarse, opt_eta_coarse)[0,1])
                print(f"    {name:<14}: width = {w_v4:.4f}  coverage = {cov_v4:.3f}  "
                      f"η range [{eta_v4_fine.min():+.3f},{eta_v4_fine.max():+.3f}]  "
                      f"corr {corr:+.3f}")
            except Exception as e:
                print(f"    {name}: failed — {e}")

        output[label] = {
            'theta_test_grid': theta_test_grid,
            'opt_eta': opt_eta_coarse,
        }

    print("\n" + "="*100)
    print("Note: All metrics on σ₀-anchored sample (matches v4 training)")
    print("="*100)


if __name__ == "__main__":
    main()
