"""Per-θ_test argmin η probe — the correct fn-min oracle.

The integrated_p loss decomposes as

    L = ∫ G(θ_test, η(θ_test)) dθ_test

where G(θ_test, η) = E_{θ_true, D | θ_true}[p(θ_test; D, η)] is the
data-marginal of the p-value at one (θ_test, η) pair. Therefore the
function-constrained minimum is achieved by a per-θ_test independent
argmin: η*(θ_test) = argmin_η G(θ_test, η).

This script computes that curve directly. For each fixed slice
(μ₀, σ₀, σ), it grids θ_test on a wider domain than σ₀-anchored
training (covering both the "near μ₀" and "far from μ₀" regions),
samples (θ_true, D), and finds the optimal η at each θ_test.

Then for each loaded v4 fixture, queries η(θ_test) on the same grid
and overlays it against the optimum. This shows:
  1. Whether the optimal η has the expected U-shape (small near μ₀,
     → 1 far from μ₀).
  2. Where v4's actual η matches or deviates from optimum.

Three slices (low/mid/high w) give the picture across the framework's
intended regime.
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


SLICES = [
    # (label, mu0, sigma0, sigma)
    ("low-w  σ₀=0.3", 0.0, 0.3, 1.0),
    ("mid-w  σ₀=1.0", 0.0, 1.0, 1.0),
    ("mid-hi σ₀=2.0", 0.0, 2.0, 1.0),
    ("high-w σ₀=4.0", 0.0, 4.0, 1.0),
]

# Fixtures to overlay
FIXTURES = [
    ("v4_default",     "artifacts/learned_eta_canonical_normal_normal_powerlaw_v4.eqx"),
    ("v4_baseline",    "artifacts/probe_v4_baseline.eqx"),
    ("v4_no_boundary", "artifacts/probe_v4_no_boundary.eqx"),
]


def _G_at_theta_test(theta_test, mu0, sigma0, sigma, eta_value, D_arr):
    """G(θ_test, η) = mean over D of p(θ_test; D, η)."""
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
                          eta_grid=np.linspace(-1.5, 2.5, 81),
                          n_d_mc=512, K=5.0, seed=0):
    """For each θ_test in `theta_test_grid`, find argmin_η G(θ_test, η).

    θ_true is sampled from a wide range (5σ₀ + 5σ around μ₀ = max prior
    or likelihood scale) so the data marginal covers both near-μ₀ and
    far-from-μ₀ regimes.
    """
    rng = np.random.default_rng(seed)
    sample_K = max(K * sigma0, K * sigma)
    theta_true = rng.uniform(mu0 - sample_K, mu0 + sample_K, n_d_mc)
    D_arr = rng.normal(theta_true, sigma)

    optimal_eta = np.empty(theta_test_grid.size)
    optimal_G = np.empty(theta_test_grid.size)
    for i, t_test in enumerate(theta_test_grid):
        G_per_eta = np.empty(eta_grid.size)
        for j, eta in enumerate(eta_grid):
            G_per_eta[j] = _G_at_theta_test(
                t_test, mu0, sigma0, sigma, float(eta), D_arr)
        best = int(np.argmin(G_per_eta))
        optimal_eta[i] = float(eta_grid[best])
        optimal_G[i] = float(G_per_eta[best])

    # Also compute G(θ_test) at η=0 (WALDO) and η=1 (Wald) for reference
    G_waldo = np.empty(theta_test_grid.size)
    G_wald = np.empty(theta_test_grid.size)
    for i, t_test in enumerate(theta_test_grid):
        G_waldo[i] = _G_at_theta_test(t_test, mu0, sigma0, sigma, 0.0, D_arr)
        G_wald[i] = _G_at_theta_test(t_test, mu0, sigma0, sigma, 1.0, D_arr)

    return optimal_eta, optimal_G, G_waldo, G_wald


def query_v4_eta(artifact, mu0, sigma0, sigma, theta_test_grid):
    """Query a loaded v4 fixture's η(θ_test, prior, lik) at the grid."""
    return artifact.predict_eta(
        theta_test_grid,
        np.asarray([mu0, sigma0]),
        np.asarray([sigma]),
    )


def main():
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    output = {}

    # Build a unified θ_test grid that covers both regimes for each slice.
    # We'll use a per-slice grid: [μ₀ - 5σ_max, μ₀ + 5σ_max] where
    # σ_max = max(σ₀, σ). 81 points.
    print("\nPer-θ_test argmin η probe — does the optimum have a U-shape?")
    print("(Wide θ_test grid covering both near-μ₀ and far-from-μ₀ regions.)\n")

    # Pre-load fixtures once
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

    for label, mu0, sigma0, sigma in SLICES:
        print(f"\n=== {label} (μ₀={mu0}, σ₀={sigma0}, σ={sigma}) ===")
        sigma_max = max(sigma0, sigma)
        K = 5.0
        theta_test_grid = np.linspace(
            mu0 - K * sigma_max, mu0 + K * sigma_max, 81)

        opt_eta, opt_G, G_waldo, G_wald = per_theta_test_argmin(
            mu0, sigma0, sigma, theta_test_grid)

        # Print summary at key θ_test values
        key_idx = [0, 20, 40, 60, 80]  # five equally-spaced points
        print(f"  {'θ_test':>8}  {'eta*':>7}  {'G*':>7}  {'G_waldo':>8}  {'G_wald':>7}")
        for idx in key_idx:
            t = theta_test_grid[idx]
            print(f"  {t:>+8.3f}  {opt_eta[idx]:>+7.3f}  {opt_G[idx]:>7.4f}  "
                  f"{G_waldo[idx]:>8.4f}  {G_wald[idx]:>7.4f}")

        # Integrated losses
        loss_opt = float(np.trapezoid(opt_G, theta_test_grid))
        loss_waldo = float(np.trapezoid(G_waldo, theta_test_grid))
        loss_wald = float(np.trapezoid(G_wald, theta_test_grid))
        print(f"  Integrated G over grid: opt={loss_opt:.4f}  "
              f"WALDO={loss_waldo:.4f}  Wald={loss_wald:.4f}")
        print(f"  Δ opt vs Wald: {loss_opt - loss_wald:+.4f}  "
              f"({100*(loss_wald-loss_opt)/loss_wald:+.1f}% improvement)")
        print(f"  η* range: [{opt_eta.min():+.3f}, {opt_eta.max():+.3f}]  "
              f"mean: {opt_eta.mean():+.3f}")

        # Detect U-shape
        in_prior = (np.abs(theta_test_grid - mu0) <= sigma0)
        out_prior = (np.abs(theta_test_grid - mu0) >= 3 * sigma0)
        if in_prior.sum() > 0 and out_prior.sum() > 0:
            eta_in = opt_eta[in_prior].mean()
            eta_out = opt_eta[out_prior].mean()
            print(f"  η* near μ₀ (within σ₀): mean {eta_in:+.3f}")
            print(f"  η* far from μ₀ (≥3σ₀): mean {eta_out:+.3f}")
            shape = "U-shape (small near μ₀, ~1 far away)" if eta_out > eta_in + 0.1 else \
                    "INVERTED (small far away, big near μ₀)" if eta_out < eta_in - 0.1 else \
                    "FLAT (similar near and far)"
            print(f"  Shape: {shape}")

        # v4 overlays
        v4_curves = {}
        for name, art in artifacts.items():
            try:
                eta_v4 = query_v4_eta(art, mu0, sigma0, sigma, theta_test_grid)
                v4_curves[name] = eta_v4
                print(f"  {name}: η range [{eta_v4.min():+.3f}, "
                      f"{eta_v4.max():+.3f}] mean {eta_v4.mean():+.3f}  "
                      f"corr-with-opt {np.corrcoef(eta_v4, opt_eta)[0,1]:+.3f}")
            except Exception as e:
                print(f"  {name}: query failed — {e}")

        output[label] = {
            'theta_test_grid': theta_test_grid,
            'mu0': mu0, 'sigma0': sigma0, 'sigma': sigma,
            'optimal_eta': opt_eta,
            'G_optimal': opt_G,
            'G_waldo': G_waldo,
            'G_wald': G_wald,
            'v4_curves': v4_curves,
        }

    # Save
    save_dict = {}
    for label, d in output.items():
        prefix = label.split()[0]  # "low-w", "mid-w", etc.
        save_dict[f'{prefix}_theta_test_grid'] = d['theta_test_grid']
        save_dict[f'{prefix}_optimal_eta'] = d['optimal_eta']
        save_dict[f'{prefix}_G_optimal'] = d['G_optimal']
        save_dict[f'{prefix}_G_waldo'] = d['G_waldo']
        save_dict[f'{prefix}_G_wald'] = d['G_wald']
        for name, eta in d['v4_curves'].items():
            save_dict[f'{prefix}_eta_{name}'] = eta

    np.savez(out_dir / "per_theta_test_argmin.npz", **save_dict)
    print(f"\nSaved to {out_dir / 'per_theta_test_argmin.npz'}")


if __name__ == "__main__":
    main()
