"""E0 probe for mixture cd_variance investigation (2026-05-10).

For a few fixed (μ₀, σ₀, σ) slices, plot cd_variance loss as a
function of constant η ∈ [0, eta_max]. If the curve has an interior
minimum, mixture cd_variance is a stable optimization target and the
training instability is fixable via hyperparameter tuning. If the
curve is monotonic in η (driving toward boundary), mixture cd_variance
is intrinsically boundary-hugging and should be skipped.

Compares against power_law cd_variance landscape on the same slices
to confirm the contrast.
"""

from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.training.cd_jax import cd_density_jax
from frasian.learned.training.pvalue_jax import (
    mixture_tilted_pvalue_jax,
    power_law_tilted_pvalue_jax,
)
from frasian.tilting.mixture import _admissibility_normal_normal


def _cd_variance_at_const_eta(
    pvalue_fn,
    theta_grid_np,
    D,
    w,
    mu0,
    sigma,
    eta_const,
):
    """cd_variance loss for one (μ₀, σ₀, σ, D) at constant η across
    theta_grid. Returns variance of the Schweder-Hjort CD."""
    theta_grid = jnp.asarray(theta_grid_np, dtype=jnp.float64)
    eta_arr = jnp.full(theta_grid.shape, float(eta_const), dtype=jnp.float64)
    p = pvalue_fn(
        theta=theta_grid,
        D=jnp.asarray(D, dtype=jnp.float64),
        w=jnp.asarray(w, dtype=jnp.float64),
        mu0=jnp.asarray(mu0, dtype=jnp.float64),
        sigma=jnp.asarray(sigma, dtype=jnp.float64),
        eta=eta_arr,
        statistic_name="waldo",
    )
    pdf = cd_density_jax(p[None, :], theta_grid[None, :])
    mu = jnp.trapezoid(theta_grid[None, :] * pdf, theta_grid, axis=-1)[0]
    centred = theta_grid - mu
    var = jnp.trapezoid(pdf[0] * centred * centred, theta_grid)
    return float(var)


def _eta_max_for_slice(mu0, sigma0, sigma, D):
    w = sigma0**2 / (sigma**2 + sigma0**2)
    _, eta_max = _admissibility_normal_normal(0.0, w, mu0, D, sigma)
    return eta_max


def landscape_for_slice(scheme_name, mu0, sigma0, sigma, D,
                         eta_grid=None, K=5.0, n_grid=401):
    if eta_grid is None:
        eta_grid = np.linspace(-0.5, 3.5, 81)
    fn = (
        mixture_tilted_pvalue_jax if scheme_name == "mixture"
        else power_law_tilted_pvalue_jax
    )
    theta_grid = np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, n_grid)
    w = sigma0**2 / (sigma**2 + sigma0**2)
    var_curve = np.empty(eta_grid.size)
    for i, eta in enumerate(eta_grid):
        var_curve[i] = _cd_variance_at_const_eta(
            fn, theta_grid, D, w, mu0, sigma, eta,
        )
    return eta_grid, var_curve


def main():
    # Slices spanning the v4 hp range; D drawn from N(θ_true, σ) with θ_true
    # σ₀-anchored to keep the sample representative of training.
    slices = [
        # (label, mu0, sigma0, sigma, theta_true offset from mu0 in σ₀ units)
        ("mid-w   σ₀=1.0  D≈μ₀", 0.0, 1.0, 1.0, 0.0),
        ("mid-w   σ₀=1.0  D=2σ₀", 0.0, 1.0, 1.0, 2.0),
        ("low-w   σ₀=0.3  D=2σ₀", 0.0, 0.3, 1.0, 2.0),
        ("high-w  σ₀=4.0  D=2σ₀", 0.0, 4.0, 1.0, 2.0),
    ]

    print("Cd_variance landscape probe (E0):")
    print(f"{'slice':<28} {'η_max':>6} {'mx_argmin':>10} {'mx_var(opt)':>12} "
          f"{'mx_var(eta_max-ε)':>18} {'mx_curve_shape':>16}")
    print("-" * 100)

    out_records = []
    for label, mu0, sigma0, sigma, theta_offset in slices:
        theta_true = mu0 + theta_offset * sigma0
        rng = np.random.default_rng(42)
        D = float(rng.normal(theta_true, sigma))

        eta_max = _eta_max_for_slice(mu0, sigma0, sigma, D)
        # Restrict the η-grid to admissible region [0, eta_max]
        if not np.isfinite(eta_max):
            eta_max = 5.0
        eta_grid = np.linspace(0.0, min(eta_max - 1e-6, 5.0), 81)

        eta_g_mx, var_mx = landscape_for_slice(
            "mixture", mu0, sigma0, sigma, D, eta_grid=eta_grid,
        )
        eta_g_pl, var_pl = landscape_for_slice(
            "power_law", mu0, sigma0, sigma, D, eta_grid=np.linspace(-0.5, 1.5, 81),
        )

        argmin_idx = int(np.argmin(var_mx))
        eta_argmin = float(eta_g_mx[argmin_idx])
        var_at_opt = float(var_mx[argmin_idx])
        var_near_boundary = float(var_mx[-1])
        # Shape: monotonic-decreasing, monotonic-increasing, U-shape, or other
        diffs = np.diff(var_mx)
        if (diffs > 0).all():
            shape = "↑ monotone-up"
        elif (diffs < 0).all():
            shape = "↓ monotone-down"
        elif argmin_idx == 0:
            shape = "min at 0"
        elif argmin_idx == len(eta_g_mx) - 1:
            shape = "min at boundary"
        elif 0 < argmin_idx < len(eta_g_mx) - 1:
            shape = "U-shape (interior)"
        else:
            shape = "other"
        print(f"{label:<28} {eta_max:>6.3f} {eta_argmin:>10.3f} {var_at_opt:>12.4g} "
              f"{var_near_boundary:>18.4g} {shape:>16}")

        out_records.append({
            'label': label,
            'mu0': mu0, 'sigma0': sigma0, 'sigma': sigma, 'D': D,
            'eta_max': eta_max,
            'eta_grid_mx': eta_g_mx,
            'var_mx': var_mx,
            'eta_grid_pl': eta_g_pl,
            'var_pl': var_pl,
            'eta_argmin': eta_argmin,
            'var_at_opt': var_at_opt,
            'shape': shape,
        })

    print()
    print("Interpretation:")
    print("  - 'U-shape (interior)': cd_var has a stable interior optimum;")
    print("    training instability is fixable via hyperparameter tuning.")
    print("  - 'min at boundary' / 'monotone-down': cd_var is boundary-hugging;")
    print("    no amount of stabilization will help — skip mixture cd_var.")

    # Save
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    np.savez(
        out_dir / "mixture_cd_var_landscape.npz",
        **{f"{r['label'].split()[0]}_eta_grid_mx": r['eta_grid_mx'] for r in out_records},
        **{f"{r['label'].split()[0]}_var_mx": r['var_mx'] for r in out_records},
        **{f"{r['label'].split()[0]}_eta_grid_pl": r['eta_grid_pl'] for r in out_records},
        **{f"{r['label'].split()[0]}_var_pl": r['var_pl'] for r in out_records},
    )
    print(f"\nSaved curves to {out_dir / 'mixture_cd_var_landscape.npz'}")


if __name__ == "__main__":
    main()
