"""Per-slice optimum sweep across the training hyperparam distribution.

Question: is the integrated_p constant-eta optimum at eta=-1 universal
across (mu0, sigma_0, sigma, D), or does it vary?

If the optimum is at eta=-1 everywhere -> trained nets at +0.85 are
genuinely stuck.
If the optimum varies widely (some slices at -1, some at +1) -> the
trained +0.85 might be a sensible "best-on-average" compromise.

For each (sigma_0, sigma) pair drawn from the v4 hyperparam range,
sweep eta_const over its admissible range and find the argmin of
integrated_p loss. Plot argmin_eta as a function of (w, |Delta|).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian._registry_bootstrap import bootstrap

bootstrap()

import jax.numpy as jnp

from frasian.learned.training.losses import integrated_pvalue_loss
from frasian.learned.training.pvalue_jax import (
    ot_tilted_pvalue_jax,
    power_law_tilted_pvalue_jax,
)


def loss_at(scheme: str, eta_const: float, mu0: float, sigma0: float,
            sigma: float, D: float) -> float:
    """integrated_p at constant eta on a sigma-anchored theta-grid."""
    K = 5.0
    theta_grid = np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401)
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
    fn = power_law_tilted_pvalue_jax if scheme == "power_law" else ot_tilted_pvalue_jax
    eta_arr = jnp.full(theta_grid.shape, float(eta_const))
    p = fn(
        theta=jnp.asarray(theta_grid), D=jnp.asarray(D),
        w=jnp.asarray(w), mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
        eta=eta_arr, statistic_name="waldo",
    )
    return float(integrated_pvalue_loss(
        jnp.asarray(p)[None, :], jnp.asarray(theta_grid)[None, :]
    ))


def find_argmin(scheme, mu0, sigma0, sigma, D,
                eta_grid=np.linspace(-1.5, 1.5, 121)):
    """Return (argmin_eta, loss_at_argmin) for the slice."""
    losses = np.array([loss_at(scheme, float(e), mu0, sigma0, sigma, D)
                       for e in eta_grid])
    valid = np.isfinite(losses)
    if not valid.any():
        return float("nan"), float("nan")
    idx = int(np.argmin(np.where(valid, losses, np.inf)))
    return float(eta_grid[idx]), float(losses[idx])


def main():
    # Sample the v4 hyperparam range. mu0 fixed at 0 (translation-invariant),
    # vary sigma_0 over loguniform(0.2, 5) and sigma over loguniform(0.5, 2).
    sigma0_grid = np.array([0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 5.0])
    sigma_grid = np.array([0.5, 0.7, 1.0, 1.4, 2.0])
    D_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # span no-conflict to conflict

    rows = []
    for scheme in ["power_law", "ot"]:
        for s0 in sigma0_grid:
            for s in sigma_grid:
                w = s0 ** 2 / (s ** 2 + s0 ** 2)
                for D in D_grid:
                    delta = (1.0 - w) * abs(0.0 - D) / s
                    a_eta, _ = find_argmin(scheme, 0.0, s0, s, D)
                    rows.append((scheme, s0, s, w, D, delta, a_eta))

    print(f"\n{'scheme':<10} {'s0':>5} {'s':>5} {'w':>6} {'D':>5} {'|Delta|':>8} {'argmin_eta':>12}")
    print("-" * 70)
    for r in rows:
        print(f"{r[0]:<10} {r[1]:5.2f} {r[2]:5.2f} {r[3]:6.3f} "
              f"{r[4]:5.2f} {r[5]:8.3f} {r[6]:+12.3f}")

    # Plot: argmin_eta as a function of (w, |Delta|), one panel per scheme.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax_idx, scheme in enumerate(["power_law", "ot"]):
        ax = axes[ax_idx]
        sub = [r for r in rows if r[0] == scheme]
        ws = np.array([r[3] for r in sub])
        deltas = np.array([r[5] for r in sub])
        argmins = np.array([r[6] for r in sub])
        sc = ax.scatter(deltas, ws, c=argmins, cmap="RdBu_r",
                        vmin=-1.5, vmax=1.5, s=80, edgecolors="k", linewidths=0.4)
        ax.set_xlabel(r"$|\Delta|$ (normalised conflict)")
        ax.set_ylabel(r"$w = \sigma_0^2 / (\sigma^2 + \sigma_0^2)$")
        ax.set_title(f"{scheme}: argmin η for `integrated_p` loss\n"
                     f"(constant-η over θ ∈ μ₀ ± 5σ₀)")
        plt.colorbar(sc, ax=ax, label="argmin η")
    fig.tight_layout()
    out = Path("output/illustrations/per_slice_optimum_sweep.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")

    # Summary statistics.
    for scheme in ["power_law", "ot"]:
        sub = [r for r in rows if r[0] == scheme]
        argmins = np.array([r[6] for r in sub])
        print(f"\n{scheme}: argmin_eta range = [{argmins.min():+.3f}, {argmins.max():+.3f}], "
              f"mean = {argmins.mean():+.3f}, median = {np.median(argmins):+.3f}")
        print(f"  fraction at eta <= -0.5: {(argmins <= -0.5).mean():.2%}")
        print(f"  fraction at -0.5 < eta < +0.5: {((argmins > -0.5) & (argmins < 0.5)).mean():.2%}")
        print(f"  fraction at eta >= +0.5: {(argmins >= 0.5).mean():.2%}")


if __name__ == "__main__":
    main()
