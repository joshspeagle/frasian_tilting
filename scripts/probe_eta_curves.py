"""Plot eta(theta) curves for all 5 probe fixtures vs the analytical optimum.

Thread 1 of the post-Stage-2 investigation. For each of the 5 trained
configs (baseline, no_boundary, no_norm, anti_wald_10, stratified) at
the demo slice (mu0=0, sigma_0=sigma=1, w=0.5), plot eta(theta) on the
[-5, +5] training range and overlay the per-theta analytical argmin
of integrated_p (computed for each theta_test as a separate slice).

Reveals whether any config has the right SHAPE even if average is wrong.
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

from frasian.learned.eta_artifact import EtaArtifact
from frasian.learned.training.losses import integrated_pvalue_loss
from frasian.learned.training.pvalue_jax import power_law_tilted_pvalue_jax


# Demo slice
MU0 = 0.0
SIGMA0 = 1.0
SIGMA = 1.0
W = SIGMA0 ** 2 / (SIGMA ** 2 + SIGMA0 ** 2)  # = 0.5
K = 5.0

THETA_GRID = np.linspace(MU0 - K * SIGMA0, MU0 + K * SIGMA0, 51)

CONFIGS = [
    ("baseline",     "tab:blue"),
    ("no_boundary",  "tab:orange"),
    ("no_norm",      "tab:green"),
    ("anti_wald_10", "tab:red"),
    ("stratified",   "tab:purple"),
]


def _per_theta_analytical_argmin(theta_test, D, mu0, sigma0, sigma,
                                 eta_grid=np.linspace(-1.5, 1.5, 121),
                                 K_loss=5.0, n_grid=401):
    """Per-slice argmin of integrated_p_loss over constant eta.

    Note: this is the SAME as the offline argmin in diagnostics — the
    integrated-p loss at a single (mu0, sigma0, sigma, D) slice. theta_test
    is unused (slice optimum doesn't depend on a single test theta).
    """
    theta_grid = np.linspace(mu0 - K_loss * sigma0, mu0 + K_loss * sigma0, n_grid)
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
    losses = np.empty(eta_grid.size, dtype=np.float64)
    for i, eta in enumerate(eta_grid):
        eta_arr = jnp.full(theta_grid.shape, float(eta))
        p = power_law_tilted_pvalue_jax(
            theta=jnp.asarray(theta_grid), D=jnp.asarray(D),
            w=jnp.asarray(w), mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
            eta=eta_arr, statistic_name="waldo",
        )
        losses[i] = float(integrated_pvalue_loss(
            jnp.asarray(p)[None, :], jnp.asarray(theta_grid)[None, :]
        ))
    return float(eta_grid[int(np.argmin(losses))])


def main():
    # First: the per-slice analytical argmin at this slice (constant in theta).
    # We'll set D = mu0 (the no-conflict reference) to get the "easy" optimum,
    # but actually the true per-slice analytical optimum depends on D (the
    # observed datum). For the trained network, eta is a function of (theta,
    # mu0, sigma0, sigma) — D is implicit (D ~ N(theta, sigma) at training).
    # So at inference, the trained network's eta(theta) is what it would
    # output for a query at that theta, integrated over hypothetical D values
    # generated during training.
    #
    # The "right" comparison is per-(theta_test, D) slice argmin, but D is
    # not a NN input. So we sweep argmin across reasonable D values for each
    # theta_test and show the BAND of analytical optima.

    D_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    analytical_argmins = np.zeros((len(THETA_GRID), len(D_values)))
    for i, theta in enumerate(THETA_GRID):
        for j, D in enumerate(D_values):
            analytical_argmins[i, j] = _per_theta_analytical_argmin(
                float(theta), float(D), MU0, SIGMA0, SIGMA,
            )
    # Note: at this slice (mu0=sigma0=sigma=1, w=0.5), theta_test doesn't
    # actually affect the per-slice argmin because the loss integrates over
    # the theta-grid. So all rows of analytical_argmins should be identical.
    # The variation is across D only.

    # Single argmin per D (collapsed across theta_test):
    argmin_by_D = analytical_argmins[0, :]  # All theta_test rows identical
    print("Analytical argmin at the demo slice, varying D:")
    for D, am in zip(D_values, argmin_by_D):
        print(f"  D={D:+.1f}  argmin_eta={am:+.3f}")

    # Now load each trained network and compute eta(theta_grid).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_eta = axes[0]

    # Left panel: eta(theta) curves
    for name, color in CONFIGS:
        path = Path(f"artifacts/probe_v4_{name}.eqx")
        if not path.exists():
            print(f"missing: {path}")
            continue
        art = EtaArtifact(
            artifact_path=path, name="probe", version=f"probe_v4_{name}",
        )
        art.load()
        eta_curve = art.predict_eta(
            theta=THETA_GRID,
            prior_hp=np.asarray([MU0, SIGMA0]),
            lik_hp=np.asarray([SIGMA]),
        )
        ax_eta.plot(THETA_GRID, eta_curve, color=color, lw=2.0, label=name)

    # Overlay the analytical argmin band: shaded region between min/max over D
    # for each theta_test (constant since rows are identical, but draw both
    # endpoints).
    # Alternative interpretation: the analytical argmin doesn't depend on
    # theta_test for this loss formulation — it's a single number per slice.
    # So plot it as a horizontal band per D.
    cmap = plt.get_cmap("Greys")
    for j, D in enumerate(D_values):
        gray = cmap(0.3 + 0.4 * (j / max(1, len(D_values) - 1)))
        ax_eta.axhline(argmin_by_D[j], color=gray, lw=1.0, ls="--",
                       label=f"argmin (D={D:+.0f})" if j in (0, 2, 4) else None)

    ax_eta.set_xlabel(r"$\theta$")
    ax_eta.set_ylabel(r"$\eta(\theta)$")
    ax_eta.set_title(rf"Trained $\eta(\theta)$ at demo slice "
                     rf"$\mu_0={MU0}, \sigma_0={SIGMA0}, \sigma={SIGMA}$ (w=0.5)")
    ax_eta.axhline(0.0, color="black", lw=0.5, ls=":")
    ax_eta.axhline(1.0, color="red", lw=0.5, ls=":", alpha=0.5)
    ax_eta.legend(fontsize=8, loc="best", ncols=2)
    ax_eta.set_ylim(-1.6, 1.6)
    ax_eta.grid(alpha=0.3)

    # Right panel: histograms of eta(theta) per fixture across hp range.
    # Note: predict_eta broadcasts a SINGLE (prior_hp, lik_hp) across the
    # theta vector; per-sample hps require a Python loop.
    ax_hist = axes[1]
    for name, color in CONFIGS:
        path = Path(f"artifacts/probe_v4_{name}.eqx")
        if not path.exists():
            continue
        art = EtaArtifact(
            artifact_path=path, name="probe", version=f"probe_v4_{name}",
        )
        art.load()
        rng = np.random.default_rng(0xCAFE)
        N = 200
        mu0_s = rng.uniform(-2.0, 2.0, N)
        sigma0_s = np.exp(rng.uniform(np.log(0.2), np.log(5.0), N))
        sigma_s = np.exp(rng.uniform(np.log(0.5), np.log(2.0), N))
        theta_s = rng.uniform(mu0_s - K * sigma0_s, mu0_s + K * sigma0_s)
        eta_s = np.empty(N, dtype=np.float64)
        for i in range(N):
            eta_s[i] = float(art.predict_eta(
                theta=theta_s[i:i + 1],
                prior_hp=np.asarray([mu0_s[i], sigma0_s[i]]),
                lik_hp=np.asarray([sigma_s[i]]),
            )[0])
        ax_hist.hist(eta_s, bins=40, alpha=0.5, color=color, label=name,
                     histtype="step", lw=1.5)
    ax_hist.set_xlabel(r"$\eta$ output")
    ax_hist.set_ylabel("count (over 200 random hp samples)")
    ax_hist.set_title(r"Distribution of $\eta$ outputs across hp range")
    ax_hist.axvline(0.0, color="black", lw=0.5, ls=":")
    ax_hist.axvline(1.0, color="red", lw=0.5, ls=":", alpha=0.5)
    ax_hist.legend(fontsize=8, loc="best")
    ax_hist.grid(alpha=0.3)
    ax_hist.set_xlim(-1.6, 1.6)

    fig.suptitle(
        "Stage-2 trained networks: η(θ) at demo slice + distribution across hp range",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path("output/illustrations/probe_eta_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
