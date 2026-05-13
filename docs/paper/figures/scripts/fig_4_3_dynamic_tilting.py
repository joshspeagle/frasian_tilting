"""F4.3 — Dynamic tilting illustration: η(θ) profile + tilted densities at three θ values.

Pedagogical figure. Top panel: a representative η(θ) function (U-shaped:
small at θ ≈ μ_0 where the prior is well-aligned and helpful, larger at
extremes where the prior is uninformative). Bottom panel: the PL-tilted
density q_{η(θ_test)}(·; D) at three θ_test values along the curve.

The key point: at each θ_test we use a *different* test (different η), but
the test at each θ_test is fixed before seeing D, preserving calibration.

Run from repo root:
    python docs/paper/figures/scripts/fig_4_3_dynamic_tilting.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# NN+Normal sandbox.
mu_0, sigma_0 = 0.0, 1.0
D, sigma = 0.5, 1.0
w = sigma_0**2 / (sigma**2 + sigma_0**2)
mu_n = w * D + (1 - w) * mu_0
sigma_n = np.sqrt(w) * sigma


def pl_tilt(eta):
    denom = 1 - eta * (1 - w)
    return ((w * D + (1 - eta) * (1 - w) * mu_0) / denom,
            np.sqrt(w * sigma**2 / denom))


def eta_dynamic(theta):
    """U-shaped η(θ): near 0 at θ = μ_0, → 1 far away (Wald fallback).

    Synthetic illustration of the dynamic-tilting hypothesis from CLAUDE.md:
    'small near μ_0 (use prior to tighten CI), → 1 far from μ_0 (Wald fallback).'
    """
    z = (theta - mu_0) / sigma_0
    return 1 - np.exp(-(z / 2.0)**2)


theta_grid = np.linspace(-4, 4, 600)
eta_curve = eta_dynamic(theta_grid)

# Pick three test points along the curve.
test_thetas = [0.0, 1.5, -2.5]
test_etas = [eta_dynamic(t) for t in test_thetas]
test_colors = ["#2ca02c", "#d6a72d", "#7e3a93"]

fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0),
                          gridspec_kw={"height_ratios": [0.9, 1.1]})

# === Top: η(θ) curve ===
ax = axes[0]
ax.plot(theta_grid, eta_curve, color="#1f4e79", lw=2.4)
ax.fill_between(theta_grid, eta_curve, alpha=0.10, color="#1f4e79")

# Mark mu_0.
ax.axvline(mu_0, color="#888", lw=0.7, ls=":", alpha=0.5)
ax.text(mu_0 + 0.05, 0.07, r"$\mu_0$", color="#888", fontsize=10, va="bottom")

# Mark test thetas.
for t, eta_val, color in zip(test_thetas, test_etas, test_colors):
    ax.scatter([t], [eta_val], s=80, color=color, zorder=5,
               edgecolor="white", linewidth=0.8)
    ax.text(t, eta_val + 0.08,
            f"$\\theta_{{\\rm test}} = {t}$\n$\\eta(\\theta) = {eta_val:.2f}$",
            color=color, fontsize=9.5, ha="center")

ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel(r"$\eta(\theta)$")
ax.set_xlim(-4, 4)
ax.set_ylim(-0.05, 1.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-3.85, 1.30,
        "dynamic tilting: η depends on θ, NOT on D\n"
        "→ test at each θ fixed before seeing data → calibrated",
        fontsize=9.5, color="#1f4e79", style="italic")
ax.set_title("Dynamic tilting: a deterministic η(θ) function", fontsize=12, pad=10)

# === Bottom: three tilted densities ===
ax = axes[1]
ax.axvline(mu_0, color="#888", lw=0.7, ls=":", alpha=0.5)
ax.axvline(D, color="#888", lw=0.7, ls=":", alpha=0.5)
ax.text(mu_0, 0.605, r"$\mu_0$", color="#888", fontsize=9.5, ha="center")
ax.text(D, 0.605, r"$D$", color="#888", fontsize=9.5, ha="center")

for t, eta_val, color in zip(test_thetas, test_etas, test_colors):
    mu_eta, sigma_eta = pl_tilt(eta_val)
    pdf = norm.pdf(theta_grid, mu_eta, sigma_eta)
    ax.fill_between(theta_grid, pdf, alpha=0.16, color=color)
    ax.plot(theta_grid, pdf, lw=1.8, color=color,
            label=fr"$\theta_{{\rm test}} = {t}$, $\eta = {eta_val:.2f}$")
    # Annotate the test-θ position with a vertical tick.
    ax.axvline(t, color=color, lw=1.2, ls="--", alpha=0.6)

ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel(r"tilted density $\pi_{\eta(\theta_{\rm test})}(\theta \mid D)$")
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.65)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

out_path = Path(__file__).parent.parent / "fig_4_3_dynamic_tilting.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
