"""F4.4 — Adaptive confidence distribution under dynamic tilting.

Two scenarios on the same NN+Normal sandbox:
  - Low conflict (D = 0.5, μ_0 = 0):  prior + data agree; the adaptive CD
    is near-WALDO (uses the prior, tight).
  - High conflict (D = 3.0, μ_0 = 0): prior + data disagree; the adaptive
    CD is near-Wald (falls back, robust).

Each panel overlays the static-WALDO p-value (gray) and the dynamic-WALDO
p-value built from the U-shaped η(θ) of F4.3 (colored).

Run from repo root:
    python docs/paper/figures/scripts/fig_4_4_adaptive_cd.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# NN+Normal sandbox parameters.
mu_0, sigma_0 = 0.0, 1.0
sigma = 1.0
alpha = 0.05
w = sigma_0**2 / (sigma**2 + sigma_0**2)


def pl_tilt_moments(eta, D):
    """Theorem-6 closed form for (μ_η, σ_η²)."""
    denom = 1 - eta * (1 - w)
    mu_eta = (w * D + (1 - eta) * (1 - w) * mu_0) / denom
    sigma2_eta = w * sigma**2 / denom
    return mu_eta, sigma2_eta


def pl_tilted_pvalue(theta, eta, D):
    """Closed-form tilted-WALDO p-value via the (a, b) parameterization,
    generalized to η-tilted family. At η = 0 it reduces to bare WALDO.

    Replicate D' ~ N(theta, sigma^2); the tilted mu_eta(D') = α(η) D' + (1-α) μ_0
    with α = w + η(1-w); s_t = α σ. Standard residual a, b form follows.
    """
    alpha_eta = w + eta * (1 - w)
    s_t = (w + eta * (1 - w)) * sigma   # OT/PL share this s_t form on NN
    # Observed tilted mean.
    mu_eta_obs = alpha_eta * D + (1 - alpha_eta) * mu_0
    a = np.abs(mu_eta_obs - theta) / s_t
    b = (1 - alpha_eta) * (mu_0 - theta) / s_t   # Note: PL uses (1-η)(1-w)
    # The unified PL/OT closed form on NN (from ot.md / power_law.md):
    b_pl = (1 - eta) * (1 - w) * (mu_0 - theta) / s_t
    return norm.cdf(b_pl - a) + norm.cdf(-a - b_pl)


def eta_dynamic(theta):
    """U-shape: small near μ_0, → 1 at extremes."""
    z = (theta - mu_0) / sigma_0
    return 1 - np.exp(-(z / 2.0)**2)


theta_grid = np.linspace(-3.5, 5.5, 800)


def make_panel(ax, D, title):
    # Static WALDO (η = 0).
    p_static = pl_tilted_pvalue(theta_grid, 0.0, D)
    # Dynamic: η = η(θ) per point.
    p_dyn = np.array([pl_tilted_pvalue(t, eta_dynamic(t), D) for t in theta_grid])

    # Reference: pure Wald (η = 1) on D.
    p_wald = 2 * (1 - norm.cdf(np.abs(D - theta_grid) / sigma))

    ax.fill_between(theta_grid, 0, p_static, where=(p_static > alpha),
                    color="#888", alpha=0.18, label=None)
    ax.fill_between(theta_grid, 0, p_dyn, where=(p_dyn > alpha),
                    color="#1f4e79", alpha=0.18, label=None)
    ax.plot(theta_grid, p_static, color="#888", lw=1.6, ls="-",
            label="static WALDO ($\\eta = 0$)")
    ax.plot(theta_grid, p_dyn, color="#1f4e79", lw=2.2,
            label="dynamic ($\\eta(\\theta)$, U-shape)")
    ax.plot(theta_grid, p_wald, color="#d62728", lw=1.2, ls=":",
            label="Wald ($\\eta = 1$)")

    ax.axhline(alpha, color="#444", lw=0.9, ls="-.", alpha=0.6)
    ax.text(theta_grid[-1] - 0.1, alpha + 0.015,
            f"$\\alpha = {alpha}$", color="#444", fontsize=9, ha="right")

    ax.axvline(mu_0, color="#2ca02c", lw=0.7, ls=":", alpha=0.5)
    ax.axvline(D, color="#d62728", lw=0.7, ls=":", alpha=0.5)
    ax.text(mu_0, 1.02, r"$\mu_0$", color="#2ca02c", fontsize=10, ha="center")
    ax.text(D, 1.02, "$D$", color="#d62728", fontsize=10, ha="center")

    ax.set_xlabel(r"parameter $\theta$")
    ax.set_ylim(0, 1.10)
    ax.set_title(title, fontsize=11.5, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)
make_panel(axes[0], D=0.5,
           title=r"Low conflict ($D = 0.5$, $|\Delta| = 0.25$): adaptive CD $\approx$ WALDO (uses prior)")
make_panel(axes[1], D=3.0,
           title=r"High conflict ($D = 3$, $|\Delta| = 1.5$): adaptive CD $\to$ Wald (drops prior)")
for ax in axes:
    ax.set_ylabel("p-value $p(\\theta; D)$")
# Only the bottom panel needs the x-label.
axes[0].set_xlabel("")
axes[0].legend(loc="upper left", frameon=False, fontsize=9)

fig.suptitle("Adaptive confidence distribution: dynamic tilting adapts to local conflict",
             fontsize=12, y=1.00)

out_path = Path(__file__).parent.parent / "fig_4_4_adaptive_cd.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
