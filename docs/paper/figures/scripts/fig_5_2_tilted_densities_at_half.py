"""F5.2 — Tilted densities at η=½ across the four geodesic schemes.

PL, FR, OT all produce Gaussian tilted distributions at η=½. MX produces a
two-component Gaussian mixture, bimodal at conflict.

Run from repo root:
    python docs/paper/figures/scripts/fig_5_2_tilted_densities_at_half.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Strong conflict to make MX bimodality visible.
mu_0, sigma_0 = -2.0, 1.0
D, sigma = 2.0, 1.0
w = sigma_0**2 / (sigma**2 + sigma_0**2)
mu_n = w * D + (1 - w) * mu_0
sigma_n = np.sqrt(w) * sigma

eta = 0.5

# === Per-scheme tilted (mu, sigma) at eta=0.5 ===
# PL.
denom_pl = 1 - eta * (1 - w)
mu_pl = (w * D + (1 - eta) * (1 - w) * mu_0) / denom_pl
sig_pl = np.sqrt(w * sigma**2 / denom_pl)

# OT.
mu_ot = (1 - eta) * mu_n + eta * D
sig_ot = (1 - eta) * sigma_n + eta * sigma

# FR (half-plane semicircle).
mt_a, mt_b = mu_n / np.sqrt(2), D / np.sqrt(2)
c = ((mt_a**2 - mt_b**2) + (sigma_n**2 - sigma**2)) / (2 * (mt_a - mt_b))
r = np.sqrt((mt_a - c)**2 + sigma_n**2)
phi_a = np.arctan2(sigma_n, mt_a - c)
phi_b = np.arctan2(sigma, mt_b - c)
s_eta = (1 - eta) * np.log(np.tan(phi_a / 2)) + eta * np.log(np.tan(phi_b / 2))
phi_eta = 2 * np.arctan(np.exp(s_eta))
mt_fr = c + r * np.cos(phi_eta)
sig_fr = r * np.sin(phi_eta)
mu_fr = mt_fr * np.sqrt(2)

# MX: two-component mixture.
def mx_pdf(theta):
    return (1 - eta) * norm.pdf(theta, mu_n, sigma_n) + eta * norm.pdf(theta, D, sigma)


theta_grid = np.linspace(-5, 5, 800)

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=True)

schemes = [
    ("Power-law (e)",     mu_pl,  sig_pl,  None,        "#d62728"),
    ("Mixture (m)",       None,   None,    mx_pdf,      "#2ca02c"),
    ("Fisher–Rao",        mu_fr,  sig_fr,  None,        "#7e3a93"),
    ("Optimal transport", mu_ot,  sig_ot,  None,        "#1f4e79"),
]

for ax, (name, mu, sig, pdf_fn, color) in zip(axes, schemes):
    if pdf_fn is not None:
        # MX: plot mixture + components dashed.
        pdf = pdf_fn(theta_grid)
        ax.fill_between(theta_grid, pdf, alpha=0.18, color=color)
        ax.plot(theta_grid, pdf, lw=2.4, color=color, label="mixture")
        # Components.
        ax.plot(theta_grid, (1 - eta) * norm.pdf(theta_grid, mu_n, sigma_n),
                lw=1.0, ls="--", color="#888", alpha=0.7, label=r"posterior $(\eta=0)$")
        ax.plot(theta_grid, eta * norm.pdf(theta_grid, D, sigma),
                lw=1.0, ls="--", color="#bbb", alpha=0.7, label=r"likelihood $(\eta=1)$")
    else:
        pdf = norm.pdf(theta_grid, mu, sig)
        ax.fill_between(theta_grid, pdf, alpha=0.18, color=color)
        ax.plot(theta_grid, pdf, lw=2.4, color=color)
        ax.text(0.04, 0.92, fr"$\mu_{{\eta}} = {mu:+.2f}$" "\n" fr"$\sigma_{{\eta}} = {sig:.3f}$",
                transform=ax.transAxes, fontsize=9.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, lw=0.8))

    ax.axvline(mu_n, color="#888", lw=0.7, ls=":", alpha=0.4)
    ax.axvline(D, color="#888", lw=0.7, ls=":", alpha=0.4)

    ax.set_title(name, fontsize=11, color=color)
    ax.set_xlabel(r"$\theta$")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(r"tilted density $\pi_{\eta=0.5}(\theta \mid D)$")
axes[1].legend(loc="upper right", frameon=False, fontsize=7.5)

fig.suptitle(
    fr"Tilted density at $\eta = 0.5$ across schemes (NN+Normal: $\mu_0={mu_0}$, $D={D}$, $w={w}$)",
    fontsize=12, y=1.05,
)

out_path = Path(__file__).parent.parent / "fig_5_2_tilted_densities_at_half.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
print(f"PL: (μ, σ) = ({mu_pl:+.3f}, {sig_pl:.3f})")
print(f"OT: (μ, σ) = ({mu_ot:+.3f}, {sig_ot:.3f})")
print(f"FR: (μ, σ) = ({mu_fr:+.3f}, {sig_fr:.3f})")
