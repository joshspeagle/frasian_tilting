"""F4.1 — Power-law tilted family π_η on NN+Normal at five η values.

Visualizes the one-parameter Gaussian family interpolating the posterior
(η = 0) and the likelihood-as-Gaussian (η = 1) via the Theorem-6 closed form.

Run from repo root:
    python docs/paper/figures/scripts/fig_4_1_pl_tilted_family.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# NN+Normal sandbox at moderate conflict.
mu_0, sigma_0 = -1.5, 1.0
D, sigma = 1.5, 1.0
w = sigma_0**2 / (sigma**2 + sigma_0**2)
mu_n = w * D + (1 - w) * mu_0
sigma_n = np.sqrt(w) * sigma


def pl_tilt(eta):
    """Theorem-6 closed form: returns (mu_eta, sigma_eta) for power-law tilting."""
    denom = 1 - eta * (1 - w)
    mu_eta = (w * D + (1 - eta) * (1 - w) * mu_0) / denom
    sigma2_eta = w * sigma**2 / denom
    return mu_eta, np.sqrt(sigma2_eta)


etas = [0.0, 0.25, 0.5, 0.75, 1.0]
# Color from posterior-blue to likelihood-red.
cmap = plt.cm.coolwarm
colors = [cmap(0.05 + 0.9 * i / (len(etas) - 1)) for i in range(len(etas))]

theta = np.linspace(-4, 4, 600)

fig, ax = plt.subplots(figsize=(7.5, 4.2))

# Mark mu_0 and D.
ax.axvline(mu_0, color="#1f77b4", lw=0.8, ls=":", alpha=0.5)
ax.axvline(D, color="#d62728", lw=0.8, ls=":", alpha=0.5)
ax.text(mu_0, 0.61, r"$\mu_0$", color="#1f77b4", fontsize=10, ha="center")
ax.text(D, 0.61, r"$D$", color="#d62728", fontsize=10, ha="center")

for eta, color in zip(etas, colors):
    mu_eta, sig_eta = pl_tilt(eta)
    pdf = norm.pdf(theta, mu_eta, sig_eta)
    ax.fill_between(theta, pdf, alpha=0.15, color=color)
    ax.plot(theta, pdf, lw=1.8 if eta in (0.0, 1.0) else 1.4,
            color=color, label=fr"$\eta = {eta:.2f}$")
    # Mark mean.
    ax.scatter([mu_eta], [norm.pdf(mu_eta, mu_eta, sig_eta)], s=30,
               color=color, zorder=5, edgecolor="white", linewidth=0.6)

# Endpoint labels.
mu_0_pdf = norm.pdf(theta, *pl_tilt(0.0))
mu_1_pdf = norm.pdf(theta, *pl_tilt(1.0))
ax.text(*pl_tilt(0.0)[:1], 0.55,
        "posterior\n($\\eta = 0$)", color=colors[0],
        fontsize=10, ha="center", va="top")
ax.text(*pl_tilt(1.0)[:1], 0.55,
        "likelihood\n($\\eta = 1$)", color=colors[-1],
        fontsize=10, ha="center", va="top")

ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel(r"tilted density $\pi_\eta(\theta \mid D)$")
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.65)
ax.legend(loc="upper left", frameon=False, fontsize=9.5,
          title=r"$\pi_\eta \propto \pi^{1-\eta} L$ (Theorem 6)",
          title_fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(fr"Power-law $\eta$-tilting family (NN+Normal: $\mu_0={mu_0}$, $D={D}$, $w={w}$)",
             fontsize=11.5, pad=10)

out_path = Path(__file__).parent.parent / "fig_4_1_pl_tilted_family.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
