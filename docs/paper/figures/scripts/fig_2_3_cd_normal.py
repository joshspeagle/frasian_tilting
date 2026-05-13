"""F2.3 — Confidence distribution for Normal location.

Two-panel figure: the CD H(θ; D) = Φ((θ − D)/σ) and its derivative h(θ; D)
on the Normal-location sandbox. Annotated with CD median, 1-α interval bounds,
and the U[0,1]-under-truth calibration property.

Run from repo root:
    python docs/paper/figures/scripts/fig_2_3_cd_normal.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Normal location sandbox.
D = 0.0
sigma = 1.0
alpha = 0.05

theta = np.linspace(-4, 4, 600)
H = norm.cdf((theta - D) / sigma)
h = norm.pdf(theta, loc=D, scale=sigma)

# CD median and (1-α) level set bounds.
median_theta = D  # since H(D) = 0.5 for symmetric Normal location
lo = D + sigma * norm.ppf(alpha / 2)
hi = D + sigma * norm.ppf(1 - alpha / 2)

fig, axes = plt.subplots(2, 1, figsize=(7, 5.4), sharex=True,
                         gridspec_kw={"height_ratios": [1.0, 0.9]})

# === Top: CDF H(θ; D) ===
ax = axes[0]
ax.plot(theta, H, color="#1f4e79", lw=2.2,
        label=r"$H(\theta; D) = \Phi\!\left(\frac{\theta - D}{\sigma}\right)$")
ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.6)
ax.axhline(alpha / 2, color="#888", lw=0.8, ls="-.", alpha=0.6)
ax.axhline(1 - alpha / 2, color="#888", lw=0.8, ls="-.", alpha=0.6)

ax.axvline(median_theta, color="#2ca02c", lw=1.2, ls="--", alpha=0.8)
ax.axvline(lo, color="#d62728", lw=1.0, ls=":", alpha=0.7)
ax.axvline(hi, color="#d62728", lw=1.0, ls=":", alpha=0.7)

ax.scatter([median_theta], [0.5], s=70, color="#2ca02c", zorder=5)
ax.scatter([lo, hi], [alpha / 2, 1 - alpha / 2], s=60, color="#d62728", zorder=5)

ax.text(-3.85, 0.52, r"$H = 0.5$", color="gray", fontsize=9, va="bottom")
ax.text(-3.85, alpha / 2 + 0.01, fr"$H = \alpha/2$", color="#888", fontsize=9, va="bottom")
ax.text(-3.85, 1 - alpha / 2 + 0.01, fr"$H = 1 - \alpha/2$", color="#888", fontsize=9, va="bottom")

ax.text(median_theta + 0.1, 0.93, "CD median\n= MLE", color="#2ca02c", fontsize=9.5,
        ha="left", va="top")
ax.text(lo - 0.1, 0.07, r"$\theta^-$", color="#d62728", fontsize=10.5, ha="right")
ax.text(hi + 0.1, 0.93, r"$\theta^+$", color="#d62728", fontsize=10.5, ha="left", va="top")

ax.set_ylabel(r"CD value $H(\theta; D)$")
ax.set_ylim(-0.02, 1.08)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Confidence distribution: CDF $H$ and density $h$ (Normal location, $\\sigma=1$)",
             fontsize=11, pad=10)

# === Bottom: density h(θ; D) ===
ax = axes[1]
ax.fill_between(theta, h, alpha=0.20, color="#1f4e79")
ax.plot(theta, h, color="#1f4e79", lw=2.0,
        label=r"$h(\theta; D) = \dfrac{\partial H}{\partial \theta}$")
ax.axvline(median_theta, color="#2ca02c", lw=1.2, ls="--", alpha=0.8)
ax.axvline(lo, color="#d62728", lw=1.0, ls=":", alpha=0.7)
ax.axvline(hi, color="#d62728", lw=1.0, ls=":", alpha=0.7)
ax.fill_between(theta, h, where=(theta < lo) | (theta > hi),
                color="#d62728", alpha=0.18)

ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel(r"CD density $h(\theta; D)$")
ax.set_ylim(0, 0.5)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(0, 0.42, r"$1-\alpha$ level set", color="#1f4e79", fontsize=10, ha="center")
ax.text(-3.0, 0.06, fr"$\alpha/2$", color="#d62728", fontsize=10, ha="center")
ax.text(3.0, 0.06, fr"$\alpha/2$", color="#d62728", fontsize=10, ha="center")

out_path = Path(__file__).parent.parent / "fig_2_3_cd_normal.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
