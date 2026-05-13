"""F2.2 — WALDO p-value as a function of θ on Normal-Normal+Normal.

Plots p(θ) = Φ(b(θ) − a(θ)) + Φ(−a(θ) − b(θ)) with a, b from the framework's
closed form (T1.7)/(T1.6). Marks the WALDO pivot θ = μ_n, the α = 0.05
threshold, and the resulting confidence interval.

Run from repo root:
    python docs/paper/figures/scripts/fig_2_2_waldo_pvalue.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# NN+Normal sandbox parameters with moderate prior-data agreement.
mu_0, sigma_0 = 0.0, 1.0
D, sigma = 1.5, 1.0
alpha = 0.05

# Posterior closed form.
w = sigma_0**2 / (sigma**2 + sigma_0**2)
mu_n = w * D + (1 - w) * mu_0
sigma_n = np.sqrt(w) * sigma


def waldo_pvalue(theta):
    a = np.abs(mu_n - theta) / (w * sigma)
    b = (1 - w) * (mu_0 - theta) / (w * sigma)
    return norm.cdf(b - a) + norm.cdf(-a - b)


theta_grid = np.linspace(-3, 3, 800)
p_grid = waldo_pvalue(theta_grid)

# Find CI bounds (where p crosses alpha) via simple bracket.
inside = p_grid > alpha
# Indices of transitions.
ci_lo_idx = np.argmax(inside)
ci_hi_idx = len(inside) - 1 - np.argmax(inside[::-1])
ci_lo = theta_grid[ci_lo_idx]
ci_hi = theta_grid[ci_hi_idx]

fig, ax = plt.subplots(figsize=(7, 3.8))

# Shade CI region.
ax.fill_between(theta_grid, 0, p_grid, where=inside,
                color="#5882c2", alpha=0.22, label=fr"$C_{{1-\alpha}}(D)$ (level $\alpha = {alpha}$)")

# Plot p(theta).
ax.plot(theta_grid, p_grid, color="#1f4e79", lw=2.0, label=r"$p_{\rm WALDO}(\theta; D)$")

# Mark pivot.
ax.axvline(mu_n, color="#2ca02c", lw=1.4, ls="--", alpha=0.8)
ax.text(mu_n + 0.05, 1.02, r"$\mu_n$", color="#2ca02c", fontsize=11)
ax.scatter([mu_n], [waldo_pvalue(mu_n)], s=70, color="#2ca02c", zorder=5)

# Mark D for context.
ax.axvline(D, color="#d62728", lw=1.0, ls=":", alpha=0.6)
ax.text(D + 0.05, 0.92, r"$D$", color="#d62728", fontsize=11)

# Mark mu_0 for context.
ax.axvline(mu_0, color="#1f77b4", lw=1.0, ls=":", alpha=0.6)
ax.text(mu_0 + 0.05, 0.92, r"$\mu_0$", color="#1f77b4", fontsize=11)

# alpha threshold line.
ax.axhline(alpha, color="#888", lw=1.0, ls="-.", alpha=0.7)
ax.text(2.95, alpha + 0.015, fr"$\alpha = {alpha}$", color="#888",
        fontsize=10, ha="right")

# CI bounds.
ax.scatter([ci_lo, ci_hi], [alpha, alpha], s=50, color="#1f4e79", zorder=5)
ax.text(ci_lo - 0.05, alpha - 0.08, fr"$\theta^-$", color="#1f4e79",
        fontsize=10, ha="right")
ax.text(ci_hi + 0.05, alpha - 0.08, fr"$\theta^+$", color="#1f4e79",
        fontsize=10, ha="left")

ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel(r"WALDO p-value")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1.08)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(fr"WALDO p-value on NN+Normal ($\mu_0={mu_0}$, $\sigma_0={sigma_0}$, $D={D}$, $\sigma={sigma}$)",
             fontsize=11, pad=10)

out_path = Path(__file__).parent.parent / "fig_2_2_waldo_pvalue.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}; CI=[{ci_lo:.3f}, {ci_hi:.3f}]")
