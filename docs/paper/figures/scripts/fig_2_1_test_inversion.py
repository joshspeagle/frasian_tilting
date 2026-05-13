"""F2.1 — Neyman test inversion diagram.

Shows the acceptance region A_α(θ) and confidence region C_α(D) as
horizontal/vertical slices of the joint accept set {(θ, D) : T(θ; D) < q_{1-α}}
on the Normal-location sandbox (Wald statistic).

Run from repo root:
    python docs/paper/figures/scripts/fig_2_1_test_inversion.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm

# Wald statistic on Normal location, sigma=1.
sigma = 1.0
alpha = 0.05
z = norm.ppf(1 - alpha / 2)

# Joint accept set: |D - θ| < z·σ. Shaded strip in (θ, D).
theta_grid = np.linspace(-3, 3, 400)
D_grid = np.linspace(-3, 3, 400)

# Fixed θ and D for the slice annotations.
theta0 = 0.5
D_obs = 1.2

fig, ax = plt.subplots(figsize=(6.2, 5.6))

# Shade the joint accept set.
TH, DD = np.meshgrid(theta_grid, D_grid)
accept = np.abs(DD - TH) < z * sigma
ax.contourf(
    TH, DD, accept.astype(float),
    levels=[0.5, 1.5], colors=["#d9e7f5"], alpha=0.6,
)
ax.contour(
    TH, DD, np.abs(DD - TH) - z * sigma,
    levels=[0.0], colors=["#1f4e79"], linewidths=1.4, linestyles="--",
)

# Horizontal slice at θ = θ0: this is the acceptance region A_α(θ0) ⊂ D-space.
A_lo, A_hi = theta0 - z * sigma, theta0 + z * sigma
ax.plot([A_lo, A_hi], [theta0, theta0],
        color="#1f4e79", lw=4.0, solid_capstyle="butt", zorder=4)

# We want the convention: x-axis is θ, y-axis is D.
# A_α(θ0) lives at y = θ0... wait. Let me redo.
# Convention: A_α(θ) = {D : T(θ; D) < threshold}. For fixed θ on the x-axis,
# A_α(θ) is a SET of D values, displayed as a vertical interval at x = θ.
# C_α(D) = {θ : ...} for fixed D, displayed as a HORIZONTAL interval at y = D.
ax.cla()
ax.contourf(
    TH, DD, accept.astype(float),
    levels=[0.5, 1.5], colors=["#d9e7f5"], alpha=0.6,
)
ax.contour(
    TH, DD, np.abs(DD - TH) - z * sigma,
    levels=[0.0], colors=["#1f4e79"], linewidths=1.4, linestyles="--",
)

# A_α(θ0): vertical interval at x = θ0, spanning D ∈ [θ0 - zσ, θ0 + zσ].
ax.plot([theta0, theta0], [theta0 - z * sigma, theta0 + z * sigma],
        color="#1f4e79", lw=4.5, solid_capstyle="butt", zorder=5, label=r"$A_\alpha(\theta_0)$")
# C_α(D_obs): horizontal interval at y = D_obs, spanning θ ∈ [D_obs - zσ, D_obs + zσ].
ax.plot([D_obs - z * sigma, D_obs + z * sigma], [D_obs, D_obs],
        color="#d62728", lw=4.5, solid_capstyle="butt", zorder=5, label=r"$C_\alpha(D_{\rm obs})$")

# Mark the intersection point and the test value.
ax.scatter([theta0], [D_obs], s=80, marker="o", color="black", zorder=6)
ax.annotate(
    r"$(\theta_0, D_{\rm obs})$",
    xy=(theta0, D_obs),
    xytext=(theta0 + 0.25, D_obs + 0.25),
    fontsize=10,
    arrowprops=dict(arrowstyle="-", lw=0.8, color="black"),
)

# Annotations for axes.
ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel(r"data $D$")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")
ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
ax.axvline(0, color="gray", lw=0.5, alpha=0.4)

# θ0 and D_obs guide lines.
ax.axvline(theta0, color="#1f4e79", lw=0.7, ls=":", alpha=0.5)
ax.axhline(D_obs, color="#d62728", lw=0.7, ls=":", alpha=0.5)

# Labels at axes.
ax.text(theta0, -3.18, r"$\theta_0$", color="#1f4e79", fontsize=11, ha="center")
ax.text(-3.18, D_obs, r"$D_{\rm obs}$", color="#d62728", fontsize=11, va="center", ha="right")

# Background label.
ax.text(-2.4, 2.4,
        r"accept set: $\{(\theta, D) : |D - \theta| < z_{1-\alpha/2}\,\sigma\}$",
        fontsize=9.5, color="#1f4e79", style="italic")

ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title("Neyman test inversion (Normal-location, Wald)", fontsize=12, pad=10)

out_path = Path(__file__).parent.parent / "fig_2_1_test_inversion.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
