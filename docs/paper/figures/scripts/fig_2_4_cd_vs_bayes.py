"""F2.4 — Confidence distribution vs Bayesian posterior side-by-side.

Same visual on Normal location; the difference is semantic, not visual.
Left: CD H(θ; D) — calibrated by repeated experimentation (U[0,1] under truth).
Right: flat-prior Bayesian posterior — calibrated by Bayes' rule (degrees of belief).

The two coincide visually under a flat prior but answer different questions.
Under informative prior, they diverge — entry point to prior-data conflict (§3).

Run from repo root:
    python docs/paper/figures/scripts/fig_2_4_cd_vs_bayes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Normal location sandbox.
D = 0.5
sigma = 1.0
theta = np.linspace(-3.5, 3.5, 600)

# CD density.
cd_density = norm.pdf(theta, loc=D, scale=sigma)

# Bayesian posterior under flat (improper uniform) prior — same Gaussian shape.
post_density = norm.pdf(theta, loc=D, scale=sigma)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

# === Left: CD ===
ax = axes[0]
ax.fill_between(theta, cd_density, alpha=0.22, color="#1f4e79")
ax.plot(theta, cd_density, color="#1f4e79", lw=2.2)
ax.axvline(D, color="#1f4e79", lw=1.2, ls="--", alpha=0.7)
ax.text(D + 0.1, 0.40, "CD median\n= MLE", color="#1f4e79", fontsize=10, va="top")

ax.set_title("Confidence distribution (frequentist)", fontsize=12, pad=10)
ax.set_xlabel(r"parameter $\theta$", labelpad=8)
ax.set_ylabel("density")
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(0, 0.48)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotation: semantics. Place below curve to avoid overlap.
ax.text(
    0.5, -0.30,
    "calibrated by repeated experiment:\n"
    r"$H(\theta_0; D) \sim U[0,1]$ when $D \sim P_{\theta_0}$",
    transform=ax.transAxes, fontsize=9.5, va="top", ha="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf0f7", edgecolor="#1f4e79", lw=0.8),
)

# === Right: Bayesian posterior ===
ax = axes[1]
ax.fill_between(theta, post_density, alpha=0.22, color="#7e3a93")
ax.plot(theta, post_density, color="#7e3a93", lw=2.2)
ax.axvline(D, color="#7e3a93", lw=1.2, ls="--", alpha=0.7)
ax.text(D + 0.1, 0.40, "posterior mode\n(= MLE under flat prior)",
        color="#7e3a93", fontsize=10, va="top")

ax.set_title("Bayesian posterior (degrees of belief)", fontsize=12, pad=10)
ax.set_xlabel(r"parameter $\theta$", labelpad=8)
ax.set_xlim(-3.5, 3.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotation: semantics. Place below curve to avoid overlap.
ax.text(
    0.5, -0.30,
    "calibrated by Bayes' rule:\n"
    r"$\pi(\theta\mid D) \propto \pi(\theta)\,L(\theta;D)$"
    "  (flat prior here)",
    transform=ax.transAxes, fontsize=9.5, va="top", ha="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3e9f8", edgecolor="#7e3a93", lw=0.8),
)

fig.suptitle("Same visual, different semantics: CD vs Bayesian posterior\n"
             "(Normal location, flat prior; both densities $= N(D, \\sigma^2)$)",
             fontsize=11.5, y=1.04)

out_path = Path(__file__).parent.parent / "fig_2_4_cd_vs_bayes.png"
# Bump bottom margin so the annotation boxes fit.
fig.subplots_adjust(bottom=0.30)
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
