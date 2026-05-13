"""F1.1 — NN sandbox cartoon: prior, likelihood, posterior under prior-data conflict.

Standalone matplotlib script. Run from repo root:
    python docs/paper/figures/scripts/fig_1_1_nn_cartoon.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# NN sandbox parameters chosen to make conflict visible.
mu_0, sigma_0 = -2.0, 1.0          # prior
D, sigma = 2.0, 1.0                # likelihood data + scale

# Posterior closed form.
w = sigma_0**2 / (sigma**2 + sigma_0**2)
mu_n = w * D + (1 - w) * mu_0
sigma_n = np.sqrt(w) * sigma

# Grid for plotting.
theta = np.linspace(-5, 5, 600)
prior_pdf = norm.pdf(theta, mu_0, sigma_0)
lik_pdf = norm.pdf(theta, D, sigma)
post_pdf = norm.pdf(theta, mu_n, sigma_n)

fig, ax = plt.subplots(figsize=(7, 3.6))

ax.fill_between(theta, prior_pdf, alpha=0.18, color="#1f77b4")
ax.fill_between(theta, lik_pdf, alpha=0.18, color="#d62728")
ax.fill_between(theta, post_pdf, alpha=0.28, color="#2ca02c")

ax.plot(theta, prior_pdf, lw=1.8, color="#1f77b4", label=r"prior $\pi(\theta)$")
ax.plot(theta, lik_pdf, lw=1.8, color="#d62728", label=r"likelihood $L(\theta;D)$")
ax.plot(theta, post_pdf, lw=2.2, color="#2ca02c", label=r"posterior $\pi(\theta\mid D)$")

# Annotation lines.
for x, label, color, dy in [
    (mu_0, r"$\mu_0$", "#1f77b4", -0.04),
    (D, "$D$", "#d62728", -0.04),
    (mu_n, r"$\mu_n$", "#2ca02c", -0.04),
]:
    ax.axvline(x, color=color, lw=1.0, ls="--", alpha=0.6)
    ax.text(x, 0.45 + dy, label, color=color, ha="center", fontsize=11)

# Conflict arrow: from D to mu_n showing "Bayes pulls toward prior".
ax.annotate(
    "",
    xy=(mu_n + 0.12, 0.06),
    xytext=(D - 0.05, 0.06),
    arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"),
)
ax.text(
    (D + mu_n) / 2,
    0.10,
    "shrinkage",
    fontsize=10,
    color="#444",
    ha="center",
    style="italic",
)

ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel("density")
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.55)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

out_path = Path(__file__).parent.parent / "fig_1_1_nn_cartoon.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
