"""F1.2 — Synthetic Prospector-β rare-object illustration.

Two-panel figure:
(left) Galaxy parameter scatter (log SFR vs log M*) showing the typical-galaxy
       cloud + a single atypical (rare) galaxy. The empirical prior is
       essentially the density of the cloud.
(right) Schematic Bayesian recovery: the atypical galaxy's true parameters
        are pulled toward the cloud center by the prior, producing biased
        point estimates and credible regions that miss the truth.

Run from repo root:
    python docs/paper/figures/scripts/fig_1_2_prospector_synth.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(20260513)

# Typical galaxy population: bivariate Gaussian in (log SFR, log M*).
n_pop = 600
mu_pop = np.array([0.4, 10.5])
cov_pop = np.array([[0.30, 0.18], [0.18, 0.45]])
pop = rng.multivariate_normal(mu_pop, cov_pop, size=n_pop)

# Atypical (rare) galaxy: low SFR at high mass (e.g., post-starburst / quenched).
rare_true = np.array([-1.5, 11.0])

# Bayesian recovery: pulled toward population mean by w_eff in (0, 1).
# At rare-object boundary, the empirical prior dominates ⇒ w_eff small.
w_eff = 0.55
rare_post = w_eff * rare_true + (1 - w_eff) * mu_pop

# Credible region (1-sigma) around posterior point, set by the prior covariance
# scaled down (posterior is tighter than prior but still misses the truth).
post_cov = 0.45 * cov_pop


def ellipse_xy(mu, cov, n=200):
    """1-sigma ellipse for visualization."""
    from numpy.linalg import eigh

    eigval, eigvec = eigh(cov)
    angle = np.linspace(0, 2 * np.pi, n)
    circle = np.array([np.cos(angle), np.sin(angle)])
    transform = eigvec @ np.diag(np.sqrt(eigval))
    pts = transform @ circle
    return mu[:, None] + pts


fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)

# Common axes.
for ax in axes:
    ax.set_xlim(-3.0, 2.0)
    ax.set_ylim(8.5, 12.0)
    ax.set_xlabel(r"$\log\,\mathrm{SFR}\,/\,M_\odot\,\mathrm{yr}^{-1}$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(r"$\log\,M_\star\,/\,M_\odot$")

# === Panel 1: population scatter + rare object ===
ax = axes[0]
ax.scatter(pop[:, 0], pop[:, 1], s=8, color="#5882c2", alpha=0.45, label="typical galaxies")
# Density contour (just the 1-sigma ellipse of the population for clarity).
pop_ell = ellipse_xy(mu_pop, cov_pop)
ax.plot(pop_ell[0], pop_ell[1], color="#1f4e79", lw=1.5, ls="--", alpha=0.7)

ax.scatter(*rare_true, s=120, marker="*", color="#d62728", zorder=5,
           edgecolor="black", linewidth=0.6, label="rare object (truth)")

ax.set_title("Population + a rare object", fontsize=12)
ax.legend(loc="lower left", frameon=False, fontsize=9)

# === Panel 2: Bayesian shrinkage of the rare object ===
ax = axes[1]
# Faded population.
ax.scatter(pop[:, 0], pop[:, 1], s=8, color="#5882c2", alpha=0.18)
ax.plot(pop_ell[0], pop_ell[1], color="#1f4e79", lw=1.3, ls="--", alpha=0.4)

# True rare object.
ax.scatter(*rare_true, s=120, marker="*", color="#d62728", zorder=6,
           edgecolor="black", linewidth=0.6, label="true rare object")

# Bayesian-recovered point + credible region.
post_ell = ellipse_xy(rare_post, post_cov)
ax.fill(post_ell[0], post_ell[1], color="#2ca02c", alpha=0.18)
ax.plot(post_ell[0], post_ell[1], color="#2ca02c", lw=1.6, alpha=0.9,
        label="68% credible region")
ax.scatter(*rare_post, s=110, marker="o", color="#2ca02c", zorder=6,
           edgecolor="black", linewidth=0.6, label="posterior mode")

# Arrow truth → posterior.
ax.annotate(
    "",
    xy=tuple(rare_post + 0.08 * (mu_pop - rare_true) / np.linalg.norm(mu_pop - rare_true)),
    xytext=tuple(rare_true + 0.08 * (mu_pop - rare_true) / np.linalg.norm(mu_pop - rare_true)),
    arrowprops=dict(arrowstyle="->", lw=1.6, color="#444"),
)
ax.text(
    (rare_true[0] + rare_post[0]) / 2 + 0.05,
    (rare_true[1] + rare_post[1]) / 2 + 0.15,
    "shrinkage\nto prior",
    fontsize=9.5,
    color="#444",
    style="italic",
    ha="left",
)

ax.set_title("Bayesian recovery (under-covers)", fontsize=12)
ax.legend(loc="lower left", frameon=False, fontsize=9)

out_path = Path(__file__).parent.parent / "fig_1_2_prospector_synth.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
