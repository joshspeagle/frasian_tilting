"""F5.1 — Four geodesic paths between posterior and likelihood in (μ, σ) space.

PL (e-geodesic), MX (m-geodesic; plot via tilted mean/std), FR (Levi-Civita
on the Gaussian half-plane), OT (W₂; linear in (μ, σ)).

All four start at the posterior (μ_n, σ_n) at η=0 and end at the likelihood-
as-Gaussian (D, σ) at η=1. The paths differ measurably under conflict.

Run from repo root:
    python docs/paper/figures/scripts/fig_5_1_four_geodesics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# NN+Normal sandbox with substantial conflict and σ_n ≠ σ.
mu_0, sigma_0 = -2.0, 1.0
D, sigma = 2.0, 1.0
w = sigma_0**2 / (sigma**2 + sigma_0**2)
mu_n = w * D + (1 - w) * mu_0
sigma_n = np.sqrt(w) * sigma

print(f"Endpoints: posterior=({mu_n:.3f}, {sigma_n:.3f})  likelihood=({D:.3f}, {sigma:.3f})")

t_grid = np.linspace(0, 1, 200)


# ----- Power-law (e-geodesic) -----
def pl_path(eta):
    denom = 1 - eta * (1 - w)
    mu_eta = (w * D + (1 - eta) * (1 - w) * mu_0) / denom
    sigma2_eta = w * sigma**2 / denom
    return mu_eta, np.sqrt(sigma2_eta)


pl_mu, pl_sig = np.array([pl_path(t) for t in t_grid]).T


# ----- Mixture (m-geodesic): plot tilted mean + std -----
def mx_moments(eta):
    mean = (1 - eta) * mu_n + eta * D
    var = (1 - eta) * sigma_n**2 + eta * sigma**2 + (1 - eta) * eta * (mu_n - D)**2
    return mean, np.sqrt(var)


mx_mu, mx_sig = np.array([mx_moments(t) for t in t_grid]).T


# ----- OT (W₂): linear in (μ, σ) -----
def ot_path(eta):
    return (1 - eta) * mu_n + eta * D, (1 - eta) * sigma_n + eta * sigma


ot_mu, ot_sig = np.array([ot_path(t) for t in t_grid]).T


# ----- Fisher–Rao (half-plane semicircle) -----
def fr_path(eta_arr):
    """FR geodesic on (μ̃, σ) half-plane with μ̃ = μ/√2."""
    mu_a, sig_a = mu_n, sigma_n
    mu_b, sig_b = D, sigma
    mt_a, mt_b = mu_a / np.sqrt(2), mu_b / np.sqrt(2)
    if abs(mt_a - mt_b) < 1e-10:
        # Vertical case: σ(t) = σ_a^{1-t} σ_b^t
        mus = mt_a * np.ones_like(eta_arr) * np.sqrt(2)
        sigs = sig_a**(1 - eta_arr) * sig_b**eta_arr
        return mus, sigs
    # Center c on σ=0 axis; radius r.
    c = ((mt_a**2 - mt_b**2) + (sig_a**2 - sig_b**2)) / (2 * (mt_a - mt_b))
    r = np.sqrt((mt_a - c)**2 + sig_a**2)
    phi_a = np.arctan2(sig_a, mt_a - c)
    phi_b = np.arctan2(sig_b, mt_b - c)
    s_a = np.log(np.tan(phi_a / 2))
    s_b = np.log(np.tan(phi_b / 2))
    s = (1 - eta_arr) * s_a + eta_arr * s_b
    phi = 2 * np.arctan(np.exp(s))
    mt = c + r * np.cos(phi)
    sig = r * np.sin(phi)
    return mt * np.sqrt(2), sig


fr_mu, fr_sig = fr_path(t_grid)

# ----- Plot -----
fig, ax = plt.subplots(figsize=(7.6, 5.4))

paths = [
    ("Power-law ($e$-geodesic)",   pl_mu, pl_sig, "#d62728"),
    ("Mixture ($m$-geodesic)*",     mx_mu, mx_sig, "#2ca02c"),
    ("Fisher–Rao (Levi-Civita)",   fr_mu, fr_sig, "#7e3a93"),
    ("Optimal transport ($W_2$)",  ot_mu, ot_sig, "#1f4e79"),
]
markers = ["o", "s", "^", "D"]

for (label, mu, sig, color), mk in zip(paths, markers):
    ax.plot(mu, sig, lw=2.2, color=color, label=label, alpha=0.9)
    # Mark η=0.5 with a marker.
    idx = len(t_grid) // 2
    ax.scatter([mu[idx]], [sig[idx]], s=70, marker=mk, color=color,
               edgecolor="white", linewidth=0.8, zorder=5)

# Endpoints.
ax.scatter([mu_n], [sigma_n], s=180, marker="*", color="black", zorder=10,
           edgecolor="white", linewidth=1.0)
ax.scatter([D], [sigma], s=180, marker="*", color="black", zorder=10,
           edgecolor="white", linewidth=1.0)
ax.annotate(r"posterior $(\mu_n, \sigma_n)$", xy=(mu_n, sigma_n),
            xytext=(mu_n - 0.55, sigma_n - 0.13), fontsize=10,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="black"))
ax.annotate(r"likelihood $(D, \sigma)$", xy=(D, sigma),
            xytext=(D + 0.05, sigma + 0.10), fontsize=10,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="black"))

# η = 0.5 annotation in the upper-right corner.
ax.text(0.98, 0.97, r"markers: $\eta = 0.5$",
        transform=ax.transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#666", lw=0.6),
        verticalalignment="top", horizontalalignment="right")

# Footnote on MX in the lower-right corner.
ax.text(0.98, 0.05,
        "*MX path: (tilted mean, std) of the 2-Gaussian\n"
        "  mixture; not itself Gaussian.",
        transform=ax.transAxes, fontsize=8.5, color="#2ca02c",
        style="italic", horizontalalignment="right")

ax.set_xlabel(r"mean $\mu$")
ax.set_ylabel(r"std $\sigma$")
ax.set_xlim(-0.7, 2.5)
ax.set_ylim(0.5, 1.4)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title("Four geodesic paths between posterior and likelihood on (μ, σ)",
             fontsize=12, pad=10)

out_path = Path(__file__).parent.parent / "fig_5_1_four_geodesics.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
