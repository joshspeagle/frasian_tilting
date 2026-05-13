"""F6.2 — Learned η(θ) curves from EtaNet on the canonical NN+Normal PL fixture.

Replacement for the legacy `results/phaseC_eta_plots/eta_curves_per_loss.png` —
same content (3 learned losses + analytic NumericalEtaSelector reference) but
with the legend placed below the plot to avoid obscuring the curves.

Loads trained v4 PL artifacts from `artifacts/`:
  - integrated_p, cd_variance, static_width

Run from repo root:
    python docs/paper/figures/scripts/fig_6_2_learned_eta_curves.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from frasian._registry_bootstrap import bootstrap  # noqa: E402
bootstrap()
from frasian.learned.eta_artifact import EtaArtifact  # noqa: E402
from frasian.models.distributions import NormalDistribution  # noqa: E402
from frasian.models.normal_normal import NormalNormalModel  # noqa: E402
from frasian.tilting.eta_selectors import NumericalEtaSelector, _NamedStatistic  # noqa: E402
from frasian.tilting.power_law import PowerLawTilting  # noqa: E402

LOSSES = [
    ("integrated_p", "#1f4e79"),
    ("cd_variance",  "#d6a72d"),
    ("static_width", "#2ca02c"),
]

# Canonical NN+Normal demo slice.
mu_0, sigma_0, sigma = 0.0, 1.0, 1.0
w = sigma_0**2 / (sigma**2 + sigma_0**2)
theta_grid = np.linspace(-5.0, 5.0, 201)

fig, ax = plt.subplots(figsize=(7.6, 4.5))

# Numerical reference (analytic per-θ argmin).
tilt = PowerLawTilting()
num = NumericalEtaSelector()
waldo = _NamedStatistic("waldo")
model = NormalNormalModel(sigma=sigma)
prior = NormalDistribution(loc=mu_0, scale=sigma_0)
eta_num = num.select_grid(theta_grid, tilt, statistic=waldo,
                          model=model, prior=prior, alpha=0.05)
ax.plot(theta_grid, eta_num, color="black", lw=1.2, ls="--",
        label="NumericalEtaSelector $\\eta^*(\\theta)$ (per-$\\theta$ argmin)")

# Learned curves.
for loss, color in LOSSES:
    path = REPO / "artifacts" / f"learned_eta_canonical_normal_normal_powerlaw_phaseC_{loss}_v4.eqx"
    if not path.exists():
        print(f"[skip] missing: {path}")
        continue
    art = EtaArtifact(artifact_path=path)
    art.load()
    eta_l = art.predict_eta(theta_grid,
                            np.array([mu_0, sigma_0]), np.array([sigma]))
    ax.plot(theta_grid, eta_l, color=color, lw=2.4, label=f"learned [{loss}]")

# Reference lines for the η-axis: η=0 (WALDO) and η=1 (Wald).
ax.axhline(0.0, color="#2ca02c", lw=0.7, ls=":", alpha=0.5)
ax.axhline(1.0, color="#d62728", lw=0.7, ls=":", alpha=0.5)
ax.text(5.05, 0.0, r"$\eta=0$ (WALDO)", color="#2ca02c", fontsize=9, va="center")
ax.text(5.05, 1.0, r"$\eta=1$ (Wald)", color="#d62728", fontsize=9, va="center")

# Mark μ_0.
ax.axvline(mu_0, color="#888", lw=0.6, ls=":", alpha=0.4)

# Shade σ₀-anchored training range.
ax.axvspan(mu_0 - 2.5 * sigma_0, mu_0 + 2.5 * sigma_0,
           color="#5882c2", alpha=0.06, label=r"$\sigma_0$-anchored training range")

ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel(r"$\eta(\theta)$")
ax.set_xlim(-5, 5)
ax.set_ylim(-1.3, 1.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title(f"Learned $\\eta(\\theta)$ curves vs analytic per-$\\theta$ reference\n"
             f"(canonical NN, power_law, $w = {w}$)", fontsize=11, pad=10)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.50), frameon=False,
          fontsize=9, ncol=2)

out_path = Path(__file__).parent.parent / "fig_6_2_learned_eta_curves.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
