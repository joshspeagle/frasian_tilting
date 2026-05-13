"""F6.4 — Learned admissibility surface P(valid | θ, η) for ValidityNet.

Replacement for the legacy `results/phaseC_eta_plots/validity_heatmaps.png` —
same content (3 panels for the 3 loss heads, with learned η(θ) trajectory
overlaid in white) but with the per-panel "learned η(θ)" legend removed (the
white curve is self-evident) and a shared colorbar moved out of plot area.

Loads trained v4 PL artifacts from `artifacts/`.

Run from repo root:
    python docs/paper/figures/scripts/fig_6_4_validity_heatmaps.py
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from frasian._registry_bootstrap import bootstrap  # noqa: E402
bootstrap()
from frasian.learned.eta_artifact import EtaArtifact  # noqa: E402

LOSSES = ["integrated_p", "cd_variance", "static_width"]

mu_0, sigma_0, sigma = 0.0, 1.0, 1.0
theta_grid = np.linspace(-4.0, 4.0, 81)
eta_grid = np.linspace(-2.0, 2.0, 81)
TH, EE = np.meshgrid(theta_grid, eta_grid)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)

prior_hp = jnp.asarray([mu_0, sigma_0])
lik_hp = jnp.asarray([sigma])

for ax, loss in zip(axes, LOSSES):
    path = REPO / "artifacts" / f"learned_eta_canonical_normal_normal_powerlaw_phaseC_{loss}_v4.eqx"
    if not path.exists():
        print(f"[skip] missing: {path}")
        continue
    art = EtaArtifact(artifact_path=path)
    art.load()

    # P(valid | theta, eta).
    flat_theta = jnp.asarray(TH.ravel())
    flat_eta = jnp.asarray(EE.ravel())
    flat_pvalid = art.predict_validity(flat_theta, prior_hp, lik_hp, flat_eta)
    pvalid_grid = np.asarray(flat_pvalid).reshape(TH.shape)

    # Learned eta trajectory.
    eta_l = np.asarray(art.predict_eta(theta_grid, np.array([mu_0, sigma_0]),
                                       np.array([sigma])))

    im = ax.pcolormesh(TH, EE, pvalid_grid, cmap="viridis",
                       vmin=0.0, vmax=1.0, shading="auto")
    ax.plot(theta_grid, eta_l, color="white", lw=2.2, zorder=5)
    ax.plot(theta_grid, eta_l, color="black", lw=0.6, zorder=6, alpha=0.6)

    ax.set_xlabel(r"$\theta$")
    ax.set_title(f"{loss}\n$P(\\mathrm{{valid}} \\mid \\theta, \\eta)$",
                 fontsize=11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 2)

axes[0].set_ylabel(r"$\eta$")

# Shared colorbar to the right of all panels.
cbar = fig.colorbar(im, ax=axes, location="right", pad=0.02, shrink=0.85)
cbar.set_label(r"$P(\mathrm{valid})$", fontsize=10)

# Inline annotation labelling the learned eta trajectory (one shared legend).
fig.text(0.5, -0.04,
         "White curve: learned $\\eta(\\theta)$ trajectory from EtaNet.",
         ha="center", fontsize=9.5, style="italic")

fig.suptitle(
    r"Learned admissibility surface $P(\mathrm{valid} \mid \theta, \eta)$ "
    r"(canonical NN power-law, $w = 0.5$)",
    fontsize=12, y=1.02,
)

out_path = Path(__file__).parent.parent / "fig_6_4_validity_heatmaps.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
