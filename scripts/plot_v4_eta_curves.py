"""Plot η(θ) curves for the Phase G v4 conditional learned-η fixtures.

Two panels:
  Panel A — η(θ) at the demo slice (μ₀=0, σ₀=σ=1) for all 3 losses
            + the numerical optimum.
  Panel B — η(θ) across σ₀ ∈ {0.5, 1, 2, 4} at (μ₀=0, σ=1) for one
            chosen loss (cd_variance, the audit's narrowest), to
            visualise how the learned selector adapts to prior strength.

Saves to ``output/illustrations/v4_eta_curves.png``.

Run from repo root:
    python -m scripts.plot_v4_eta_curves
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.eta_artifact import EtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.eta_selectors import (
    NumericalEtaSelector,
    _NamedStatistic,
)
from frasian.tilting.power_law import PowerLawTilting

_FIXTURES = [
    ("integrated_p", "C0",
     "artifacts/learned_eta_canonical_normal_normal_powerlaw_phaseC_integrated_p_v4.eqx"),
    ("cd_variance", "C1",
     "artifacts/learned_eta_canonical_normal_normal_powerlaw_phaseC_cd_variance_v4.eqx"),
    ("static_width", "C2",
     "artifacts/learned_eta_canonical_normal_normal_powerlaw_phaseC_static_width_v4.eqx"),
]


def _load_fixtures():
    out = []
    for label, color, path in _FIXTURES:
        p = Path(path)
        if not p.exists():
            print(f"[skip] fixture missing: {p}")
            continue
        art = EtaArtifact(artifact_path=p)
        art.load()
        out.append((label, color, art))
    return out


def _numerical_eta(theta_grid, mu0, sigma0, sigma):
    scheme = PowerLawTilting()
    num = NumericalEtaSelector()
    waldo_named = _NamedStatistic("waldo")
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    return num.select_grid(
        theta_grid, scheme,
        statistic=waldo_named, model=model, prior=prior, alpha=0.05,
    )


def main(out: Path | None = None) -> Path:
    fixtures = _load_fixtures()
    if not fixtures:
        raise FileNotFoundError("No v4 fixtures present in artifacts/.")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- Panel A: demo slice, all 3 losses + numerical ---
    mu0_a, sigma0_a, sigma_a = 0.0, 1.0, 1.0
    w_a = sigma0_a**2 / (sigma_a**2 + sigma0_a**2)
    eta_min_a = -w_a / (1.0 - w_a)
    eta_max_a = 1.0 / (1.0 - w_a)
    theta_grid = np.linspace(-3.0, 3.0, 121)
    eta_num = _numerical_eta(theta_grid, mu0_a, sigma0_a, sigma_a)
    # Display-clip the numerical curve to keep it from squashing the plot.
    # The actual selector clips at admissibility too; we show a slightly
    # tighter window for legibility.
    eta_num_disp = np.clip(eta_num, eta_min_a + 0.02, eta_max_a - 0.02)
    ax_a.plot(
        theta_grid, eta_num_disp,
        color="black", lw=1.0, ls=":",
        label="numerical optimum (clipped to admissibility)",
    )
    for label, color, art in fixtures:
        eta_l = art.predict_eta(
            theta_grid, np.array([mu0_a, sigma0_a]), np.array([sigma_a]),
        )
        ax_a.plot(theta_grid, eta_l, color=color, lw=2.0, label=f"learned [{label}]")
    ax_a.axhline(1.0, color="red", lw=0.6, alpha=0.5, ls="--", label="η=1 (Wald)")
    ax_a.axhline(0.0, color="green", lw=0.6, alpha=0.5, ls="--", label="η=0 (WALDO)")
    ax_a.axhline(eta_min_a, color="grey", lw=0.5, alpha=0.5)
    ax_a.axhline(eta_max_a, color="grey", lw=0.5, alpha=0.5)
    ax_a.axvline(mu0_a, color="purple", lw=0.5, alpha=0.4, ls=":")
    ax_a.set_xlabel("θ")
    ax_a.set_ylabel("η(θ)")
    ax_a.set_title(
        f"A. Demo slice  (μ₀={mu0_a:g}, σ₀={sigma0_a:g}, σ={sigma_a:g}, w={w_a:.2f})\n"
        f"admissibility: ({eta_min_a:+.2f}, {eta_max_a:+.2f})",
        fontsize=10,
    )
    ax_a.set_ylim(-1.5, 1.5)
    ax_a.legend(loc="lower right", fontsize=8)

    # --- Panel B: σ₀ sweep at (μ₀=0, σ=1) for cd_variance fixture ---
    cd_fix = next((f for f in fixtures if f[0] == "cd_variance"), fixtures[0])
    label_b, _, art_b = cd_fix
    mu0_b, sigma_b = 0.0, 1.0
    sigma0_grid = [0.5, 1.0, 2.0, 4.0]
    cmap = plt.colormaps["viridis"]
    theta_b = np.linspace(-4.0, 4.0, 121)
    for i, sigma0 in enumerate(sigma0_grid):
        w = sigma0**2 / (sigma_b**2 + sigma0**2)
        color = cmap(0.15 + 0.7 * i / max(len(sigma0_grid) - 1, 1))
        eta_l = art_b.predict_eta(
            theta_b, np.array([mu0_b, sigma0]), np.array([sigma_b]),
        )
        ax_b.plot(
            theta_b, eta_l, color=color, lw=2.0,
            label=f"σ₀={sigma0:g} (w={w:.2f})",
        )
    ax_b.axhline(1.0, color="red", lw=0.6, alpha=0.5, ls="--")
    ax_b.axhline(0.0, color="green", lw=0.6, alpha=0.5, ls="--")
    ax_b.axvline(mu0_b, color="purple", lw=0.5, alpha=0.4, ls=":")
    ax_b.set_xlabel("θ")
    ax_b.set_ylabel("η(θ)")
    ax_b.set_title(
        f"B. σ₀ sweep at (μ₀={mu0_b:g}, σ={sigma_b:g})  —  fixture: [{label_b}]\n"
        "stronger prior (smaller σ₀) → smaller η (more prior reliance)",
        fontsize=10,
    )
    ax_b.set_ylim(0.5, 1.2)
    ax_b.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Phase G v4 conditional learned-η: η(θ) curves",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = out or Path("output/illustrations/v4_eta_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(out=args.out)
    print(f"wrote {path}")
