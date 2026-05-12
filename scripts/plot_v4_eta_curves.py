"""Plot η(θ) curves for the Phase G v4 conditional learned-η fixtures.

Two panels:
  Panel A — η(θ) at the demo slice (μ₀=0, σ₀=σ=1) for all 3 losses
            + the numerical optimum.
  Panel B — η(θ) across σ₀ ∈ {0.5, 1, 2, 4} at (μ₀=0, σ=1) for one
            chosen loss (cd_variance, the audit's narrowest), to
            visualise how the learned selector adapts to prior strength.

Saves to ``output/illustrations/v4_eta_curves_<scheme>.png``.

Run from repo root::

    python -m scripts.plot_v4_eta_curves                  # default power_law
    python -m scripts.plot_v4_eta_curves --scheme ot      # different scheme
"""
from __future__ import annotations

import argparse
import importlib
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

_LOSSES = ["integrated_p", "cd_variance", "static_width"]
_COLORS = ["C0", "C1", "C2"]

# scheme name → (artifact-suffix, TiltingClass module:qualname). Artifact
# files use "powerlaw" for the power_law scheme; everything else matches
# the registry name verbatim.
_SCHEMES = {
    "power_law": ("powerlaw", "frasian.tilting.power_law:PowerLawTilting"),
    "ot": ("ot", "frasian.tilting.ot:OTTilting"),
    "mixture": ("mixture", "frasian.tilting.mixture:MixtureTilting"),
    "fisher_rao": ("fisher_rao", "frasian.tilting.fisher_rao:FisherRaoTilting"),
}


def _import_tilting_class(scheme: str):
    _, dotted = _SCHEMES[scheme]
    mod_name, cls_name = dotted.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def _fixtures_for(scheme: str) -> list[tuple[str, str, str]]:
    suffix, _ = _SCHEMES[scheme]
    return [
        (
            loss,
            color,
            f"artifacts/learned_eta_canonical_normal_normal_{suffix}_phaseC_{loss}_v4.eqx",
        )
        for loss, color in zip(_LOSSES, _COLORS)
    ]


def _load_fixtures(scheme: str):
    out = []
    for label, color, path in _fixtures_for(scheme):
        p = Path(path)
        if not p.exists():
            print(f"[skip] fixture missing: {p}")
            continue
        art = EtaArtifact(artifact_path=p)
        art.load()
        out.append((label, color, art))
    return out


def _numerical_eta(scheme: str, theta_grid, mu0, sigma0, sigma):
    TiltingCls = _import_tilting_class(scheme)
    tilt = TiltingCls()
    num = NumericalEtaSelector()
    waldo_named = _NamedStatistic("waldo")
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    return num.select_grid(
        theta_grid, tilt,
        statistic=waldo_named, model=model, prior=prior, alpha=0.05,
    )


def _admissibility_bounds(scheme: str, w: float) -> tuple[float, float] | None:
    """Return (eta_min, eta_max) for display, or None if scheme is unbounded.

    Fisher-Rao is geodesically complete on the half-plane (η ∈ ℝ); the
    other three schemes have finite admissibility and benefit from an
    explicit clip + axhlines on the demo plot.
    """
    if scheme == "power_law":
        return (-w / (1.0 - w), 1.0 / (1.0 - w))
    if scheme in ("ot", "mixture"):
        return (0.0, 1.0)
    if scheme == "fisher_rao":
        return None
    raise ValueError(f"unknown scheme: {scheme!r}")


def main(scheme: str = "power_law", out: Path | None = None) -> Path:
    fixtures = _load_fixtures(scheme)
    if not fixtures:
        raise FileNotFoundError(
            f"No phaseC v4 fixtures present in artifacts/ for scheme {scheme!r}."
        )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- Panel A: demo slice, all 3 losses + numerical ---
    mu0_a, sigma0_a, sigma_a = 0.0, 1.0, 1.0
    w_a = sigma0_a**2 / (sigma_a**2 + sigma0_a**2)
    bounds_a = _admissibility_bounds(scheme, w_a)
    theta_grid = np.linspace(-3.0, 3.0, 121)
    eta_num = _numerical_eta(scheme, theta_grid, mu0_a, sigma0_a, sigma_a)
    if bounds_a is not None:
        eta_min_a, eta_max_a = bounds_a
        # Display-clip the numerical curve to keep it from squashing the plot.
        # The actual selector clips at admissibility too; we show a slightly
        # tighter window for legibility.
        eta_num_disp = np.clip(eta_num, eta_min_a + 0.02, eta_max_a - 0.02)
        num_label = "numerical optimum (clipped to admissibility)"
    else:
        eta_num_disp = eta_num
        num_label = "numerical optimum"
    ax_a.plot(
        theta_grid, eta_num_disp,
        color="black", lw=1.0, ls=":",
        label=num_label,
    )
    learned_curves = []
    for label, color, art in fixtures:
        eta_l = art.predict_eta(
            theta_grid, np.array([mu0_a, sigma0_a]), np.array([sigma_a]),
        )
        ax_a.plot(theta_grid, eta_l, color=color, lw=2.0, label=f"learned [{label}]")
        learned_curves.append(np.asarray(eta_l))
    ax_a.axhline(1.0, color="red", lw=0.6, alpha=0.5, ls="--", label="η=1 (Wald)")
    ax_a.axhline(0.0, color="green", lw=0.6, alpha=0.5, ls="--", label="η=0 (WALDO)")
    if bounds_a is not None:
        ax_a.axhline(bounds_a[0], color="grey", lw=0.5, alpha=0.5)
        ax_a.axhline(bounds_a[1], color="grey", lw=0.5, alpha=0.5)
    ax_a.axvline(mu0_a, color="purple", lw=0.5, alpha=0.4, ls=":")
    ax_a.set_xlabel("θ")
    ax_a.set_ylabel("η(θ)")
    if bounds_a is not None:
        admiss_line = f"admissibility: ({bounds_a[0]:+.2f}, {bounds_a[1]:+.2f})"
    else:
        admiss_line = "admissibility: η ∈ ℝ (geodesically complete)"
    ax_a.set_title(
        f"A. {scheme} demo slice  "
        f"(μ₀={mu0_a:g}, σ₀={sigma0_a:g}, σ={sigma_a:g}, w={w_a:.2f})\n"
        + admiss_line,
        fontsize=10,
    )
    # Compute ylim from data to accommodate FR's unbounded η range
    # (cd_variance fixtures can drift well outside [-1.5, 1.5]).
    if learned_curves:
        all_eta = np.concatenate([eta_num_disp, *learned_curves])
        finite = all_eta[np.isfinite(all_eta)]
        if finite.size:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            pad = max(0.1, 0.1 * (hi - lo))
            ax_a.set_ylim(min(lo - pad, -1.5), max(hi + pad, 1.5))
        else:
            ax_a.set_ylim(-1.5, 1.5)
    else:
        ax_a.set_ylim(-1.5, 1.5)
    ax_a.legend(loc="lower right", fontsize=8)

    # --- Panel B: σ₀ sweep at (μ₀=0, σ=1) for cd_variance fixture ---
    cd_fix = next((f for f in fixtures if f[0] == "cd_variance"), fixtures[0])
    label_b, _, art_b = cd_fix
    mu0_b, sigma_b = 0.0, 1.0
    sigma0_grid = [0.5, 1.0, 2.0, 4.0]
    cmap = plt.colormaps["viridis"]
    theta_b = np.linspace(-4.0, 4.0, 121)
    panel_b_eta = []
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
        panel_b_eta.append(np.asarray(eta_l))
    ax_b.axhline(1.0, color="red", lw=0.6, alpha=0.5, ls="--")
    ax_b.axhline(0.0, color="green", lw=0.6, alpha=0.5, ls="--")
    ax_b.axvline(mu0_b, color="purple", lw=0.5, alpha=0.4, ls=":")
    ax_b.set_xlabel("θ")
    ax_b.set_ylabel("η(θ)")
    ax_b.set_title(
        f"B. {scheme} σ₀ sweep at (μ₀={mu0_b:g}, σ={sigma_b:g})  —  "
        f"fixture: [{label_b}]\n"
        "stronger prior (smaller σ₀) → smaller η (more prior reliance)",
        fontsize=10,
    )
    if panel_b_eta:
        all_b = np.concatenate(panel_b_eta)
        finite_b = all_b[np.isfinite(all_b)]
        if finite_b.size:
            lo_b = float(np.min(finite_b))
            hi_b = float(np.max(finite_b))
            pad_b = max(0.1, 0.1 * (hi_b - lo_b))
            ax_b.set_ylim(min(lo_b - pad_b, 0.5), max(hi_b + pad_b, 1.2))
        else:
            ax_b.set_ylim(0.5, 1.2)
    else:
        ax_b.set_ylim(0.5, 1.2)
    ax_b.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"Phase G v4 conditional learned-η: η(θ) curves — scheme = {scheme}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = out or Path(f"output/illustrations/v4_eta_curves_{scheme}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheme",
        choices=tuple(_SCHEMES),
        default="power_law",
        help="tilting scheme to plot (default power_law)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(scheme=args.scheme, out=args.out)
    print(f"wrote {path}")
