"""Illustration: learned-η dynamic selector vs the legacy NumericalEtaSelector.

Two panels:
  Panel A — η*(|Δ|; w) curves at three w values for both selectors.
            Shows the smooth + monotone curve from the trained MLP vs
            the kinky clamp-then-jump curve from NumericalEtaSelector.

  Panel B — Width vs Wald baseline at three w values, computed via
            tilting.confidence_regions for each (w, |Δ|).

Requires the v0_smoke or v1 checkpoint to exist. If neither exists,
the script trains a tiny smoke checkpoint on the fly so the demo is
runnable without prior setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.learned.monotonic_eta import MonotonicEtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import (DynamicNumericalEtaSelector,
                                              LearnedDynamicEtaSelector,
                                              _NamedStatistic)
from frasian.tilting.power_law import PowerLawTilting


def _ensure_checkpoint(smoke: bool) -> Path:
    """Pick the best available checkpoint; train a smoke one if missing."""
    candidates = [
        Path("artifacts/learned_eta_power_law_v1.pt"),
        Path("artifacts/learned_eta_power_law_v0_smoke.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: train a tiny one on the fly.
    print("[learned_eta_demo] no checkpoint found; training a smoke one...")
    from frasian.learned.training.train import fit_monotonic_eta_artifact
    out = Path("artifacts/learned_eta_power_law_smoke_demo.pt")
    fit_monotonic_eta_artifact(
        scheme_name="power_law", loss_kind="integrated_p",
        n_lhs=512, n_mc=2, n_epochs=10 if smoke else 30,
        batch_size=16, theta_grid_n=51,
        out_path=out, seed=42, verbose=False,
    )
    return out


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma, mu0 = 1.0, 0.0
    alpha = 0.05
    ckpt_path = _ensure_checkpoint(smoke=smoke)

    artifact = MonotonicEtaArtifact(artifact_path=ckpt_path)
    artifact.load()
    learned_selector = LearnedDynamicEtaSelector(artifact=artifact)
    legacy_selector = DynamicNumericalEtaSelector(sigma=sigma, mu0=mu0)
    scheme = PowerLawTilting()

    n_grid = 51 if smoke else 201
    abs_delta_grid = np.linspace(0.0, 5.0, n_grid)
    w_values = [0.2, 0.5, 0.8]

    fig, (ax_eta, ax_w) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    cmap = plt.cm.viridis
    for k, w_val in enumerate(w_values):
        color = cmap(0.2 + 0.6 * k / max(1, len(w_values) - 1))

        # Panel A — eta curves.
        eta_learned = learned_selector.select_grid(
            abs_delta_grid, scheme, statistic=WaldoStatistic(),
            w=w_val, alpha=alpha,
        )
        eta_legacy = legacy_selector.select_grid(
            abs_delta_grid, scheme, statistic=_NamedStatistic("waldo"),
            w=w_val, alpha=alpha,
        )
        ax_eta.plot(abs_delta_grid, eta_learned, color=color, lw=2.0,
                     label=fr"learned w={w_val}")
        ax_eta.plot(abs_delta_grid, eta_legacy, color=color, lw=1.0,
                     linestyle="--", alpha=0.7,
                     label=fr"legacy w={w_val}")

    ax_eta.axhline(0.0, color="grey", lw=0.5, alpha=0.5)
    ax_eta.axhline(1.0, color="grey", lw=0.5, alpha=0.5)
    ax_eta.set_xlabel(r"$|\Delta|$")
    ax_eta.set_ylabel(r"$\eta^*(|\Delta|; w)$")
    ax_eta.set_title(
        "η selection: learned (solid) vs legacy NumericalEtaSelector (dashed)\n"
        "(η=1 → Wald limit; η<0 → oversharpening regime)"
    )
    ax_eta.legend(loc="best", fontsize=7, ncol=2)

    # Panel B — width comparison via confidence_regions.
    sigma0_for = lambda w: float(np.sqrt(w / max(1.0 - w, 1e-9)) * sigma)
    n_D = 9 if smoke else 21
    for k, w_val in enumerate(w_values):
        color = cmap(0.2 + 0.6 * k / max(1, len(w_values) - 1))
        sigma0 = sigma0_for(w_val)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        model = NormalNormalModel(sigma=sigma)
        D_grid = np.linspace(-4.0, 4.0, n_D)
        widths_learned = np.empty(n_D)
        widths_legacy = np.empty(n_D)
        for i, D in enumerate(D_grid):
            data = np.asarray([D])
            for label, sel, store in [
                ("learned", learned_selector, widths_learned),
                ("legacy", legacy_selector, widths_legacy),
            ]:
                pl = PowerLawTilting(selector=sel)
                regions = pl.confidence_regions(
                    alpha, data, model, prior, WaldoStatistic(),
                )
                store[i] = sum(hi - lo for lo, hi in regions)
        abs_delta_D = np.abs((1.0 - w_val) * (mu0 - D_grid) / sigma)
        ax_w.plot(abs_delta_D, widths_learned, color=color, lw=2.0,
                    label=fr"learned w={w_val}")
        ax_w.plot(abs_delta_D, widths_legacy, color=color, lw=1.0,
                    linestyle="--", alpha=0.7,
                    label=fr"legacy w={w_val}")
    ax_w.axhline(2 * 1.96, color="black", lw=1.0, alpha=0.5,
                   label="Wald baseline")
    ax_w.set_xlabel(r"$|\Delta_D|$")
    ax_w.set_ylabel("CI width (union)")
    ax_w.set_title("Dynamic-WALDO CI width: learned (solid) vs legacy (dashed)")
    ax_w.legend(loc="best", fontsize=7, ncol=2)

    fig.tight_layout()
    out = out or Path("output/illustrations/learned_eta_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(smoke=args.smoke, out=args.out)
    print(f"wrote {path}")
