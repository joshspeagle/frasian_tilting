"""Illustration: confidence distributions and confidence curves on the
conjugate-Normal sandbox.

`python -m frasian.experiments.illustrations.confidence_distribution_demo
--smoke` runs in fast mode and produces a PNG at
`output/illustrations/confidence_distribution_demo.png`. For four
canonical D values at w=0.5, side-by-side panels overlay:
  - left:  the CDFs of (identity, wald), (identity, waldo),
           (power_law[dynamic_numerical], waldo)
  - right: the confidence curves cc(θ) = 2·min(F(θ), 1−F(θ)) for the
           same three cells, with the α=0.05 horizontal marked.

The Dyn-WALDO curve is the visual punchline: at small |Δ| it sits
near (identity, waldo); at large |Δ| it asymptotes back toward Wald
*after* a transient detour. The same shape that appears in our
smoothness diagnostic now visible in distributional form.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.cd.from_pvalue import build_cd_from_pvalue
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    alpha = 0.05
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=1.0)  # w = 0.5

    # Three cells mirroring `default_tiltings()`.
    n_grid_dyn = 201 if smoke else 401
    coarse_n = 11 if smoke else 25
    cells = [
        ("Wald", IdentityTilting(), WaldStatistic(), "#DC3545"),
        ("WALDO", IdentityTilting(), WaldoStatistic(), "#2E86AB"),
        ("Dyn-WALDO",
            PowerLawTilting(selector=DynamicNumericalEtaSelector(
                sigma=sigma, mu0=0.0, n_grid=n_grid_dyn, coarse_n=coarse_n,
            )),
            WaldoStatistic(),
            "#7B2CBF"),
    ]

    n_cd = 401 if smoke else 1001
    canonical_Ds = [0.0, 1.0, 2.0, 3.0]

    fig, axes = plt.subplots(len(canonical_Ds), 2,
                              figsize=(11, 2.6 * len(canonical_Ds)),
                              squeeze=False)

    for r, D in enumerate(canonical_Ds):
        ax_cdf = axes[r][0]
        ax_cc = axes[r][1]
        for label, tilting, statistic, color in cells:
            cd = build_cd_from_pvalue(
                tilting, statistic, D, model, prior,
                n_grid=n_cd,
            )
            theta = cd.theta_grid
            cdf = cd.cdf_values
            cc = 2.0 * np.minimum(cdf, 1.0 - cdf)
            ax_cdf.plot(theta, cdf, color=color, lw=1.6, label=label)
            ax_cc.plot(theta, cc, color=color, lw=1.6, label=label)
        # Reference markers.
        ax_cdf.axvline(D, color="0.4", lw=1, ls=":", alpha=0.7)
        ax_cdf.axhline(0.5, color="0.6", lw=0.7, ls="--", alpha=0.5)
        ax_cc.axvline(D, color="0.4", lw=1, ls=":", alpha=0.7)
        ax_cc.axhline(alpha, color="0.4", lw=1, ls="--",
                       label=fr"$\alpha={alpha}$")
        # Cosmetics.
        ax_cdf.set_xlim(D - 6, D + 6)
        ax_cdf.set_ylim(-0.02, 1.02)
        ax_cdf.set_ylabel("CDF $F(\\theta)$", fontsize=9)
        ax_cdf.set_title(f"D = {D:.2f}, w = 0.5", fontsize=9)
        ax_cc.set_xlim(D - 6, D + 6)
        ax_cc.set_ylim(-0.02, 1.02)
        ax_cc.set_ylabel(r"cc$(\theta) = 2\min(F, 1-F)$", fontsize=9)
        if r == 0:
            ax_cdf.legend(loc="lower right", frameon=False, fontsize=7)
            ax_cc.legend(loc="upper right", frameon=False, fontsize=7)
        if r == len(canonical_Ds) - 1:
            ax_cdf.set_xlabel(r"$\theta$")
            ax_cc.set_xlabel(r"$\theta$")

    fig.suptitle(
        "Confidence distributions and confidence curves: "
        "Wald vs WALDO vs Dyn-WALDO at canonical D values",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = out or Path("output/illustrations/confidence_distribution_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="fast mode used by CI")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(smoke=args.smoke, out=args.out)
    print(f"wrote {path}")
