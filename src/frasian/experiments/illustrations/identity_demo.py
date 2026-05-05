"""Illustration: IdentityTilting passthrough on the conjugate-Normal sandbox.

`python -m frasian.experiments.illustrations.identity_demo --smoke` runs
in fast mode (small grid, no figure window), producing a PNG at
`output/illustrations/identity_demo.png`. Side-by-side panels show that
`(identity, wald)` and `(identity, waldo)` reproduce the bare statistic
CIs exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.identity import IdentityTilting


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    D = 1.5
    alpha = 0.05
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    ident = IdentityTilting()

    n_grid = 121 if smoke else 401
    thetas = np.linspace(-3, 6, n_grid)

    wald = WaldStatistic()
    waldo = WaldoStatistic()
    p_wald = wald.pvalue(thetas, np.asarray([D]), model)
    p_waldo = waldo.pvalue(thetas, np.asarray([D]), model, prior)

    ci_wald = ident.confidence_interval(alpha, np.asarray([D]), model, prior, wald)
    ci_waldo = ident.confidence_interval(alpha, np.asarray([D]), model, prior, waldo)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (p, ci, color, label) in zip(
        axes,
        [
            (p_wald, ci_wald, "#DC3545", "Wald"),
            (p_waldo, ci_waldo, "#2E86AB", "WALDO"),
        ],
    ):
        ax.plot(thetas, p, color=color, lw=2, label=r"$p(\theta)$")
        ax.axhline(alpha, ls="--", color="0.5", lw=1, label=rf"$\alpha={alpha}$")
        ax.axvspan(
            ci[0], ci[1], color=color, alpha=0.12, label=f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
        )
        ax.axvline(D, color="0.2", lw=1, ls=":", label=f"D={D}")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$p(\theta)$")
        ax.set_title(f"identity x {label}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("IdentityTilting: passthrough to bare statistic", fontsize=11)
    fig.tight_layout()

    out = out or Path("output/illustrations/identity_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="fast mode used by CI")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(smoke=args.smoke, out=args.out)
    print(f"wrote {path}")
