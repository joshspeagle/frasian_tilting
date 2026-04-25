"""Illustration: Wald p-value and CI on the conjugate-Normal sandbox.

`python -m frasian.experiments.illustrations.wald_demo --smoke` runs in
fast mode (small grid, no figure window), producing a PNG at
`output/illustrations/wald_demo.png` for the CI completeness check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    D = 1.5
    alpha = 0.05
    model = NormalNormalModel(sigma=sigma)
    stat = WaldStatistic()

    n_grid = 121 if smoke else 401
    thetas = np.linspace(-3, 6, n_grid)
    ps = stat.pvalue(thetas, np.asarray([D]), model)

    z = stats.norm.ppf(1 - alpha / 2)
    lo, hi = D - z * sigma, D + z * sigma

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(thetas, ps, color="#DC3545", lw=2, label=r"Wald $p(\theta)$")
    ax.axhline(alpha, ls="--", color="0.5", lw=1, label=fr"$\alpha={alpha}$")
    ax.axvspan(lo, hi, color="#DC3545", alpha=0.12, label="95% CI")
    ax.axvline(D, color="0.2", lw=1, ls=":", label="MLE $= D$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p(\theta)$")
    ax.set_title(rf"Wald: $D={D}$, $\sigma={sigma}$")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()

    out = out or Path("output/illustrations/wald_demo.png")
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
