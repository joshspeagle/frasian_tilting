"""Illustration: Score statistic on the conjugate-Normal sandbox.

Demonstrates the trinity collapse:
    Score == Wald == LRT  on NN+n=1

The figure overlays all three p-value curves; they sit on top of
each other to floating-point precision. The residual panel
confirms `max|p_score - p_wald| ~ machine epsilon`.

`python -m frasian.experiments.illustrations.score_demo --smoke`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.lrt import LRTStatistic
from frasian.statistics.score import ScoreStatistic
from frasian.statistics.wald import WaldStatistic


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    D = 2.0
    alpha = 0.05
    model = NormalNormalModel(sigma=sigma)
    data = np.asarray([D])

    n_grid = 161 if smoke else 601
    thetas = np.linspace(-3, 6, n_grid)
    p_score = np.asarray(ScoreStatistic().pvalue(thetas, data, model),
                         dtype=np.float64)
    p_wald = np.asarray(WaldStatistic().pvalue(thetas, data, model),
                        dtype=np.float64)
    p_lrt = np.asarray(LRTStatistic().pvalue(thetas, data, model),
                       dtype=np.float64)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.30)

    # Top: three curves overlaid.
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(thetas, p_wald, color="#999999", lw=4,
                label="Wald (closed form)")
    ax_top.plot(thetas, p_lrt, color="#2E86AB", lw=2,
                label="LRT (closed form)")
    ax_top.plot(thetas, p_score, color="#DC3545", lw=1, ls="--",
                label="Score (closed form)")
    ax_top.axhline(alpha, ls=":", color="0.4", lw=1, label=rf"$\alpha={alpha}$")
    ax_top.axvline(D, color="#28A745", ls=":", lw=1.5,
                   label=rf"$D={D}$ (MLE)")
    ax_top.set_xlabel(r"$\theta$")
    ax_top.set_ylabel(r"$p(\theta)$")
    max_diff = float(max(np.max(np.abs(p_score - p_wald)),
                         np.max(np.abs(p_score - p_lrt))))
    ax_top.set_title(
        r"Score $\equiv$ Wald $\equiv$ LRT on NN+$n=1$  "
        rf"($\sigma={sigma}$; max $|\Delta p|={max_diff:.1e}$)",
        fontsize=10,
    )
    ax_top.legend(loc="upper right", frameon=False, fontsize=8)
    ax_top.set_ylim(-0.02, 1.02)

    # Bottom: residuals on log scale.
    ax_bot = fig.add_subplot(gs[1])
    diff_sw = np.maximum(np.abs(p_score - p_wald), 1e-18)
    diff_sl = np.maximum(np.abs(p_score - p_lrt), 1e-18)
    ax_bot.semilogy(thetas, diff_sw, color="#999999", lw=1,
                    label=r"$|p_{\rm Score} - p_{\rm Wald}|$")
    ax_bot.semilogy(thetas, diff_sl, color="#DC3545", lw=1, ls="--",
                    label=r"$|p_{\rm Score} - p_{\rm LRT}|$")
    ax_bot.axhline(np.finfo(np.float64).eps, color="0.4", ls=":", lw=1,
                   label=r"$\epsilon_{\rm machine}$")
    ax_bot.set_xlabel(r"$\theta$")
    ax_bot.set_ylabel(r"residual")
    ax_bot.legend(loc="upper right", frameon=False, fontsize=8)
    ax_bot.set_ylim(1e-18, 1e-10)

    out = out or Path("output/illustrations/score_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(smoke=args.smoke, out=args.out)
    print(f"wrote {path}")
