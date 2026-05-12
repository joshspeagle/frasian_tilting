"""Illustration: SCOREO statistic on the conjugate-Normal sandbox.

Demonstrates the **Bayesian trinity collapse**:
    Scoreo == WALDO == LRTO  on NN+Normal

The figure overlays all three p-value curves; they sit on top of
each other to floating-point precision (Bayesian counterpart of
the frequentist Score = Wald = LRT collapse). Bottom panel
plots the flat-prior limit convergence.

`python -m frasian.experiments.illustrations.scoreo_demo --smoke`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel, posterior_params
from frasian.statistics.lrto import LRTOStatistic
from frasian.statistics.score import ScoreStatistic
from frasian.statistics.scoreo import ScoreoStatistic
from frasian.statistics.waldo import WaldoStatistic


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    sigma0 = 1.5
    mu0 = 0.0
    D = 2.0
    alpha = 0.05

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    data = np.asarray([D])

    n_grid = 161 if smoke else 601
    thetas = np.linspace(-3, 6, n_grid)
    p_scoreo = np.asarray(
        ScoreoStatistic().pvalue(thetas, data, model, prior), dtype=np.float64
    )
    p_waldo = np.asarray(
        WaldoStatistic().pvalue(thetas, data, model, prior), dtype=np.float64
    )
    p_lrto = np.asarray(
        LRTOStatistic().pvalue(thetas, data, model, prior), dtype=np.float64
    )
    mu_n, _, w = posterior_params(D, mu0, sigma, sigma0)

    # Flat-prior limit sweep
    theta_q = -0.5
    sigma0_grid = np.logspace(-0.5, 4, 19 if smoke else 41)
    diffs = []
    for s0 in sigma0_grid:
        prior_s0 = NormalDistribution(loc=mu0, scale=float(s0))
        p_scoreo_s0 = float(ScoreoStatistic().pvalue(theta_q, data, model, prior_s0))
        p_score_lim = float(ScoreStatistic().pvalue(theta_q, data, model))
        diffs.append(abs(p_scoreo_s0 - p_score_lim))
    diffs = np.asarray(diffs)

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.42)

    # Top: Bayesian trinity overlay.
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(thetas, p_waldo, color="#999999", lw=4, label="WALDO (closed form)")
    ax_top.plot(thetas, p_lrto, color="#2E86AB", lw=2, label="LRTO (closed form)")
    ax_top.plot(thetas, p_scoreo, color="#DC3545", lw=1, ls="--",
                label="SCOREO (closed form)")
    ax_top.axhline(alpha, ls=":", color="0.4", lw=1, label=rf"$\alpha={alpha}$")
    ax_top.axvline(float(mu_n), color="#2E86AB", ls=":", lw=1.2,
                   label=rf"$\mu_n={float(mu_n):.2f}$ (MAP)")
    ax_top.axvline(D, color="#28A745", ls=":", lw=1.2, label=rf"$D={D}$ (MLE)")
    ax_top.set_xlabel(r"$\theta$")
    ax_top.set_ylabel(r"$p(\theta)$")
    max_diff = float(max(np.max(np.abs(p_scoreo - p_waldo)),
                         np.max(np.abs(p_scoreo - p_lrto))))
    ax_top.set_title(
        r"Bayesian trinity: Scoreo $\equiv$ WALDO $\equiv$ LRTO on NN+Normal  "
        rf"($\sigma={sigma}$, $\sigma_0={sigma0}$, $w={w:.2f}$; "
        rf"max $|\Delta p|={max_diff:.1e}$)",
        fontsize=10,
    )
    ax_top.legend(loc="upper right", frameon=False, fontsize=8)
    ax_top.set_ylim(-0.02, 1.02)

    # Bottom: flat-prior limit convergence.
    ax_bot = fig.add_subplot(gs[1])
    ax_bot.loglog(sigma0_grid, np.maximum(diffs, 1e-16), "o-", color="#DC3545",
                  lw=1.5, ms=4,
                  label=r"$|p_{\rm Scoreo}(\sigma_0) - p_{\rm Score}|$")
    ref = diffs[len(diffs) // 2] * (sigma0_grid[len(diffs) // 2] ** 2) / sigma0_grid**2
    ax_bot.loglog(sigma0_grid, ref, "--", color="0.4", lw=1,
                  label=r"$\propto 1/\sigma_0^2$ reference")
    ax_bot.set_xlabel(r"$\sigma_0$ (prior scale)")
    ax_bot.set_ylabel(r"$|p_{\rm Scoreo} - p_{\rm Score}|$")
    ax_bot.set_title(
        rf"Flat-prior limit: SCOREO $\to$ Score at $\theta_0={theta_q}$",
        fontsize=10,
    )
    ax_bot.legend(loc="lower left", frameon=False, fontsize=8)
    ax_bot.grid(True, which="both", alpha=0.3)

    out = out or Path("output/illustrations/scoreo_demo.png")
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
