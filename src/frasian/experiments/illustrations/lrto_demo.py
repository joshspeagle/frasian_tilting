"""Illustration: LRTO statistic on the conjugate-Normal sandbox.

Two demonstrations on one figure:
  1. **NN+Normal collapse** — lrto.pvalue == waldo.pvalue exactly
     (Derivation Step 3 of `docs/methods/lrto.md`). Top panel overlays
     both curves and shows the residual.
  2. **Flat-prior limit** — as sigma_0 -> infty, lrto -> lrt (= wald
     on NN). Bottom panel sweeps sigma_0 and shows the convergence.

`python -m frasian.experiments.illustrations.lrto_demo --smoke` runs
in fast mode and emits `output/illustrations/lrto_demo.png`.
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
from frasian.statistics.lrt import LRTStatistic
from frasian.statistics.lrto import LRTOStatistic
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
    p_lrto = np.asarray(LRTOStatistic().pvalue(thetas, data, model, prior),
                        dtype=np.float64)
    p_waldo = np.asarray(WaldoStatistic().pvalue(thetas, data, model, prior),
                         dtype=np.float64)
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    # Flat-prior limit sweep: at theta_0 fixed, sweep sigma_0.
    theta_query = -0.5
    sigma0_grid = np.logspace(-0.5, 4, 19 if smoke else 41)
    diffs = []
    for s0 in sigma0_grid:
        prior_s0 = NormalDistribution(loc=mu0, scale=float(s0))
        p_lrto_s0 = float(LRTOStatistic().pvalue(theta_query, data, model, prior_s0))
        p_lrt_lim = float(LRTStatistic().pvalue(theta_query, data, model))
        diffs.append(abs(p_lrto_s0 - p_lrt_lim))
    diffs = np.asarray(diffs)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.42)

    # Top: NN+Normal collapse (lrto overlays waldo).
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(thetas, p_waldo, color="#999999", lw=3, label="WALDO (NN closed form)")
    ax_top.plot(thetas, p_lrto, color="#2E86AB", lw=1.5, ls="-",
                label="LRTO (NN closed form)")
    ax_top.axhline(alpha, ls=":", color="0.4", lw=1, label=rf"$\alpha={alpha}$")
    ax_top.axvline(float(mu_n), color="#2E86AB", ls=":", lw=1.2,
                   label=rf"$\theta_{{MAP}} = \mu_n = {float(mu_n):.2f}$")
    ax_top.axvline(D, color="#DC3545", ls=":", lw=1.2, label=rf"$D={D}$ (MLE)")
    ax_top.set_xlabel(r"$\theta$")
    ax_top.set_ylabel(r"$p(\theta)$")
    max_diff = float(np.max(np.abs(p_lrto - p_waldo)))
    ax_top.set_title(
        rf"LRTO $\equiv$ WALDO on NN+Normal "
        rf"($\sigma={sigma}$, $\sigma_0={sigma0}$, $w={w:.2f}$; "
        rf"max $|\Delta p|={max_diff:.1e}$)",
        fontsize=10,
    )
    ax_top.legend(loc="upper right", frameon=False, fontsize=8)
    ax_top.set_ylim(-0.02, 1.02)

    # Bottom: flat-prior limit convergence.
    ax_bot = fig.add_subplot(gs[1])
    # Reference: O(1/sigma_0^2) — Derivation Step 6 says ~3.7/sigma_0^2
    # for the brief's (sigma=1, mu0=0.5, D=0.7, theta_0=-0.2). The
    # absolute constant depends on (D, theta_0, mu0, sigma); plot the
    # measured curve plus a 1/sigma_0^2 reference slope for shape.
    ax_bot.loglog(sigma0_grid, np.maximum(diffs, 1e-16),
                  "o-", color="#2E86AB", lw=1.5, ms=4,
                  label=r"$|p_{\rm LRTO}(\sigma_0) - p_{\rm LRT}|$")
    ref = diffs[len(diffs) // 2] * (sigma0_grid[len(diffs) // 2] ** 2) / sigma0_grid**2
    ax_bot.loglog(sigma0_grid, ref, "--", color="0.4", lw=1,
                  label=r"$\propto 1/\sigma_0^2$ reference")
    ax_bot.set_xlabel(r"$\sigma_0$ (prior scale)")
    ax_bot.set_ylabel(r"$|p_{\rm LRTO} - p_{\rm LRT}|$")
    ax_bot.set_title(
        rf"Flat-prior limit: LRTO $\to$ LRT at $\theta_0={theta_query}$",
        fontsize=10,
    )
    ax_bot.legend(loc="lower left", frameon=False, fontsize=8)
    ax_bot.grid(True, which="both", alpha=0.3)

    out = out or Path("output/illustrations/lrto_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="fast mode used by CI")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(smoke=args.smoke, out=args.out)
    print(f"wrote {path}")
