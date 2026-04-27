"""Illustration: dynamic-η CI on the canonical Normal-Normal sandbox.

Shows the dynamic p-value `p_dyn(theta) = tilted_pvalue(theta, D, ...,
eta*(|Delta(theta)|))` next to the static WALDO and Wald p-values for
the same observation. The dynamic CI is the level-α set; visible as
the shaded region.
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
from frasian.tilting.eta_selectors import NumericalEtaSelector
from frasian.tilting.power_law import PowerLawTilting


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma, mu0, sigma0 = 1.0, 0.0, 1.0
    D = 1.5
    alpha = 0.05

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    scheme = PowerLawTilting()
    selector = NumericalEtaSelector(sigma=sigma, mu0=mu0)

    n = 81 if smoke else 201
    thetas = np.linspace(D - 5, D + 5, n)

    # Static p-values.
    wald_p = WaldStatistic().pvalue(thetas, np.asarray([D]), model)
    waldo_p = WaldoStatistic().pvalue(thetas, np.asarray([D]), model, prior)

    # Dynamic p-value: precompute η* on a coarse |Δ| grid, interpolate.
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
    abs_delta = np.abs((1.0 - w) * (mu0 - thetas) / sigma)
    coarse_grid = np.linspace(0.0, abs_delta.max() + 1e-6, 15)
    coarse_eta = selector.select_grid(
        coarse_grid, scheme,
        statistic=type("S", (), {"name": "waldo"})(), w=w, alpha=alpha,
    )
    eta_at_theta = np.interp(abs_delta, coarse_grid, coarse_eta)
    dyn_p = scheme.dynamic_tilted_pvalue(
        thetas, D, model, prior, "waldo", eta_at_theta,
    )

    # Dynamic CI (single call to the inversion routine).
    regions, total_width, _ = scheme.dynamic_tilted_confidence_interval(
        alpha, D, model, prior, "waldo", selector,
        n_grid=n, coarse_n=15,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(thetas, wald_p, color="#DC3545", lw=1.4, ls="--", label="Wald")
    ax.plot(thetas, waldo_p, color="#2E86AB", lw=1.4, ls=":",
             label=r"WALDO ($\eta=0$)")
    ax.plot(thetas, dyn_p, color="#28A745", lw=2.2,
             label=r"dynamic $\eta^\ast(|\Delta(\theta)|)$")
    ax.axhline(alpha, color="0.5", lw=1, ls="-",
                label=fr"$\alpha={alpha}$")
    for lo, hi in regions:
        ax.axvspan(lo, hi, color="#28A745", alpha=0.12)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p(\theta)$")
    ax.set_title(
        fr"Dynamic vs static p-values at $D={D}$, $\sigma_0={sigma0}$ "
        fr"(dyn-CI total width = {total_width:.3f})"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()

    out = out or Path("output/illustrations/dynamic_ci_demo.png")
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
