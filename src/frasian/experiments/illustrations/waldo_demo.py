"""Illustration: WALDO p-value, posterior mean, and CD on the sandbox.

Demonstrates the closed-form `Phi(b - a) + Phi(-a - b)` p-value and shows
where its mode (theta = mu_n) sits relative to the Wald MLE.
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
from frasian.statistics.waldo import WaldoStatistic


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    sigma0 = 1.5
    mu0 = 0.0
    D = 2.0
    alpha = 0.05

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    stat = WaldoStatistic()

    n_grid = 161 if smoke else 601
    thetas = np.linspace(-3, 6, n_grid)
    ps = stat.pvalue(thetas, np.asarray([D]), model, prior)

    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(thetas, ps, color="#2E86AB", lw=2, label=r"WALDO $p(\theta)$")
    ax.axhline(alpha, ls="--", color="0.5", lw=1, label=rf"$\alpha={alpha}$")
    ax.axvline(float(mu_n), color="#2E86AB", ls=":", lw=1.5, label=rf"$\mu_n={float(mu_n):.2f}$")
    ax.axvline(D, color="#DC3545", ls=":", lw=1.5, label=rf"$D={D}$ (MLE)")
    ax.axvline(mu0, color="#28A745", ls=":", lw=1.5, label=rf"$\mu_0={mu0}$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p(\theta)$")
    ax.set_title(rf"WALDO: $D={D}$, $\sigma={sigma}$, $\sigma_0={sigma0}$, $w={w:.2f}$")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()

    out = out or Path("output/illustrations/waldo_demo.png")
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
