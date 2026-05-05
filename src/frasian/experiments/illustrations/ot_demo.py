"""Illustration: OT (Wasserstein-2) tilted posteriors.

Two panels:

  Panel A — Gaussian fast path. W2 geodesic between the WALDO posterior
            and the likelihood-induced Gaussian N(D, sigma^2). The
            interpolation is linear in (mu, sigma), so the family slides
            smoothly from posterior (eta=0) to Wald (eta=1) without a
            clamp — contrast with `power_law_demo.py` where the family
            crowds at low |Delta|.

  Panel B — General 1D quantile-mixture. W2 geodesic between Beta(2, 5)
            and Beta(5, 2) computed via QuantileMixturePath. Demonstrates
            that the OT machinery is endpoint-agnostic: it runs on any
            two distributions exposing `quantile`, not just Gaussians.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.models.distributions import BetaDistribution, GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.ot import OTTilting
from frasian.tilting.quantile_mixture import QuantileMixturePath


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    sigma0 = 1.0
    mu0 = 0.0
    D = 2.0

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)
    posterior = model.posterior(np.asarray([D]), prior)

    scheme = OTTilting()
    etas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_grid = 401 if smoke else 801
    thetas = np.linspace(-2.5, 5.5, n_grid)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 3.7))

    # Panel A: Gaussian fast path on the Normal-Normal sandbox.
    cmap = plt.get_cmap("viridis")
    for i, eta in enumerate(etas):
        tilted = scheme.tilt(posterior, prior, likelihood, float(eta))
        color = cmap(i / max(1, len(etas) - 1))
        label = (
            r"$\eta=0$ (WALDO)"
            if eta == 0.0
            else r"$\eta=1$ (Wald)" if eta == 1.0 else rf"$\eta={eta:.1f}$"
        )
        ax_a.plot(thetas, tilted.pdf(thetas), color=color, lw=1.6, label=label)
    ax_a.set_xlabel(r"$\theta$")
    ax_a.set_ylabel("density")
    ax_a.set_title(
        rf"OT (Gaussian fast path): linear in $(\mu, \sigma)$, no clamp"
        f"\n$D={D}$, $\\sigma_0={sigma0}$"
    )
    ax_a.legend(loc="upper right", frameon=False, fontsize=8)

    # Panel B: General 1D quantile-mixture between two Betas.
    p = BetaDistribution(alpha=2.0, beta=5.0)
    q = BetaDistribution(alpha=5.0, beta=2.0)
    for i, t in enumerate(etas):
        path = QuantileMixturePath(p=p, q=q, t=float(t))
        color = cmap(i / max(1, len(etas) - 1))
        # Plot the quantile function rather than the pdf for visual
        # clarity on Beta endpoints — pdf computation requires numerical
        # CDF inversion at every theta and is ~1000x slower than quantile.
        u = np.linspace(0.001, 0.999, 201)
        ax_b.plot(u, path.quantile(u), color=color, lw=1.6, label=rf"$t={t:.1f}$")
    ax_b.set_xlabel(r"$u$ (uniform parameter)")
    ax_b.set_ylabel(r"$F_t^{-1}(u)$")
    ax_b.set_title(
        r"OT (general 1D): quantile-mixture $F_t^{-1}(u) = (1-t)F_p^{-1}(u) + tF_q^{-1}(u)$"
        "\nendpoints: Beta(2,5) and Beta(5,2)"
    )
    ax_b.legend(loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()

    out = out or Path("output/illustrations/ot_demo.png")
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
