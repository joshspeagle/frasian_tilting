"""Fisher-Rao tilting illustration — eta-curve on the Gaussian half-plane.

Plots the Fisher-Rao tilted density at several eta values for a fixed
Normal-Normal sandbox setup with prior-data conflict, demonstrating
the curvature-aware interpolation between the posterior (eta=0) and
the likelihood-induced Gaussian N(D, sigma^2) (eta=1) along the
constant-curvature semi-circular arc on the Gaussian half-plane.

Run:
    python -m frasian.experiments.illustrations.fisher_rao_demo
    python -m frasian.experiments.illustrations.fisher_rao_demo --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.fisher_rao import FisherRaoTilting


def main(smoke: bool = False, out: Path | None = None) -> Path:
    # Conflict configuration: informative prior at mu0=0, data pulled to D=1.5.
    sigma = 1.0
    sigma0 = 0.5
    mu0 = 0.0
    D = 1.5

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)
    posterior = model.posterior(np.asarray([D]), prior)

    scheme = FisherRaoTilting()
    etas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_grid = 81 if smoke else 401
    thetas = np.linspace(-2.0, 3.0, n_grid)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    cmap = plt.get_cmap("viridis")
    for i, eta in enumerate(etas):
        tilted = scheme.tilt(posterior, prior, likelihood, float(eta))
        color = cmap(i / max(1, len(etas) - 1))
        label = (
            r"$\eta=0$ (posterior)"
            if eta == 0.0
            else r"$\eta=1$ (Wald)" if eta == 1.0 else rf"$\eta={eta:.1f}$"
        )
        ax.plot(thetas, tilted.pdf(thetas), color=color, lw=1.6, label=label)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.set_title(
        r"Fisher-Rao geodesic: $\eta$-curve on the Gaussian half-plane"
        f"\n$D={D}$, $\\sigma_0={sigma0}$, $\\sigma={sigma}$"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out = out or Path("output/illustrations/fisher_rao_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80 if smoke else 150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(smoke=args.smoke, out=args.out)
    print(f"wrote {path}")
