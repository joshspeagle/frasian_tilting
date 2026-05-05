"""Illustration: mixture (m-geodesic) tilted posteriors at a sweep of eta.

Visualises the linear-density interpolation between the WALDO posterior
(eta=0) and the likelihood-induced Gaussian (eta=1). Unlike the
e-geodesic (`power_law_demo`) and the W2 geodesic (`ot_demo`), the
m-geodesic leaves the Normal family — at intermediate eta the density
is a 2-component Gaussian mixture and is bimodal beyond the Behboodian
threshold `|mu_n - D| > 2 min(sigma_n, sigma)`.

We pick `D=4, sigma0=1, sigma=1` (so `|Delta| = 2`, in the bimodal
band per Step 2 of the derivation) so the bimodality is visually
apparent at eta=0.5.
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
from frasian.tilting.mixture import MixtureTilting


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    sigma0 = 1.0
    mu0 = 0.0
    D = 4.0  # |Delta| = (1-w)|mu0-D|/sigma = 0.5*4 = 2 (bimodal band).

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)
    posterior = model.posterior(np.asarray([D]), prior)

    scheme = MixtureTilting()
    etas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    n_grid = 401 if smoke else 801
    thetas = np.linspace(-3.0, 7.0, n_grid)

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for i, eta in enumerate(etas):
        tilted = scheme.tilt(posterior, prior, likelihood, float(eta))
        color = cmap(i / max(1, len(etas) - 1))
        label = (
            r"$\eta=0$ (posterior)"
            if eta == 0.0
            else r"$\eta=1$ (likelihood)"
            if eta == 1.0
            else rf"$\eta={eta:.2f}$"
        )
        ax.plot(thetas, tilted.pdf(thetas), color=color, lw=1.6, label=label)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.set_title(
        rf"Mixture (m-geodesic): $D={D}$, $\sigma_0={sigma0}$ "
        rf"($|\Delta|=2$, bimodal at $\eta\!\in\!(0,1)$)"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()

    out = out or Path("output/illustrations/mixture_demo.png")
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
