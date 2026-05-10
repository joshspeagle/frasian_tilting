"""Illustration: m-geodesic (mixture) tilted reference at a sweep of eta values.

Visualises how the linear-density mixture tilted reference interpolates
between WALDO posterior (eta=0) and likelihood-as-Gaussian (eta=1).
Bimodality emerges in the conflict band where |mu_n - D| > 2 * min(sigma_n, sigma)
(Behboodian 1970), the qualitative signature the smoothness diagnostic
quantifies for the mixture cell.
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
    D = 2.0  # mild conflict; |Delta| = (1-w)|mu0-D|/sigma = 0.5*2 = 1.0

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)
    posterior = model.posterior(np.asarray([D]), prior)

    scheme = MixtureTilting()
    etas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_grid = 401 if smoke else 801
    thetas = np.linspace(-2.5, 5.5, n_grid)

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for i, eta in enumerate(etas):
        tilted = scheme.tilt(posterior, prior, likelihood, float(eta))
        color = cmap(i / max(1, len(etas) - 1))
        label = (
            r"$\eta=0$ (posterior / WALDO ref)"
            if eta == 0.0
            else r"$\eta=1$ (likelihood / Wald ref)"
            if eta == 1.0
            else rf"$\eta={eta:.2f}$"
        )
        ax.plot(thetas, np.asarray(tilted.pdf(thetas)), color=color, lw=1.6, label=label)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.set_title(
        rf"Mixture (m-geodesic) tilting: $D={D}$, $\sigma_0={sigma0}$ "
        rf"(linear interpolation in density space)"
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
