"""Illustration: power-law tilted posteriors at a sweep of eta values.

Visualises how the tilted Normal posterior interpolates between WALDO
(eta=0) and Wald (eta=1), and explores the oversharpened regime
(eta < 0). The visible separation between curves at low `|Delta|` is the
qualitative signature the Step-5 smoothness diagnostic will quantify.
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
from frasian.tilting.power_law import PowerLawTilting


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    sigma0 = 1.0
    mu0 = 0.0
    D = 2.0

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)
    posterior = model.posterior(np.asarray([D]), prior)

    scheme = PowerLawTilting()
    etas = np.array([-0.4, -0.1, 0.0, 0.3, 0.6, 0.9])
    n_grid = 401 if smoke else 801
    thetas = np.linspace(-2.5, 5.5, n_grid)

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for i, eta in enumerate(etas):
        tilted = scheme.tilt(posterior, prior, likelihood, float(eta))
        color = cmap(i / max(1, len(etas) - 1))
        label = (
            r"$\eta=0$ (WALDO)" if eta == 0.0 else
            r"$\eta=1$ (Wald)" if eta == 1.0 else
            fr"$\eta={eta:+.2f}$"
        )
        ax.plot(thetas, tilted.pdf(thetas), color=color, lw=1.6, label=label)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.set_title(
        fr"Power-law tilting: $D={D}$, $\sigma_0={sigma0}$ "
        fr"(crowding at low $|\Delta|$ is the smoothness concern)"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()

    out = out or Path("output/illustrations/power_law_demo.png")
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
