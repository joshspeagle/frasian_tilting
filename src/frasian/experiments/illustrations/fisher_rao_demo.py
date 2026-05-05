"""Illustration: Fisher-Rao (Levi-Civita) geodesic tilted posteriors.

Two panels:

  Panel A — Densities along the FR geodesic on the Normal-Normal
            sandbox at a sweep of eta. The geodesic stays in the
            Gaussian family (the manifold *is* the Gaussian family),
            so the family slides smoothly from posterior (eta=0) to
            `N(D, sigma^2)` (eta=1). Curvature-aware path through
            `(mu, sigma)`-space — distinct from `ot`'s straight line.

  Panel B — Path through `(mu, sigma)`-space at finely spaced eta,
            overlaid against `ot`'s straight-line W2 path. The
            visible difference between the two curves is the
            geometry-vs-mass-displacement contrast (Takatsu 2011).
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
from frasian.tilting.ot import OTTilting


def main(smoke: bool = False, out: Path | None = None) -> Path:
    # Pick sigma0 such that posterior sigma_n differs noticeably from
    # likelihood sigma — that's the regime where FR and OT split.
    sigma = 1.0
    sigma0 = 0.5  # informative prior -> small sigma_n, contrast with sigma.
    mu0 = 0.0
    D = 2.0

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)
    posterior = model.posterior(np.asarray([D]), prior)

    fr = FisherRaoTilting()
    ot = OTTilting()
    etas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    n_grid = 401 if smoke else 801
    thetas = np.linspace(-2.0, 5.0, n_grid)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 3.7))

    # Panel A — density at sweep of eta along the FR geodesic.
    cmap = plt.get_cmap("viridis")
    for i, eta in enumerate(etas):
        tilted = fr.tilt(posterior, prior, likelihood, float(eta))
        color = cmap(i / max(1, len(etas) - 1))
        label = (
            r"$\eta=0$ (posterior)"
            if eta == 0.0
            else r"$\eta=1$ (likelihood)"
            if eta == 1.0
            else rf"$\eta={eta:.2f}$"
        )
        ax_a.plot(thetas, tilted.pdf(thetas), color=color, lw=1.6, label=label)
    ax_a.set_xlabel(r"$\theta$")
    ax_a.set_ylabel("density")
    ax_a.set_title(
        rf"Fisher-Rao geodesic densities ($D={D}$, $\sigma_0={sigma0}$)"
    )
    ax_a.legend(loc="upper right", frameon=False, fontsize=8)

    # Panel B — paths through (mu, sigma)-space. Both schemes return
    # NormalDistribution on this Gaussian-endpoint sandbox, so we read
    # `(mu, sigma)` via the Distribution protocol's `mean()` / `std()`
    # equivalents (mean for mu, sqrt(var) for sigma) — works whether
    # the concrete return type is NormalDistribution or any other
    # `Distribution`-conformant wrapper.
    eta_fine = np.linspace(0.0, 1.0, 51)

    def _mu_sigma(scheme, t: float) -> tuple[float, float]:
        d = scheme.tilt(posterior, prior, likelihood, t)
        return float(d.mean()), float(np.sqrt(d.var()))

    fr_path = np.asarray([_mu_sigma(fr, float(t)) for t in eta_fine])
    ot_path = np.asarray([_mu_sigma(ot, float(t)) for t in eta_fine])
    ax_b.plot(fr_path[:, 0], fr_path[:, 1], color="C0", lw=2.0, label="Fisher-Rao")
    ax_b.plot(ot_path[:, 0], ot_path[:, 1], color="C3", lw=2.0, ls="--", label="OT (W2)")
    # Mark endpoints.
    ax_b.scatter(
        [posterior.loc, likelihood.D],
        [posterior.scale, likelihood.sigma],
        color="black",
        s=40,
        zorder=5,
    )
    ax_b.annotate(r"posterior ($\eta=0$)",
                  (posterior.loc, posterior.scale),
                  textcoords="offset points", xytext=(8, -14), fontsize=9)
    ax_b.annotate(r"$N(D,\sigma^2)$ ($\eta=1$)",
                  (likelihood.D, likelihood.sigma),
                  textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax_b.set_xlabel(r"$\mu$")
    ax_b.set_ylabel(r"$\sigma$")
    ax_b.set_title(
        r"Path through $(\mu, \sigma)$ — FR is curvature-aware, OT is straight"
    )
    ax_b.legend(loc="best", frameon=False, fontsize=9)

    fig.tight_layout()

    out = out or Path("output/illustrations/fisher_rao_demo.png")
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
