"""Illustration: η*(|Δ|) curve for (power_law, waldo) on a fine grid.

Visualises the kink at low |Δ| where power-law tilting hits the
admissible-boundary clamp — the central pathology that motivates the
framework's search for smoother tilting families.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.config import GridSpec
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.base import TiltingContext
from frasian.tilting.eta_selectors import NumericalEtaSelector
from frasian.tilting.power_law import PowerLawTilting


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma, mu0, w, alpha = 1.0, 0.0, 0.5, 0.05
    grid = GridSpec("abs_delta", 0.0, 5.0, 21 if smoke else 101)
    delta = grid.to_array()

    scheme = PowerLawTilting()
    statistic = WaldoStatistic()
    selector = NumericalEtaSelector(sigma=sigma, mu0=mu0)

    eta_star = np.empty_like(delta)
    for i, d in enumerate(delta):
        ctx = TiltingContext(w=w, abs_delta=float(d), alpha=alpha)
        eta_star[i] = selector.select(ctx, scheme, statistic=statistic)

    eta_low_bound = -w / (1.0 - w)

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(delta, eta_star, color="#2E86AB", lw=2,
            label=r"$\eta^\ast(|\Delta|)$, power-law")
    ax.axhline(0.0, ls=":", color="0.6", lw=1, label=r"WALDO ($\eta=0$)")
    ax.axhline(1.0, ls=":", color="#DC3545", lw=1, label=r"Wald ($\eta=1$)")
    ax.axhline(eta_low_bound, ls="--", color="0.4", lw=1,
               label=fr"clamp: $-w/(1-w)={eta_low_bound:.2f}$")
    ax.set_xlabel(r"$|\Delta|$")
    ax.set_ylabel(r"$\eta^\ast$")
    ax.set_title(
        fr"Power-law optimal $\eta^\ast(|\Delta|)$ at $w={w}$, "
        fr"$\alpha={alpha}$ (kink near $|\Delta|\approx 0.3$)"
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()

    out = out or Path("output/illustrations/smoothness_demo.png")
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
