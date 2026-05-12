"""Illustration: LRT statistic on the conjugate-Normal sandbox.

Shows the closed-form NN path (which is identical to Wald by
Derivation Step 3 of `docs/methods/lrt.md`) overlaid with the
generic likelihood-based path (`tau = -2[loglik(theta_hat) -
loglik(theta)]`, `chi^2_1` calibration). The two coincide on NN to
numerical tolerance — that equivalence is the point of the figure.

`python -m frasian.experiments.illustrations.lrt_demo --smoke` runs
in fast mode and emits `output/illustrations/lrt_demo.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.lrt import LRTStatistic
from frasian.statistics.wald import WaldStatistic


def main(smoke: bool = False, out: Path | None = None) -> Path:
    sigma = 1.0
    D = 1.5
    alpha = 0.05
    model = NormalNormalModel(sigma=sigma)
    data = np.asarray([D])

    n_grid = 121 if smoke else 401
    thetas = np.linspace(-3, 6, n_grid)

    # Both LRT paths (closed-form and generic) plus Wald for the overlay.
    lrt = LRTStatistic()
    lrt_g = LRTStatistic(force_generic=True)
    wald = WaldStatistic()
    p_lrt_cf = np.asarray(lrt.pvalue(thetas, data, model), dtype=np.float64)
    p_lrt_g = np.asarray(
        [float(lrt_g.pvalue(t, data, model)) for t in thetas], dtype=np.float64
    )
    p_wald = np.asarray(wald.pvalue(thetas, data, model), dtype=np.float64)

    ci_lrt = lrt.confidence_interval(alpha, data, model)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 5.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top: p-value curves.
    ax_top.plot(thetas, p_wald, color="#999999", lw=3, label="Wald (NN closed form)")
    ax_top.plot(thetas, p_lrt_cf, color="#DC3545", lw=1.5, ls="-", label="LRT (NN closed form)")
    ax_top.plot(
        thetas, p_lrt_g, color="#1F77B4", lw=1.0, ls="--", label="LRT (generic path)"
    )
    ax_top.axhline(alpha, ls=":", color="0.4", lw=1, label=rf"$\alpha={alpha}$")
    ax_top.axvspan(ci_lrt[0], ci_lrt[1], color="#DC3545", alpha=0.10, label="95% CI (LRT)")
    ax_top.axvline(D, color="0.2", lw=1, ls=":", label="MLE $= D$")
    ax_top.set_ylabel(r"$p(\theta)$")
    ax_top.set_title(rf"LRT on Normal-Normal: $D={D}$, $\sigma={sigma}$")
    ax_top.legend(loc="upper right", frameon=False, fontsize=8)
    ax_top.set_ylim(-0.02, 1.02)

    # Bottom: residual between closed-form and generic LRT paths.
    residual = p_lrt_cf - p_lrt_g
    ax_bot.plot(thetas, residual, color="#1F77B4", lw=1)
    ax_bot.axhline(0.0, color="0.4", lw=0.5, ls=":")
    ax_bot.set_xlabel(r"$\theta$")
    ax_bot.set_ylabel("CF $-$ generic")
    max_abs = float(np.max(np.abs(residual)))
    ax_bot.set_title(
        rf"closed-form $-$ generic residual (max $|\Delta p| = {max_abs:.2e}$)",
        fontsize=9,
    )

    fig.tight_layout()
    out = out or Path("output/illustrations/lrt_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="fast mode used by CI")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = main(smoke=args.smoke, out=args.out)
    print(f"wrote {path}")
