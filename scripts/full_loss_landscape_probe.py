"""Full-loss landscape probe for the learned-eta puzzle.

Companion to ``loss_landscape_probe.py`` (per-theta marginal). This
script evaluates the FULL integrated-p, cd-variance, and static-width
losses as a function of CONSTANT eta (one value held across all theta),
at fixed (mu0, sigma_0, sigma, D). Three loss curves per scheme.

Mark on each:
- argmin (the best constant-eta solution).
- Trained network's "average" eta (mean of eta(theta) over the
  theta-grid). The trained output varies in theta; we compare a
  single-number summary.

Saves a 3 (loss) x 2 (scheme) grid of panels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian._registry_bootstrap import bootstrap

bootstrap()

import jax
import jax.numpy as jnp

from frasian.learned.eta_artifact import EtaArtifact
from frasian.learned.training.cd_jax import cd_density_jax
from frasian.learned.training.losses import (
    cd_variance_loss,
    integrated_pvalue_loss,
    static_width_loss,
)
from frasian.learned.training.pvalue_jax import (
    ot_tilted_pvalue_jax,
    power_law_tilted_pvalue_jax,
)


# Fixed configuration (demo slice from CLAUDE.md / v4 plot).
MU0 = 0.0
SIGMA0 = 1.0
SIGMA = 1.0
D_OBS = 0.0
W = SIGMA0 ** 2 / (SIGMA ** 2 + SIGMA0 ** 2)  # = 0.5
K_THETA = 5.0  # matches SigmaAnchoredUniformThetaDistribution
ALPHA = 0.05  # for static_width

# Bug fix 2026-05-10: previously [-1.0, 1.5]; aligned with
# diagnostics._compute_argmin_constant_eta and trained_vs_optimal_sweep.py
# so the per-slice argmin search range is consistent across probes.
ETA_GRID = np.linspace(-1.5, 1.5, 201)
THETA_GRID = np.linspace(MU0 - K_THETA * SIGMA0, MU0 + K_THETA * SIGMA0, 401)


def _pvalue_curve(scheme: str, eta_const: float) -> np.ndarray:
    """Return p(theta_grid; D, eta_const) for the chosen scheme."""
    fn = power_law_tilted_pvalue_jax if scheme == "power_law" else ot_tilted_pvalue_jax
    eta_arr = jnp.full(THETA_GRID.shape, float(eta_const))
    p = fn(
        theta=jnp.asarray(THETA_GRID),
        D=jnp.asarray(D_OBS),
        w=jnp.asarray(W),
        mu0=jnp.asarray(MU0),
        sigma=jnp.asarray(SIGMA),
        eta=eta_arr,
        statistic_name="waldo",
    )
    return np.asarray(p, dtype=np.float64)


def _loss_at_eta(scheme: str, loss_name: str, eta_const: float) -> float:
    p = _pvalue_curve(scheme, eta_const)
    p_2d = jnp.asarray(p[None, :])         # (B=1, N)
    grid_2d = jnp.asarray(THETA_GRID[None, :])
    if loss_name == "integrated_p":
        return float(integrated_pvalue_loss(p_2d, grid_2d))
    if loss_name == "cd_variance":
        return float(cd_variance_loss(p_2d, grid_2d))
    if loss_name == "static_width":
        return float(static_width_loss(p_2d, grid_2d, ALPHA))
    raise ValueError(loss_name)


def _trained_eta_summary(scheme: str, loss_name: str):
    """Load the trained net and compute (mean, min, max) of eta over THETA_GRID."""
    if scheme == "power_law":
        path = Path(
            f"artifacts/learned_eta_canonical_normal_normal_powerlaw_phaseC_{loss_name}_v4.eqx"
        )
        version = f"phaseC_powerlaw_{loss_name}_v4"
    elif scheme == "ot":
        path = Path(
            f"artifacts/learned_eta_canonical_normal_normal_ot_phaseC_{loss_name}_v4.eqx"
        )
        version = f"phaseC_ot_{loss_name}_v4"
    else:
        return None
    if not path.exists():
        return None
    art = EtaArtifact(artifact_path=path, name="learned", version=version)
    art.load()
    eta_curve = art.predict_eta(
        theta=THETA_GRID,
        prior_hp=np.asarray([MU0, SIGMA0]),
        lik_hp=np.asarray([SIGMA]),
    )
    return float(np.mean(eta_curve)), float(np.min(eta_curve)), float(np.max(eta_curve))


def main():
    schemes = ["power_law", "ot"]
    losses = ["integrated_p", "cd_variance", "static_width"]
    n_l = len(losses)
    n_s = len(schemes)

    fig, axes = plt.subplots(n_s, n_l, figsize=(4.0 * n_l, 3.4 * n_s),
                             squeeze=False)

    print(f"\n=== Full-loss probe at mu0={MU0}, sigma0={SIGMA0}, sigma={SIGMA}, D={D_OBS} ===")
    print(f"Theta-grid: [-5, +5] (matches K=5 sigma-anchored training)")
    print()

    for r, scheme in enumerate(schemes):
        for c, loss_name in enumerate(losses):
            ax = axes[r, c]
            # Sweep loss over eta_const.
            loss_vals = np.array([
                _loss_at_eta(scheme, loss_name, float(e)) for e in ETA_GRID
            ])
            valid = np.isfinite(loss_vals)
            if not valid.any():
                ax.set_title(f"{scheme} / {loss_name}\n(all-NaN)")
                continue
            ax.plot(ETA_GRID, loss_vals, color="black", lw=1.5)

            # Argmin over constant-eta.
            idx = int(np.argmin(np.where(valid, loss_vals, np.inf)))
            argmin_eta = float(ETA_GRID[idx])
            argmin_loss = float(loss_vals[idx])
            ax.plot(argmin_eta, argmin_loss, "kx", markersize=10,
                    label=f"argmin: eta={argmin_eta:+.2f}")

            # Trained net summary at this scheme/loss.
            summary = _trained_eta_summary(scheme, loss_name)
            if summary is not None:
                eta_mean, eta_min, eta_max = summary
                ax.axvline(eta_mean, color="tab:blue", lw=1.4,
                           label=f"trained mean: {eta_mean:+.3f}")
                ax.axvspan(eta_min, eta_max, color="tab:blue", alpha=0.12,
                           label=f"trained range: [{eta_min:+.3f}, {eta_max:+.3f}]")

            ax.axvline(0.0, color="gray", lw=0.6, ls=":")
            ax.axvline(1.0, color="red", lw=0.6, ls=":", alpha=0.5)
            ax.set_title(f"{scheme} / {loss_name}", fontsize=10)
            if c == 0:
                ax.set_ylabel("loss(eta_const)", fontsize=9)
            if r == n_s - 1:
                ax.set_xlabel(r"$\eta_{const}$", fontsize=9)
            ax.legend(loc="best", fontsize=7, frameon=False)

            # Numerical print.
            sum_str = "n/a"
            if summary is not None:
                sum_str = f"mean={summary[0]:+.3f} range=[{summary[1]:+.3f}, {summary[2]:+.3f}]"
            print(f"{scheme:<10} / {loss_name:<14}  argmin_eta={argmin_eta:+.3f}  "
                  f"loss_at_argmin={argmin_loss:.6g}  trained={sum_str}")

    fig.suptitle(
        "Full-loss landscape: loss(eta_const) over [-1, 1.5]\n"
        f"black: loss with constant eta; X: argmin; "
        f"blue line: trained-net mean eta; blue band: trained-net (min, max) over theta",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path("output/illustrations/full_loss_landscape_probe.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
