"""Compare trained-net eta vs per-slice constant-eta argmin across the
hyperparam grid.

For each (mu0=0, sigma_0, sigma, D) slice, computes:
  - argmin_eta: best constant eta for integrated_p loss on that slice
  - trained_eta_mean: trained EtaNet's predicted eta averaged over the
    sigma-anchored theta-grid

Plots:
  - left: argmin_eta vs (w, |Delta|) (the structured optimum)
  - middle: trained_eta_mean vs (w, |Delta|) (the network's actual output)
  - right: residual (trained - argmin) (the systematic gap)

If the trained net tracks the optimum's structure but with a bias,
the residual will be roughly constant. If the network has lost the
structure entirely, the trained values will be roughly constant
regardless of (w, |Delta|).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian._registry_bootstrap import bootstrap

bootstrap()

import jax.numpy as jnp

from frasian.learned.eta_artifact import EtaArtifact
from frasian.learned.training.losses import integrated_pvalue_loss
from frasian.learned.training.pvalue_jax import (
    ot_tilted_pvalue_jax,
    power_law_tilted_pvalue_jax,
)


def loss_at(scheme, eta_const, mu0, sigma0, sigma, D):
    K = 5.0
    theta_grid = np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 401)
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
    fn = power_law_tilted_pvalue_jax if scheme == "power_law" else ot_tilted_pvalue_jax
    eta_arr = jnp.full(theta_grid.shape, float(eta_const))
    p = fn(
        theta=jnp.asarray(theta_grid), D=jnp.asarray(D),
        w=jnp.asarray(w), mu0=jnp.asarray(mu0), sigma=jnp.asarray(sigma),
        eta=eta_arr, statistic_name="waldo",
    )
    return float(integrated_pvalue_loss(
        jnp.asarray(p)[None, :], jnp.asarray(theta_grid)[None, :]
    ))


def find_argmin(scheme, mu0, sigma0, sigma, D,
                eta_grid=np.linspace(-1.5, 1.5, 121)):
    losses = np.array([loss_at(scheme, float(e), mu0, sigma0, sigma, D)
                       for e in eta_grid])
    valid = np.isfinite(losses)
    if not valid.any():
        return float("nan")
    idx = int(np.argmin(np.where(valid, losses, np.inf)))
    return float(eta_grid[idx])


def trained_eta_mean(art: EtaArtifact, mu0, sigma0, sigma):
    """Mean of trained eta(theta) over a sigma-anchored theta grid."""
    K = 5.0
    theta_grid = np.linspace(mu0 - K * sigma0, mu0 + K * sigma0, 51)
    eta_curve = art.predict_eta(
        theta=theta_grid,
        prior_hp=np.asarray([mu0, sigma0]),
        lik_hp=np.asarray([sigma]),
    )
    return float(np.mean(eta_curve))


def main(loss_kind: str = "integrated_p"):
    sigma0_grid = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    sigma_grid = np.array([0.5, 0.7, 1.0, 1.4, 2.0])
    D_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for r_idx, scheme in enumerate(["power_law", "ot"]):
        path = Path(
            f"artifacts/learned_eta_canonical_normal_normal_"
            f"{'powerlaw' if scheme=='power_law' else 'ot'}_phaseC_{loss_kind}_v4.eqx"
        )
        if not path.exists():
            print(f"missing: {path}")
            continue
        art = EtaArtifact(
            artifact_path=path, name="learned",
            version=f"phaseC_{scheme.replace('power_law','powerlaw')}_{loss_kind}_v4",
        )
        art.load()

        rows = []
        for s0 in sigma0_grid:
            for s in sigma_grid:
                for D in D_grid:
                    w = s0 ** 2 / (s ** 2 + s0 ** 2)
                    delta = (1.0 - w) * abs(0.0 - D) / s
                    a_eta = find_argmin(scheme, 0.0, s0, s, D)
                    t_eta = trained_eta_mean(art, 0.0, s0, s)
                    rows.append((w, delta, a_eta, t_eta))

        ws = np.array([r[0] for r in rows])
        ds = np.array([r[1] for r in rows])
        argmins = np.array([r[2] for r in rows])
        traineds = np.array([r[3] for r in rows])
        residuals = traineds - argmins

        for c_idx, (vals, label, vmin, vmax, cmap) in enumerate([
            (argmins, "argmin η (per-slice optimum)", -1.5, 1.5, "RdBu_r"),
            (traineds, f"trained η_mean ({loss_kind})", -1.5, 1.5, "RdBu_r"),
            (residuals, "residual (trained − argmin)", -2.5, 2.5, "RdBu_r"),
        ]):
            ax = axes[r_idx, c_idx]
            sc = ax.scatter(ds, ws, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                            s=70, edgecolors="k", linewidths=0.4)
            ax.set_xlabel(r"$|\Delta|$")
            if c_idx == 0:
                ax.set_ylabel(f"{scheme}\n$w = \\sigma_0^2/(\\sigma^2+\\sigma_0^2)$")
            ax.set_title(label, fontsize=10)
            plt.colorbar(sc, ax=ax)

        # Numerical summary.
        print(f"\n=== {scheme} / {loss_kind} ===")
        print(f"argmin range:    [{argmins.min():+.3f}, {argmins.max():+.3f}], "
              f"mean {argmins.mean():+.3f}, median {np.median(argmins):+.3f}")
        print(f"trained range:   [{traineds.min():+.3f}, {traineds.max():+.3f}], "
              f"mean {traineds.mean():+.3f}, median {np.median(traineds):+.3f}")
        print(f"residual range:  [{residuals.min():+.3f}, {residuals.max():+.3f}], "
              f"mean {residuals.mean():+.3f}, median {np.median(residuals):+.3f}")
        # Correlation: does trained track argmin's structure?
        corr = float(np.corrcoef(argmins, traineds)[0, 1])
        print(f"correlation(argmin, trained): {corr:+.4f}")

    fig.suptitle(f"Trained vs optimal eta across (w, |Delta|) — loss = {loss_kind}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(f"output/illustrations/trained_vs_optimal_{loss_kind}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import sys
    loss = sys.argv[1] if len(sys.argv) > 1 else "integrated_p"
    main(loss)
