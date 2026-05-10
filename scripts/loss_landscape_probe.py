"""Loss-landscape probe for the learned-eta tail-decay puzzle.

For a fixed (mu0, sigma_0, sigma, D), at multiple theta_test values
spanning the central regime and the boundary regime, sweep eta_test
over its admissible range and plot:

  1. p(theta_test; D, eta_test) -- the per-theta-test marginal
     contribution to integrated_pvalue_loss. The minimum over eta_test
     is what an integrated-p-trained network would want at that
     theta_test (in the marginal sense).

  2. The trained network's actual prediction eta_hat(theta_test) for
     each of the three losses, drawn as vertical lines.

If for theta_test=4 the loss-landscape minimum is at eta near 1 but
the trained networks predict eta near 0.5, that's strong evidence
of a training pathology (sparse-data extrapolation or similar). If
the loss minimum is itself at eta near 0.5, the trained behavior
matches what the loss prefers and our intuition was wrong.

Saves a side-by-side panel plot with one column per theta_test.

Run from repo root:
    python -m scripts.loss_landscape_probe
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
from frasian.learned.training.pvalue_jax import (
    power_law_tilted_pvalue_jax,
    ot_tilted_pvalue_jax,
)
from frasian.tilting.mixture import _mixture_tilted_pvalue_numpy_scalar


# Fixed (mu0, sigma_0, sigma, D); demo slice from CLAUDE.md / v4 plot.
MU0 = 0.0
SIGMA0 = 1.0
SIGMA = 1.0
D_OBS = 0.0
W = SIGMA0 ** 2 / (SIGMA ** 2 + SIGMA0 ** 2)  # = 0.5

# theta_test values spanning the central + boundary regimes.
THETA_TEST = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# eta sweep, schemes & training admissibility.
ETA_GRID = np.linspace(-1.0, 1.5, 251)


def _scheme_pvalue_at_eta(scheme, theta_test, eta_arr):
    """Return p(theta_test; D, eta_arr) for one scheme."""
    out = np.empty_like(eta_arr)
    if scheme == "power_law":
        for i, eta in enumerate(eta_arr):
            out[i] = float(
                power_law_tilted_pvalue_jax(
                    theta=jnp.asarray(theta_test),
                    D=jnp.asarray(D_OBS),
                    w=jnp.asarray(W),
                    mu0=jnp.asarray(MU0),
                    sigma=jnp.asarray(SIGMA),
                    eta=jnp.asarray(float(eta)),
                    statistic_name="waldo",
                )
            )
    elif scheme == "ot":
        for i, eta in enumerate(eta_arr):
            out[i] = float(
                ot_tilted_pvalue_jax(
                    theta=jnp.asarray(theta_test),
                    D=jnp.asarray(D_OBS),
                    w=jnp.asarray(W),
                    mu0=jnp.asarray(MU0),
                    sigma=jnp.asarray(SIGMA),
                    eta=jnp.asarray(float(eta)),
                    statistic_name="waldo",
                )
            )
    elif scheme == "mixture":
        for i, eta in enumerate(eta_arr):
            try:
                out[i] = _mixture_tilted_pvalue_numpy_scalar(
                    float(theta_test), float(eta), float(D_OBS),
                    W, MU0, SIGMA, "waldo",
                )
            except Exception:
                out[i] = np.nan
    else:
        raise ValueError(scheme)
    return out


def _load_eta_at(scheme, loss, theta_test):
    """Load the trained EtaArtifact and predict eta at theta_test."""
    if scheme == "power_law":
        path = Path(
            f"artifacts/learned_eta_canonical_normal_normal_powerlaw_phaseC_{loss}_v4.eqx"
        )
        version = f"phaseC_powerlaw_{loss}_v4"
    elif scheme == "ot":
        path = Path(
            f"artifacts/learned_eta_canonical_normal_normal_ot_phaseC_{loss}_v4.eqx"
        )
        version = f"phaseC_ot_{loss}_v4"
    else:
        return None
    if not path.exists():
        return None
    art = EtaArtifact(artifact_path=path, name="learned", version=version)
    art.load()
    eta_hat = art.predict_eta(
        theta=np.asarray([float(theta_test)]),
        prior_hp=np.asarray([MU0, SIGMA0]),
        lik_hp=np.asarray([SIGMA]),
    )
    return float(eta_hat[0])


def main():
    schemes = ["power_law", "ot"]
    losses = ["integrated_p", "cd_variance", "static_width"]
    loss_colors = {"integrated_p": "tab:blue", "cd_variance": "tab:orange",
                   "static_width": "tab:green"}

    n_theta = len(THETA_TEST)
    n_schemes = len(schemes)
    fig, axes = plt.subplots(
        n_schemes, n_theta, figsize=(2.6 * n_theta, 3.0 * n_schemes),
        sharex=True, sharey="row", squeeze=False,
    )

    for r, scheme in enumerate(schemes):
        for c, theta_test in enumerate(THETA_TEST):
            ax = axes[r, c]
            p_curve = _scheme_pvalue_at_eta(scheme, theta_test, ETA_GRID)

            # Plot p(theta_test; D, eta) as the integrated-p marginal proxy.
            ax.plot(ETA_GRID, p_curve, color="black", lw=1.5,
                    label="p(theta_test; D, eta)")

            # Mark trained-net predictions for each loss.
            for loss in losses:
                eta_hat = _load_eta_at(scheme, loss, theta_test)
                if eta_hat is None:
                    continue
                ax.axvline(eta_hat, color=loss_colors[loss], lw=1.4,
                           label=f"{loss}: eta={eta_hat:+.2f}")

            # Mark argmin of p(theta_test; D, eta) -- the integrated-p
            # marginal optimum.
            valid = np.isfinite(p_curve)
            if valid.any():
                idx = int(np.argmin(np.where(valid, p_curve, np.inf)))
                argmin_eta = float(ETA_GRID[idx])
                argmin_p = float(p_curve[idx])
                ax.plot(argmin_eta, argmin_p, "kx", markersize=8,
                        label=f"argmin: eta={argmin_eta:+.2f}")

            # Reference markers: eta=0 (WALDO), eta=1 (Wald).
            ax.axvline(0.0, color="gray", lw=0.6, ls=":")
            ax.axvline(1.0, color="red", lw=0.6, ls=":", alpha=0.5)

            ax.set_title(rf"$\theta={theta_test:+.0f}$", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{scheme}\np(theta_test; D, eta)", fontsize=9)
            if r == n_schemes - 1:
                ax.set_xlabel(r"$\eta$", fontsize=9)
            ax.set_xlim(-1.0, 1.5)
            ax.tick_params(labelsize=8)
            if r == 0 and c == n_theta - 1:
                ax.legend(loc="upper right", fontsize=7, frameon=False)

    fig.suptitle(
        rf"Loss-landscape probe at $\mu_0={MU0}, \sigma_0={SIGMA0}, "
        rf"\sigma={SIGMA}, D={D_OBS}, w={W}$"
        "\nblack: p(theta_test; D, eta) = integrated-p marginal at theta_test; "
        "vertical lines: trained net predictions",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path("output/illustrations/loss_landscape_probe.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

    # Also print a numerical table for the report.
    print("\n=== Numerical summary ===")
    print(f"{'scheme':<10} {'theta':>6} {'argmin_p_eta':>14} "
          f"{'p_at_argmin':>14} {'eta_intp':>10} {'eta_cdvar':>10} {'eta_statw':>10}")
    print("-" * 90)
    for scheme in schemes:
        for theta_test in THETA_TEST:
            p_curve = _scheme_pvalue_at_eta(scheme, theta_test, ETA_GRID)
            valid = np.isfinite(p_curve)
            idx = int(np.argmin(np.where(valid, p_curve, np.inf)))
            argmin_eta = float(ETA_GRID[idx])
            argmin_p = float(p_curve[idx])
            eta_ip = _load_eta_at(scheme, "integrated_p", theta_test)
            eta_cv = _load_eta_at(scheme, "cd_variance", theta_test)
            eta_sw = _load_eta_at(scheme, "static_width", theta_test)
            ip_str = f"{eta_ip:+.3f}" if eta_ip is not None else "  --"
            cv_str = f"{eta_cv:+.3f}" if eta_cv is not None else "  --"
            sw_str = f"{eta_sw:+.3f}" if eta_sw is not None else "  --"
            print(f"{scheme:<10} {theta_test:+6.1f} {argmin_eta:+14.3f} "
                  f"{argmin_p:14.6f} {ip_str:>10} {cv_str:>10} {sw_str:>10}")


if __name__ == "__main__":
    main()
