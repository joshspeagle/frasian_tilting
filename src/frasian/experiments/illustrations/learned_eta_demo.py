"""Illustration: Phase E learned-η selector — η(θ) curve + boundary heatmap.

Two panels:
  Panel A — η_pred(θ) vs the legacy NumericalEtaSelector. Shows the
            smooth GELU-MLP curve from EtaNet vs the kinky clamp-then-
            jump curve from NumericalEtaSelector.

  Panel B — ValidityNet's predicted P(valid | θ, η) on a (θ, η) grid,
            with the trained η_pred curve overlaid. Visualises the
            *learned* admissible boundary that the boundary penalty
            pushes EtaNet inside.

Requires a Phase E checkpoint (format v2) with ValidityNet state.
Trains a smoke checkpoint on the fly when none is available so the
demo is runnable without prior setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import (DynamicNumericalEtaSelector,
                                              LearnedDynamicEtaSelector,
                                              _NamedStatistic)
from frasian.tilting.power_law import PowerLawTilting


def _ensure_phase_e_checkpoint(smoke: bool) -> Path:
    """Locate or train a Phase E checkpoint (format v2)."""
    # Anchor at project root so paths don't depend on CWD.
    project_root = Path(__file__).resolve().parents[4]
    candidates = [
        project_root / "artifacts" / "learned_eta_canonical_normal_normal_powerlaw_v0_smoke.pt",
        project_root / "artifacts" / "learned_eta_canonical_normal_normal_powerlaw_v1.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    print("[learned_eta_demo] no Phase E checkpoint found; training one...")
    from frasian._registry_bootstrap import bootstrap
    bootstrap()
    from frasian.learned.training.sampling import ExperimentConfig
    from frasian.learned.training.train import fit_eta_artifact

    cfg_path = project_root / "experiments" / "canonical_normal_normal_powerlaw.yaml"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    if smoke:
        cfg = ExperimentConfig(
            scheme_name=cfg.scheme_name, statistic_name=cfg.statistic_name,
            prior=cfg.prior, model=cfg.model,
            theta_distribution=cfg.theta_distribution,
            n_grid=51, n_lhs=512, eta_explore_box=cfg.eta_explore_box,
            seed=cfg.seed,
        )
    out = project_root / "artifacts" / "learned_eta_canonical_normal_normal_powerlaw_v0_smoke.pt"
    fit_eta_artifact(
        config=cfg, out_path=out,
        n_epochs=10 if smoke else 30,
        batch_size=32 if smoke else 256,
        n_aux=32 if smoke else 256,
        lambda_max=2.0 if smoke else 10.0,
        lambda_warmup_frac=0.4,
        patience=10 if smoke else 8,
        verbose=False,
    )
    return out


def main(smoke: bool = False, out: Path | None = None) -> Path:
    alpha = 0.05
    ckpt_path = _ensure_phase_e_checkpoint(smoke=smoke)

    from frasian.learned.eta_artifact import EtaArtifact
    artifact = EtaArtifact(
        artifact_path=ckpt_path,
        name=f"learned_eta_{ckpt_path.stem}",
    )
    artifact.load()

    # Read the trained experiment from the checkpoint config so the
    # demo works for any (μ₀, σ) the user trained at, not just (0, 1).
    cfg = artifact.metadata["experiment_config"]
    theta_lo, theta_hi = float(
        cfg["theta_distribution_fingerprint"][1]
    ), float(cfg["theta_distribution_fingerprint"][2])
    mu0 = float(cfg["prior_fingerprint"][1])
    sigma0 = float(cfg["prior_fingerprint"][2])
    sigma = float(cfg["model_fingerprint"][1])
    w_trained = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

    learned_selector = LearnedDynamicEtaSelector(
        artifact=artifact, sigma=sigma, mu0=mu0,
    )
    legacy_selector = DynamicNumericalEtaSelector(sigma=sigma, mu0=mu0)
    scheme = PowerLawTilting()

    n_grid = 51 if smoke else 201
    theta_grid = np.linspace(theta_lo, theta_hi, n_grid)
    abs_delta_grid = np.abs((1.0 - w_trained) * (mu0 - theta_grid) / sigma)

    fig, (ax_eta, ax_boundary) = plt.subplots(1, 2, figsize=(12.0, 4.5))

    # Panel A — η(θ) curves.
    eta_learned = artifact.predict_eta(theta_grid)
    eta_legacy = legacy_selector.select_grid(
        abs_delta_grid, scheme, statistic=_NamedStatistic("waldo"),
        w=w_trained, alpha=alpha,
    )
    ax_eta.plot(theta_grid, eta_learned, color="C0", lw=2.0,
                  label=f"EtaNet (Phase E, w={w_trained:.2f})")
    ax_eta.plot(theta_grid, eta_legacy, color="C1", lw=1.5,
                  linestyle="--", alpha=0.8,
                  label=f"NumericalEtaSelector (legacy, w={w_trained:.2f})")
    # Show the admissible boundary.
    eta_min = (2 * w_trained - 1) / w_trained if w_trained > 0 else float("-inf")
    eta_max = 1.0 / max(1.0 - w_trained, 1e-9)
    ax_eta.axhline(eta_min, color="grey", lw=0.6, alpha=0.5,
                     label=f"η_min = {eta_min:.2f}")
    ax_eta.axhline(eta_max, color="grey", lw=0.6, alpha=0.5,
                     label=f"η_max = {eta_max:.2f}")
    ax_eta.set_xlabel("θ")
    ax_eta.set_ylabel("η(θ)")
    ax_eta.set_title("Panel A — η(θ) (Phase E EtaNet vs legacy)")
    ax_eta.legend(loc="best", fontsize=8)

    # Panel B — ValidityNet boundary heatmap with EtaNet's η_pred curve.
    n_eta = 31 if smoke else 81
    eta_min_plot = max(eta_min - 0.5, -3.0) if np.isfinite(eta_min) else -3.0
    eta_max_plot = min(eta_max + 0.5, 3.0) if np.isfinite(eta_max) else 3.0
    eta_axis = np.linspace(eta_min_plot, eta_max_plot, n_eta)
    Theta, Eta = np.meshgrid(theta_grid, eta_axis, indexing="xy")
    p_valid = artifact.predict_validity(Theta, Eta)
    im = ax_boundary.imshow(
        p_valid, origin="lower", aspect="auto",
        extent=[theta_lo, theta_hi, eta_min_plot, eta_max_plot],
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
    )
    fig.colorbar(im, ax=ax_boundary, label="ValidityNet P(valid | θ, η)")
    # Overlay η_pred and the analytical admissible boundary.
    ax_boundary.plot(theta_grid, eta_learned, color="black", lw=2.0,
                       label="η_pred (EtaNet)")
    if np.isfinite(eta_min):
        ax_boundary.axhline(eta_min, color="black", lw=0.8, ls=":",
                              label="analytical boundary")
    if np.isfinite(eta_max):
        ax_boundary.axhline(eta_max, color="black", lw=0.8, ls=":")
    ax_boundary.set_xlabel("θ")
    ax_boundary.set_ylabel("η")
    ax_boundary.set_title(
        "Panel B — ValidityNet's learned admissible boundary"
    )
    ax_boundary.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = out or Path("output/illustrations/learned_eta_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("learned_eta_demo: torch not available; skipping.")
        raise SystemExit(0)
    path = main(smoke=args.smoke, out=args.out)
    print(f"wrote {path}")
