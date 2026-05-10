"""Illustration: Phase G learned-η selector — η(θ) curve + boundary heatmap.

Two panels:
  Panel A — η_pred(θ) vs the legacy NumericalEtaSelector. Shows the
            smooth GELU-MLP curve from EtaNet vs the kinky clamp-then-
            jump curve from NumericalEtaSelector.

  Panel B — ValidityNet's predicted P(valid | θ, η) on a (θ, η) grid,
            with the trained η_pred curve overlaid. Visualises the
            *learned* admissible boundary that the boundary penalty
            pushes EtaNet inside.

Requires a Phase G v4 checkpoint (conditional architecture). The
demo evaluates at a fixed canonical (μ₀=0, σ₀=1, σ=1) which sits
inside the trained hyperparam range; pass --mu0/--sigma0/--sigma to
explore other in-range slices. Trains a fixture on the fly when none
is available.
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
from frasian.tilting.eta_selectors import (
    DynamicNumericalEtaSelector,
    LearnedDynamicEtaSelector,
    _NamedStatistic,
)
from frasian.tilting.power_law import PowerLawTilting


_V4_CONFIG_NAME = "canonical_normal_normal_powerlaw_v4"


def _ensure_v4_checkpoint(smoke: bool) -> Path:
    """Locate or train the canonical NN+power_law v4 checkpoint."""
    project_root = Path(__file__).resolve().parents[4]
    out = project_root / "artifacts" / f"learned_eta_{_V4_CONFIG_NAME}.eqx"
    if out.exists():
        return out
    print("[learned_eta_demo] no v4 checkpoint found; training one...")
    from frasian._registry_bootstrap import bootstrap

    bootstrap()
    from frasian.learned.training.sampling import ExperimentConfig
    from frasian.learned.training.train import fit_eta_artifact

    cfg_path = project_root / "experiments" / f"{_V4_CONFIG_NAME}.yaml"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    if smoke:
        cfg = ExperimentConfig(
            scheme_name=cfg.scheme_name,
            statistic_name=cfg.statistic_name,
            prior_cls=cfg.prior_cls,
            model_cls=cfg.model_cls,
            hyperparam_distribution=cfg.hyperparam_distribution,
            theta_distribution=cfg.theta_distribution,
            n_grid=51,
            n_lhs=512,
            n_data=cfg.n_data,
            seed=cfg.seed,
            eta_explore_box=cfg.eta_explore_box,
            theta_grid_lo=cfg.theta_grid_lo,
            theta_grid_hi=cfg.theta_grid_hi,
        )
    fit_eta_artifact(
        config=cfg,
        out_path=out,
        n_epochs=10 if smoke else 30,
        batch_size=32 if smoke else 256,
        n_aux=32 if smoke else 256,
        lambda_max=2.0 if smoke else 10.0,
        lambda_warmup_frac=0.4,
        patience=10 if smoke else 8,
        verbose=False,
    )
    return out


def main(
    smoke: bool = False,
    out: Path | None = None,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    sigma: float = 1.0,
) -> Path:
    alpha = 0.05
    ckpt_path = _ensure_v4_checkpoint(smoke=smoke)

    from frasian.learned.eta_artifact import EtaArtifact

    artifact = EtaArtifact(
        artifact_path=ckpt_path,
        name=f"learned_eta_{ckpt_path.stem}",
    )
    artifact.load()

    cfg = artifact.metadata["experiment_config"]
    theta_lo = float(cfg["theta_distribution_fingerprint"][1])
    theta_hi = float(cfg["theta_distribution_fingerprint"][2])
    w_eval = sigma0**2 / (sigma**2 + sigma0**2)

    demo_model = NormalNormalModel(sigma=sigma)
    demo_prior = NormalDistribution(loc=mu0, scale=sigma0)
    prior_hp = demo_prior.hyperparams()
    lik_hp = demo_model.hyperparams()

    _learned_selector = LearnedDynamicEtaSelector(artifact=artifact)
    legacy_selector = DynamicNumericalEtaSelector()
    scheme = PowerLawTilting()

    n_grid = 51 if smoke else 201
    theta_grid = np.linspace(theta_lo, theta_hi, n_grid)

    fig, (ax_eta, ax_boundary) = plt.subplots(1, 2, figsize=(12.0, 4.5))

    # Panel A — η(θ) curves at the demo (mu0, sigma0, sigma) slice.
    eta_learned = artifact.predict_eta(theta_grid, prior_hp, lik_hp)
    eta_legacy = legacy_selector.select_grid(
        theta_grid,
        scheme,
        statistic=_NamedStatistic("waldo"),  # type: ignore[arg-type]
        model=demo_model,
        prior=demo_prior,
        alpha=alpha,
    )
    ax_eta.plot(
        theta_grid, eta_learned, color="C0", lw=2.0,
        label=f"EtaNet v4 (μ₀={mu0}, σ₀={sigma0}, σ={sigma}, w={w_eval:.2f})",
    )
    ax_eta.plot(
        theta_grid,
        eta_legacy,
        color="C1",
        lw=1.5,
        linestyle="--",
        alpha=0.8,
        label=f"NumericalEtaSelector (legacy, w={w_eval:.2f})",
    )
    eta_min = (2 * w_eval - 1) / w_eval if w_eval > 0 else float("-inf")
    eta_max = 1.0 / max(1.0 - w_eval, 1e-9)
    ax_eta.axhline(eta_min, color="grey", lw=0.6, alpha=0.5, label=f"η_min = {eta_min:.2f}")
    ax_eta.axhline(eta_max, color="grey", lw=0.6, alpha=0.5, label=f"η_max = {eta_max:.2f}")
    ax_eta.set_xlabel("θ")
    ax_eta.set_ylabel("η(θ)")
    ax_eta.set_title("Panel A — η(θ) (Phase G EtaNet vs legacy)")
    ax_eta.legend(loc="best", fontsize=8)

    # Panel B — ValidityNet boundary heatmap with EtaNet's η_pred curve.
    n_eta = 31 if smoke else 81
    eta_min_plot = max(eta_min - 0.5, -3.0) if np.isfinite(eta_min) else -3.0
    eta_max_plot = min(eta_max + 0.5, 3.0) if np.isfinite(eta_max) else 3.0
    eta_axis = np.linspace(eta_min_plot, eta_max_plot, n_eta)
    Theta, Eta = np.meshgrid(theta_grid, eta_axis, indexing="xy")
    p_valid = artifact.predict_validity(Theta, Eta, prior_hp, lik_hp)
    im = ax_boundary.imshow(
        p_valid,
        origin="lower",
        aspect="auto",
        extent=[theta_lo, theta_hi, eta_min_plot, eta_max_plot],
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=ax_boundary, label="ValidityNet P(valid | θ, η)")
    ax_boundary.plot(theta_grid, eta_learned, color="black", lw=2.0, label="η_pred (EtaNet)")
    if np.isfinite(eta_min):
        ax_boundary.axhline(eta_min, color="black", lw=0.8, ls=":", label="analytical boundary")
    if np.isfinite(eta_max):
        ax_boundary.axhline(eta_max, color="black", lw=0.8, ls=":")
    ax_boundary.set_xlabel("θ")
    ax_boundary.set_ylabel("η")
    ax_boundary.set_title("Panel B — ValidityNet's learned admissible boundary")
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
    parser.add_argument("--mu0", type=float, default=0.0)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()
    try:
        import jax  # noqa: F401
        import equinox  # noqa: F401
    except ImportError:
        print("learned_eta_demo: jax/equinox not available; skipping.")
        raise SystemExit(0) from None
    path = main(
        smoke=args.smoke, out=args.out,
        mu0=args.mu0, sigma0=args.sigma0, sigma=args.sigma,
    )
    print(f"wrote {path}")
