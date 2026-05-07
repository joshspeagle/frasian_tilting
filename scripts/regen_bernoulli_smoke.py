"""Regenerate the Bernoulli v0_smoke learned-eta checkpoint.

Reproduces ``artifacts/learned_eta_canonical_bernoulli_powerlaw_v0_smoke.eqx``
from a fresh training run on ``experiments/canonical_bernoulli_powerlaw.yaml``.

The committed checkpoint is **not bit-equal** to a fresh re-train —
JAX's PRNG primitives can drift across versions, and the per-run RNG
seed only fixes the *initial* network weights and exploration draws;
training metrics (val_width, head_b_accuracy) typically reproduce
within ~1× MC standard error.

Calibration on Bernoulli is verified at inference time, NOT at
training time, against the MC reference path
(``power_law._generic_tilted_pvalue``); see
``tests/regression/test_bernoulli_coverage.py``. Phase 4 skeptic #8.

Usage::

    python -m scripts.regen_bernoulli_smoke
    python -m scripts.regen_bernoulli_smoke --out artifacts/<custom>.eqx

The default budget (n_lhs=128, n_grid=51, 6 epochs, batch_size=16,
n_aux=16) takes ~5 min on dev hardware and matches the committed
fixture's budget exactly. Larger budgets converge tighter but
shouldn't change the qualitative shape of the trained η(θ).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/learned_eta_canonical_bernoulli_powerlaw_v0_smoke.eqx"),
        help="Output checkpoint path.",
    )
    parser.add_argument("--n-lhs", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=51)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-aux", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    config_yaml = Path("experiments/canonical_bernoulli_powerlaw.yaml")
    if not config_yaml.exists():
        sys.stderr.write(
            f"\nERROR: experiment YAML not found at {config_yaml}.\n"
            f"Run from the repo root, or check out a branch where the\n"
            f"experiment file is committed.\n\n"
        )
        raise SystemExit(1)

    from frasian._registry_bootstrap import bootstrap
    bootstrap()

    from frasian.learned.training.sampling import ExperimentConfig
    from frasian.learned.training.train import fit_eta_artifact

    cfg = ExperimentConfig.from_yaml(config_yaml)
    # Apply smoke-budget overrides.
    cfg = ExperimentConfig(
        scheme_name=cfg.scheme_name,
        statistic_name=cfg.statistic_name,
        prior=cfg.prior,
        model=cfg.model,
        theta_distribution=cfg.theta_distribution,
        n_grid=args.n_grid,
        n_lhs=args.n_lhs,
        n_data=cfg.n_data,
        eta_explore_box=cfg.eta_explore_box,
        seed=cfg.seed,
        name=cfg.name,
        description=cfg.description,
    )

    print(f"[regen] training to {args.out} (budget: n_lhs={args.n_lhs}, "
          f"n_grid={args.n_grid}, {args.n_epochs} epochs, batch_size="
          f"{args.batch_size}, n_aux={args.n_aux})")

    result = fit_eta_artifact(
        config=cfg,
        out_path=args.out,
        loss_kind="integrated_p",
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        n_aux=args.n_aux,
        patience=args.patience,
        antithetic=False,
        verbose=True,
    )

    print(f"[regen] final val_width    = {result.final_val_loss:.4f}")
    print(f"[regen] head_b accuracy    = {result.metadata['final_head_b_accuracy']:.3f}")
    print(f"[regen] eta_pred_valid_rate= {result.metadata['final_eta_pred_valid_rate']:.3f}")
    print(f"[regen] checkpoint saved to {result.artifact_path}")
    print()
    print("[regen] To verify the fresh checkpoint passes inference-time")
    print("[regen] calibration on Bernoulli, run:")
    print("[regen]   python -m pytest tests/regression/test_bernoulli_coverage.py -m L5")


if __name__ == "__main__":
    main()
