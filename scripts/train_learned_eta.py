"""CLI for the Phase E learned-η selector training.

Drives the dual-head training loop (``EtaNet`` + ``ValidityNet``) off
an ``ExperimentConfig`` YAML file.

Example:
    python -m scripts.train_learned_eta \\
        --config experiments/canonical_normal_normal_powerlaw.yaml \\
        --n-epochs 30 --patience 8 \\
        --out artifacts/learned_eta_canonical_normal_normal_powerlaw_v1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from frasian._registry_bootstrap import bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Phase E learned-η selector "
        "(EtaNet + ValidityNet) on an ExperimentConfig."
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="ExperimentConfig YAML path.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loss", default="integrated_p",
                        choices=["integrated_p", "cd_variance", "static_width"])
    parser.add_argument("--alpha", type=float, default=None,
                        help="required iff --loss=static_width")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-aux", type=int, default=None,
                        help="aux samples for boundary probing (defaults to batch_size)")
    parser.add_argument("--lr-a", type=float, default=1e-3,
                        help="learning rate for EtaNet (Head A)")
    parser.add_argument("--lr-b", type=float, default=1e-3,
                        help="learning rate for ValidityNet (Head B)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-max", type=float, default=10.0,
                        help="boundary-penalty max weight")
    parser.add_argument("--lambda-warmup-frac", type=float, default=0.3,
                        help="fraction of epochs to ramp λ from 0 to lambda_max")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--eta-hidden-sizes", type=int, nargs="+",
                        default=[64, 64])
    parser.add_argument("--validity-hidden-sizes", type=int, nargs="+",
                        default=[64, 64])
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--version", default="v0")
    parser.add_argument("--fast", action="store_true",
                        help="smoke budgets: 512 LHS, 51-pt grid, 10 epochs.")
    parser.add_argument(
        "--n-lhs", type=int, default=None,
        help="Override config.n_lhs (e.g. for smoke training).",
    )
    parser.add_argument(
        "--n-grid", type=int, default=None,
        help="Override config.n_grid (e.g. for smoke training).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    bootstrap()

    from frasian.learned.training.sampling import ExperimentConfig
    from frasian.learned.training.train import fit_eta_artifact

    config = ExperimentConfig.from_yaml(args.config)

    # Apply CLI overrides on n_lhs / n_grid if provided.
    overrides = {}
    if args.n_lhs is not None:
        overrides["n_lhs"] = int(args.n_lhs)
    if args.n_grid is not None:
        overrides["n_grid"] = int(args.n_grid)
    if overrides:
        config = ExperimentConfig(
            scheme_name=config.scheme_name,
            statistic_name=config.statistic_name,
            prior=config.prior,
            model=config.model,
            theta_distribution=config.theta_distribution,
            n_grid=overrides.get("n_grid", config.n_grid),
            n_lhs=overrides.get("n_lhs", config.n_lhs),
            eta_explore_box=config.eta_explore_box,
            seed=config.seed,
            name=config.name,
            description=config.description,
        )

    if args.fast:
        config = ExperimentConfig(
            scheme_name=config.scheme_name,
            statistic_name=config.statistic_name,
            prior=config.prior,
            model=config.model,
            theta_distribution=config.theta_distribution,
            n_grid=51,
            n_lhs=512,
            eta_explore_box=config.eta_explore_box,
            seed=config.seed,
            name=config.name,
            description=config.description,
        )

    result = fit_eta_artifact(
        config=config,
        out_path=args.out,
        loss_kind=args.loss,
        alpha=args.alpha,
        n_epochs=10 if args.fast else args.n_epochs,
        batch_size=args.batch_size,
        n_aux=args.n_aux,
        lr_a=args.lr_a,
        lr_b=args.lr_b,
        weight_decay=args.weight_decay,
        lambda_max=args.lambda_max,
        lambda_warmup_frac=args.lambda_warmup_frac,
        patience=args.patience,
        min_delta=args.min_delta,
        eta_hidden_sizes=tuple(args.eta_hidden_sizes),
        validity_hidden_sizes=tuple(args.validity_hidden_sizes),
        device=args.device,
        version=args.version,
        verbose=not args.quiet,
    )

    print(f"[done] final val width  = {result.final_val_loss:.4f}")
    print(
        f"[done] head_b accuracy   = "
        f"{result.metadata['final_head_b_accuracy']:.3f}"
    )
    print(
        f"[done] η_pred valid rate = "
        f"{result.metadata['final_eta_pred_valid_rate']:.3f}"
    )
    print(f"[done] artifact at       = {result.artifact_path}")


if __name__ == "__main__":
    main()
