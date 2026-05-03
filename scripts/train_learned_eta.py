"""CLI for the Phase E learned-η selector training.

Drives the dual-head training loop (``EtaNet`` + ``ValidityNet``) off
an ``ExperimentConfig`` YAML file. The legacy MonotonicEtaNet
training is reachable via ``--legacy``; that path is kept until E.4
removes the ``MonotonicEtaArtifact`` artifact loader and the
``test_train_smoke.py`` regression.

Examples
--------
New (Phase E):
    python -m scripts.train_learned_eta \\
        --config experiments/canonical_normal_normal_powerlaw.yaml \\
        --n-epochs 30 --patience 8 \\
        --out artifacts/learned_eta_canonical_normal_normal_powerlaw_v0_smoke.pt

Legacy (Phase D, until removed):
    python -m scripts.train_learned_eta --legacy --scheme power_law \\
        --loss integrated_p \\
        --out artifacts/learned_eta_power_law_v0_smoke.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from frasian._registry_bootstrap import bootstrap


def _phase_e_main(args: argparse.Namespace) -> None:
    """Phase E dual-head training driven by ExperimentConfig."""
    from frasian.learned.training.sampling import ExperimentConfig
    from frasian.learned.training.train import fit_eta_artifact

    config = ExperimentConfig.from_yaml(args.config)

    if args.fast:
        # Smoke budgets: small LHS, few epochs, small grid.
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
        n_epochs=args.n_epochs,
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


def _legacy_main(args: argparse.Namespace) -> None:
    """LEGACY (Phase D) training. Removed in E.4."""
    from frasian.learned.training.sampling import TrainingDistribution
    from frasian.learned.training.train import fit_monotonic_eta_artifact

    if args.fast:
        args.n_lhs = 512
        args.n_mc = 4
        args.n_epochs = 20

    train_dist = TrainingDistribution(
        w_range=(args.w_min, args.w_max),
        theta_true_half_width=args.theta_true_half_width,
    )
    result = fit_monotonic_eta_artifact(
        scheme_name=args.scheme,
        statistic_name=args.statistic,
        loss_kind=args.loss,
        alpha=args.alpha,
        train_dist=train_dist,
        n_lhs=args.n_lhs,
        n_mc=args.n_mc,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr_a,
        weight_decay=args.weight_decay,
        theta_grid_n=args.theta_grid_n,
        search_mult=args.search_mult,
        patience=args.patience,
        min_delta=args.min_delta,
        device=args.device,
        seed=args.seed,
        out_path=args.out,
        version=args.version,
        architecture_kwargs={
            "shared_sizes": tuple(args.shared_sizes),
            "mono_sizes": tuple(args.mono_sizes),
        },
        verbose=not args.quiet,
    )
    print(f"[done] (legacy) final val loss = {result.final_val_loss:.4f}")
    print(f"[done] (legacy) artifact at    = {result.artifact_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a learned-η selector (Phase E dual-head, "
        "or legacy MonotonicEtaNet via --legacy)."
    )

    # --- Phase E (default) ---
    parser.add_argument(
        "--config", type=Path,
        help="ExperimentConfig YAML (Phase E). Mutually exclusive with --legacy.",
    )
    parser.add_argument("--n-aux", type=int, default=None,
                        help="aux samples for boundary probing (defaults to batch_size)")
    parser.add_argument("--lambda-max", type=float, default=10.0,
                        help="boundary-penalty max weight")
    parser.add_argument("--lambda-warmup-frac", type=float, default=0.3,
                        help="fraction of epochs to ramp λ from 0 to lambda_max")
    parser.add_argument("--lr-a", type=float, default=1e-3,
                        help="learning rate for EtaNet (Head A)")
    parser.add_argument("--lr-b", type=float, default=1e-3,
                        help="learning rate for ValidityNet (Head B)")
    parser.add_argument("--eta-hidden-sizes", type=int, nargs="+",
                        default=[64, 64])
    parser.add_argument("--validity-hidden-sizes", type=int, nargs="+",
                        default=[64, 64])

    # --- Legacy (Phase D) ---
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy MonotonicEtaNet training (Phase D).")
    parser.add_argument("--scheme", choices=["power_law", "ot"], default=None,
                        help="(legacy) scheme name")
    parser.add_argument("--theta-true-half-width", type=float, default=5.0)
    parser.add_argument("--w-min", type=float, default=0.05)
    parser.add_argument("--w-max", type=float, default=0.95)
    parser.add_argument("--n-mc", type=int, default=8)
    parser.add_argument("--theta-grid-n", type=int, default=401)
    parser.add_argument("--search-mult", type=float, default=8.0)
    parser.add_argument("--shared-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--mono-sizes", type=int, nargs="+", default=[64, 64])

    # --- Shared ---
    parser.add_argument("--statistic", default="waldo",
                        choices=["waldo", "wald"],
                        help="(legacy only) test statistic")
    parser.add_argument("--loss", default="integrated_p",
                        choices=["integrated_p", "cd_variance", "static_width"])
    parser.add_argument("--alpha", type=float, default=None,
                        help="required iff --loss=static_width")
    parser.add_argument("--n-lhs", type=int, default=10000)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", default="v0")
    parser.add_argument("--fast", action="store_true",
                        help="smoke budgets")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    bootstrap()

    if args.legacy:
        if args.scheme is None:
            parser.error("--legacy requires --scheme")
        _legacy_main(args)
    else:
        if args.config is None:
            parser.error(
                "--config <path/to/experiment.yaml> is required "
                "(or pass --legacy to use the deprecated path)"
            )
        _phase_e_main(args)


if __name__ == "__main__":
    main()
