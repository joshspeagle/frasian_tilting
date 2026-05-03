"""CLI for training the learned dynamic-η selector.

Example:
    python -m scripts.train_learned_eta --scheme power_law \\
        --loss integrated_p --out artifacts/learned_eta_power_law_v1.pt

Use --fast for a smoke training run (small budgets).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from frasian.learned.training.sampling import TrainingDistribution
from frasian.learned.training.train import fit_monotonic_eta_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a MonotonicEtaArtifact for the dynamic-η selector",
    )
    parser.add_argument("--scheme", choices=["power_law", "ot"], required=True)
    parser.add_argument("--statistic", default="waldo",
                        choices=["waldo", "wald"])
    parser.add_argument("--loss", default="integrated_p",
                        choices=["integrated_p", "cd_variance", "static_width"])
    parser.add_argument("--alpha", type=float, default=None,
                        help="required iff --loss=static_width")
    parser.add_argument("--n-lhs", type=int, default=10000)
    parser.add_argument("--n-mc", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--theta-grid-n", type=int, default=401)
    parser.add_argument("--search-mult", type=float, default=8.0)
    parser.add_argument("--patience", type=int, default=15,
                        help="early stop after this many epochs with no val improvement")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="minimum val-loss improvement counted as progress")
    parser.add_argument("--theta-true-half-width", type=float, default=5.0)
    parser.add_argument("--w-min", type=float, default=0.05)
    parser.add_argument("--w-max", type=float, default=0.95)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", default="v0")
    parser.add_argument("--shared-sizes", type=int, nargs="+",
                        default=[64, 64],
                        help="hidden sizes for MonotonicEtaNet shared (w) pathway")
    parser.add_argument("--mono-sizes", type=int, nargs="+",
                        default=[64, 64],
                        help="hidden sizes for MonotonicEtaNet monotonic (Δ') pathway")
    parser.add_argument("--fast", action="store_true",
                        help="smoke mode: 512 LHS, 4 MC, 20 epochs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

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
        lr=args.lr,
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

    print(f"[done] final val loss = {result.final_val_loss:.4f}")
    print(f"[done] artifact at    = {result.artifact_path}")


if __name__ == "__main__":
    main()
