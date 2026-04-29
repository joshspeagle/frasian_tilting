#!/usr/bin/env python
"""Train MLP for optimal tilting and generate lookup tables.

NOTE: Requires UTF-8 encoding for output (Greek letters in diagnostics).

This script provides a complete pipeline to:
1. Generate training data via Latin Hypercube sampling + Monte Carlo
2. Train a PyTorch MLP to predict E[W]/W_Wald from (w, α, Δ', η')
3. Generate lookup tables by minimizing MLP over η' for each (w, α, Δ')

Model features:
- GELU activations
- AdamW optimizer
- OneCycleLR scheduler
- Uniform distribution standardization (mean=0.5, std=1/√12)
- Loss curve plotting

Usage:
    python scripts/train_optimal_eta_mlp.py --generate-data
    python scripts/train_optimal_eta_mlp.py --train
    python scripts/train_optimal_eta_mlp.py --generate-lookup
    python scripts/train_optimal_eta_mlp.py --all
    python scripts/train_optimal_eta_mlp.py --fast --all  # Quick test
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time

# Configure UTF-8 output for Greek letters
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python < 3.7

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP for optimal tilting parameter η*",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--generate-data', action='store_true',
        help='Generate training data via Latin Hypercube + MC'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Train MLP on generated data'
    )
    parser.add_argument(
        '--generate-lookup', action='store_true',
        help='Generate lookup tables from trained MLP'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run full pipeline: data → train → lookup'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Quick test mode with reduced samples'
    )
    parser.add_argument(
        '--n-samples', type=int, default=10000,
        help='Number of training samples (default: 10000)'
    )
    parser.add_argument(
        '--n-mc', type=int, default=100,
        help='MC samples per point (default: 100)'
    )
    parser.add_argument(
        '--n-jobs', type=int, default=-1,
        help='Parallel workers (-1 for all cores)'
    )
    # Training hyperparameters
    parser.add_argument(
        '--epochs', type=int, default=1000,
        help='Number of training epochs (default: 1000)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Batch size (default: 256)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Peak learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=1e-4,
        help='AdamW weight decay (default: 1e-4)'
    )

    args = parser.parse_args()

    # Output directory
    OUTPUT_DIR = Path(__file__).parent.parent / "output" / "simulations" / "mlp"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # File paths
    data_path = OUTPUT_DIR / "training_data.h5"
    model_path = OUTPUT_DIR / "mlp_model.pt"  # Changed to .pt for PyTorch
    lookup_path = OUTPUT_DIR / "optimal_eta_lookup.h5"
    loss_plot_path = OUTPUT_DIR / "loss_curve.png"

    # Handle --all flag
    if args.all:
        args.generate_data = True
        args.train = True
        args.generate_lookup = True

    # Handle --fast flag
    if args.fast:
        args.n_samples = 1000
        args.n_mc = 50
        args.epochs = 200

    # Check that at least one action is specified
    if not (args.generate_data or args.train or args.generate_lookup):
        parser.print_help()
        print("\nError: Specify at least one action (--generate-data, --train, --generate-lookup, or --all)")
        sys.exit(1)

    total_start = time.time()

    # =========================================================================
    # Step 1: Generate Training Data
    # =========================================================================
    if args.generate_data:
        print(f"\n{'='*60}")
        print("STEP 1: Generating Training Data")
        print(f"{'='*60}")
        print(f"  n_samples: {args.n_samples}")
        print(f"  n_mc: {args.n_mc}")
        print(f"  n_jobs: {args.n_jobs}")
        print()

        from frasian.simulations.mlp_data import (
            generate_training_data, save_training_data
        )

        start = time.time()
        data = generate_training_data(
            n_samples=args.n_samples,
            n_mc=args.n_mc,
            seed=42,
            n_jobs=args.n_jobs,
            verbose=True,
        )
        elapsed = time.time() - start

        save_training_data(data, str(data_path))
        print(f"\n  Saved training data to {data_path}")
        print(f"  Valid samples: {data['metadata']['n_valid']}/{args.n_samples}")
        print(f"  Time: {elapsed:.1f}s")

        # Quick stats
        y = data['y']
        print(f"\n  Target stats (log width ratio):")
        print(f"    min:  {y.min():.4f}")
        print(f"    max:  {y.max():.4f}")
        print(f"    mean: {y.mean():.4f}")
        print(f"    std:  {y.std():.4f}")

    # =========================================================================
    # Step 2: Train MLP (PyTorch)
    # =========================================================================
    if args.train:
        print(f"\n{'='*60}")
        print("STEP 2: Training PyTorch MLP")
        print(f"{'='*60}")
        print(f"  Architecture: (64, 64, 64) with GELU activations")
        print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
        print(f"  Scheduler: OneCycleLR")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print()

        from frasian.simulations.mlp_data import load_training_data
        from frasian.simulations.mlp_model import (
            train_pytorch_mlp, evaluate_model, WidthRatioMLP
        )

        # Load data
        data = load_training_data(str(data_path))
        X, y = data['X'], data['y']
        print(f"  Training samples: {len(X)}")
        print()

        # Train
        start = time.time()
        mlp, history = train_pytorch_mlp(
            X, y,
            hidden_sizes=(64, 64, 64),
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            verbose=True,
        )
        elapsed = time.time() - start

        # Save model
        mlp.save(str(model_path))
        print(f"\n  Saved model to {model_path}")
        print(f"  Time: {elapsed:.1f}s")

        # Plot and save loss curve
        mlp.plot_loss_curve(str(loss_plot_path))

        # Evaluate
        metrics = evaluate_model(mlp, X, y)
        print(f"\n  Metrics:")
        print(f"    R² score: {metrics['r2']:.4f}")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    MAE: {metrics['mae']:.4f}")

    # =========================================================================
    # Step 3: Generate Lookup Tables
    # =========================================================================
    if args.generate_lookup:
        print(f"\n{'='*60}")
        print("STEP 3: Generating Lookup Tables")
        print(f"{'='*60}")

        from frasian.simulations.mlp_model import WidthRatioMLP, create_predictor
        from frasian.simulations.mlp_lookup import (
            generate_lookup_table, save_lookup_table, enforce_monotonicity, LOOKUP_CONFIG
        )

        # Load model
        mlp = WidthRatioMLP.load(str(model_path))
        mlp_predict = create_predictor(mlp)

        # Define grids
        config_key = 'fast' if args.fast else 'default'
        config = LOOKUP_CONFIG[config_key]

        w_grid = config['w_grid']
        alpha_grid = config['alpha_grid']
        delta_prime_grid = config['delta_prime_grid']

        print(f"  w grid: {len(w_grid)} points [{w_grid[0]:.2f}, {w_grid[-1]:.2f}]")
        print(f"  alpha grid: {len(alpha_grid)} points [{alpha_grid[0]:.2f}, {alpha_grid[-1]:.2f}]")
        print(f"  delta' grid: {len(delta_prime_grid)} points [{delta_prime_grid[0]:.2f}, {delta_prime_grid[-1]:.2f}]")
        print(f"  Total: {len(w_grid) * len(alpha_grid) * len(delta_prime_grid):,} lookups")
        print()

        start = time.time()
        lookup = generate_lookup_table(
            mlp_predict,
            w_grid,
            alpha_grid,
            delta_prime_grid,
            verbose=True,
        )

        # Apply cumulative max for monotonicity
        print("\n  Applying cumulative max for monotonicity...")
        lookup = enforce_monotonicity(lookup)
        elapsed = time.time() - start

        save_lookup_table(lookup, str(lookup_path))
        print(f"\n  Saved lookup table to {lookup_path}")
        print(f"  Time: {elapsed:.1f}s")

        # Quick verification
        from frasian.simulations.mlp_lookup import OptimalEtaLookup
        lookup_obj = OptimalEtaLookup(lookup)

        print(f"\n  Verification (w=0.5, alpha=0.05):")
        for delta in [0.0, 0.5, 1.0, 2.0, 5.0]:
            eta = lookup_obj.get_optimal_eta(0.5, 0.05, delta)
            ratio = lookup_obj.get_width_ratio(0.5, 0.05, delta)
            print(f"    |Delta|={delta:.1f}: eta*={eta:.3f}, E[W]/W_Wald={ratio:.3f}")

    # =========================================================================
    # Summary
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # List output files
    print(f"\nOutput files:")
    for p in [data_path, model_path, loss_plot_path, lookup_path]:
        if p.exists():
            size = p.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {p.name}: {size_str}")


if __name__ == "__main__":
    main()
