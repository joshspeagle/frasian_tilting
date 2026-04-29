#!/usr/bin/env python
"""Train monotonic MLP for optimal eta prediction.

This script trains a secondary neural network that:
1. Samples from the trained width ratio MLP using Latin Hypercube Sampling
2. Finds optimal eta' for each sample via grid search
3. Trains a monotonic neural network to predict eta'*(w, alpha, Delta')

The monotonic architecture guarantees d(eta*)/d(|Delta|) >= 0 by construction,
eliminating the need for post-hoc monotonicity enforcement.

Prerequisites:
    - Trained width ratio MLP at output/simulations/mlp/mlp_model.pt

Usage:
    python scripts/train_monotonic_eta_mlp.py
    python scripts/train_monotonic_eta_mlp.py --fast  # Quick test
    python scripts/train_monotonic_eta_mlp.py --n-samples 100000  # More samples
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_training_data(
    width_ratio_mlp,
    n_samples: int = 50000,
    seed: int = 99,
    verbose: bool = True,
):
    """Generate training data by sampling optimal eta' from width ratio MLP.

    Parameters
    ----------
    width_ratio_mlp : WidthRatioMLP
        Trained width ratio predictor
    n_samples : int
        Number of training samples
    seed : int
        Random seed (different from width ratio MLP for independence)
    verbose : bool
        Print progress

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Input features [w, alpha, delta_prime]
    y : ndarray of shape (n_samples,)
        Target values (eta_prime_star)
    """
    from scipy.stats import qmc
    from frasian.simulations.mlp_lookup import find_optimal_eta_prime

    if verbose:
        print(f"Generating {n_samples:,} samples...")

    # Latin Hypercube Sampling in (w, alpha, Delta') space
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale to valid ranges
    w_samples = 0.01 + 0.98 * samples[:, 0]          # [0.01, 0.99]
    alpha_samples = 0.01 + 0.98 * samples[:, 1]      # [0.01, 0.99]
    delta_prime_samples = 0.99 * samples[:, 2]       # [0, 0.99]

    # Find optimal eta' for each sample
    if verbose:
        print("Finding optimal eta' for each sample...")

    eta_prime_star = np.zeros(n_samples)

    from tqdm import tqdm
    iterator = tqdm(range(n_samples), desc="Grid search") if verbose else range(n_samples)

    for i in iterator:
        eta_star, _ = find_optimal_eta_prime(
            lambda X: width_ratio_mlp.predict(X),
            w_samples[i],
            alpha_samples[i],
            delta_prime_samples[i],
            n_grid=100,
        )
        eta_prime_star[i] = eta_star

    X = np.column_stack([w_samples, alpha_samples, delta_prime_samples])
    y = eta_prime_star

    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Train monotonic MLP for optimal eta prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Quick test mode with reduced samples and epochs'
    )
    parser.add_argument(
        '--n-samples', type=int, default=50000,
        help='Number of training samples (default: 50000)'
    )
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
        '--seed', type=int, default=99,
        help='Random seed for LHS sampling (default: 99, different from width MLP)'
    )

    args = parser.parse_args()

    # Output directory
    OUTPUT_DIR = Path(__file__).parent.parent / "output" / "simulations" / "mlp"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # File paths
    width_mlp_path = OUTPUT_DIR / "mlp_model.pt"
    mono_mlp_path = OUTPUT_DIR / "monotonic_eta_mlp.pt"
    loss_plot_path = OUTPUT_DIR / "monotonic_loss_curve.png"

    # Fast mode settings
    if args.fast:
        args.n_samples = 5000
        args.epochs = 200

    # Check prerequisites
    if not width_mlp_path.exists():
        print(f"ERROR: Width ratio MLP not found at {width_mlp_path}")
        print("Run 'python scripts/train_optimal_eta_mlp.py --train' first")
        sys.exit(1)

    print(f"{'='*60}")
    print("Training Monotonic Eta MLP")
    print(f"{'='*60}")
    print(f"  Width MLP: {width_mlp_path}")
    print(f"  Output: {mono_mlp_path}")
    print(f"  Samples: {args.n_samples:,}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    print()

    total_start = time.time()

    # =========================================================================
    # Step 1: Load width ratio MLP
    # =========================================================================
    print("Step 1: Loading width ratio MLP...")
    from frasian.simulations.mlp_model import WidthRatioMLP
    width_mlp = WidthRatioMLP.load(str(width_mlp_path))
    print(f"  Loaded model with architecture: {width_mlp.hidden_sizes}")

    # =========================================================================
    # Step 2: Generate training data
    # =========================================================================
    print(f"\nStep 2: Generating training data...")
    start = time.time()
    X, y = generate_training_data(
        width_mlp,
        n_samples=args.n_samples,
        seed=args.seed,
        verbose=True,
    )
    elapsed = time.time() - start
    print(f"  Data generation time: {elapsed:.1f}s")
    print(f"  X shape: {X.shape}")
    print(f"  y (eta'*) range: [{y.min():.3f}, {y.max():.3f}]")

    # =========================================================================
    # Step 3: Train monotonic MLP
    # =========================================================================
    print(f"\nStep 3: Training monotonic MLP...")
    print(f"  Architecture: shared=(64, 64), mono=(64, 64), cross=32")
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=1e-4)")
    print(f"  Scheduler: OneCycleLR")
    print()

    from frasian.simulations.mlp_monotonic import MonotonicEtaMLP

    mono_mlp = MonotonicEtaMLP(
        shared_sizes=(64, 64),
        mono_sizes=(64, 64),
        cross_size=32,
    )

    start = time.time()
    history = mono_mlp.fit(
        X, y,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        verbose=True,
    )
    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s")

    # Save model
    mono_mlp.save(str(mono_mlp_path))
    print(f"  Saved model to: {mono_mlp_path}")

    # =========================================================================
    # Step 4: Plot loss curve
    # =========================================================================
    print(f"\nStep 4: Plotting loss curve...")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = np.arange(1, len(history['train_losses']) + 1)

    # Full loss curve
    ax1 = axes[0]
    ax1.plot(epochs, history['train_losses'], label='Train', alpha=0.7)
    ax1.plot(epochs, history['val_losses'], label='Validation', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Monotonic MLP Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Zoomed
    ax2 = axes[1]
    start_idx = len(epochs) // 2
    ax2.plot(epochs[start_idx:], history['train_losses'][start_idx:], label='Train', alpha=0.7)
    ax2.plot(epochs[start_idx:], history['val_losses'][start_idx:], label='Validation', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Loss (Last 50%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved loss curve to: {loss_plot_path}")

    # =========================================================================
    # Step 5: Verify monotonicity
    # =========================================================================
    print(f"\nStep 5: Verifying monotonicity...")

    from frasian.simulations.mlp_data import delta_transform, eta_inverse

    all_monotonic = True
    for w_test in [0.2, 0.5, 0.8]:
        for alpha_test in [0.01, 0.05, 0.1]:
            delta_prime_fine = np.linspace(0, 0.99, 201)
            X_test = np.column_stack([
                np.full(201, w_test),
                np.full(201, alpha_test),
                delta_prime_fine
            ])
            eta_prime_pred = mono_mlp.predict(X_test)
            diffs = np.diff(eta_prime_pred)
            violations = np.sum(diffs < -1e-6)
            if violations > 0:
                print(f"  WARNING: w={w_test}, alpha={alpha_test}: {violations} violations")
                all_monotonic = False

    if all_monotonic:
        print("  All test slices are monotonically increasing!")

    # =========================================================================
    # Step 6: Compare with lookup table
    # =========================================================================
    print(f"\nStep 6: Comparison with lookup table...")

    lookup_path = OUTPUT_DIR / "optimal_eta_lookup.h5"
    if lookup_path.exists():
        from frasian.simulations.mlp_lookup import OptimalEtaLookup
        from frasian.simulations.mlp_monotonic import OptimalEtaPredictor

        lookup = OptimalEtaLookup.from_file(str(lookup_path))
        predictor = OptimalEtaPredictor(mono_mlp)

        # Compare predictions
        w, alpha = 0.5, 0.05
        delta_values = np.array([0, 0.5, 1, 2, 5])

        print(f"  Comparison at w={w}, alpha={alpha}:")
        print(f"  {'|Delta|':>8} {'Lookup':>10} {'MonoMLP':>10} {'Diff':>10}")
        print(f"  {'-'*40}")

        for d in delta_values:
            eta_lookup = lookup.get_optimal_eta(w, alpha, d)
            eta_mono = predictor.get_optimal_eta(w, alpha, d)
            diff = eta_mono - eta_lookup
            print(f"  {d:>8.1f} {eta_lookup:>10.4f} {eta_mono:>10.4f} {diff:>10.4f}")
    else:
        print("  Lookup table not found, skipping comparison")

    # =========================================================================
    # Summary
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Final metrics
    final_val_rmse = np.sqrt(history['val_losses'][-1])
    print(f"\nFinal validation RMSE (eta' space): {final_val_rmse:.4f}")

    print(f"\nOutput files:")
    for p in [mono_mlp_path, loss_plot_path]:
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
