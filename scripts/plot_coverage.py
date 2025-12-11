#!/usr/bin/env python3
"""
Coverage and Efficiency Figures (Category 2)

Generates figures 2.1-2.3:
- Figure 2.1: Coverage Heatmaps (w vs |Δ|) for Wald, Posterior, WALDO, Dynamic
- Figure 2.2: Coverage Curves vs θ for three w values (0.2, 0.5, 0.8)
- Figure 2.3: CI Width Heatmaps (w vs |Δ|) - efficiency counterpart to 2.1

Usage:
    python scripts/plot_coverage.py [--no-save] [--show] [--fast] [--figure 2.X]
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

from frasian.core import prior_residual
from frasian.figure_style import (
    COLORS, FIGSIZE, MC_CONFIG, setup_style, save_figure,
    plot_with_error_band, format_with_uncertainty, annotate_with_uncertainty,
)
from frasian.simulations import (
    load_raw_simulation,
    raw_simulation_exists,
    generate_all_raw_simulations,
    compute_coverage_indicators,
    compute_sigma0_from_w,
    bootstrap_proportion,
    get_or_compute_coverage,
    FAST_CONFIG,
    DEFAULT_CONFIG,
)


# =============================================================================
# Data Loading
# =============================================================================

def ensure_raw_data_exists(fast: bool = False):
    """Ensure raw D samples exist, generating if needed."""
    if not raw_simulation_exists("coverage_raw"):
        print("Raw coverage data not found, generating...")
        config = FAST_CONFIG if fast else DEFAULT_CONFIG
        generate_all_raw_simulations(config=config, force=False, verbose=True)


def load_coverage_data(fast: bool = False):
    """Load raw D samples for coverage analysis."""
    ensure_raw_data_exists(fast)
    data, metadata = load_raw_simulation("coverage_raw")
    return data, metadata


# =============================================================================
# Figure 2.1: Coverage Heatmaps (w vs |Delta|)
# =============================================================================

def figure_2_1_coverage_heatmap(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 2.1: Coverage Heatmaps for Four Methods.

    Four 2D heatmaps showing coverage as a function of w (prior weight) and |Delta|
    (prior-truth conflict) for Wald, Posterior, WALDO, and Dynamic methods.
    """
    print("\n" + "="*60)
    print("Figure 2.1: Coverage Heatmaps (w vs |Delta|)")
    print("="*60)

    # Model parameters
    mu0, sigma = 0.0, 1.0
    alpha = 0.2  # 80% coverage target

    # Grid for w and |Delta| - more bins
    n_w = 12 if fast else 25
    n_delta = 15 if fast else 30
    n_reps = 1000 if fast else 5000

    w_values = np.linspace(0.1, 0.9, n_w)
    delta_values = np.linspace(0, 4, n_delta)

    methods = ['Wald', 'Posterior', 'WALDO', 'Dynamic']
    method_keys = ['wald', 'posterior', 'waldo', 'dynamic']

    print(f"Grid: {n_w} w values x {n_delta} |Delta| values")
    print(f"Replicates per cell: {n_reps}")
    print(f"Target coverage: {1-alpha:.0%}")

    # Compute coverage for each method on (w, |Delta|) grid
    # EFFICIENT: Check p(theta_true) >= alpha instead of computing full CIs
    coverage_grids = {m: np.zeros((n_delta, n_w)) for m in method_keys}

    from frasian.tilting import dynamic_tilted_pvalue
    from frasian.waldo import pvalue as waldo_pvalue
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)

    np.random.seed(42)

    total_cells = n_w * n_delta
    pbar = tqdm(total=total_cells, desc="Computing coverage")

    for i, delta in enumerate(delta_values):
        for j, w in enumerate(w_values):
            # Compute sigma0 from w
            sigma0 = sigma * np.sqrt(w / (1 - w))

            # Compute theta_true from |Delta|: |Delta| = (1-w)|mu0 - theta|/sigma
            theta_true = mu0 + sigma * delta / (1 - w) if delta > 0 else mu0

            # Generate D samples (vectorized)
            D_samples = np.random.normal(theta_true, sigma, n_reps)

            # Compute posterior means for all samples (vectorized)
            mu_n_samples = w * D_samples + (1 - w) * mu0
            sigma_n = np.sqrt(w) * sigma

            # Wald: theta_true in CI iff |D - theta_true| <= z * sigma
            wald_covered = np.abs(D_samples - theta_true) <= z * sigma
            coverage_grids['wald'][i, j] = np.mean(wald_covered)

            # Posterior: theta_true in CI iff |mu_n - theta_true| <= z * sigma_n
            post_covered = np.abs(mu_n_samples - theta_true) <= z * sigma_n
            coverage_grids['posterior'][i, j] = np.mean(post_covered)

            # WALDO: p(theta_true) >= alpha
            # Correct call: pvalue(theta, mu_n, mu0, w, sigma)
            waldo_pvals = np.array([waldo_pvalue(theta_true, mu_n, mu0, w, sigma)
                                    for mu_n in mu_n_samples])
            coverage_grids['waldo'][i, j] = np.mean(waldo_pvals >= alpha)

            # Dynamic: p_dynamic(theta_true) >= alpha
            dynamic_pvals = np.array([dynamic_tilted_pvalue(theta_true, D, mu0, sigma, sigma0, alpha)
                                      for D in D_samples])
            coverage_grids['dynamic'][i, j] = np.mean(dynamic_pvals >= alpha)

            pbar.update(1)

    pbar.close()

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Diverging colormap centered at target coverage (0.8), spanning 0 to 1
    # Red = undercoverage, White = target, Blue = overcoverage
    from matplotlib.colors import TwoSlopeNorm
    target_coverage = 1 - alpha  # 0.8
    norm = TwoSlopeNorm(vmin=0.0, vcenter=target_coverage, vmax=1.0)
    cmap = plt.cm.RdBu

    for idx, (ax, method, method_key) in enumerate(zip(axes, methods, method_keys)):
        coverage = coverage_grids[method_key]

        # Plot heatmap - NO contour lines
        im = ax.imshow(coverage, aspect='auto', cmap=cmap, norm=norm,
                       origin='lower', extent=[w_values[0], w_values[-1],
                                               delta_values[0], delta_values[-1]])

        ax.set_xlabel(r'Prior weight $w$', fontsize=11)
        ax.set_ylabel(r'Prior-truth conflict $|\Delta|$', fontsize=11)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')

        # Colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Coverage', fontsize=9)

    # Add definitions
    fig.text(0.5, -0.02,
             r'$w = \sigma_0^2/(\sigma^2 + \sigma_0^2)$ (prior weight),  '
             r'$|\Delta| = (1-w)|\mu_0 - \theta_{\mathrm{true}}|/\sigma$ (prior-truth conflict)',
             ha='center', fontsize=10, style='italic')

    fig.suptitle(f'Coverage Rates Across $(w, |\\Delta|)$ Parameter Space\n'
                 f'Target = {target_coverage:.0%} (white), Red = undercoverage, Blue = overcoverage',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save:
        save_figure(fig, "fig_2_1_coverage_heatmap", "coverage")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 2.2: Coverage Curves with Error Bands
# =============================================================================

def figure_2_2_coverage_curves(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 2.2: Coverage Curves with error bands.

    Shows coverage as continuous function of theta for four methods,
    with shaded ±1.96*SE bands. Three rows for w=0.2, 0.5, 0.8.
    """
    print("\n" + "="*60)
    print("Figure 2.2: Coverage Curves with Error Bands")
    print("="*60)

    # Model parameters
    mu0, sigma = 0.0, 1.0
    alpha = 0.2  # 80% coverage to match figure 2.1

    # Fixed w values for 3-row layout
    w_values = [0.2, 0.5, 0.8]
    w_labels = ['Strong prior (w=0.2)', 'Balanced (w=0.5)', 'Weak prior (w=0.8)']

    # Theta grid and replicates
    n_theta = 15 if fast else 30
    n_reps = 500 if fast else 2000
    theta_grid = np.linspace(-4, 6, n_theta)

    methods = ['wald', 'posterior', 'waldo', 'dynamic']
    method_labels = {
        'wald': 'Wald',
        'posterior': 'Posterior',
        'waldo': 'WALDO',
        'dynamic': 'Dynamic',
    }

    print(f"Theta grid: {n_theta} points")
    print(f"W values: {w_values}")
    print(f"Replicates per point: {n_reps}")
    print(f"Target coverage: {1-alpha:.0%}")

    # Create 3-row figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharey=True)

    from frasian.waldo import pvalue as waldo_pvalue
    from frasian.tilting import dynamic_tilted_pvalue
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    np.random.seed(42)

    for row_idx, (ax, w, w_label) in enumerate(zip(axes, w_values, w_labels)):
        sigma0 = sigma * np.sqrt(w / (1 - w))
        sigma_n = np.sqrt(w) * sigma
        print(f"\nComputing coverage for {w_label}...")

        # Compute coverage for each method
        results = {m: {'coverage': [], 'se': []} for m in methods}

        for theta in tqdm(theta_grid, desc=f"  w={w}"):
            # Generate D samples for this theta
            D_samples = np.random.normal(theta, sigma, n_reps)
            mu_n_samples = w * D_samples + (1 - w) * mu0

            # Wald coverage
            wald_covered = np.abs(D_samples - theta) <= z * sigma
            results['wald']['coverage'].append(np.mean(wald_covered))
            results['wald']['se'].append(np.std(wald_covered) / np.sqrt(n_reps))

            # Posterior coverage
            post_covered = np.abs(mu_n_samples - theta) <= z * sigma_n
            results['posterior']['coverage'].append(np.mean(post_covered))
            results['posterior']['se'].append(np.std(post_covered) / np.sqrt(n_reps))

            # WALDO coverage: p(theta) >= alpha
            waldo_pvals = np.array([waldo_pvalue(theta, mu_n, mu0, w, sigma)
                                    for mu_n in mu_n_samples])
            waldo_covered = waldo_pvals >= alpha
            results['waldo']['coverage'].append(np.mean(waldo_covered))
            results['waldo']['se'].append(np.std(waldo_covered) / np.sqrt(n_reps))

            # Dynamic coverage: p_dynamic(theta) >= alpha
            dynamic_pvals = np.array([dynamic_tilted_pvalue(theta, D, mu0, sigma, sigma0, alpha)
                                      for D in D_samples])
            dynamic_covered = dynamic_pvals >= alpha
            results['dynamic']['coverage'].append(np.mean(dynamic_covered))
            results['dynamic']['se'].append(np.std(dynamic_covered) / np.sqrt(n_reps))

        # Convert to arrays
        for method in methods:
            results[method]['coverage'] = np.array(results[method]['coverage'])
            results[method]['se'] = np.array(results[method]['se'])

        # Plot curves with error bands
        for method in methods:
            cov = results[method]['coverage']
            se = results[method]['se']
            plot_with_error_band(
                ax, theta_grid, cov, se,
                color=COLORS[method],
                label=method_labels[method],
                alpha=0.25,
            )

        # Reference lines
        target = 1 - alpha
        ax.axhline(y=target, color='black', linestyle='--', linewidth=1, alpha=0.7,
                   label=f'{target:.0%} target')
        ax.axhspan(target - 0.02, target + 0.02, alpha=0.1, color='gray')
        ax.axvline(x=mu0, color='gray', linestyle=':', alpha=0.5, label=r'$\mu_0$')

        # Formatting
        ax.set_ylabel('Coverage', fontsize=11)
        ax.set_title(w_label, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if row_idx == 0:
            ax.legend(loc='lower right', fontsize=9, ncol=3)
        if row_idx == 2:
            ax.set_xlabel(r'True parameter $\theta$', fontsize=11)

    fig.suptitle(f'Coverage vs True Parameter Value (target = {1-alpha:.0%})\n'
                 'WALDO and Wald maintain nominal coverage; Posterior fails away from prior mean',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_2_2_coverage_curves", "coverage")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 2.3: CI Width Heatmaps (Efficiency)
# =============================================================================

def figure_2_3_width_heatmap(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 2.3: CI Width Heatmaps for Four Methods.

    Same grid as 2.1 (w vs |Δ|) but showing expected CI width instead of coverage.
    This shows efficiency - narrower CIs are better (given valid coverage).
    """
    print("\n" + "="*60)
    print("Figure 2.3: CI Width Heatmaps (w vs |Delta|)")
    print("="*60)

    # Model parameters (same as 2.1)
    mu0, sigma = 0.0, 1.0
    alpha = 0.2  # 80% CI to match figure 2.1

    # Grid for w and |Delta| - same as 2.1
    n_w = 12 if fast else 25
    n_delta = 15 if fast else 30
    n_reps = 500 if fast else 2000

    w_values = np.linspace(0.1, 0.9, n_w)
    delta_values = np.linspace(0, 4, n_delta)

    methods = ['Wald', 'Posterior', 'WALDO', 'Dynamic']
    method_keys = ['wald', 'posterior', 'waldo', 'dynamic']

    print(f"Grid: {n_w} w values x {n_delta} |Delta| values")
    print(f"Replicates per cell: {n_reps}")
    print(f"CI level: {1-alpha:.0%}")

    # Compute expected CI width for each method on (w, |Delta|) grid
    width_grids = {m: np.zeros((n_delta, n_w)) for m in method_keys}

    from frasian.waldo import pvalue as waldo_pvalue
    from frasian.tilting import dynamic_tilted_pvalue_batch
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    np.random.seed(42)

    # Fine theta grid for finding CI bounds via grid search
    n_theta_grid = 200 if fast else 500

    total_cells = n_w * n_delta
    pbar = tqdm(total=total_cells, desc="Computing widths")

    for i, delta in enumerate(delta_values):
        for j, w in enumerate(w_values):
            # Compute sigma0 from w
            sigma0 = sigma * np.sqrt(w / (1 - w))

            # Compute theta_true from |Delta| (for generating D samples)
            theta_true = mu0 + sigma * delta / (1 - w) if delta > 0 else mu0

            # Generate D samples (vectorized)
            D_samples = np.random.normal(theta_true, sigma, n_reps)

            # Wald width: constant = 2 * z * sigma
            width_grids['wald'][i, j] = 2 * z * sigma

            # Posterior width: 2 * z * sigma_n = 2 * z * sqrt(w) * sigma
            width_grids['posterior'][i, j] = 2 * z * np.sqrt(w) * sigma

            # For WALDO and Dynamic, use grid-based CI width estimation
            # Create theta grid centered on typical posterior means
            mu_n_samples = w * D_samples + (1 - w) * mu0
            theta_min = np.min(mu_n_samples) - 5 * sigma
            theta_max = np.max(mu_n_samples) + 5 * sigma
            theta_grid = np.linspace(theta_min, theta_max, n_theta_grid)
            d_theta = theta_grid[1] - theta_grid[0]

            # WALDO: fully vectorized - pvalue() accepts arrays for theta
            waldo_widths = np.zeros(n_reps)
            for k, D in enumerate(D_samples):
                mu_n = w * D + (1 - w) * mu0
                # waldo_pvalue is vectorized over theta
                pvals = waldo_pvalue(theta_grid, mu_n, mu0, w, sigma)
                in_ci = pvals >= alpha
                waldo_widths[k] = np.sum(in_ci) * d_theta
            width_grids['waldo'][i, j] = np.mean(waldo_widths)

            # Dynamic: use batched MLP for efficiency
            n_dynamic = min(100, n_reps)  # Use subset of samples
            dynamic_widths = np.zeros(n_dynamic)
            for k in range(n_dynamic):
                D = D_samples[k]
                # Use batched p-value computation (MLP inference is batched)
                pvals = dynamic_tilted_pvalue_batch(theta_grid, D, mu0, sigma, sigma0, alpha)
                in_ci = pvals >= alpha
                dynamic_widths[k] = np.sum(in_ci) * d_theta
            width_grids['dynamic'][i, j] = np.mean(dynamic_widths)

            pbar.update(1)

    pbar.close()

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Diverging colormap centered on Wald width
    # Wald width is constant: 2 * z * sigma
    wald_width = 2 * z * sigma

    # Find global min/max for colormap range
    all_widths = np.concatenate([width_grids[m].flatten() for m in method_keys])
    vmin, vmax = np.min(all_widths), np.max(all_widths)

    # Use TwoSlopeNorm to center colormap on Wald width (yellow)
    # Green = narrower than Wald (better), Red = wider than Wald (worse)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=wald_width, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # Reversed: green = small (good), red = large (bad)

    for idx, (ax, method, method_key) in enumerate(zip(axes, methods, method_keys)):
        widths = width_grids[method_key]

        # Plot heatmap
        im = ax.imshow(widths, aspect='auto', cmap=cmap, norm=norm,
                       origin='lower', extent=[w_values[0], w_values[-1],
                                               delta_values[0], delta_values[-1]])

        ax.set_xlabel(r'Prior weight $w$', fontsize=11)
        ax.set_ylabel(r'Prior-truth conflict $|\Delta|$', fontsize=11)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')

        # Colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('CI Width', fontsize=9)

    # Add definitions
    fig.text(0.5, -0.02,
             r'$w = \sigma_0^2/(\sigma^2 + \sigma_0^2)$ (prior weight),  '
             r'$|\Delta| = (1-w)|\mu_0 - \theta_{\mathrm{true}}|/\sigma$ (prior-truth conflict)',
             ha='center', fontsize=10, style='italic')

    fig.suptitle(f'Expected CI Width Across $(w, |\\Delta|)$ Parameter Space\n'
                 f'Yellow = Wald width ({wald_width:.2f}), Green = narrower (better), Red = wider',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save:
        save_figure(fig, "fig_2_3_width_heatmap", "coverage")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate coverage figures")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--fast", action="store_true", help="Use fast config (fewer samples)")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (3.1, 3.2, 3.3)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    fast = args.fast

    print("="*60)
    print("COVERAGE ANALYSIS FIGURES")
    print("="*60)
    print(f"Configuration: {'FAST' if fast else 'PRODUCTION'}")

    figures_to_generate = ['2.1', '2.2', '2.3']
    if args.figure:
        figures_to_generate = [args.figure]

    if '2.1' in figures_to_generate:
        figure_2_1_coverage_heatmap(save=save, show=show, fast=fast)

    if '2.2' in figures_to_generate:
        figure_2_2_coverage_curves(save=save, show=show, fast=fast)

    if '2.3' in figures_to_generate:
        figure_2_3_width_heatmap(save=save, show=show, fast=fast)

    print("\n" + "="*60)
    print("DONE - Coverage figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
