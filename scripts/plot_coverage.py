#!/usr/bin/env python3
"""
Coverage and Efficiency Figures (Category 2)

Generates figures 2.1-2.4:
- Figure 2.1: Coverage Curves vs θ for three w values (0.2, 0.5, 0.8)
- Figure 2.2: Coverage Heatmaps (w vs |Δ|) for Wald, Posterior, WALDO, Dynamic
- Figure 2.3: CI Width Heatmaps (w vs |Δ|) - efficiency counterpart to 2.2
- Figure 2.4: Pairwise Width Ratio Heatmaps - 6 method comparisons

Figures 2.2-2.4 share the same underlying simulations for efficiency.

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
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import norm
from tqdm import tqdm

from frasian.core import prior_residual
from frasian.waldo import pvalue as waldo_pvalue
from frasian.tilting import dynamic_tilted_pvalue_batch, dynamic_tilted_pvalue
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
# Shared Data Cache for Figures 2.2-2.4
# =============================================================================

_CACHED_HEATMAP_DATA = None
_CACHED_FAST_MODE = None


def compute_shared_heatmap_data(fast: bool = False):
    """
    Compute coverage and width grids for all methods on (w, |Δ|) grid.

    This is shared by Figures 2.2, 2.3, and 2.4 to avoid redundant computation.
    Results are cached at module level.

    Returns:
        dict with keys:
            'coverage_grids': {method_key: (n_delta, n_w) array}
            'width_grids': {method_key: (n_delta, n_w) array}
            'w_values': array of w values
            'delta_values': array of |Δ| values
            'alpha': significance level
            'z': z-score for CI
    """
    global _CACHED_HEATMAP_DATA, _CACHED_FAST_MODE

    # Return cached data if available and fast mode matches
    if _CACHED_HEATMAP_DATA is not None and _CACHED_FAST_MODE == fast:
        print("Using cached heatmap data...")
        return _CACHED_HEATMAP_DATA

    print("\n" + "="*60)
    print("Computing shared heatmap data for Figures 2.2-2.4")
    print("="*60)

    # Model parameters
    mu0, sigma = 0.0, 1.0
    alpha = 0.2  # 80% coverage/CI target

    # Grid for w and |Delta|
    n_w = 12 if fast else 25
    n_delta = 15 if fast else 30
    n_reps = 500 if fast else 2000  # Shared across coverage and width

    w_values = np.linspace(0.1, 0.9, n_w)
    delta_values = np.linspace(0, 7, n_delta)

    method_keys = ['wald', 'posterior', 'waldo', 'dynamic']

    print(f"Grid: {n_w} w values x {n_delta} |Delta| values")
    print(f"Replicates per cell: {n_reps}")
    print(f"Target coverage: {1-alpha:.0%}")

    # Initialize grids
    coverage_grids = {m: np.zeros((n_delta, n_w)) for m in method_keys}
    width_grids = {m: np.zeros((n_delta, n_w)) for m in method_keys}

    z = norm.ppf(1 - alpha / 2)
    np.random.seed(42)

    # Fine theta grid for CI width estimation
    n_theta_grid = 200 if fast else 500

    total_cells = n_w * n_delta
    pbar = tqdm(total=total_cells, desc="Computing coverage & widths")

    for i, delta in enumerate(delta_values):
        for j, w in enumerate(w_values):
            # Compute sigma0 from w
            sigma0 = sigma * np.sqrt(w / (1 - w))
            sigma_n = np.sqrt(w) * sigma

            # Compute theta_true from |Delta|: |Delta| = (1-w)|mu0 - theta|/sigma
            theta_true = mu0 + sigma * delta / (1 - w) if delta > 0 else mu0

            # Generate D samples (shared for coverage and width)
            D_samples = np.random.normal(theta_true, sigma, n_reps)
            mu_n_samples = w * D_samples + (1 - w) * mu0

            # ===== COVERAGE COMPUTATION =====

            # Wald coverage: theta_true in CI iff |D - theta_true| <= z * sigma
            wald_covered = np.abs(D_samples - theta_true) <= z * sigma
            coverage_grids['wald'][i, j] = np.mean(wald_covered)

            # Posterior coverage: theta_true in CI iff |mu_n - theta_true| <= z * sigma_n
            post_covered = np.abs(mu_n_samples - theta_true) <= z * sigma_n
            coverage_grids['posterior'][i, j] = np.mean(post_covered)

            # WALDO coverage: p(theta_true) >= alpha
            waldo_pvals = np.array([waldo_pvalue(theta_true, mu_n, mu0, w, sigma)
                                    for mu_n in mu_n_samples])
            coverage_grids['waldo'][i, j] = np.mean(waldo_pvals >= alpha)

            # Dynamic coverage: p_dynamic(theta_true) >= alpha
            dynamic_pvals = np.array([dynamic_tilted_pvalue(theta_true, D, mu0, sigma, sigma0, alpha)
                                      for D in D_samples])
            coverage_grids['dynamic'][i, j] = np.mean(dynamic_pvals >= alpha)

            # ===== WIDTH COMPUTATION =====

            # Wald width: constant = 2 * z * sigma
            width_grids['wald'][i, j] = 2 * z * sigma

            # Posterior width: 2 * z * sigma_n
            width_grids['posterior'][i, j] = 2 * z * sigma_n

            # For WALDO and Dynamic, use grid-based CI width estimation
            # Grid must cover both mu_n (CI center) AND D (where theta_true is)
            # At large |Delta|, CI can extend far beyond mu_n toward D
            theta_min = min(np.min(mu_n_samples), np.min(D_samples)) - 5 * sigma
            theta_max = max(np.max(mu_n_samples), np.max(D_samples)) + 5 * sigma
            theta_grid = np.linspace(theta_min, theta_max, n_theta_grid)
            d_theta = theta_grid[1] - theta_grid[0]

            # WALDO widths
            waldo_widths = np.zeros(n_reps)
            for k, D in enumerate(D_samples):
                mu_n = w * D + (1 - w) * mu0
                pvals = waldo_pvalue(theta_grid, mu_n, mu0, w, sigma)
                in_ci = pvals >= alpha
                waldo_widths[k] = np.sum(in_ci) * d_theta
            width_grids['waldo'][i, j] = np.mean(waldo_widths)

            # Dynamic widths (use subset for efficiency)
            n_dynamic = min(100, n_reps) if fast else min(200, n_reps)
            dynamic_widths = np.zeros(n_dynamic)
            for k in range(n_dynamic):
                D = D_samples[k]
                pvals = dynamic_tilted_pvalue_batch(theta_grid, D, mu0, sigma, sigma0, alpha)
                in_ci = pvals >= alpha
                dynamic_widths[k] = np.sum(in_ci) * d_theta
            width_grids['dynamic'][i, j] = np.mean(dynamic_widths)

            pbar.update(1)

    pbar.close()

    # Cache the results
    _CACHED_HEATMAP_DATA = {
        'coverage_grids': coverage_grids,
        'width_grids': width_grids,
        'w_values': w_values,
        'delta_values': delta_values,
        'alpha': alpha,
        'z': z,
    }
    _CACHED_FAST_MODE = fast

    return _CACHED_HEATMAP_DATA


# =============================================================================
# Figure 2.1: Coverage Curves with Error Bands (standalone)
# =============================================================================

def figure_2_1_coverage_curves(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 2.1: Coverage Curves with error bands.

    Shows coverage as continuous function of theta for four methods,
    with shaded ±1.96*SE bands. Three rows for w=0.2, 0.5, 0.8.
    """
    print("\n" + "="*60)
    print("Figure 2.1: Coverage Curves with Error Bands")
    print("="*60)

    # Model parameters
    mu0, sigma = 0.0, 1.0
    alpha = 0.2  # 80% coverage

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
        save_figure(fig, "fig_2_1_coverage_curves", "coverage")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 2.2: Coverage Heatmaps (uses shared data)
# =============================================================================

def figure_2_2_coverage_heatmap(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
    shared_data: dict = None,
) -> plt.Figure:
    """
    Generate Figure 2.2: Coverage Heatmaps for Four Methods.

    Four 2D heatmaps showing coverage as a function of w (prior weight) and |Delta|
    (prior-truth conflict) for Wald, Posterior, WALDO, and Dynamic methods.
    """
    print("\n" + "="*60)
    print("Figure 2.2: Coverage Heatmaps (w vs |Delta|)")
    print("="*60)

    # Get shared data
    if shared_data is None:
        shared_data = compute_shared_heatmap_data(fast)

    coverage_grids = shared_data['coverage_grids']
    w_values = shared_data['w_values']
    delta_values = shared_data['delta_values']
    alpha = shared_data['alpha']

    methods = ['Wald', 'Posterior', 'WALDO', 'Dynamic']
    method_keys = ['wald', 'posterior', 'waldo', 'dynamic']

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Diverging colormap centered at target coverage (0.8), spanning 0 to 1
    target_coverage = 1 - alpha  # 0.8
    color_norm = TwoSlopeNorm(vmin=0.0, vcenter=target_coverage, vmax=1.0)
    cmap = plt.cm.RdBu

    for idx, (ax, method, method_key) in enumerate(zip(axes, methods, method_keys)):
        coverage = coverage_grids[method_key]

        # Plot heatmap
        im = ax.imshow(coverage, aspect='auto', cmap=cmap, norm=color_norm,
                       origin='lower', extent=[w_values[0], w_values[-1],
                                               delta_values[0], delta_values[-1]])

        ax.set_xlabel(r'Prior weight $w$', fontsize=11)
        ax.set_ylabel(r'Prior-truth conflict $|\Delta|$', fontsize=11)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')

        # Colorbar with specific coverage ticks (including 20%)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Coverage', fontsize=9)
        cbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{t:.0%}' for t in cbar_ticks])

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
        save_figure(fig, "fig_2_2_coverage_heatmap", "coverage")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 2.3: CI Width Heatmaps (uses shared data)
# =============================================================================

def figure_2_3_width_heatmap(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
    shared_data: dict = None,
) -> plt.Figure:
    """
    Generate Figure 2.3: CI Width Heatmaps for Four Methods.

    Same grid as 2.2 (w vs |Δ|) but showing expected CI width instead of coverage.
    This shows efficiency - narrower CIs are better (given valid coverage).
    """
    print("\n" + "="*60)
    print("Figure 2.3: CI Width Heatmaps (w vs |Delta|)")
    print("="*60)

    # Get shared data
    if shared_data is None:
        shared_data = compute_shared_heatmap_data(fast)

    width_grids = shared_data['width_grids']
    w_values = shared_data['w_values']
    delta_values = shared_data['delta_values']
    z = shared_data['z']

    methods = ['Wald', 'Posterior', 'WALDO', 'Dynamic']
    method_keys = ['wald', 'posterior', 'waldo', 'dynamic']

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Diverging colormap centered on Wald width
    sigma = 1.0  # Standard parameter
    wald_width = 2 * z * sigma

    # Find global min/max for colormap range
    all_widths = np.concatenate([width_grids[m].flatten() for m in method_keys])
    vmin, vmax = np.min(all_widths), np.max(all_widths)

    # Use TwoSlopeNorm to center colormap on Wald width
    color_norm = TwoSlopeNorm(vmin=vmin, vcenter=wald_width, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # Green = small (good), red = large (bad)

    for idx, (ax, method, method_key) in enumerate(zip(axes, methods, method_keys)):
        widths = width_grids[method_key]

        # Plot heatmap
        im = ax.imshow(widths, aspect='auto', cmap=cmap, norm=color_norm,
                       origin='lower', extent=[w_values[0], w_values[-1],
                                               delta_values[0], delta_values[-1]])

        ax.set_xlabel(r'Prior weight $w$', fontsize=11)
        ax.set_ylabel(r'Prior-truth conflict $|\Delta|$', fontsize=11)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')

        # Colorbar
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
# Figure 2.4: Pairwise Width Ratio Heatmaps (uses shared data)
# =============================================================================

def figure_2_4_width_ratios(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
    shared_data: dict = None,
) -> plt.Figure:
    """
    Generate Figure 2.4: Pairwise Width Ratio Heatmaps.

    Shows 6 pairwise comparisons of CI width ratios across (w, |Delta|) grid:
    Upper triangular 3x3 layout with empty cells for definitions.

    Ratio < 1 means numerator method has narrower CIs (better).
    Diverging colormap centered on 1.0 (yellow = equal).
    """
    print("\n" + "="*60)
    print("Figure 2.4: Pairwise Width Ratio Heatmaps")
    print("="*60)

    # Get shared data
    if shared_data is None:
        shared_data = compute_shared_heatmap_data(fast)

    width_grids = shared_data['width_grids']
    w_values = shared_data['w_values']
    delta_values = shared_data['delta_values']

    # Define pairwise comparisons in upper triangular 3x3 layout
    row_methods = ['Posterior', 'Wald', 'WALDO']
    col_methods = ['Wald', 'WALDO', 'Dynamic']
    row_keys = ['posterior', 'wald', 'waldo']
    col_keys = ['wald', 'waldo', 'dynamic']

    comparisons = {}
    for row in range(3):
        for col in range(row, 3):
            num_key = col_keys[col]  # column method in numerator
            denom_key = row_keys[row]  # row method in denominator
            title = f'{col_methods[col]} / {row_methods[row]}'
            comparisons[(row, col)] = (num_key, denom_key, title)

    # Create 3x3 figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    # Global colormap range - find min/max ratio across all comparisons
    all_ratios = []
    for (row, col), (num_key, denom_key, _) in comparisons.items():
        ratio = width_grids[num_key] / width_grids[denom_key]
        all_ratios.append(ratio)
    all_ratios = np.concatenate([r.flatten() for r in all_ratios])

    # Use log-symmetric range
    log_max = np.log(np.nanmax(all_ratios))
    log_min = np.log(np.nanmin(all_ratios))
    log_extent = max(abs(log_max), abs(log_min))
    vmin_sym = np.exp(-log_extent)
    vmax_sym = np.exp(log_extent)

    # TwoSlopeNorm centered on 1.0
    color_norm = TwoSlopeNorm(vmin=vmin_sym, vcenter=1.0, vmax=vmax_sym)
    cmap = plt.cm.RdYlGn_r

    # Plot each comparison
    for (row, col), (num_key, denom_key, title) in comparisons.items():
        ax = axes[row, col]
        ratio = width_grids[num_key] / width_grids[denom_key]

        im = ax.imshow(ratio, aspect='auto', cmap=cmap, norm=color_norm,
                       origin='lower', extent=[w_values[0], w_values[-1],
                                               delta_values[0], delta_values[-1]])

        ax.set_xlabel(r'$w$', fontsize=11)
        ax.set_ylabel(r'$|\Delta|$', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Colorbar with evenly-spaced ticks
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        n_ticks = 11
        positions = np.linspace(0, 1, n_ticks)
        cbar_ticks = []
        for pos in positions:
            if pos <= 0.5:
                tick = vmin_sym + 2 * pos * (1.0 - vmin_sym)
            else:
                tick = 1.0 + 2 * (pos - 0.5) * (vmax_sym - 1.0)
            cbar_ticks.append(tick)
        cbar_ticks = np.array(cbar_ticks)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{t:.2g}' for t in cbar_ticks])
        cbar.ax.tick_params(labelsize=7)

        # Contour at ratio = 1.0
        ax.contour(w_values, delta_values, ratio, levels=[1.0],
                   colors='black', linewidths=1.5, linestyles='--')

    # Layout and labels
    plt.tight_layout(rect=[0.05, 0.02, 1, 0.94])
    fig.canvas.draw()

    # Row and column labels
    for row, label in enumerate(row_methods):
        ref_col = row
        ax_ref = axes[row, ref_col]
        bbox = ax_ref.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(bbox.x0 - 0.025, y_center, label, fontsize=15, fontweight='bold',
                 ha='right', va='center', rotation=90, color='#333333')

    for col, label in enumerate(col_methods):
        ref_row = 0 if col == 0 else (1 if col == 1 else 2)
        ax_ref = axes[ref_row, col]
        bbox = ax_ref.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        top_bbox = axes[0, col].get_position()
        fig.text(x_center, top_bbox.y1 + 0.025, label, fontsize=15, fontweight='bold',
                 ha='center', va='bottom', color='#333333')

    # Empty lower-triangle cells for definitions
    ax_def = axes[1, 0]
    ax_def.axis('off')
    ax_def.text(0.5, 0.88, 'Definitions:', fontsize=16, fontweight='bold',
                ha='center', va='top', transform=ax_def.transAxes)
    ax_def.text(0.5, 0.68, r'$w = \frac{\sigma_0^2}{\sigma^2 + \sigma_0^2}$ (prior weight)',
                fontsize=14, ha='center', va='top', transform=ax_def.transAxes)
    ax_def.text(0.5, 0.38, r'$|\Delta| = \frac{(1-w)|\mu_0 - \theta_{\mathrm{true}}|}{\sigma}$',
                fontsize=14, ha='center', va='top', transform=ax_def.transAxes)
    ax_def.text(0.5, 0.12, '(prior-truth conflict)',
                fontsize=12, ha='center', va='top', transform=ax_def.transAxes, style='italic')

    ax_color = axes[2, 0]
    ax_color.axis('off')
    ax_color.text(0.5, 0.90, 'Color Key:', fontsize=16, fontweight='bold',
                  ha='center', va='top', transform=ax_color.transAxes)
    ax_color.text(0.5, 0.68, 'Green (< 1): col narrower',
                  fontsize=14, ha='center', va='top', transform=ax_color.transAxes,
                  color='#228B22')
    ax_color.text(0.5, 0.46, 'Yellow (= 1): equal',
                  fontsize=14, ha='center', va='top', transform=ax_color.transAxes,
                  color='#B8860B')
    ax_color.text(0.5, 0.24, 'Red (> 1): col wider',
                  fontsize=14, ha='center', va='top', transform=ax_color.transAxes,
                  color='#DC143C')

    ax_insight = axes[2, 1]
    ax_insight.axis('off')
    ax_insight.text(0.5, 0.88, 'Key Insight:', fontsize=16, fontweight='bold',
                    ha='center', va='top', transform=ax_insight.transAxes)
    ax_insight.text(0.5, 0.58, 'Dynamic tilting adapts to conflict:\n'
                    r'narrow at low $|\Delta|$, Wald-like at high $|\Delta|$',
                    fontsize=14, ha='center', va='top', transform=ax_insight.transAxes,
                    linespacing=1.2)

    fig.suptitle('Pairwise CI Width Ratios: Column Method / Row Method',
                 fontsize=16, fontweight='bold', y=0.99)

    if save:
        save_figure(fig, "fig_2_4_width_ratios", "coverage")

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
    parser.add_argument("--figure", type=str, help="Generate only specific figure (2.1, 2.2, 2.3, 2.4)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    fast = args.fast

    print("="*60)
    print("COVERAGE ANALYSIS FIGURES")
    print("="*60)
    print(f"Configuration: {'FAST' if fast else 'PRODUCTION'}")

    figures_to_generate = ['2.1', '2.2', '2.3', '2.4']
    if args.figure:
        figures_to_generate = [args.figure]

    # Compute shared data once for figures 2.2-2.4
    shared_data = None
    if any(f in figures_to_generate for f in ['2.2', '2.3', '2.4']):
        shared_data = compute_shared_heatmap_data(fast)

    if '2.1' in figures_to_generate:
        figure_2_1_coverage_curves(save=save, show=show, fast=fast)

    if '2.2' in figures_to_generate:
        figure_2_2_coverage_heatmap(save=save, show=show, fast=fast, shared_data=shared_data)

    if '2.3' in figures_to_generate:
        figure_2_3_width_heatmap(save=save, show=show, fast=fast, shared_data=shared_data)

    if '2.4' in figures_to_generate:
        figure_2_4_width_ratios(save=save, show=show, fast=fast, shared_data=shared_data)

    print("\n" + "="*60)
    print("DONE - Coverage figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
