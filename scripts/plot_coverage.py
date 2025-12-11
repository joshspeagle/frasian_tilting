#!/usr/bin/env python3
"""
Coverage Analysis Figures (Category 3)

Generates figures 3.1-3.3 with proper uncertainty quantification:
- Figure 3.1: Coverage Table Heatmap with SE annotations
- Figure 3.2: Coverage Curves vs theta with error bands
- Figure 3.3: Conditional Coverage Given Prior Residual with error bands

Uses the three-layer simulation architecture:
- Layer 0: Load raw D samples
- Layer 1: Compute coverage indicators from D samples
- Layer 1.5: Cache processed results

Usage:
    python scripts/plot_coverage.py [--no-save] [--show] [--fast]
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
# Figure 3.1: Coverage Table Heatmap
# =============================================================================

def figure_3_1_coverage_heatmap(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 3.1: Coverage Table Visualization with SE annotations.

    A heatmap showing coverage rates for Wald, Posterior, WALDO across theta values.
    Color scale from red (severe undercoverage) to green (correct coverage).
    Now includes standard error in annotations.
    """
    print("\n" + "="*60)
    print("Figure 3.1: Coverage Table Heatmap")
    print("="*60)

    # Model parameters (Section 7 setup: mu0=0, sigma=sigma0=1, w=0.5)
    mu0, sigma, sigma0 = 0.0, 1.0, 1.0
    w = 0.5
    alpha = 0.05

    # Standard theta values from document
    theta_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
    methods = ['Wald', 'Posterior', 'WALDO', 'Dynamic']
    method_keys = ['wald', 'posterior', 'waldo', 'dynamic']

    # Load raw data
    data, metadata = load_coverage_data(fast)
    D_samples = data["D_samples"]  # [n_theta, n_w, n_reps]
    theta_grid = data["theta_grid"]
    w_values = data["w_values"]

    # Find w=0.5 index (or closest)
    w_idx = np.argmin(np.abs(w_values - w))
    actual_w = w_values[w_idx]
    sigma0_actual = compute_sigma0_from_w(sigma, actual_w)

    print(f"Using w={actual_w:.2f} (sigma0={sigma0_actual:.3f})")
    print(f"Number of replicates: {D_samples.shape[2]}")

    # Compute coverage for selected theta values
    coverage_matrix = np.zeros((len(methods), len(theta_values)))
    se_matrix = np.zeros((len(methods), len(theta_values)))

    for j, theta in enumerate(tqdm(theta_values, desc="Computing coverage")):
        # Find closest theta in grid
        theta_idx = np.argmin(np.abs(theta_grid - theta))
        D_row = D_samples[theta_idx, w_idx, :]

        for i, method in enumerate(method_keys):
            # Compute coverage indicators
            indicators = compute_coverage_indicators(
                D_row, theta, mu0, sigma, sigma0_actual, method, alpha
            )
            # Bootstrap for proper uncertainty
            cov, se, ci_lo, ci_hi = bootstrap_proportion(
                indicators, n_boot=1000, seed=42 + j
            )
            coverage_matrix[i, j] = cov
            se_matrix[i, j] = se

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Custom colormap centered around 95%
    from matplotlib.colors import LinearSegmentedColormap

    colors_list = ['#DC3545', '#FFC107', '#28A745']  # red, yellow, green
    cmap = LinearSegmentedColormap.from_list("coverage", colors_list, N=256)

    # Plot heatmap
    im = ax.imshow(coverage_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    # Add text annotations with SE
    for i in range(len(methods)):
        for j in range(len(theta_values)):
            cov = coverage_matrix[i, j]
            se = se_matrix[i, j]
            # White text on dark backgrounds, black on light
            text_color = 'white' if cov < 0.5 else 'black'
            # Format with uncertainty
            text = format_with_uncertainty(cov * 100, se * 100, decimals=1, percent=True)
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=9, fontweight='bold', color=text_color)

    # Axis configuration
    ax.set_xticks(range(len(theta_values)))
    ax.set_xticklabels([f'{t:.0f}' for t in theta_values])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    ax.set_xlabel(r'True parameter value $\theta$', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('Coverage Rates: WALDO Maintains 95% Coverage Everywhere\n'
                 r'($\mu_0=0$, $\sigma=\sigma_0=1$, $\alpha=0.05$)',
                 fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coverage Rate', fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 0.95, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '95%', '100%'])
    cbar.ax.axhline(y=0.95, color='black', linestyle='--', linewidth=2)

    # Add legend explaining colors
    legend_elements = [
        Patch(facecolor='#28A745', label='Correct (>93.5%)'),
        Patch(facecolor='#FFC107', label='Marginal (70-93.5%)'),
        Patch(facecolor='#DC3545', label='Severe undercoverage (<70%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.45, 1.0))

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_3_1_coverage_heatmap", "coverage")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 3.2: Coverage Curves with Error Bands
# =============================================================================

def figure_3_2_coverage_curves(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 3.2: Coverage Curves with error bands.

    Shows coverage as continuous function of theta for three methods,
    with shaded ±1.96*SE bands and multiple panels for different w values.
    """
    print("\n" + "="*60)
    print("Figure 3.2: Coverage Curves with Error Bands")
    print("="*60)

    # Load raw data
    data, metadata = load_coverage_data(fast)
    D_samples = data["D_samples"]  # [n_theta, n_w, n_reps]
    theta_grid = data["theta_grid"]
    w_values = data["w_values"]
    mu0 = metadata["mu0"]
    sigma = metadata["sigma"]
    alpha = 0.05

    methods = ['wald', 'posterior', 'waldo', 'dynamic']
    method_labels = {
        'wald': 'Wald',
        'posterior': 'Posterior',
        'waldo': 'WALDO',
        'dynamic': 'Dynamic',
    }

    print(f"Theta grid: {len(theta_grid)} points")
    print(f"W values: {w_values}")
    print(f"Replicates: {D_samples.shape[2]}")

    fig, axes = plt.subplots(1, len(w_values), figsize=FIGSIZE["panel_1x3"], sharey=True)

    # Handle single w value case
    if len(w_values) == 1:
        axes = [axes]

    for ax, (w_idx, w) in zip(axes, enumerate(w_values)):
        sigma0 = compute_sigma0_from_w(sigma, w)
        print(f"\nComputing coverage for w={w}...")

        # Compute coverage for each method
        results = {}
        for method in methods:
            coverage_rates = []
            coverage_se = []

            for i, theta in enumerate(tqdm(theta_grid, desc=f"  {method}")):
                D_row = D_samples[i, w_idx, :]
                indicators = compute_coverage_indicators(
                    D_row, theta, mu0, sigma, sigma0, method, alpha
                )
                cov, se, _, _ = bootstrap_proportion(indicators, n_boot=500, seed=42 + i)
                coverage_rates.append(cov)
                coverage_se.append(se)

            results[method] = {
                'coverage': np.array(coverage_rates),
                'se': np.array(coverage_se),
            }

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
        ax.axhline(y=0.95, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhspan(0.935, 0.965, alpha=0.1, color='gray')
        ax.axvline(x=mu0, color='gray', linestyle=':', alpha=0.5)

        # Formatting
        ax.set_xlabel(r'$\theta$', fontsize=11)
        prior_strength = "strong" if w < 0.4 else "weak" if w > 0.6 else "balanced"
        ax.set_title(f'w = {w} ({prior_strength} prior)', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if ax == axes[0]:
            ax.set_ylabel('Coverage', fontsize=11)
            ax.legend(loc='lower left', fontsize=9)

    fig.suptitle('Coverage vs True Parameter Value (with 95% CI bands)\n'
                 'WALDO and Wald maintain 95% coverage; Posterior fails away from prior mean',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_3_2_coverage_curves", "coverage")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 3.3: Conditional Coverage Given Prior Residual
# =============================================================================

def figure_3_3_conditional_coverage(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 3.3: Conditional Coverage Given Prior Residual with error bands.

    Illustrates the "prior ancillary" concept from Section 8.3.
    """
    print("\n" + "="*60)
    print("Figure 3.3: Conditional Coverage Given Prior Residual")
    print("="*60)

    # Load raw data
    data, metadata = load_coverage_data(fast)
    D_samples = data["D_samples"]
    theta_grid = data["theta_grid"]
    w_values = data["w_values"]
    mu0 = metadata["mu0"]
    sigma = metadata["sigma"]
    alpha = 0.05

    # Use w=0.5 (or closest available)
    w_idx = np.argmin(np.abs(w_values - 0.5))
    w = w_values[w_idx]
    sigma0 = compute_sigma0_from_w(sigma, w)

    # Compute delta values
    delta_values = np.array([prior_residual(t, mu0, sigma0) for t in theta_grid])

    print(f"Computing conditional coverage (w={w})...")

    # Compute coverage at each theta for both methods
    waldo_coverage = []
    waldo_se = []
    posterior_coverage = []
    posterior_se = []

    for i, theta in enumerate(tqdm(theta_grid, desc="Computing")):
        D_row = D_samples[i, w_idx, :]

        # WALDO
        ind_waldo = compute_coverage_indicators(
            D_row, theta, mu0, sigma, sigma0, 'waldo', alpha
        )
        cov, se, _, _ = bootstrap_proportion(ind_waldo, n_boot=500, seed=42 + i)
        waldo_coverage.append(cov)
        waldo_se.append(se)

        # Posterior
        ind_post = compute_coverage_indicators(
            D_row, theta, mu0, sigma, sigma0, 'posterior', alpha
        )
        cov, se, _, _ = bootstrap_proportion(ind_post, n_boot=500, seed=43 + i)
        posterior_coverage.append(cov)
        posterior_se.append(se)

    waldo_coverage = np.array(waldo_coverage)
    waldo_se = np.array(waldo_se)
    posterior_coverage = np.array(posterior_coverage)
    posterior_se = np.array(posterior_se)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Coverage vs delta with error bands
    ax1 = axes[0]
    plot_with_error_band(ax1, delta_values, waldo_coverage, waldo_se,
                         color=COLORS['waldo'], label='WALDO', alpha=0.25)
    plot_with_error_band(ax1, delta_values, posterior_coverage, posterior_se,
                         color=COLORS['posterior'], label='Posterior', alpha=0.25)

    ax1.axhline(y=0.95, color='black', linestyle='--', linewidth=1)
    ax1.axhspan(0.935, 0.965, alpha=0.1, color='gray')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label=r'$\delta=0$')

    ax1.set_xlabel(r'Prior residual $\delta(\theta) = (\theta - \mu_0)/\sigma_0$', fontsize=11)
    ax1.set_ylabel('Coverage', fontsize=11)
    ax1.set_title('Coverage vs Prior Residual (with 95% CI)', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)

    # Panel B: Coverage difference with error bars
    ax2 = axes[1]
    coverage_diff = waldo_coverage - posterior_coverage
    diff_se = np.sqrt(waldo_se**2 + posterior_se**2)  # Propagated SE

    ax2.bar(delta_values, coverage_diff, width=0.4,
            color=[COLORS['waldo'] if d >= 0 else COLORS['posterior'] for d in coverage_diff],
            alpha=0.7, edgecolor='black', linewidth=0.5,
            yerr=1.96 * diff_se, capsize=2, error_kw={'elinewidth': 0.5})

    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel(r'Prior residual $\delta(\theta)$', fontsize=11)
    ax2.set_ylabel('Coverage(WALDO) - Coverage(Posterior)', fontsize=11)
    ax2.set_title('WALDO Advantage Over Posterior', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax2.annotate('WALDO\nbetter', xy=(np.max(delta_values)*0.6, 0.35), fontsize=10, ha='center',
                color=COLORS['waldo'], fontweight='bold')

    fig.suptitle('Prior Ancillary Property: WALDO Coverage is Uniform in $\\delta(\\theta)$\n'
                 'Posterior coverage degrades as $|\\delta|$ increases',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_3_3_conditional_coverage", "coverage")

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

    figures_to_generate = ['3.1', '3.2', '3.3']
    if args.figure:
        figures_to_generate = [args.figure]

    if '3.1' in figures_to_generate:
        figure_3_1_coverage_heatmap(save=save, show=show, fast=fast)

    if '3.2' in figures_to_generate:
        figure_3_2_coverage_curves(save=save, show=show, fast=fast)

    if '3.3' in figures_to_generate:
        figure_3_3_conditional_coverage(save=save, show=show, fast=fast)

    print("\n" + "="*60)
    print("DONE - Coverage figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
