#!/usr/bin/env python3
"""
Tilting Framework Figures (Category 5)

Generates figures 5.1-5.2:
- Figure 5.1: Optimal Tilting eta*(|Delta|) (formula-based, formerly 5.4)
- Figure 5.2: Expected Width Ratio (uses MC simulation, formerly 5.5)

Note: Foundational tilting figures (parameter space, p-value family, non-centrality)
have been moved to the Theory section as figures 1.4-1.6.

Usage:
    python scripts/plot_tilting.py [--no-save] [--show]
    python scripts/plot_tilting.py --fast  # Reduced MC samples for quick testing
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from frasian.core import posterior_params, scaled_conflict
from frasian.waldo import pvalue, wald_ci_width, confidence_interval_width
from frasian.tilting import tilted_ci_width
from frasian.figure_style import (
    COLORS, MC_CONFIG, setup_style, save_figure,
)
from frasian.simulations import (
    load_raw_simulation,
    raw_simulation_exists,
    generate_all_raw_simulations,
    DEFAULT_CONFIG,
    FAST_CONFIG,
    bootstrap_mean,
    optimal_eta_empirical,
    get_optimal_eta_interpolator,
)


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_raw_data_exists(fast: bool = False):
    """Ensure raw simulation data exists, generating if needed."""
    if not raw_simulation_exists("width_raw"):
        print("Raw width data not found, generating...")
        config = FAST_CONFIG if fast else DEFAULT_CONFIG
        generate_all_raw_simulations(config=config, force=False, verbose=True)


def get_model_params(w: float, sigma: float = 1.0, mu0: float = 0.0):
    """Get model parameters for a given weight."""
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return mu0, sigma, sigma0


def data_for_conflict(delta: float, mu0: float, w: float, sigma: float) -> float:
    """Compute D value that produces a given Delta."""
    return mu0 - sigma * delta / (1 - w)


# =============================================================================
# Figure 5.1: Optimal Tilting eta*(|Delta|) (formerly 5.4)
# =============================================================================

def figure_5_1_optimal_tilting(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.1: Optimal Tilting eta*(|Delta|).

    The key practical result - how much to tilt.
    Uses numerically computed optimal eta from precomputed grid.
    """
    print("\n" + "="*60)
    print("Figure 5.1: Optimal Tilting eta*(|Delta|)")
    print("="*60)

    # Ensure optimal eta data exists
    ensure_raw_data_exists(fast=fast)

    # Load precomputed optimal eta grid
    data, metadata = load_raw_simulation("optimal_eta")
    delta_grid = data["delta_grid"]
    optimal_eta = data["optimal_eta"]

    print(f"Loaded optimal eta* for {len(delta_grid)} |Delta| values")
    print(f"  Computed with n_sims={metadata['n_sims']}, w={metadata['w']}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot numerically computed curve
    ax.plot(delta_grid, optimal_eta, color=COLORS['tilted'], linewidth=2.5,
            label=r'$\eta^*(|\Delta|)$ (numerically computed)')

    # Mark computed points (every 10th to avoid clutter)
    ax.scatter(delta_grid[::10], optimal_eta[::10], color=COLORS['tilted'],
               s=30, alpha=0.5, zorder=4)

    # Key reference line: eta = 0 boundary
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color=COLORS['wald'], linestyle='--', linewidth=1, alpha=0.5,
               label=r'$\eta=1$ (Wald)')

    # Mark regime boundaries with updated colors
    ax.axvspan(0, 0.5, alpha=0.1, color='purple', label=r'Oversharpening ($\eta^* < 0$)')
    ax.axvspan(0.5, 2, alpha=0.1, color=COLORS['tilted'], label='Transition')
    ax.axvspan(2, max(delta_grid), alpha=0.1, color=COLORS['wald'], label=r'Near-Wald ($\eta^* > 0.9$)')

    # Annotate key findings
    ax.annotate(r'Oversharpening: $\eta^* \approx -0.98$' + '\n(15% narrower than Wald)',
                xy=(0.1, optimal_eta[np.argmin(np.abs(delta_grid - 0.1))]),
                xytext=(1.0, -0.7), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.annotate(r'Approaches Wald',
                xy=(4.5, optimal_eta[np.argmin(np.abs(delta_grid - 4.5))]),
                xytext=(3.5, 0.7), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel(r'Prior-Data Conflict $|\Delta|$', fontsize=12)
    ax.set_ylabel(r'Optimal Tilting $\eta^*$', fontsize=12)
    ax.set_title('Optimal Tilting Parameter (Numerically Computed)\n'
                 r'$\eta^* < 0$ (oversharpening) when conflict is low; $\eta^* \to 1$ when high',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='center right')
    ax.set_xlim(0, max(delta_grid) + 0.2)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_1_optimal_tilting", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.2: Expected Width Ratio (formerly 5.5)
# =============================================================================

def figure_5_2_width_ratio(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.2: Expected Width Ratio.

    Shows E[W_eta*] / W_Wald as function of |Delta|.
    Uses raw D samples from simulation infrastructure.
    """
    print("\n" + "="*60)
    print("Figure 5.2: Expected Width Ratio")
    print("="*60)

    # Load raw simulation data
    ensure_raw_data_exists(fast=fast)
    data, metadata = load_raw_simulation("width_raw")

    D_samples = data["D_samples"]  # [n_theta, n_samples]
    theta_grid = data["theta_grid"]
    mu0 = metadata["mu0"]
    sigma = metadata["sigma"]
    w = metadata["w"]
    sigma0 = metadata["sigma0"]  # Use stored sigma0 directly

    n_theta, n_samples = D_samples.shape
    print(f"Loaded {n_samples} D samples for {n_theta} theta values")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reference widths
    w_wald = wald_ci_width(sigma)

    # Compute expected widths for each theta
    # theta maps to expected |Delta| = |0.5 * (mu0 - theta) / sigma| = |0.5 * (-theta)| for mu0=0, sigma=1
    expected_waldo_ratios = []
    expected_tilted_ratios = []
    expected_deltas = []
    waldo_ses = []
    tilted_ses = []

    print("Computing expected widths from raw D samples...")

    for i, theta in enumerate(theta_grid):
        D_row = D_samples[i, :]  # n_samples D values for this theta

        waldo_widths = []
        tilted_widths = []

        for D in D_row:
            # Compute actual delta for this D
            actual_delta = abs(scaled_conflict(D, mu0, w, sigma))

            # WALDO width
            w_waldo = confidence_interval_width(D, mu0, sigma, sigma0)
            waldo_widths.append(w_waldo / w_wald)

            # Optimally tilted width (using numerically computed optimal eta)
            eta_star = optimal_eta_empirical(actual_delta, fast=fast)
            w_tilted = tilted_ci_width(D, mu0, sigma, sigma0, eta_star)
            tilted_widths.append(w_tilted / w_wald)

        # Compute mean and bootstrap SE
        waldo_mean, waldo_se, _, _ = bootstrap_mean(np.array(waldo_widths), n_boot=500, seed=42 + i)
        tilted_mean, tilted_se, _, _ = bootstrap_mean(np.array(tilted_widths), n_boot=500, seed=43 + i)

        expected_waldo_ratios.append(waldo_mean)
        expected_tilted_ratios.append(tilted_mean)
        waldo_ses.append(waldo_se)
        tilted_ses.append(tilted_se)

        # Expected delta for this theta
        expected_deltas.append(abs((1 - w) * (mu0 - theta) / sigma))

    expected_waldo_ratios = np.array(expected_waldo_ratios)
    expected_tilted_ratios = np.array(expected_tilted_ratios)
    waldo_ses = np.array(waldo_ses)
    tilted_ses = np.array(tilted_ses)
    expected_deltas = np.array(expected_deltas)

    # Sort by expected delta for cleaner plotting
    sort_idx = np.argsort(expected_deltas)
    expected_deltas = expected_deltas[sort_idx]
    expected_waldo_ratios = expected_waldo_ratios[sort_idx]
    expected_tilted_ratios = expected_tilted_ratios[sort_idx]
    waldo_ses = waldo_ses[sort_idx]
    tilted_ses = tilted_ses[sort_idx]

    # Plot with error bands
    ax.plot(expected_deltas, expected_waldo_ratios, color=COLORS['waldo'], linewidth=2,
            label='WALDO E[W]/W_Wald')
    ax.fill_between(expected_deltas,
                    expected_waldo_ratios - 1.96 * waldo_ses,
                    expected_waldo_ratios + 1.96 * waldo_ses,
                    color=COLORS['waldo'], alpha=0.2)

    ax.plot(expected_deltas, expected_tilted_ratios, color=COLORS['tilted'], linewidth=2,
            label=r'Tilted($\eta^*$) E[W]/W_Wald')
    ax.fill_between(expected_deltas,
                    expected_tilted_ratios - 1.96 * tilted_ses,
                    expected_tilted_ratios + 1.96 * tilted_ses,
                    color=COLORS['tilted'], alpha=0.2)

    ax.axhline(y=1.0, color=COLORS['wald'], linestyle='--', linewidth=1.5,
               label='Wald (reference)')

    # Highlight where tilted is better
    ax.fill_between(expected_deltas, expected_tilted_ratios, 1.0,
                    where=expected_tilted_ratios < 1.0,
                    alpha=0.1, color=COLORS['tilted'],
                    label='Tilted more efficient than Wald')

    ax.set_xlabel(r'Expected Prior-Data Conflict $|\Delta|$', fontsize=12)
    ax.set_ylabel('E[Width] / W_Wald', fontsize=12)
    ax.set_title('Expected Width Ratio (with 95% CI)\n'
                 'Optimal tilting provides efficiency without sacrificing coverage',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, max(expected_deltas))
    ax.set_ylim(0.5, max(2.5, max(expected_waldo_ratios) + 0.2))
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Tilted always\nnarrower than Wald\n(on average)',
               xy=(max(expected_deltas) * 0.8, 0.95), fontsize=10, ha='center', color=COLORS['tilted'])

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_2_width_ratio", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate tilting figures")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (reduced MC samples)")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (5.1-5.2)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    fast = args.fast

    print("="*60)
    print("TILTING FRAMEWORK FIGURES")
    if fast:
        print("(Fast mode - reduced MC samples)")
    print("="*60)

    figures_to_generate = ['5.1', '5.2']
    if args.figure:
        figures_to_generate = [args.figure]

    if '5.1' in figures_to_generate:
        figure_5_1_optimal_tilting(save=save, show=show, fast=fast)

    if '5.2' in figures_to_generate:
        figure_5_2_width_ratio(save=save, show=show, fast=fast)

    print("\n" + "="*60)
    print("DONE - Tilting figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
