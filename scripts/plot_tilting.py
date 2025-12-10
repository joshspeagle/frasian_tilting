#!/usr/bin/env python3
"""
Tilting Framework Figures (Category 5)

Generates figures 5.1-5.5:
- Figure 5.1: The Tilting Parameter Space (formula-based)
- Figure 5.2: Tilted P-value Family (formula-based)
- Figure 5.3: Non-centrality Reduction (formula-based)
- Figure 5.4: Optimal Tilting eta*(|Delta|) (formula-based)
- Figure 5.5: Expected Width Ratio (uses MC simulation)

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
from frasian.tilting import (
    tilted_params,
    tilted_pvalue,
    tilted_ci_width,
)
from frasian.figure_style import (
    COLORS, MC_CONFIG, setup_style, save_figure,
    get_tilting_colors,
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
# Figure 5.1: The Tilting Parameter Space
# =============================================================================

def figure_5_1_tilting_space(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.1: The Tilting Parameter Space.

    Schematic showing eta in [-1, 1] continuum from oversharpening through WALDO to Wald.
    """
    print("\n" + "="*60)
    print("Figure 5.1: The Tilting Parameter Space")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)
    D = 3.0  # Example data point with conflict

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Panel A: Schematic of extended eta spectrum (now -1 to 1)
    ax1 = axes[0]

    # Draw the spectrum bar - now extended to include oversharpening
    # Oversharpening region: purple
    ax1.axvspan(-1, 0, color='purple', alpha=0.4)
    # Standard tilting: gradient from WALDO to Wald
    eta_vals = np.linspace(0, 1, 100)
    colors = get_tilting_colors(100)
    for i, (eta, color) in enumerate(zip(eta_vals[:-1], colors[:-1])):
        ax1.axvspan(eta, eta_vals[i+1], color=color, alpha=0.8)

    # Add vertical lines at key boundaries
    ax1.axvline(x=-1, color='black', linewidth=2)
    ax1.axvline(x=0, color='black', linewidth=2)
    ax1.axvline(x=1, color='black', linewidth=2)

    # Labels
    ax1.text(-0.5, 1.15, 'Oversharpening\n($\\eta < 0$)', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='purple')
    ax1.text(0, 1.15, 'WALDO\n($\\eta = 0$)', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color=COLORS['waldo'])
    ax1.text(1, 1.15, 'Wald\n($\\eta = 1$)', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color=COLORS['wald'])
    ax1.text(0.5, 1.15, 'Tilted\nPosterior', ha='center', va='bottom',
             fontsize=11, color=COLORS['tilted'])

    # Add key eta values
    key_etas = [-1, -0.5, 0, 0.5, 1.0]
    for eta in key_etas:
        ax1.plot(eta, 0.5, 'ko', markersize=8)
        ax1.text(eta, 0.3, f'{eta}', ha='center', fontsize=10)

    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(0, 1.5)
    ax1.set_xlabel(r'Tilting Parameter $\eta$', fontsize=12)
    ax1.set_title(r'The Extended $\eta$ Spectrum: Oversharpening to Wald', fontsize=13, fontweight='bold')
    ax1.set_yticks([])

    # Add annotations about what changes with eta
    ax1.annotate('Mode past prior', xy=(-0.5, 0.7), fontsize=10, ha='center')
    ax1.annotate('CI narrows', xy=(-0.5, 0.55), fontsize=10, ha='center', color='gray')
    ax1.annotate('Mode moves: mu_n -> D', xy=(0.5, 0.7), fontsize=10, ha='center')
    ax1.annotate('CI widens', xy=(0.5, 0.55), fontsize=10, ha='center', color='gray')

    # Panel B: Mode position vs eta (extended range)
    ax2 = axes[1]

    etas = np.linspace(-1, 1, 100)
    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

    modes = []
    for eta in etas:
        mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta)
        modes.append(mu_eta)

    ax2.plot(etas, modes, color=COLORS['tilted'], linewidth=2.5)

    # Shade oversharpening region
    ax2.axvspan(-1, 0, color='purple', alpha=0.1)

    ax2.axhline(y=mu_n, color=COLORS['waldo'], linestyle='--', linewidth=1.5,
                label=f'WALDO mode $\\mu_n$ = {mu_n:.2f}')
    ax2.axhline(y=D, color=COLORS['mle'], linestyle='--', linewidth=1.5,
                label=f'MLE D = {D:.2f}')
    ax2.axhline(y=mu0, color=COLORS['prior_mean'], linestyle=':', linewidth=1.5,
                label=f'Prior mean $\\mu_0$ = {mu0:.2f}')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # Mark key points
    mu_neg1, _, _ = tilted_params(D, mu0, sigma, sigma0, -1)
    ax2.scatter([-1, 0, 1], [mu_neg1, mu_n, D], color='black', s=80, zorder=5)
    ax2.annotate(f'$\\eta$=-1: {mu_neg1:.2f}', xy=(-1, mu_neg1), xytext=(-0.8, mu_neg1-0.3),
                fontsize=9)
    ax2.annotate(f'$\\eta$=0: {mu_n:.2f}', xy=(0, mu_n), xytext=(0.1, mu_n+0.2),
                fontsize=9)
    ax2.annotate(f'$\\eta$=1: {D:.2f}', xy=(1, D), xytext=(0.85, D+0.2),
                fontsize=9)

    ax2.set_xlabel(r'Tilting Parameter $\eta$', fontsize=12)
    ax2.set_ylabel(r'Tilted Mode $\mu_\eta$', fontsize=12)
    ax2.set_title(r'Mode Position: $\eta < 0$ pushes past $\mu_n$ toward prior; $\eta > 0$ pushes toward D',
                  fontsize=12)
    ax2.legend(loc='center right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.1, 1.1)

    fig.suptitle('Extended Tilting: From Oversharpening Through WALDO to Wald',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_1_tilting_space", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.2: Tilted P-value Family
# =============================================================================

def figure_5_2_tilted_pvalue_family(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.2: Tilted P-value Family.

    Shows p_eta(theta) for multiple eta values.
    """
    print("\n" + "="*60)
    print("Figure 5.2: Tilted P-value Family")
    print("="*60)

    # Model parameters with conflict
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)
    D = 3.0  # Data with conflict

    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
    delta = scaled_conflict(D, mu0, w, sigma)
    print(f"D = {D}, mu_n = {mu_n:.2f}, Delta = {delta:.2f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Theta range
    thetas = np.linspace(-1, 5, 200)

    # Eta values to plot
    etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = get_tilting_colors(len(etas))

    for eta, color in zip(etas, colors):
        pvals = [tilted_pvalue(theta, D, mu0, sigma, sigma0, eta) for theta in thetas]

        if eta == 0:
            label = r'$\eta=0$ (WALDO)'
        elif eta == 1:
            label = r'$\eta=1$ (Wald)'
        else:
            label = f'$\\eta={eta}$'

        ax.plot(thetas, pvals, color=color, linewidth=2, label=label)

        # Mark the mode
        mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta)
        ax.scatter([mu_eta], [1.0], color=color, s=50, zorder=5)

    # Reference lines
    ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=r'$\alpha=0.05$')
    ax.axvline(x=D, color=COLORS['mle'], linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'MLE D={D}')
    ax.axvline(x=mu0, color=COLORS['prior_mean'], linestyle=':', linewidth=1.5, alpha=0.5,
               label=f'Prior $\\mu_0$={mu0}')

    ax.set_xlabel(r'$\theta$', fontsize=12)
    ax.set_ylabel('p-value', fontsize=12)
    ax.set_title(f'Tilted P-value Functions (D={D}, $\\Delta$={delta:.1f})\n'
                 'Mode shifts from $\\mu_n$ toward D as $\\eta$ increases',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-1, 5)
    ax.grid(True, alpha=0.3)

    # Add arrow showing mode movement
    ax.annotate('', xy=(D-0.1, 0.95), xytext=(mu_n+0.1, 0.95),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text((mu_n + D)/2, 0.88, 'Mode moves', ha='center', fontsize=10)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_2_tilted_pvalue_family", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.3: Non-centrality Reduction (Theorem 7)
# =============================================================================

def figure_5_3_noncentrality_reduction(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.3: Non-centrality Reduction (Theorem 7).

    Shows lambda_eta = (1-eta)^2 * lambda_0.
    """
    print("\n" + "="*60)
    print("Figure 5.3: Non-centrality Reduction")
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extended eta range to show oversharpening
    etas = np.linspace(-1, 1, 200)

    # Relative non-centrality: lambda_eta / lambda_0 = (1-eta)^2
    relative_lambda = (1 - etas) ** 2

    # Plot the full curve
    ax.plot(etas, relative_lambda, color=COLORS['tilted'], linewidth=2.5)

    # Fill regions differently
    ax.fill_between(etas[etas <= 0], 1, relative_lambda[etas <= 0],
                    alpha=0.2, color='purple', label=r'Oversharpening ($\eta < 0$)')
    ax.fill_between(etas[etas >= 0], 0, relative_lambda[etas >= 0],
                    alpha=0.2, color=COLORS['tilted'], label=r'Standard tilting ($\eta \geq 0$)')

    # Reference line at lambda/lambda_0 = 1
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # Mark key points
    key_etas = [-1, -0.5, 0, 0.5, 1.0]
    key_lambdas = [(1-e)**2 for e in key_etas]
    ax.scatter(key_etas, key_lambdas, color='black', s=80, zorder=5)

    for eta, lam in zip(key_etas, key_lambdas):
        offset = (5, 10) if eta >= 0 else (-40, 10)
        ax.annotate(f'({eta}, {lam:.1f})', xy=(eta, lam),
                   xytext=offset, textcoords='offset points', fontsize=10)

    # Add formula annotation
    ax.text(0.0, 2.5, r'$\lambda_\eta = (1-\eta)^2 \cdot \lambda_0$',
           fontsize=14, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Tilting Parameter $\eta$', fontsize=12)
    ax.set_ylabel(r'Relative Non-centrality $\lambda_\eta / \lambda_0$', fontsize=12)
    ax.set_title('Non-centrality vs Tilting Parameter (Theorem 7)\n'
                 r'$\eta < 0$: Oversharpening (increases $\lambda$); $\eta > 0$: Reduces $\lambda$',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 4.2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add interpretations
    ax.annotate(r'$\eta=-1$: Maximum oversharpening' + '\n' + r'($\lambda_\eta = 4\lambda_0$)',
               xy=(-1, 4), xytext=(-0.5, 3.5), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(r'$\eta=0$: WALDO ($\lambda_0$)', xy=(0, 1), xytext=(0.3, 1.5),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(r'$\eta=1$: Wald ($\lambda=0$)', xy=(1, 0), xytext=(0.5, 0.3),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_3_noncentrality_reduction", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.4: Optimal Tilting eta*(|Delta|)
# =============================================================================

def figure_5_4_optimal_tilting(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.4: Optimal Tilting eta*(|Delta|).

    The key practical result - how much to tilt.
    Uses numerically computed optimal eta from precomputed grid.
    """
    print("\n" + "="*60)
    print("Figure 5.4: Optimal Tilting eta*(|Delta|)")
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
        save_figure(fig, "fig_5_4_optimal_tilting", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.5: Expected Width Ratio
# =============================================================================

def figure_5_5_width_ratio(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.5: Expected Width Ratio.

    Shows E[W_eta*] / W_Wald as function of |Delta|.
    Uses raw D samples from simulation infrastructure.
    """
    print("\n" + "="*60)
    print("Figure 5.5: Expected Width Ratio")
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
        save_figure(fig, "fig_5_5_width_ratio", "tilting")

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
    parser.add_argument("--figure", type=str, help="Generate only specific figure (5.1-5.5)")
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

    figures_to_generate = ['5.1', '5.2', '5.3', '5.4', '5.5']
    if args.figure:
        figures_to_generate = [args.figure]

    if '5.1' in figures_to_generate:
        figure_5_1_tilting_space(save=save, show=show)

    if '5.2' in figures_to_generate:
        figure_5_2_tilted_pvalue_family(save=save, show=show)

    if '5.3' in figures_to_generate:
        figure_5_3_noncentrality_reduction(save=save, show=show)

    if '5.4' in figures_to_generate:
        figure_5_4_optimal_tilting(save=save, show=show, fast=fast)

    if '5.5' in figures_to_generate:
        figure_5_5_width_ratio(save=save, show=show, fast=fast)

    print("\n" + "="*60)
    print("DONE - Tilting figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
