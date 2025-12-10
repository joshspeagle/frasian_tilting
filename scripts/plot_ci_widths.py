#!/usr/bin/env python3
"""
CI Width Analysis Figures (Category 4)

Generates figures 4.1-4.3:
- Figure 4.1: CI Width Comparison Bar Chart
- Figure 4.2: CI Width vs Prior-Data Conflict
- Figure 4.3: CI Asymmetry Visualization

These figures use analytical formulas (no Monte Carlo).

Usage:
    python scripts/plot_ci_widths.py [--no-save] [--show]
    python scripts/plot_ci_widths.py --fast  # Same output, for consistency
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from frasian.core import posterior_params
from frasian.waldo import (
    confidence_interval,
    confidence_interval_width,
    wald_ci_width,
    posterior_ci_width,
    ci_asymmetry,
)
from frasian.figure_style import (
    COLORS, setup_style, save_figure,
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_params(w: float, sigma: float = 1.0, mu0: float = 0.0):
    """Get model parameters for a given weight."""
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return mu0, sigma, sigma0


def data_for_conflict(delta: float, mu0: float, w: float, sigma: float) -> float:
    """Compute D value that produces a given Delta."""
    # Delta = (1-w)(mu0 - D) / sigma
    # D = mu0 - sigma * Delta / (1-w)
    return mu0 - sigma * delta / (1 - w)


# =============================================================================
# Figure 4.1: CI Width Comparison Bar Chart
# =============================================================================

def figure_4_1_ci_width_bars(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 4.1: CI Width Comparison Bar Chart.

    Side-by-side comparison of Posterior, WALDO, Wald widths
    at different conflict levels.
    """
    print("\n" + "="*60)
    print("Figure 4.1: CI Width Comparison Bar Chart")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    # Conflict levels to show
    deltas = [0, -1, -2.5, -5]
    delta_labels = ['0', '-1', '-2.5', '-5']

    # Compute widths
    w_wald = wald_ci_width(sigma)
    w_post = posterior_ci_width(sigma, sigma0)

    waldo_widths = []
    for delta in deltas:
        D = data_for_conflict(delta, mu0, w, sigma)
        width = confidence_interval_width(D, mu0, sigma, sigma0)
        waldo_widths.append(width)

    print(f"Wald width: {w_wald:.2f}")
    print(f"Posterior width: {w_post:.2f}")
    print(f"WALDO widths at Delta={deltas}: {[f'{w:.2f}' for w in waldo_widths]}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(deltas))
    bar_width = 0.25

    # Plot bars
    bars1 = ax.bar(x - bar_width, [w_post] * len(deltas), bar_width,
                   label='Posterior', color=COLORS['posterior'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, waldo_widths, bar_width,
                   label='WALDO', color=COLORS['waldo'], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + bar_width, [w_wald] * len(deltas), bar_width,
                   label='Wald', color=COLORS['wald'], edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    # Reference line at Wald width
    ax.axhline(y=w_wald, color=COLORS['wald'], linestyle='--', alpha=0.5,
               linewidth=1.5, label=f'Wald reference ({w_wald:.2f})')

    # Formatting
    ax.set_xlabel(r'Prior-Data Conflict $\Delta$', fontsize=12)
    ax.set_ylabel('CI Width', fontsize=12)
    ax.set_title('Confidence Interval Width Comparison\n'
                 'WALDO width grows with conflict; trades width for correct coverage',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(delta_labels)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 10)
    ax.grid(True, axis='y', alpha=0.3)

    # Add annotation
    ax.annotate('Posterior is narrowest\nbut has wrong coverage!',
               xy=(0 - bar_width, w_post), xytext=(-1.5, 5),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_4_1_ci_width_bars", "widths")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 4.2: CI Width vs Prior-Data Conflict
# =============================================================================

def figure_4_2_width_vs_conflict(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 4.2: CI Width vs Prior-Data Conflict.

    Shows how WALDO CI width varies with |Δ|, with constant
    Wald and Posterior references.
    """
    print("\n" + "="*60)
    print("Figure 4.2: CI Width vs Prior-Data Conflict")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    # Range of Delta values
    deltas = np.linspace(-5, 5, 101)

    # Compute constant widths
    w_wald = wald_ci_width(sigma)
    w_post = posterior_ci_width(sigma, sigma0)

    # Compute WALDO widths
    waldo_widths = []
    for delta in deltas:
        D = data_for_conflict(delta, mu0, w, sigma)
        width = confidence_interval_width(D, mu0, sigma, sigma0)
        waldo_widths.append(width)

    # Find crossover point (where WALDO = Wald)
    crossover_idx = np.argmin(np.abs(np.array(waldo_widths) - w_wald))
    crossover_delta = deltas[crossover_idx]

    print(f"Wald width: {w_wald:.2f}")
    print(f"Posterior width: {w_post:.2f}")
    print(f"WALDO crosses Wald at |Delta| ~ {abs(crossover_delta):.2f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot WALDO curve
    ax.plot(deltas, waldo_widths, color=COLORS['waldo'], linewidth=2.5,
            label='WALDO', zorder=3)

    # Plot constant references
    ax.axhline(y=w_wald, color=COLORS['wald'], linestyle='--', linewidth=2,
               label=f'Wald ({w_wald:.2f})')
    ax.axhline(y=w_post, color=COLORS['posterior'], linestyle=':', linewidth=2,
               label=f'Posterior ({w_post:.2f})')

    # Mark crossover points
    ax.scatter([crossover_delta, -crossover_delta],
               [w_wald, w_wald], color='black', s=80, zorder=5,
               label=f'Crossover (|Δ| ≈ {abs(crossover_delta):.1f})')

    # Shade regions
    ax.fill_between(deltas, w_post, waldo_widths,
                    where=np.array(waldo_widths) <= w_wald,
                    alpha=0.1, color=COLORS['waldo'],
                    label='WALDO more efficient than Wald')

    # Mark key points from document
    key_deltas = [0, -1, -2.5]
    key_widths = [3.29, 3.63, 5.53]
    ax.scatter(key_deltas, key_widths, color='red', s=60, marker='s', zorder=5,
               label='Reference values')
    for d, w_val in zip(key_deltas, key_widths):
        ax.annotate(f'({d}, {w_val})', xy=(d, w_val), xytext=(5, 5),
                   textcoords='offset points', fontsize=9)

    # Formatting
    ax.set_xlabel(r'Prior-Data Conflict $\Delta = (1-w)(\mu_0 - D)/\sigma$', fontsize=12)
    ax.set_ylabel('CI Width', fontsize=12)
    ax.set_title('CI Width vs Prior-Data Conflict\n'
                 'Width penalty is the cost of frequentist validity',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3)

    # Add annotation about asymptotic behavior
    ax.annotate('WALDO width grows\nlinearly with |Δ|',
               xy=(4, 8), fontsize=10, ha='center', color=COLORS['waldo'])

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_4_2_width_vs_conflict", "widths")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 4.3: CI Asymmetry Visualization
# =============================================================================

def figure_4_3_ci_asymmetry(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 4.3: CI Asymmetry Visualization.

    Shows how CIs become asymmetric with conflict, extending toward MLE.
    """
    print("\n" + "="*60)
    print("Figure 4.3: CI Asymmetry Visualization")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Symmetric CI at Δ=0
    ax1 = axes[0]
    D1 = mu0  # Delta = 0
    mu_n1, sigma_n1, _ = posterior_params(D1, mu0, sigma, sigma0)
    lower1, upper1 = confidence_interval(D1, mu0, sigma, sigma0)

    # Draw CI
    ax1.barh(0, upper1 - lower1, left=lower1, height=0.3,
             color=COLORS['waldo'], alpha=0.7, edgecolor='black')
    ax1.axvline(x=mu_n1, color='black', linestyle='-', linewidth=2, label=f'Mode μₙ={mu_n1:.2f}')
    ax1.axvline(x=D1, color=COLORS['mle'], linestyle='--', linewidth=2, label=f'MLE D={D1:.2f}')
    ax1.axvline(x=mu0, color=COLORS['prior_mean'], linestyle=':', linewidth=2, label=f'Prior μ₀={mu0:.2f}')

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel(r'$\theta$', fontsize=11)
    ax1.set_title(f'Δ = 0 (No Conflict)\nCI: [{lower1:.2f}, {upper1:.2f}]', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_yticks([])

    # Add symmetry annotation
    left_dist = mu_n1 - lower1
    right_dist = upper1 - mu_n1
    ax1.annotate(f'Left: {left_dist:.2f}', xy=(lower1, 0.2), fontsize=9)
    ax1.annotate(f'Right: {right_dist:.2f}', xy=(upper1-0.8, 0.2), fontsize=9)

    # Panel B: Asymmetric CI at Δ=-2
    ax2 = axes[1]
    delta2 = -2.0
    D2 = data_for_conflict(delta2, mu0, w, sigma)
    mu_n2, sigma_n2, _ = posterior_params(D2, mu0, sigma, sigma0)
    lower2, upper2 = confidence_interval(D2, mu0, sigma, sigma0)

    ax2.barh(0, upper2 - lower2, left=lower2, height=0.3,
             color=COLORS['waldo'], alpha=0.7, edgecolor='black')
    ax2.axvline(x=mu_n2, color='black', linestyle='-', linewidth=2, label=f'Mode μₙ={mu_n2:.2f}')
    ax2.axvline(x=D2, color=COLORS['mle'], linestyle='--', linewidth=2, label=f'MLE D={D2:.2f}')
    ax2.axvline(x=mu0, color=COLORS['prior_mean'], linestyle=':', linewidth=2, label=f'Prior μ₀={mu0:.2f}')

    ax2.set_xlim(-2, 6)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel(r'$\theta$', fontsize=11)
    ax2.set_title(f'Δ = {delta2} (Moderate Conflict)\nCI: [{lower2:.2f}, {upper2:.2f}]', fontsize=11)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_yticks([])

    # Add asymmetry annotation
    left_dist2 = mu_n2 - lower2
    right_dist2 = upper2 - mu_n2
    ax2.annotate(f'Left: {left_dist2:.2f}', xy=(lower2, 0.2), fontsize=9)
    ax2.annotate(f'Right: {right_dist2:.2f}', xy=(upper2-0.8, 0.2), fontsize=9)
    ax2.annotate('Extends toward MLE →', xy=(upper2-1, -0.3), fontsize=9,
                color=COLORS['mle'], fontweight='bold')

    # Panel C: Asymmetry measure vs |Δ|
    ax3 = axes[2]
    deltas = np.linspace(-5, 5, 51)
    asymmetries = []

    for delta in deltas:
        D = data_for_conflict(delta, mu0, w, sigma)
        asym = ci_asymmetry(D, mu0, sigma, sigma0)
        asymmetries.append(asym)

    ax3.plot(deltas, asymmetries, color=COLORS['waldo'], linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    # Fill regions to indicate direction
    ax3.fill_between(deltas, 0, asymmetries,
                     where=np.array(asymmetries) > 0,
                     alpha=0.3, color=COLORS['waldo'],
                     label='CI extends toward higher θ')
    ax3.fill_between(deltas, 0, asymmetries,
                     where=np.array(asymmetries) < 0,
                     alpha=0.3, color=COLORS['posterior'],
                     label='CI extends toward lower θ')

    ax3.set_xlabel(r'Prior-Data Conflict $\Delta$', fontsize=11)
    ax3.set_ylabel('Asymmetry (upper - lower extension)', fontsize=11)
    ax3.set_title('CI Asymmetry vs Conflict\nCI reaches toward MLE (D)', fontsize=11)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('WALDO CIs Become Asymmetric When Prior-Data Conflict Exists',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_4_3_ci_asymmetry", "widths")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate CI width figures")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--fast", action="store_true", help="Fast mode (same output, for consistency)")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (4.1, 4.2, 4.3)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show

    print("="*60)
    print("CI WIDTH ANALYSIS FIGURES")
    print("(Formula-based, no Monte Carlo)")
    print("="*60)

    figures_to_generate = ['4.1', '4.2', '4.3']
    if args.figure:
        figures_to_generate = [args.figure]

    if '4.1' in figures_to_generate:
        figure_4_1_ci_width_bars(save=save, show=show)

    if '4.2' in figures_to_generate:
        figure_4_2_width_vs_conflict(save=save, show=show)

    if '4.3' in figures_to_generate:
        figure_4_3_ci_asymmetry(save=save, show=show)

    print("\n" + "="*60)
    print("DONE - CI width figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
