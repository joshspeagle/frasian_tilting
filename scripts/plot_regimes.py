#!/usr/bin/env python3
"""
Three Regimes Figures (Category 6)

Generates figures 6.1-6.2:
- Figure 6.1: Regime Structure Diagram (Theorem 9)
- Figure 6.2: Estimator Hierarchy (Section 11.3)

Usage:
    python scripts/plot_regimes.py [--no-save] [--show]
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from frasian.core import posterior_params, scaled_conflict
from frasian.waldo import pvalue
from frasian.confidence import pvalue_mean
from frasian.tilting import tilted_params, dynamic_tilted_mode
from frasian.simulations import optimal_eta_empirical
from frasian.figure_style import (
    COLORS, FIGSIZE, setup_style, save_figure,
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
    return mu0 - sigma * delta / (1 - w)


# =============================================================================
# Figure 6.1: Regime Structure Diagram (Theorem 9)
# =============================================================================

def figure_6_1_regime_structure(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 6.1: Regime Structure Diagram (Theorem 9).

    Visualizes Low/Transition/High |Delta| regimes.
    """
    print("\n" + "="*60)
    print("Figure 6.1: Regime Structure Diagram (Theorem 9)")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: eta* with regime coloring
    ax1 = axes[0]

    deltas = np.linspace(0, 5, 200)
    etas = [optimal_eta_empirical(d, fast=fast) for d in deltas]

    # Define regime boundaries
    low_high = 0.5
    trans_high = 2.0

    # Color by regime
    low_mask = deltas <= low_high
    trans_mask = (deltas > low_high) & (deltas <= trans_high)
    high_mask = deltas > trans_high

    # Background shading
    ax1.axvspan(0, low_high, alpha=0.2, color=COLORS['waldo'], label='Low |Δ|: Near-WALDO')
    ax1.axvspan(low_high, trans_high, alpha=0.2, color=COLORS['tilted'], label='Transition')
    ax1.axvspan(trans_high, 5, alpha=0.2, color=COLORS['wald'], label='High |Δ|: Near-Wald')

    # Plot eta* curve
    ax1.plot(deltas, etas, color='black', linewidth=2.5, zorder=5)

    # Mark regime boundaries
    for boundary, label in [(low_high, '|Δ|=0.5'), (trans_high, '|Δ|=2.0')]:
        ax1.axvline(x=boundary, color='gray', linestyle='--', linewidth=1)
        eta_at_bound = optimal_eta_empirical(boundary, fast=fast)
        ax1.scatter([boundary], [eta_at_bound], color='black', s=60, zorder=6)
        ax1.annotate(f'{label}\nη*={eta_at_bound:.2f}', xy=(boundary, eta_at_bound),
                    xytext=(boundary+0.2, eta_at_bound-0.15), fontsize=9)

    # Regime descriptions
    ax1.text(0.25, 0.15, 'Use WALDO\n(η* small)', ha='center', fontsize=10,
            color=COLORS['waldo'], fontweight='bold')
    ax1.text(1.25, 0.5, 'Rapid\ntransition', ha='center', fontsize=10,
            color=COLORS['tilted'], fontweight='bold')
    ax1.text(3.5, 0.85, 'Use Wald\n(η* → 1)', ha='center', fontsize=10,
            color=COLORS['wald'], fontweight='bold')

    ax1.set_xlabel(r'Prior-Data Conflict $|\Delta|$', fontsize=12)
    ax1.set_ylabel(r'Optimal Tilting $\eta^*$', fontsize=12)
    ax1.set_title('Theorem 9: Three-Regime Structure', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Panel B: Behavior in each regime
    ax2 = axes[1]

    # Create visual table
    regimes = [
        ('Low', '|Δ| < 0.5', 'η* ≈ 0.2-0.5', 'Near WALDO', 'Narrowest CIs'),
        ('Transition', '0.5 < |Δ| < 2', 'η* ≈ 0.5-0.9', 'Interpolating', 'Adaptive'),
        ('High', '|Δ| > 2', 'η* ≈ 0.9-1.0', 'Near Wald', 'Conservative'),
    ]

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 4)
    ax2.axis('off')

    # Header
    headers = ['Regime', 'Range', 'η*', 'Behavior', 'CI Width']
    for i, h in enumerate(headers):
        ax2.text(i*2 + 1, 3.7, h, ha='center', va='center', fontweight='bold', fontsize=11)

    # Rows
    row_colors = [COLORS['waldo'], COLORS['tilted'], COLORS['wald']]
    for row_idx, (regime, range_str, eta_str, behavior, width) in enumerate(regimes):
        y = 2.8 - row_idx * 0.8
        color = row_colors[row_idx]

        # Background
        rect = Rectangle((0.1, y-0.3), 9.8, 0.6, facecolor=color, alpha=0.2)
        ax2.add_patch(rect)

        # Data
        data = [regime, range_str, eta_str, behavior, width]
        for i, d in enumerate(data):
            ax2.text(i*2 + 1, y, d, ha='center', va='center', fontsize=10)

    ax2.set_title('Regime Characteristics', fontsize=13, fontweight='bold', y=1.05)

    fig.suptitle('The Three Regimes of Optimal Tilting',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_6_1_regime_structure", "regimes")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 6.2: Estimator Hierarchy (Section 11.3)
# =============================================================================

def figure_6_2_estimator_hierarchy(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 6.2: Estimator Hierarchy (Section 11.3).

    Shows ordering of estimators between mu_n and D.
    """
    print("\n" + "="*60)
    print("Figure 6.2: Estimator Hierarchy (Section 11.3)")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Panel A: Number line for multiple |Delta| values
    ax1 = axes[0]

    D_values = [1, 2, 3, 4]
    y_positions = [3, 2, 1, 0]

    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.5, 4)

    for D, y in zip(D_values, y_positions):
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        delta = abs(scaled_conflict(D, mu0, w, sigma))

        # Compute estimators
        try:
            mean_waldo = pvalue_mean(D, mu0, sigma, sigma0, method='closed_form')
        except:
            mean_waldo = (mu_n + D) / 2

        # Tilted mode (using empirical optimal eta)
        eta_star = optimal_eta_empirical(delta, fast=fast)
        mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta_star)
        mode_tilt = mu_eta

        # Draw line segment
        ax1.plot([mu_n, D], [y, y], 'k-', linewidth=1, alpha=0.5)

        # Mark estimators
        markers = [
            (mu0, 'gray', 'o', 'Prior'),
            (mu_n, COLORS['waldo'], 's', 'Mode (μn)'),
            (mode_tilt, COLORS['tilted'], '^', 'Tilt Mode'),
            (mean_waldo, 'purple', 'd', 'Mean'),
            (D, COLORS['mle'], 'x', 'MLE (D)'),
        ]

        for x, color, marker, label in markers:
            if mu_n <= x <= D or x == mu0:
                ax1.scatter([x], [y], color=color, marker=marker, s=80, zorder=5)

        # Label
        ax1.text(-0.3, y, f'D={D}\n|Δ|={delta:.1f}', ha='right', va='center', fontsize=10)

    # Legend
    for color, marker, label in [
        (COLORS['waldo'], 's', 'Mode $\\mu_n$'),
        (COLORS['tilted'], '^', 'Tilted Mode'),
        ('purple', 'd', 'WALDO Mean'),
        (COLORS['mle'], 'x', 'MLE D'),
    ]:
        ax1.scatter([], [], color=color, marker=marker, s=80, label=label)
    ax1.legend(loc='upper right', fontsize=9)

    ax1.set_xlabel(r'$\theta$', fontsize=12)
    ax1.set_title('Estimator Positions for Various Conflict Levels\n'
                  'All estimators lie between $\\mu_n$ and D', fontsize=12, fontweight='bold')
    ax1.set_yticks([])

    # Panel B: Spacing as function of |Delta|
    ax2 = axes[1]

    deltas = np.linspace(0.1, 3, 30)

    mode_positions = []
    tilt_mode_positions = []
    mean_positions = []
    mle_positions = []

    for delta in deltas:
        D = data_for_conflict(-delta, mu0, w, sigma)
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

        try:
            mean_waldo = pvalue_mean(D, mu0, sigma, sigma0, method='closed_form')
        except:
            mean_waldo = (mu_n + D) / 2

        eta_star = optimal_eta_empirical(delta, fast=fast)
        mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta_star)

        mode_positions.append(mu_n)
        tilt_mode_positions.append(mu_eta)
        mean_positions.append(mean_waldo)
        mle_positions.append(D)

    # Normalize: show fraction of distance from mode to MLE
    mode_positions = np.array(mode_positions)
    mle_positions = np.array(mle_positions)
    span = mle_positions - mode_positions

    tilt_frac = (np.array(tilt_mode_positions) - mode_positions) / span
    mean_frac = (np.array(mean_positions) - mode_positions) / span

    ax2.plot(deltas, np.zeros_like(deltas), 's-', color=COLORS['waldo'],
             linewidth=2, label='Mode $\\mu_n$ (0%)')
    ax2.plot(deltas, tilt_frac, '^-', color=COLORS['tilted'],
             linewidth=2, label='Tilted Mode')
    ax2.plot(deltas, mean_frac, 'd-', color='purple',
             linewidth=2, label='WALDO Mean')
    ax2.plot(deltas, np.ones_like(deltas), 'x-', color=COLORS['mle'],
             linewidth=2, label='MLE D (100%)')

    ax2.fill_between(deltas, tilt_frac, mean_frac, alpha=0.2, color=COLORS['tilted'])

    ax2.set_xlabel(r'$|\Delta|$ (prior-data conflict)', fontsize=12)
    ax2.set_ylabel('Fraction of distance from Mode to MLE', fontsize=11)
    ax2.set_title('How Estimators Spread Between Mode and MLE', fontsize=12, fontweight='bold')
    ax2.legend(loc='right', fontsize=9)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Section 11.3: Estimator Hierarchy\n'
                 r'$\mu_n \leq$ Mode$_{tilt} \leq$ Mean$_{WALDO} \leq D$',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_6_2_estimator_hierarchy", "regimes")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate regime figures")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--fast", action="store_true", help="Use fast config")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (6.1, 6.2)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    fast = args.fast

    print("="*60)
    print("THREE REGIMES FIGURES")
    print("="*60)

    figures_to_generate = ['6.1', '6.2']
    if args.figure:
        figures_to_generate = [args.figure]

    if '6.1' in figures_to_generate:
        figure_6_1_regime_structure(save=save, show=show, fast=fast)

    if '6.2' in figures_to_generate:
        figure_6_2_estimator_hierarchy(save=save, show=show, fast=fast)

    print("\n" + "="*60)
    print("DONE - Regime figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
