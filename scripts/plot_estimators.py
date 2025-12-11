#!/usr/bin/env python3
"""
Estimator Properties Figures (Category 4)

Generates figures 4.1-4.3:
- Figure 4.1: CD Mean Comparison Across Methods
- Figure 4.2: CD Mode Comparison Across Methods
- Figure 4.3: Estimator Ordering vs Conflict Level

Usage:
    python scripts/plot_estimators.py [--no-save] [--show] [--fast]
    python scripts/plot_estimators.py --figure 4.1
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from frasian.core import posterior_params, scaled_conflict
from frasian.waldo import pvalue
from frasian.confidence import (
    wald_cd_mean,
    wald_cd_mode,
    waldo_cd_mean,
    waldo_cd_mode,
    dynamic_cd_mean,
    dynamic_cd_mode,
)
from frasian.tilting import tilted_params, optimal_eta_mlp
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
# Figure 4.1: CD Mean Comparison Across Methods
# =============================================================================

def figure_4_1_cd_mean(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 4.1: CD Mean Comparison Across Methods.

    Shows how the mean estimator from each CD varies with prior-data conflict.

    - Wald CD mean = D (always at MLE)
    - Posterior mean = μ_n (Bayesian estimate)
    - WALDO CD mean = (μ_n + (1-w)D)/(2-w) (between μ_n and D)
    - Dynamic CD mean (numerical, adapts to conflict)
    """
    print("\n" + "="*60)
    print("Figure 4.1: CD Mean Comparison Across Methods")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ==========================================================================
    # Panel A: Mean estimators as function of D
    # ==========================================================================
    ax1 = axes[0]

    D_values = np.linspace(-2, 5, 30)  # Reduced for faster dynamic computation

    wald_means = []
    posterior_means = []
    waldo_means = []
    dynamic_means = []

    print("  Computing mean estimators...")
    for i, D in enumerate(D_values):
        # Wald CD mean = D
        wald_means.append(wald_cd_mean(D))

        # Posterior mean = μ_n
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        posterior_means.append(mu_n)

        # WALDO CD mean (closed form)
        waldo_means.append(waldo_cd_mean(D, mu0, sigma, sigma0))

        # Dynamic CD mean (numerical)
        dynamic_means.append(dynamic_cd_mean(D, mu0, sigma, sigma0, n_grid=300))

    # Plot all mean estimators
    ax1.plot(D_values, wald_means, color=COLORS['wald'], linewidth=2.5,
             label='Wald CD mean = D')
    ax1.plot(D_values, posterior_means, color=COLORS['posterior'], linewidth=2.5,
             label=r'Posterior mean = $\mu_n$')
    ax1.plot(D_values, waldo_means, color=COLORS['waldo'], linewidth=2.5,
             label='WALDO CD mean')
    ax1.plot(D_values, dynamic_means, color=COLORS['tilted'], linewidth=2,
             linestyle='--', label='Dynamic CD mean')

    # Reference lines
    ax1.axhline(y=mu0, color=COLORS['prior_mean'], linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Prior mean $\\mu_0$ = {mu0}')
    ax1.plot(D_values, D_values, 'k--', linewidth=1, alpha=0.5, label='45° line (D)')

    ax1.set_xlabel('Observed data D', fontsize=12)
    ax1.set_ylabel('Mean estimator', fontsize=12)
    ax1.set_title('(A) CD Mean vs Observed Data\n'
                  'WALDO & Dynamic means lie between posterior mean and MLE',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 5)

    # Add formula box
    ax1.text(0.98, 0.02,
             r'WALDO mean: $\frac{\mu_n + (1-w)D}{2-w}$',
             transform=ax1.transAxes, fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ==========================================================================
    # Panel B: Normalized mean position vs conflict level
    # ==========================================================================
    ax2 = axes[1]

    deltas = np.linspace(0.1, 4, 25)  # Reduced for faster computation

    # Compute normalized positions: 0 = mu_n (posterior), 1 = D (MLE)
    waldo_norm = []
    dynamic_norm = []

    print("  Computing normalized positions vs conflict...")
    for delta in deltas:
        # D > mu0 case (positive conflict)
        D = data_for_conflict(-delta, mu0, w, sigma)
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

        # Span from μ_n to D
        span = D - mu_n if abs(D - mu_n) > 1e-6 else 1.0

        # WALDO CD mean
        waldo_mean = waldo_cd_mean(D, mu0, sigma, sigma0)
        waldo_norm.append((waldo_mean - mu_n) / span)

        # Dynamic CD mean (numerical)
        dyn_mean = dynamic_cd_mean(D, mu0, sigma, sigma0, n_grid=300)
        dynamic_norm.append((dyn_mean - mu_n) / span)

    # Plot normalized positions
    ax2.axhline(y=0, color=COLORS['posterior'], linestyle='-', linewidth=2,
                label=r'$\mu_n$ (posterior mean = 0)')
    ax2.axhline(y=1, color=COLORS['wald'], linestyle='-', linewidth=2,
                label=r'$D$ (MLE = 1)')

    ax2.plot(deltas, waldo_norm, color=COLORS['waldo'], linewidth=2.5,
             label='WALDO CD mean')
    ax2.plot(deltas, dynamic_norm, color=COLORS['tilted'], linewidth=2, linestyle='--',
             label='Dynamic CD mean')

    # The WALDO mean position should be approximately constant for fixed w
    # E[θ] = (μ_n + (1-w)D)/(2-w)
    # Normalized: (E[θ] - μ_n) / (D - μ_n) = (1-w)/(2-w) for D > μ_n
    theoretical_pos = (1 - w) / (2 - w)
    ax2.axhline(y=theoretical_pos, color='gray', linestyle=':', linewidth=1.5,
                label=f'$(1-w)/(2-w) = {theoretical_pos:.2f}$')

    ax2.fill_between(deltas, 0, waldo_norm, alpha=0.15, color=COLORS['waldo'])

    ax2.set_xlabel(r'Prior-data conflict $|\Delta|$', fontsize=12)
    ax2.set_ylabel('Normalized position (0=μ_n, 1=D)', fontsize=12)
    ax2.set_title('(B) Mean Position vs Conflict Level\n'
                  'WALDO mean is pulled toward MLE by constant fraction',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlim(0, 4)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('CD Mean Estimators: Each Method Has a Different Mean\n'
                 'Wald: D | Posterior: μ_n | WALDO & Dynamic: between',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_4_1_cd_mean", "estimators")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 4.2: CD Mode Comparison Across Methods
# =============================================================================

def figure_4_2_cd_mode(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 4.2: CD Mode Comparison Across Methods.

    Shows how the mode estimator from each CD varies with prior-data conflict.

    Key insight: WALDO mode = μ_n (posterior mean) - the Bayesian estimator
    emerges as the frequentist modal estimator!
    """
    print("\n" + "="*60)
    print("Figure 4.2: CD Mode Comparison Across Methods")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ==========================================================================
    # Panel A: P-value curves showing mode = mu_n for multiple D values
    # ==========================================================================
    ax1 = axes[0]

    # Multiple D values to show how mode shifts
    D_values = [0, 2, 4]
    colors = [plt.cm.viridis(x) for x in [0.2, 0.5, 0.8]]

    theta_range = np.linspace(-2, 6, 300)

    for D, color in zip(D_values, colors):
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)

        # Compute p-value curve
        pvals = np.array([pvalue(t, mu_n, mu0, w, sigma) for t in theta_range])

        # Plot curve
        ax1.plot(theta_range, pvals, color=color, linewidth=2.5,
                label=f'D={D}')

        # Mark mode at mu_n (peak of p-value)
        ax1.scatter([mu_n], [1.0], color=color, s=120, zorder=5,
                   edgecolor='black', linewidth=2, marker='o')

        # Mark MLE at D
        p_at_D = pvalue(D, mu_n, mu0, w, sigma)
        ax1.scatter([D], [p_at_D], color=color, s=100, zorder=5,
                   marker='x', linewidth=3)

    # Reference lines
    ax1.axhline(y=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=r'$\alpha=0.05$')
    ax1.axvline(x=mu0, color=COLORS['prior_mean'], linestyle=':', linewidth=2, alpha=0.7,
               label=f'Prior mean $\\mu_0$={mu0}')

    # Annotation explaining the key insight
    ax1.annotate('Mode always at $\\mu_n$\n(Bayesian posterior mean\nemerges as frequentist mode)',
                xy=(3.5, 0.65), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

    # Legend for markers
    ax1.plot([], [], 'ko', markersize=8, label='Mode ($\\mu_n$)')
    ax1.plot([], [], 'kx', markersize=8, markeredgewidth=2, label='MLE (D)')

    ax1.set_xlabel(r'$\theta$', fontsize=12)
    ax1.set_ylabel('p-value', fontsize=12)
    ax1.set_title('(A) P-value Curves: Mode = Posterior Mean\n'
                  'Each curve peaks at $\\mu_n$, not at MLE D',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlim(-2, 6)
    ax1.grid(True, alpha=0.3)

    # ==========================================================================
    # Panel B: Mode positions for all methods
    # ==========================================================================
    ax2 = axes[1]

    D_values_full = np.linspace(-2, 5, 30)  # Reduced for dynamic computation

    wald_modes = []
    posterior_modes = []
    waldo_modes = []
    dynamic_modes = []

    print("  Computing mode estimators...")
    for D in D_values_full:
        wald_modes.append(wald_cd_mode(D))  # = D

        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        posterior_modes.append(mu_n)  # = μ_n

        waldo_modes.append(waldo_cd_mode(D, mu0, sigma, sigma0))  # = μ_n

        # Dynamic CD mode (numerical)
        dynamic_modes.append(dynamic_cd_mode(D, mu0, sigma, sigma0, n_grid=300))

    # Plot all mode estimators
    ax2.plot(D_values_full, wald_modes, color=COLORS['wald'], linewidth=2.5,
             label='Wald CD mode = D')
    ax2.plot(D_values_full, posterior_modes, color=COLORS['posterior'], linewidth=2.5,
             label=r'Posterior mode = $\mu_n$')
    ax2.plot(D_values_full, waldo_modes, color=COLORS['waldo'], linewidth=2.5,
             linestyle='--', label=r'WALDO CD mode = $\mu_n$')
    ax2.plot(D_values_full, dynamic_modes, color=COLORS['tilted'], linewidth=2,
             linestyle=':', label='Dynamic CD mode')

    # Reference
    ax2.axhline(y=mu0, color=COLORS['prior_mean'], linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Prior mean $\\mu_0$ = {mu0}')
    ax2.plot(D_values_full, D_values_full, 'k--', linewidth=1, alpha=0.5, label='45° line')

    # Highlight the key insight
    ax2.annotate('WALDO mode = Posterior mode\n(Bayesian-frequentist bridge!)',
                xy=(3, posterior_modes[int(0.7*len(D_values_full))]),
                xytext=(4, 0),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    ax2.set_xlabel('Observed data D', fontsize=12)
    ax2.set_ylabel('Mode estimator', fontsize=12)
    ax2.set_title('(B) CD Mode vs Observed Data\n'
                  'WALDO & Dynamic modes track posterior mean',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 5)

    fig.suptitle('CD Mode Estimators: WALDO Mode = Posterior Mean\n'
                 'The Bayesian estimator emerges as the frequentist modal point',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_4_2_cd_mode", "estimators")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 4.3: Estimator Ordering vs Conflict Level
# =============================================================================

def figure_4_3_estimator_ordering(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 4.3: Estimator Ordering vs Conflict Level.

    Panel A: Mean estimators (normalized position) vs |Δ|
    Panel B: Number line showing ordering for a specific case

    Key insight: Mode = μ_n for all Bayesian-frequentist hybrids,
    but mean varies: μ_n < WALDO mean < D
    """
    print("\n" + "="*60)
    print("Figure 4.3: Estimator Ordering vs Conflict Level")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ==========================================================================
    # Panel A: Estimator positions as function of |Delta|
    # ==========================================================================
    ax1 = axes[0]

    deltas = np.linspace(0.1, 4, 20)  # Reduced for faster dynamic computation

    # Normalized positions: 0 = μ₀ (prior), 1 = D (MLE)
    # This normalization shows how estimators position relative to prior-data span

    posterior_norm = []
    waldo_mode_norm = []
    waldo_mean_norm = []
    dynamic_mode_norm = []
    dynamic_mean_norm = []

    print("  Computing normalized estimator positions...")
    for delta in deltas:
        # D > mu0 case (positive conflict)
        D = data_for_conflict(-delta, mu0, w, sigma)
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

        # Span from prior to MLE
        span = D - mu0 if abs(D - mu0) > 1e-6 else 1.0

        # Posterior mean (= WALDO mode)
        posterior_norm.append((mu_n - mu0) / span)

        # WALDO mode (= mu_n)
        waldo_mode_norm.append((waldo_cd_mode(D, mu0, sigma, sigma0) - mu0) / span)

        # WALDO mean
        waldo_mean_val = waldo_cd_mean(D, mu0, sigma, sigma0)
        waldo_mean_norm.append((waldo_mean_val - mu0) / span)

        # Dynamic CD mode and mean (numerical)
        dyn_mode = dynamic_cd_mode(D, mu0, sigma, sigma0, n_grid=300)
        dyn_mean = dynamic_cd_mean(D, mu0, sigma, sigma0, n_grid=300)
        dynamic_mode_norm.append((dyn_mode - mu0) / span)
        dynamic_mean_norm.append((dyn_mean - mu0) / span)

    # Plot normalized positions
    ax1.axhline(y=0, color=COLORS['prior_mean'], linestyle='-', linewidth=2,
                label=r'Prior mean $\mu_0$ (0)')
    ax1.axhline(y=1, color=COLORS['wald'], linestyle='-', linewidth=2,
                label=r'MLE $D$ (1)')

    ax1.plot(deltas, posterior_norm, color=COLORS['posterior'], linewidth=2.5,
             label=r'Posterior mean = $\mu_n$')
    ax1.plot(deltas, waldo_mean_norm, color=COLORS['waldo'], linewidth=2.5, linestyle='-',
             label='WALDO CD mean')
    ax1.plot(deltas, dynamic_mean_norm, color=COLORS['tilted'], linewidth=2, linestyle='--',
             label='Dynamic CD mean')

    # Shade region between mode and WALDO mean
    ax1.fill_between(deltas, posterior_norm, waldo_mean_norm,
                     alpha=0.15, color=COLORS['waldo'],
                     label='WALDO mode-mean gap')

    # Mark the constant normalized position of posterior mean = w
    ax1.axhline(y=w, color='gray', linestyle=':', linewidth=1.5,
                label=f'$w = {w}$ (posterior weight)')

    # Annotation
    ax1.annotate('Mode = posterior mean\n(constant fraction w)',
                xy=(2, posterior_norm[len(deltas)//2]),
                xytext=(3, 0.3), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel(r'Prior-data conflict $|\Delta|$', fontsize=12)
    ax1.set_ylabel('Normalized position (0=μ₀, 1=D)', fontsize=12)
    ax1.set_title('(A) Estimator Positions vs Conflict\n'
                  'Mean > Mode = posterior mean',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(0, 4)
    ax1.grid(True, alpha=0.3)

    # ==========================================================================
    # Panel B: Number line showing ordering for specific D
    # ==========================================================================
    ax2 = axes[1]

    # High-conflict case to show clear ordering
    D = 4.0
    mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)
    delta = abs(scaled_conflict(D, mu0, w, sigma))

    # Compute all estimators
    waldo_mean = waldo_cd_mean(D, mu0, sigma, sigma0)
    waldo_mode = waldo_cd_mode(D, mu0, sigma, sigma0)
    dyn_mean = dynamic_cd_mean(D, mu0, sigma, sigma0, n_grid=300)
    dyn_mode = dynamic_cd_mode(D, mu0, sigma, sigma0, n_grid=300)

    print(f"  D={D}: mu_n={mu_n:.2f}, WALDO mean={waldo_mean:.2f}, WALDO mode={waldo_mode:.2f}")
    print(f"         Dynamic mean={dyn_mean:.2f}, Dynamic mode={dyn_mode:.2f}")

    # Draw number line
    ax2.axhline(y=0.5, color='black', linewidth=3, zorder=1)
    ax2.set_xlim(-0.5, 5)
    ax2.set_ylim(0, 1)

    # Estimator positions and labels (show key estimators)
    estimators = [
        (mu0, r'$\mu_0$', COLORS['prior_mean'], 'Prior'),
        (waldo_mode, r'Mode', COLORS['posterior'], f'Mode\n$\\mu_n$={waldo_mode:.2f}'),
        (waldo_mean, r'WALDO', COLORS['waldo'], f'WALDO mean\n{waldo_mean:.2f}'),
        (dyn_mean, r'Dyn', COLORS['tilted'], f'Dyn mean\n{dyn_mean:.2f}'),
        (D, r'$D$', COLORS['mle'], 'MLE'),
    ]

    y_offsets = [0.25, 0.78, 0.22, 0.78, 0.22]  # Alternating above/below for clarity

    for (x, symbol, color, desc), y_offset in zip(estimators, y_offsets):
        # Marker on number line
        ax2.scatter([x], [0.5], color=color, s=150, zorder=5, edgecolor='black', linewidth=2)

        # Label with value
        ax2.annotate(f'{symbol}\n{desc}',
                    xy=(x, 0.5), xytext=(x, y_offset),
                    fontsize=8, ha='center', va='center',
                    color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=color, lw=1.5))

    # Draw bracket showing the key ordering
    bracket_y = 0.12
    ax2.annotate('', xy=(waldo_mode, bracket_y), xytext=(waldo_mean, bracket_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax2.text((waldo_mode + waldo_mean)/2, bracket_y - 0.06,
             r'Mode < Mean',
             ha='center', fontsize=10, fontweight='bold')

    # Additional annotation
    ax2.text(2.5, 0.92, f'$|\\Delta| = {delta:.2f}$\n$w = {w}$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax2.set_xlabel(r'$\theta$', fontsize=12)
    ax2.set_title(f'(B) Estimator Ordering (D={D})\n'
                  r'$\mu_0 < \mu_n = $ Mode < Means < D',
                  fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    fig.suptitle('Estimator Ordering: Mode = Posterior Mean, Means Pulled Toward MLE\n'
                 r'WALDO & Dynamic CDs have mode at $\mu_n$ but mean between $\mu_n$ and $D$',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_4_3_estimator_ordering", "estimators")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate estimator figures")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--fast", action="store_true", help="Fast mode (no effect here)")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (4.1, 4.2, 4.3)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show

    print("="*60)
    print("ESTIMATOR PROPERTIES FIGURES")
    print("="*60)

    figures_to_generate = ['4.1', '4.2', '4.3']
    if args.figure:
        figures_to_generate = [args.figure]

    if '4.1' in figures_to_generate:
        figure_4_1_cd_mean(save=save, show=show)

    if '4.2' in figures_to_generate:
        figure_4_2_cd_mode(save=save, show=show)

    if '4.3' in figures_to_generate:
        figure_4_3_estimator_ordering(save=save, show=show)

    print("\n" + "="*60)
    print("DONE - Estimator figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
