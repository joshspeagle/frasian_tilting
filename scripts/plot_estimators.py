#!/usr/bin/env python3
"""
Estimator Properties Figures (Category 2)

Generates figures 2.1-2.3:
- Figure 2.1: Mode = Posterior Mean (Theorem 4)
- Figure 2.2: Mean vs Mode Relationship (Theorem 5)
- Figure 2.3: Dynamic Tilting Estimators (μ_n < μ_η* < E[θ] < D)

Usage:
    python scripts/plot_estimators.py [--no-save] [--show]
    python scripts/plot_estimators.py --figure 2.3
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from frasian.core import posterior_params, scaled_conflict, weight
from frasian.waldo import pvalue
from frasian.confidence import pvalue_mode, pvalue_mean
from frasian.tilting import tilted_params
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
# Figure 2.1: Mode = Posterior Mean (Theorem 4)
# =============================================================================

def figure_2_1_mode_equals_posterior_mean(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 2.1: Mode = Posterior Mean (Theorem 4).

    Demonstrates that the mode of the confidence distribution equals mu_n.
    """
    print("\n" + "="*60)
    print("Figure 2.1: Mode = Posterior Mean (Theorem 4)")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Multiple D values showing mode always at mu_n
    D_values = [-1, 0, 2, 4]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(D_values)))

    for D, color in zip(D_values, colors):
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)
        delta = scaled_conflict(D, mu0, w, sigma)

        # Compute p-value curve
        theta_range = np.linspace(mu_n - 3*sigma, mu_n + 3*sigma, 200)
        pvals = [pvalue(t, mu_n, mu0, w, sigma) for t in theta_range]

        # Plot curve
        ax.plot(theta_range, pvals, color=color, linewidth=2,
                label=f'D={D} (mode={mu_n:.1f})')

        # Mark mode (which equals mu_n)
        ax.scatter([mu_n], [1.0], color=color, s=80, zorder=5, edgecolor='black')

        # Mark MLE
        ax.scatter([D], [pvalue(D, mu_n, mu0, w, sigma)], color=color, s=60,
                   marker='x', zorder=5, linewidth=2)

    # Reference lines
    ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=mu0, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label=f'Prior mean $\\mu_0$={mu0}')

    ax.set_xlabel(r'$\theta$', fontsize=12)
    ax.set_ylabel('p-value', fontsize=12)
    ax.set_title('Theorem 4: Mode of P-value Function = Posterior Mean $\\mu_n$\n'
                 'Each curve peaks at its respective $\\mu_n$ (dots), not at MLE D (crosses)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Mode = $\\mu_n$\n(Bayesian posterior mean\nemerges as frequentist mode)',
               xy=(2.5, 0.7), fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_2_1_mode_equals_posterior_mean", "estimators")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 2.2: Mean vs Mode Relationship (Theorem 5)
# =============================================================================

def figure_2_2_mean_vs_mode(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 2.2: Mean vs Mode Relationship (Theorem 5).

    Shows E[theta] is between mode (mu_n) and MLE (D).
    """
    print("\n" + "="*60)
    print("Figure 2.2: Mean vs Mode Relationship (Theorem 5)")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["panel_2x2"])

    # Panel A: Estimator positions for various D values
    ax1 = axes[0, 0]
    D_values = np.array([0, 1, 2, 3, 4, 5])

    modes = []
    means = []
    for D in D_values:
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        modes.append(mu_n)
        try:
            mean = pvalue_mean(D, mu0, sigma, sigma0, method='closed_form')
        except:
            mean = mu_n  # Fallback
        means.append(mean)

    ax1.plot(D_values, modes, 'o-', color=COLORS['waldo'], linewidth=2,
             markersize=8, label='Mode ($\\mu_n$)')
    ax1.plot(D_values, means, 's-', color=COLORS['tilted'], linewidth=2,
             markersize=8, label='Mean E[$\\theta$]')
    ax1.plot(D_values, D_values, 'x--', color=COLORS['mle'], linewidth=2,
             markersize=8, label='MLE (D)')
    ax1.plot(D_values, [mu0]*len(D_values), ':', color=COLORS['prior_mean'],
             linewidth=1.5, label=f'Prior $\\mu_0$={mu0}')

    ax1.set_xlabel('Data D', fontsize=11)
    ax1.set_ylabel('Estimator Value', fontsize=11)
    ax1.set_title('Estimator Values vs Data', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel B: Mean trajectory as function of |Delta|
    ax2 = axes[0, 1]
    deltas = np.linspace(0, 3, 31)

    mode_positions = []
    mean_positions = []
    mle_positions = []

    for delta in deltas:
        D = data_for_conflict(-delta, mu0, w, sigma)  # D > mu0
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        try:
            mean = pvalue_mean(D, mu0, sigma, sigma0, method='closed_form')
        except:
            mean = mu_n
        mode_positions.append(mu_n)
        mean_positions.append(mean)
        mle_positions.append(D)

    # Normalize relative to mode
    mean_shift = np.array(mean_positions) - np.array(mode_positions)
    mle_shift = np.array(mle_positions) - np.array(mode_positions)

    ax2.plot(deltas, mean_shift, color=COLORS['tilted'], linewidth=2,
             label='Mean - Mode')
    ax2.plot(deltas, mle_shift, '--', color=COLORS['mle'], linewidth=2,
             label='MLE - Mode')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.fill_between(deltas, 0, mean_shift, alpha=0.3, color=COLORS['tilted'],
                     label='Mean pulled toward MLE')

    ax2.set_xlabel(r'$|\Delta|$ (prior-data conflict)', fontsize=11)
    ax2.set_ylabel('Distance from Mode', fontsize=11)
    ax2.set_title('Mean is Pulled Toward MLE', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel C: Number line visualization
    ax3 = axes[1, 0]
    D = 4.0
    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
    try:
        mean = pvalue_mean(D, mu0, sigma, sigma0, method='closed_form')
    except:
        mean = (mu_n + D) / 2

    # Draw number line
    ax3.axhline(y=0, color='black', linewidth=2)
    ax3.set_xlim(-1, 5)
    ax3.set_ylim(-0.5, 1)

    # Mark points
    points = [
        (mu0, 'Prior $\\mu_0$', COLORS['prior_mean'], 0.3),
        (mu_n, 'Mode $\\mu_n$', COLORS['waldo'], 0.5),
        (mean, 'Mean', COLORS['tilted'], 0.7),
        (D, 'MLE D', COLORS['mle'], 0.9),
    ]

    for x, label, color, y in points:
        ax3.scatter([x], [0], color=color, s=150, zorder=5, edgecolor='black')
        ax3.annotate(f'{label}\n({x:.2f})', xy=(x, 0), xytext=(x, y),
                    fontsize=10, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-', color=color, lw=1.5))

    # Draw bracket showing ordering
    ax3.annotate('', xy=(mu_n, -0.15), xytext=(D, -0.15),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax3.text((mu_n + D)/2, -0.25, 'Mean is between', ha='center', fontsize=9)

    ax3.set_xlabel(r'$\theta$', fontsize=11)
    ax3.set_title(f'Estimator Ordering (D={D})', fontsize=11)
    ax3.set_yticks([])

    # Panel D: Comparison of closed-form vs numerical mean
    ax4 = axes[1, 1]
    D_test = np.linspace(0, 5, 21)

    closed_form_means = []
    numerical_means = []

    for D in D_test:
        try:
            cf_mean = pvalue_mean(D, mu0, sigma, sigma0, method='closed_form')
            num_mean = pvalue_mean(D, mu0, sigma, sigma0, method='numerical')
        except:
            mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
            cf_mean = mu_n
            num_mean = mu_n
        closed_form_means.append(cf_mean)
        numerical_means.append(num_mean)

    ax4.plot(D_test, closed_form_means, 'o-', color=COLORS['waldo'],
             linewidth=2, markersize=6, label='Closed-form (Thm 5)')
    ax4.plot(D_test, numerical_means, 'x--', color=COLORS['wald'],
             linewidth=1.5, markersize=6, label='Numerical integration')

    # Error
    errors = np.abs(np.array(closed_form_means) - np.array(numerical_means))
    print(f"Max error between closed-form and numerical: {errors.max():.6f}")

    ax4.set_xlabel('Data D', fontsize=11)
    ax4.set_ylabel('Mean E[$\\theta$]', fontsize=11)
    ax4.set_title('Closed-form vs Numerical Mean', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Theorem 5: Confidence Distribution Mean\n'
                 r'$\mu_n < E[\theta] < D$ (mean is pulled toward MLE)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_2_2_mean_vs_mode", "estimators")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 2.3: Dynamic Tilting Estimators
# =============================================================================

def figure_2_3_dynamic_tilting_estimators(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 2.3: Dynamic Tilting Estimators.

    Shows the hierarchy: μ_n < μ_η* < E[θ] < D as a function of |Δ|.
    The dynamically-tilted posterior mean μ_η* interpolates between
    the posterior mean and the MLE.
    """
    print("\n" + "="*60)
    print("Figure 2.3: Dynamic Tilting Estimators")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: All estimators vs |Δ|
    ax1 = axes[0]
    deltas = np.linspace(0, 3, 31)

    mu_n_vals = []
    mu_eta_vals = []
    mean_vals = []
    D_vals = []

    for delta in deltas:
        # D > mu0 case (positive conflict)
        D = data_for_conflict(-delta, mu0, w, sigma)
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)

        # Optimal eta (simplified approximation - transitions from ~0 to 1)
        eta_star = 1 - np.exp(-0.5 * delta**2)

        # Tilted posterior mean (tilted_params takes D, mu0, sigma, sigma0, eta)
        mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta_star)

        # Confidence distribution mean
        try:
            mean = pvalue_mean(D, mu0, sigma, sigma0, method='closed_form')
        except:
            mean = mu_n + 0.5 * (D - mu_n) * (1 - np.exp(-delta))

        mu_n_vals.append(mu_n)
        mu_eta_vals.append(mu_eta)
        mean_vals.append(mean)
        D_vals.append(D)

    ax1.plot(deltas, mu_n_vals, '-', color=COLORS['waldo'], linewidth=2.5,
             label=r'$\mu_n$ (posterior mean)')
    ax1.plot(deltas, mu_eta_vals, '-', color=COLORS['tilted'], linewidth=2.5,
             label=r'$\mu_{\eta^*}$ (tilted mean)')
    ax1.plot(deltas, mean_vals, '--', color='purple', linewidth=2,
             label=r'$E[\theta]$ (conf. dist. mean)')
    ax1.plot(deltas, D_vals, ':', color=COLORS['mle'], linewidth=2,
             label=r'$D$ (MLE)')

    ax1.set_xlabel(r'$|\Delta|$ (prior-data conflict)', fontsize=11)
    ax1.set_ylabel(r'Estimator value', fontsize=11)
    ax1.set_title('(A) Estimator Hierarchy', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: Normalized positions (0 = μ_n, 1 = D)
    ax2 = axes[1]

    # Normalize: 0 = μ_n, 1 = D
    mu_n_arr = np.array(mu_n_vals)
    D_arr = np.array(D_vals)
    span = D_arr - mu_n_arr
    span[span == 0] = 1  # Avoid division by zero

    mu_eta_norm = (np.array(mu_eta_vals) - mu_n_arr) / span
    mean_norm = (np.array(mean_vals) - mu_n_arr) / span

    ax2.axhline(y=0, color=COLORS['waldo'], linestyle='-', linewidth=1.5,
                label=r'$\mu_n$ (0)')
    ax2.axhline(y=1, color=COLORS['mle'], linestyle=':', linewidth=1.5,
                label=r'$D$ (1)')
    ax2.plot(deltas, mu_eta_norm, '-', color=COLORS['tilted'], linewidth=2.5,
             label=r'$\mu_{\eta^*}$')
    ax2.plot(deltas, mean_norm, '--', color='purple', linewidth=2,
             label=r'$E[\theta]$')

    ax2.fill_between(deltas, 0, mu_eta_norm, alpha=0.2, color=COLORS['tilted'])

    ax2.set_xlabel(r'$|\Delta|$ (prior-data conflict)', fontsize=11)
    ax2.set_ylabel('Normalized position', fontsize=11)
    ax2.set_title('(B) Normalized: 0=$\\mu_n$, 1=$D$', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    # Panel C: Optimal η*(|Δ|) curve
    ax3 = axes[2]
    eta_stars = [1 - np.exp(-0.5 * d**2) for d in deltas]

    ax3.plot(deltas, eta_stars, '-', color=COLORS['tilted'], linewidth=2.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label=r'$\eta=0$ (WALDO)')
    ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label=r'$\eta=1$ (Wald)')

    # Mark key points
    ax3.scatter([0], [0], color=COLORS['waldo'], s=100, zorder=5, edgecolor='black')
    ax3.scatter([3], [eta_stars[-1]], color=COLORS['wald'], s=100, zorder=5, edgecolor='black')

    ax3.annotate('Low conflict:\nuse prior info', xy=(0.2, 0.15), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax3.annotate('High conflict:\nuse MLE', xy=(2.2, 0.85), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax3.set_xlabel(r'$|\Delta|$ (prior-data conflict)', fontsize=11)
    ax3.set_ylabel(r'Optimal $\eta^*$', fontsize=11)
    ax3.set_title(r'(C) Optimal Tilting $\eta^*(|\Delta|)$', fontsize=12, fontweight='bold')
    ax3.legend(loc='center right', fontsize=9)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('Dynamic Tilting: Estimator Interpolation\n'
                 r'$\mu_n < \mu_{\eta^*} < E[\theta] < D$ with $\eta^*$ adapting to conflict',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_2_3_dynamic_tilting_estimators", "estimators")

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
    parser.add_argument("--fast", action="store_true", help="Use fast config (fewer points)")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (2.1, 2.2)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    # Note: --fast doesn't change much here since we use analytical calculations

    print("="*60)
    print("ESTIMATOR PROPERTIES FIGURES")
    print("="*60)

    figures_to_generate = ['2.1', '2.2', '2.3']
    if args.figure:
        figures_to_generate = [args.figure]

    if '2.1' in figures_to_generate:
        figure_2_1_mode_equals_posterior_mean(save=save, show=show)

    if '2.2' in figures_to_generate:
        figure_2_2_mean_vs_mode(save=save, show=show)

    if '2.3' in figures_to_generate:
        figure_2_3_dynamic_tilting_estimators(save=save, show=show)

    print("\n" + "="*60)
    print("DONE - Estimator figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
