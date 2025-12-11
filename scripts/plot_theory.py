#!/usr/bin/env python3
"""
Core Theory Figures (Category 1)

Generates figures 1.1-1.6:
- Figure 1.1: Posterior Mean Bias (Theorem 1)
- Figure 1.2: WALDO Statistic Non-centrality (Theorem 2)
- Figure 1.3: P-value Function and CI Widths (Theorem 3)
- Figure 1.4: Tilting Parameter Space (Theorem 6)
- Figure 1.5: Tilted P-value Family (Theorem 8)
- Figure 1.6: Non-centrality Reduction (Theorem 7)

Usage:
    python scripts/plot_theory.py [--no-save] [--show]
    python scripts/plot_theory.py --fast  # Reduced MC samples
    python scripts/plot_theory.py --figure 1.4  # Specific figure
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from frasian.core import (
    posterior_params, bias, scaled_conflict,
    posterior_mean_distribution_params,
)
from frasian.waldo import (
    waldo_statistic, noncentrality, pvalue, pvalue_components,
    confidence_interval, confidence_interval_width, wald_ci_width, posterior_ci_width,
)
from frasian.tilting import (
    tilted_params,
    tilted_pvalue,
    tilted_ci_width,
    tilted_ci,
    dynamic_tilted_pvalue,
    dynamic_tilted_ci,
    optimal_eta_approximation,
    optimal_eta_mlp,
)
from frasian.figure_style import (
    COLORS, FIGSIZE, setup_style, save_figure,
    get_tilting_colors,
)
from frasian.simulations import (
    load_raw_simulation,
    raw_simulation_exists,
    generate_all_raw_simulations,
    DEFAULT_CONFIG,
    FAST_CONFIG,
    compute_sigma0_from_w,
    optimal_eta_empirical,
    get_optimal_eta_interpolator,
)


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_raw_data_exists(fast: bool = False):
    """Ensure raw simulation data exists, generating if needed."""
    if not raw_simulation_exists("distribution_raw"):
        print("Raw distribution data not found, generating...")
        config = FAST_CONFIG if fast else DEFAULT_CONFIG
        generate_all_raw_simulations(config=config, force=False, verbose=True)


def get_model_params(w: float, sigma: float = 1.0, mu0: float = 0.0):
    """Get model parameters for a given weight."""
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return mu0, sigma, sigma0


def load_distribution_data(fast: bool = False):
    """Load raw D samples from simulation infrastructure."""
    ensure_raw_data_exists(fast)
    data, metadata = load_raw_simulation("distribution_raw")
    return data, metadata


# =============================================================================
# Figure 1.1: Posterior Mean Distribution (Theorem 1)
# =============================================================================

def figure_1_1_posterior_mean_dist(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.1: Posterior Mean Bias in Canonical Coordinates.

    Shows how bias b(θ) = (1-w)(μ₀ - θ) depends on:
    - w (prior weight / strength)
    - (μ₀ - θ) (deviation from prior mean)

    Panel A: 2D heatmap of bias as function of (w, μ₀-θ)
    Panel B: Shrinkage factor (1-w) vs w - the fraction pulled toward prior
    """
    print("\n" + "="*60)
    print("Figure 1.1: Posterior Mean Bias in Canonical Coordinates")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ==========================================================================
    # Panel A: 2D Heatmap of bias b(θ) = (1-w)(μ₀ - θ)
    # ==========================================================================
    ax1 = axes[0]

    # Create grid
    w_values = np.linspace(0.05, 0.95, 50)
    deviation_values = np.linspace(-4, 4, 50)  # (μ₀ - θ)
    W, Dev = np.meshgrid(w_values, deviation_values)

    # Compute bias: b(θ) = (1-w)(μ₀ - θ)
    Bias = (1 - W) * Dev

    # Plot heatmap
    im = ax1.contourf(W, Dev, Bias, levels=30, cmap='RdBu_r')
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(r'Bias $b(\theta) = (1-w)(\mu_0 - \theta)$', fontsize=11)

    # Add contour lines
    contours = ax1.contour(W, Dev, Bias, levels=[-2, -1, 0, 1, 2],
                           colors='black', linewidths=0.5, linestyles='--')
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

    # Mark key regions
    ax1.axhline(y=0, color='black', linewidth=1.5, label=r'$\theta = \mu_0$ (no bias)')
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)

    ax1.set_xlabel(r'Prior weight $w = \sigma_0^2/(\sigma^2 + \sigma_0^2)$', fontsize=11)
    ax1.set_ylabel(r'Deviation from prior $(\mu_0 - \theta)$', fontsize=11)
    ax1.set_title('(A) Bias Landscape: How Prior Strength and Deviation\n'
                  'Combine to Determine Shrinkage', fontsize=11, fontweight='bold')

    # Annotations for interpretation
    ax1.annotate('Strong prior\n(large bias)', xy=(0.15, 2.5), fontsize=9,
                ha='center', color='darkblue')
    ax1.annotate('Weak prior\n(small bias)', xy=(0.85, 2.5), fontsize=9,
                ha='center', color='darkred')

    # ==========================================================================
    # Panel B: Shrinkage Factor (1-w) vs w
    # ==========================================================================
    ax2 = axes[1]

    w_fine = np.linspace(0, 1, 100)
    shrinkage = 1 - w_fine

    ax2.plot(w_fine, shrinkage, color=COLORS['waldo'], linewidth=3)
    ax2.fill_between(w_fine, 0, shrinkage, alpha=0.3, color=COLORS['waldo'])

    # Mark key points
    key_w = [0.2, 0.5, 0.8]
    for w in key_w:
        s = 1 - w
        ax2.scatter([w], [s], color='red', s=80, zorder=5)
        ax2.annotate(f'w={w}\n{s:.0%} shrinkage',
                    xy=(w, s), xytext=(w + 0.08, s + 0.05),
                    fontsize=9, ha='left')

    ax2.set_xlabel(r'Prior weight $w$', fontsize=11)
    ax2.set_ylabel(r'Shrinkage factor $(1-w) = b(\theta)/(\mu_0 - \theta)$', fontsize=11)
    ax2.set_title('(B) Shrinkage Factor: Fraction of Deviation\n'
                  'That Becomes Bias', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Interpretation text
    ax2.text(0.15, 0.3, 'Strong prior:\nMost deviation\nbecomes bias',
             fontsize=9, ha='center', style='italic')
    ax2.text(0.85, 0.7, 'Weak prior:\nLittle shrinkage',
             fontsize=9, ha='center', style='italic')

    fig.suptitle('Theorem 1: Posterior Mean is Biased Toward Prior\n'
                 r'$\mu_n - \theta = (1-w)(\mu_0 - \theta)$ — bias from shrinkage toward prior',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_1_posterior_mean_bias", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.2: WALDO Statistic Distribution (Theorem 2)
# =============================================================================

def figure_1_2_waldo_statistic_dist(
    save: bool = True,
    show: bool = False,
    n_samples: int = None,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.2: Non-centrality in Canonical Coordinates.

    Shows how the WALDO statistic's non-centrality λ(θ) = δ²/w depends on:
    - w (prior weight)
    - δ = (θ - μ₀)/σ₀ (prior residual / standardized deviation from prior)

    Panel A: λ surface as function of (w, δ)
    Panel B: How λ controls the χ² distribution shape
    """
    print("\n" + "="*60)
    print("Figure 1.2: Non-centrality in Canonical Coordinates")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ==========================================================================
    # Panel A: Non-centrality Surface λ = δ²/w
    # ==========================================================================
    ax1 = axes[0]

    # Create grid
    w_values = np.linspace(0.1, 0.9, 50)
    delta_values = np.linspace(-3, 3, 50)  # δ = (θ - μ₀)/σ₀
    W, Delta = np.meshgrid(w_values, delta_values)

    # Compute non-centrality: λ = δ²/w
    Lambda = Delta**2 / W

    # Plot heatmap with log scale for better visualization
    levels = [0, 0.5, 1, 2, 4, 8, 16, 32]
    im = ax1.contourf(W, Delta, Lambda, levels=levels, cmap='YlOrRd', extend='max')
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(r'Non-centrality $\lambda = \delta^2/w$', fontsize=11)

    # Add contour lines
    contours = ax1.contour(W, Delta, Lambda, levels=[1, 4, 9, 16],
                           colors='black', linewidths=0.5, linestyles='--')
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

    # Mark key regions
    ax1.axhline(y=0, color='black', linewidth=1.5, label=r'$\delta=0$ ($\theta=\mu_0$)')

    ax1.set_xlabel(r'Prior weight $w$', fontsize=11)
    ax1.set_ylabel(r'Prior residual $\delta = (\theta - \mu_0)/\sigma_0$', fontsize=11)
    ax1.set_title('(A) Non-centrality Landscape\n'
                  r'$\lambda(\theta) = \delta(\theta)^2/w$ controls test statistic distribution',
                  fontsize=11, fontweight='bold')

    # Annotations with white background for visibility
    ax1.annotate('High λ:\nstrong signal', xy=(0.2, 2.5), fontsize=9,
                ha='center', color='darkred',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.annotate('Low λ:\nweak signal', xy=(0.8, 0.5), fontsize=9,
                ha='center', color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # ==========================================================================
    # Panel B: χ² distributions for different λ values
    # ==========================================================================
    ax2 = axes[1]

    x = np.linspace(0, 15, 200)
    lambda_values = [0, 1, 4, 9]
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(lambda_values)))

    for lam, color in zip(lambda_values, colors):
        if lam == 0:
            pdf = stats.chi2.pdf(x, df=1)
            label = r'$\lambda=0$ (central $\chi^2_1$)'
        else:
            pdf = stats.ncx2.pdf(x, df=1, nc=lam)
            label = rf'$\lambda={lam}$'
        ax2.plot(x, pdf, linewidth=2.5, color=color, label=label)
        ax2.fill_between(x, 0, pdf, alpha=0.15, color=color)

    # Mark the critical value for α=0.05
    crit = stats.chi2.ppf(0.95, df=1)
    ax2.axvline(x=crit, color='gray', linestyle='--', linewidth=1.5,
               label=f'Critical value (α=0.05): {crit:.2f}')

    ax2.set_xlabel(r'$\tau/w$ (scaled WALDO statistic)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('(B) How Non-centrality Shifts the Distribution\n'
                  'Higher λ → distribution shifts right → easier to reject',
                  fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 15)
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Theorem 2: WALDO Statistic Has Known Distribution\n'
                 r'$\tau_{WALDO}/w \sim \chi^2_1(\lambda)$ where $\lambda = \delta^2/w$ measures "surprise"',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_2_noncentrality", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.3: The P-value Function (Theorem 3)
# =============================================================================

def figure_1_3_pvalue_function(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.3: P-value Functions and CI Width Ratios.

    2x2 layout mirroring Figures 1.1 and 1.2:
    - Left column: Example p-value curves (Wald top, WALDO bottom)
    - Right column: CI width ratio heatmaps (w vs |Δ|)
    """
    print("\n" + "="*60)
    print("Figure 1.3: P-value Functions and CI Width Ratios")
    print("="*60)

    alpha = 0.05
    sigma = 1.0  # Fixed for simplicity

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ==========================================================================
    # Left column: P-value curve examples
    # ==========================================================================

    # Example scenarios: (w, D) pairs showing different regimes
    scenarios = [
        (0.2, 2.0, 'Strong prior, moderate conflict'),
        (0.5, 3.0, 'Balanced, high conflict'),
        (0.8, 1.0, 'Weak prior, low conflict'),
    ]
    colors = [plt.cm.viridis(x) for x in [0.2, 0.5, 0.8]]

    # Top-left: Wald p-value curves
    ax_wald = axes[0, 0]
    theta_range = np.linspace(-3, 6, 200)

    for (w, D, label), color in zip(scenarios, colors):
        mu0, _, sigma0 = get_model_params(w)
        delta = scaled_conflict(D, mu0, w, sigma)

        # Wald p-value: 2*Φ(-|D-θ|/σ)
        wald_pvals = 2 * stats.norm.cdf(-np.abs(D - theta_range) / sigma)
        ax_wald.plot(theta_range, wald_pvals, linewidth=2, color=color,
                    label=f'w={w}, D={D}')
        ax_wald.axvline(x=D, color=color, linestyle=':', linewidth=1, alpha=0.5)

    ax_wald.axhline(y=alpha, color='black', linestyle='--', linewidth=1)
    ax_wald.set_xlabel(r'$\theta$', fontsize=11)
    ax_wald.set_ylabel('p-value', fontsize=11)
    ax_wald.set_title('(A) Wald P-value: Symmetric Around D\n'
                      '(ignores prior entirely)', fontsize=11, fontweight='bold')
    ax_wald.set_ylim(0, 1.05)
    ax_wald.set_xlim(-3, 6)
    ax_wald.legend(fontsize=8, loc='upper right')
    ax_wald.grid(True, alpha=0.3)

    # Bottom-left: WALDO p-value curves
    ax_waldo = axes[1, 0]

    for (w, D, label), color in zip(scenarios, colors):
        mu0, _, sigma0 = get_model_params(w)
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)
        delta = scaled_conflict(D, mu0, w, sigma)

        # WALDO p-value
        waldo_pvals = np.array([pvalue(t, mu_n, mu0, w, sigma) for t in theta_range])
        ax_waldo.plot(theta_range, waldo_pvals, linewidth=2, color=color,
                     label=rf'w={w}, $|\Delta|$={abs(delta):.1f}')
        ax_waldo.axvline(x=mu_n, color=color, linestyle=':', linewidth=1, alpha=0.5)

    ax_waldo.axhline(y=alpha, color='black', linestyle='--', linewidth=1)
    ax_waldo.set_xlabel(r'$\theta$', fontsize=11)
    ax_waldo.set_ylabel('p-value', fontsize=11)
    ax_waldo.set_title('(B) WALDO P-value: Asymmetric, Mode at Posterior Mean\n'
                       '(incorporates prior)', fontsize=11, fontweight='bold')
    ax_waldo.set_ylim(0, 1.05)
    ax_waldo.set_xlim(-3, 6)
    ax_waldo.legend(fontsize=8, loc='upper right')
    ax_waldo.grid(True, alpha=0.3)

    # ==========================================================================
    # Right column: CI width ratio heatmaps
    # ==========================================================================

    # Grid for heatmaps
    w_values = np.linspace(0.1, 0.9, 30)
    delta_values = np.linspace(0, 4, 30)
    W_grid, Delta_grid = np.meshgrid(w_values, delta_values)

    # Compute width ratios
    posterior_ratio = np.zeros_like(W_grid)
    waldo_ratio = np.zeros_like(W_grid)

    for i, abs_delta in enumerate(delta_values):
        for j, w in enumerate(w_values):
            mu0, _, sigma0 = get_model_params(w)

            # Wald width (constant for fixed sigma)
            wald_width = wald_ci_width(sigma, alpha)

            # Posterior width
            post_width = posterior_ci_width(sigma0, w, alpha)
            posterior_ratio[i, j] = post_width / wald_width

            # WALDO width at this conflict level
            D_val = mu0 + abs_delta * sigma / (1 - w)
            mu_n, _, _ = posterior_params(D_val, mu0, sigma, sigma0)
            waldo_ci = confidence_interval(mu_n, mu0, w, sigma, alpha)
            waldo_width = waldo_ci[1] - waldo_ci[0]
            waldo_ratio[i, j] = waldo_width / wald_width

    # Top-right: Posterior/Wald ratio (ranges 0.1 to 0.9)
    ax_post = axes[0, 1]
    im1 = ax_post.contourf(W_grid, Delta_grid, posterior_ratio,
                           levels=np.linspace(0.1, 0.9, 17), cmap='RdYlGn_r',
                           extend='both')
    cbar1 = plt.colorbar(im1, ax=ax_post)
    cbar1.set_label('Posterior / Wald Width Ratio', fontsize=10)

    # Add contour lines
    contours1 = ax_post.contour(W_grid, Delta_grid, posterior_ratio,
                                levels=[0.2, 0.4, 0.6, 0.8],
                                colors='black', linewidths=0.5)
    ax_post.clabel(contours1, inline=True, fontsize=8, fmt='%.1f')

    ax_post.set_xlabel(r'Prior weight $w$', fontsize=11)
    ax_post.set_ylabel(r'Prior-data conflict $|\Delta| = (1-w)|\mu_0 - D|/\sigma$', fontsize=11)
    ax_post.set_title('(C) Posterior CI: Always Narrow\n'
                      r'Ratio $\approx \sqrt{w}$ (horizontal bands = no conflict dependence)',
                      fontsize=11, fontweight='bold')

    # Bottom-right: WALDO/Wald ratio (cap at 2.0 for visibility)
    ax_waldo_ratio = axes[1, 1]
    # Cap ratio at 2.0 for better visualization
    waldo_ratio_capped = np.clip(waldo_ratio, 0, 2.0)
    im2 = ax_waldo_ratio.contourf(W_grid, Delta_grid, waldo_ratio_capped,
                                   levels=np.linspace(0.1, 2.0, 20), cmap='RdYlGn_r',
                                   extend='max')
    cbar2 = plt.colorbar(im2, ax=ax_waldo_ratio)
    cbar2.set_label('WALDO / Wald Width Ratio', fontsize=10)

    # Add contour lines including the critical ratio=1 line
    contours2 = ax_waldo_ratio.contour(W_grid, Delta_grid, waldo_ratio,
                                        levels=[0.5, 1.0, 1.5, 2.0],
                                        colors='black', linewidths=[0.5, 1.5, 0.5, 0.5])
    ax_waldo_ratio.clabel(contours2, inline=True, fontsize=8, fmt='%.1f')

    ax_waldo_ratio.set_xlabel(r'Prior weight $w$', fontsize=11)
    ax_waldo_ratio.set_ylabel(r'Prior-data conflict $|\Delta| = (1-w)|\mu_0 - D|/\sigma$', fontsize=11)
    ax_waldo_ratio.set_title('(D) WALDO CI: Adapts to Conflict\n'
                             'Green = narrower than Wald; Red = wider (conflicting prior hurts)',
                             fontsize=11, fontweight='bold')

    fig.suptitle('Theorem 3: P-value Function Determines Confidence Intervals\n'
                 'WALDO adapts CI width to prior-data agreement',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_3_pvalue_ci_ratios", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.4: What is Tilting?
# =============================================================================

def get_model_params(w: float, sigma: float = 1.0, mu0: float = 0.0):
    """Get model parameters for a given weight."""
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return mu0, sigma, sigma0


def figure_1_4_what_is_tilting(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.4: What is Tilting?

    Panel A: Tilted posterior densities for several eta values
    Panel B: Tilted mode as function of eta
    """
    print("\n" + "="*60)
    print("Figure 1.4: What is Tilting?")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)
    D = 3.0  # Example data point with conflict

    mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Tilted posterior densities
    ax1 = axes[0]

    theta_range = np.linspace(-2, 5, 300)

    # Key eta values to show
    etas_to_show = [-0.5, 0.0, 0.5, 1.0]
    colors_map = {
        -0.5: 'purple',
        0.0: COLORS['waldo'],
        0.5: COLORS['tilted'],
        1.0: COLORS['wald']
    }
    labels_map = {
        -0.5: r'$\eta=-0.5$ (oversharpening)',
        0.0: r'$\eta=0$ (WALDO/Posterior)',
        0.5: r'$\eta=0.5$ (tilted)',
        1.0: r'$\eta=1$ (Wald/Likelihood)'
    }

    for eta in etas_to_show:
        mu_eta, sigma_eta, _ = tilted_params(D, mu0, sigma, sigma0, eta)
        density = stats.norm.pdf(theta_range, mu_eta, sigma_eta)
        ax1.plot(theta_range, density, color=colors_map[eta], linewidth=2.5,
                 label=labels_map[eta])
        # Mark the mode
        ax1.axvline(x=mu_eta, color=colors_map[eta], linestyle=':', alpha=0.5)

    # Reference lines for key locations
    ax1.axvline(x=mu0, color=COLORS['prior_mean'], linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Prior mean $\\mu_0$={mu0}')
    ax1.axvline(x=D, color=COLORS['mle'], linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'MLE D={D}')

    ax1.set_xlabel(r'$\theta$', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('(A) Tilted Posterior Densities\n'
                  r'$\eta$ interpolates between posterior ($\eta=0$) and likelihood ($\eta=1$)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(-2, 5)
    ax1.grid(True, alpha=0.3)

    # Add annotation showing the formula
    ax1.text(0.98, 0.95, r'$\mu_\eta = (1-\eta)\mu_n + \eta D$' + '\n' +
             r'$\sigma_\eta^2 = \frac{w\sigma^2}{1 - \eta(1-w)}$',
             transform=ax1.transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Panel B: Mode position vs eta
    ax2 = axes[1]

    etas = np.linspace(-1, 1, 100)

    modes = []
    for eta in etas:
        mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta)
        modes.append(mu_eta)

    ax2.plot(etas, modes, color=COLORS['tilted'], linewidth=2.5)

    # Shade oversharpening region
    ax2.axvspan(-1, 0, color='purple', alpha=0.1, label='Oversharpening')

    # Reference lines
    ax2.axhline(y=mu_n, color=COLORS['waldo'], linestyle='--', linewidth=1.5,
                label=f'Posterior mean $\\mu_n$ = {mu_n:.2f}')
    ax2.axhline(y=D, color=COLORS['mle'], linestyle='--', linewidth=1.5,
                label=f'MLE D = {D:.2f}')
    ax2.axhline(y=mu0, color=COLORS['prior_mean'], linestyle=':', linewidth=1.5,
                label=f'Prior mean $\\mu_0$ = {mu0:.2f}')

    # Mark key points
    for eta in etas_to_show:
        mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta)
        ax2.scatter([eta], [mu_eta], color=colors_map[eta], s=100, zorder=5, edgecolor='black')

    ax2.set_xlabel(r'Tilting Parameter $\eta$', fontsize=12)
    ax2.set_ylabel(r'Tilted Mode $\mu_\eta$', fontsize=12)
    ax2.set_title('(B) Mode Position vs Tilting Parameter\n'
                  r'$\eta < 0$: past posterior toward prior; $\eta > 0$: toward MLE',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.1, 1.1)

    fig.suptitle('What is Tilting? Reweighting Prior vs Likelihood Influence',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_4_what_is_tilting", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.5: Why Tilt? The Width Problem
# =============================================================================

def figure_1_5_why_tilt(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.5: Why Tilt? The Width Problem.

    Panel A: CI width vs |Delta| for Wald, WALDO, Posterior
    Panel B: E[W]/W_Wald vs eta for several |Delta| values (U-shaped curves)
    """
    print("\n" + "="*60)
    print("Figure 1.5: Why Tilt? The Width Problem")
    print("="*60)

    # Model parameters
    w = 0.5
    sigma = 1.0
    mu0 = 0.0
    sigma0 = sigma * np.sqrt(w / (1 - w))
    alpha = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: CI width vs |Delta| for different methods
    ax1 = axes[0]

    deltas = np.linspace(0, 4, 50)

    # Wald width (constant)
    wald_width = wald_ci_width(sigma, alpha)
    wald_widths = np.full_like(deltas, wald_width)

    # Posterior width (constant)
    post_width = posterior_ci_width(sigma, sigma0, alpha)
    post_widths = np.full_like(deltas, post_width)

    # WALDO width (varies with |Delta|)
    waldo_widths = []
    for delta in deltas:
        D = mu0 - sigma * delta / (1 - w)  # Compute D for this |Delta|
        w_width = confidence_interval_width(D, mu0, sigma, sigma0, alpha)
        waldo_widths.append(w_width)
    waldo_widths = np.array(waldo_widths)

    ax1.plot(deltas, wald_widths, color=COLORS['wald'], linewidth=2.5,
             label=f'Wald (constant = {wald_width:.2f})')
    ax1.plot(deltas, post_widths, color=COLORS['posterior'], linewidth=2.5,
             label=f'Posterior (constant = {post_width:.2f})')
    ax1.plot(deltas, waldo_widths, color=COLORS['waldo'], linewidth=2.5,
             label='WALDO (varies with conflict)')

    # Shade the problem region
    ax1.fill_between(deltas, wald_widths, waldo_widths,
                     where=waldo_widths > wald_widths,
                     alpha=0.2, color='red', label='WALDO wider than Wald')

    ax1.set_xlabel(r'Prior-Data Conflict $|\Delta|$', fontsize=12)
    ax1.set_ylabel('CI Width', fontsize=12)
    ax1.set_title('(A) The Problem: WALDO Width Explodes at High Conflict\n'
                  'No single method is optimal everywhere',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, max(waldo_widths) * 1.1)

    # Panel B: Heatmap of W_eta / W_Wald as function of (|Delta|, eta)
    ax2 = axes[1]

    # Create grid for heatmap
    n_delta = 40
    n_eta = 40
    delta_grid = np.linspace(0, 3, n_delta)
    eta_grid = np.linspace(-0.9, 1.0, n_eta)

    # Compute width ratio for each (delta, eta) pair
    ratio_matrix = np.zeros((n_delta, n_eta))
    optimal_eta_curve = []

    for i, delta in enumerate(delta_grid):
        D = mu0 - sigma * delta / (1 - w)
        row_ratios = []
        for j, eta in enumerate(eta_grid):
            try:
                tilted_width = tilted_ci_width(D, mu0, sigma, sigma0, eta, alpha)
                ratio = tilted_width / wald_width
                ratio_matrix[i, j] = ratio
                row_ratios.append(ratio)
            except:
                ratio_matrix[i, j] = np.nan
                row_ratios.append(np.nan)

        # Find optimal eta for this delta
        row_ratios = np.array(row_ratios)
        if not np.all(np.isnan(row_ratios)):
            min_idx = np.nanargmin(row_ratios)
            optimal_eta_curve.append((delta, eta_grid[min_idx]))

    optimal_eta_curve = np.array(optimal_eta_curve)

    # Plot heatmap with yellow centered at 1.0
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=1.5)
    im = ax2.imshow(ratio_matrix.T, extent=[0, 3, -0.9, 1.0],
                    origin='lower', aspect='auto', cmap='RdYlBu_r',
                    norm=norm)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, label=r'$W_\eta / W_{Wald}$')

    # Overlay optimal eta* curve
    if len(optimal_eta_curve) > 0:
        ax2.plot(optimal_eta_curve[:, 0], optimal_eta_curve[:, 1],
                 color='black', linewidth=2.5, linestyle='-',
                 label=r'Optimal $\eta^*(|\Delta|)$')

    # Reference lines
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=1, color='white', linestyle=':', linewidth=1.5, alpha=0.8)

    ax2.set_xlabel(r'Prior-Data Conflict $|\Delta|$', fontsize=12)
    ax2.set_ylabel(r'Tilting Parameter $\eta$', fontsize=12)
    ax2.set_title(r'(B) Width Ratio Landscape with Optimal $\eta^*$ Path' + '\n'
                  'Blue = narrower than Wald; Red = wider',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)

    fig.suptitle('Why Tilt? Adapting to Prior-Data Agreement/Disagreement',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add definitions box
    defn_text = (r'$w = \sigma_0^2/(\sigma^2 + \sigma_0^2)$ (prior weight)' + '\n'
                 r'$|\Delta| = (1-w)|\mu_0 - D|/\sigma$ (prior-data conflict)')
    fig.text(0.5, -0.02, defn_text, ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_5_why_tilt", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.6: The Optimal Tilt eta*(|Delta|)
# =============================================================================

def figure_1_6_optimal_tilt(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.6: The Optimal Tilt eta*(|Delta|).

    Panel A: eta*(|Delta|) curve showing transition from oversharpening to Wald
    Panel B: Resulting efficiency gain E[W_eta*]/W_Wald
    """
    print("\n" + "="*60)
    print("Figure 1.6: The Optimal Tilt")
    print("="*60)

    # Model parameters for MLP lookup
    w = 0.5
    alpha = 0.05

    # Generate delta grid and compute optimal eta using MLP
    delta_grid = np.linspace(0, 5, 100)
    optimal_eta = np.array([optimal_eta_mlp(d, w=w, alpha=alpha) for d in delta_grid])
    print(f"Computed optimal eta using MLP for {len(delta_grid)} points")
    print(f"  w={w}, alpha={alpha}")
    print(f"  eta* range: [{optimal_eta.min():.3f}, {optimal_eta.max():.3f}]")

    # Compute width ratios for Panel B
    sigma = 1.0
    sigma0 = sigma * np.sqrt(w / (1 - w))
    mu0 = 0.0
    wald_width = wald_ci_width(sigma, alpha)

    width_ratios = []
    for delta, eta_star in zip(delta_grid, optimal_eta):
        D = mu0 - sigma * delta / (1 - w)
        try:
            tilted_width = tilted_ci_width(D, mu0, sigma, sigma0, eta_star, alpha)
            width_ratios.append(tilted_width / wald_width)
        except:
            width_ratios.append(np.nan)
    width_ratios = np.array(width_ratios)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: eta*(|Delta|) curve
    ax1 = axes[0]

    ax1.plot(delta_grid, optimal_eta, color=COLORS['tilted'], linewidth=2.5,
             label=r'Optimal $\eta^*(|\Delta|)$')

    # Reference lines
    ax1.axhline(y=0, color=COLORS['waldo'], linestyle='--', linewidth=1.5, alpha=0.7,
                label=r'$\eta=0$ (WALDO)')
    ax1.axhline(y=1, color=COLORS['wald'], linestyle='--', linewidth=1.5, alpha=0.7,
                label=r'$\eta=1$ (Wald)')

    # Shade regions
    ax1.axvspan(0, 0.5, alpha=0.1, color='purple', label='Oversharpening regime')
    ax1.axvspan(2, max(delta_grid), alpha=0.1, color=COLORS['wald'], label='Near-Wald regime')

    # Annotate key behavior
    ax1.annotate('Low conflict:\noversharpening\n' + r'$\eta^* < 0$',
                 xy=(0.2, optimal_eta[np.argmin(np.abs(delta_grid - 0.2))]),
                 xytext=(0.8, -0.6), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax1.annotate('High conflict:\nuse Wald\n' + r'$\eta^* \to 1$',
                 xy=(4, optimal_eta[np.argmin(np.abs(delta_grid - 4))]),
                 xytext=(3, 0.5), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax1.set_xlabel(r'Prior-Data Conflict $|\Delta|$', fontsize=12)
    ax1.set_ylabel(r'Optimal Tilting $\eta^*$', fontsize=12)
    ax1.set_title(r'(A) Optimal Tilting Parameter $\eta^*(|\Delta|)$' + '\n'
                  'Adapts from oversharpening to Wald as conflict increases',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(delta_grid))
    ax1.set_ylim(-1.1, 1.1)

    # Panel B: Width ratio / efficiency gain
    ax2 = axes[1]

    ax2.plot(delta_grid, width_ratios, color=COLORS['tilted'], linewidth=2.5,
             label=r'$W_{\eta^*} / W_{Wald}$')

    # Reference line at 1.0
    ax2.axhline(y=1.0, color=COLORS['wald'], linestyle='--', linewidth=1.5, alpha=0.7,
                label='Wald baseline')

    # Shade efficiency region
    ax2.fill_between(delta_grid, width_ratios, 1.0,
                     where=width_ratios < 1.0,
                     alpha=0.2, color=COLORS['tilted'],
                     label='Efficiency gain over Wald')

    # Annotate max gain
    min_ratio = np.nanmin(width_ratios)
    min_idx = np.nanargmin(width_ratios)
    ax2.annotate(f'Max efficiency:\n{(1-min_ratio)*100:.0f}% narrower',
                 xy=(delta_grid[min_idx], min_ratio),
                 xytext=(delta_grid[min_idx] + 1, min_ratio + 0.1),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'),
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    ax2.set_xlabel(r'Prior-Data Conflict $|\Delta|$', fontsize=12)
    ax2.set_ylabel(r'Width Ratio', fontsize=12)
    ax2.set_title('(B) Efficiency of Optimal Tilting\n'
                  r'$\eta^*$ always achieves $\leq$ Wald width',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(delta_grid))
    ax2.set_ylim(0.7, 1.1)

    fig.suptitle(r'The Optimal Tilt: $\eta^*$ Adapts to Prior-Data Conflict',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add definitions box
    defn_text = (r'$w = \sigma_0^2/(\sigma^2 + \sigma_0^2)$ (prior weight)' + '\n'
                 r'$|\Delta| = (1-w)|\mu_0 - D|/\sigma$ (prior-data conflict)')
    fig.text(0.5, -0.02, defn_text, ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_6_optimal_tilt", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.7: Dynamic Tilting - Why theta, Not D
# =============================================================================

def figure_1_7_dynamic_tilting(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.7: Dynamic Tilting - Why theta, Not D.

    Uses simpler prior residual |delta(theta)| = |theta - mu0|/sigma0 framing.

    Panel A: Prior residual varies as we scan theta
    Panel B: eta*(theta) varies with theta - near prior mean use oversharpening,
             far from prior mean use Wald
    """
    print("\n" + "="*60)
    print("Figure 1.7: Dynamic Tilting")
    print("="*60)

    # Model parameters
    w = 0.5
    sigma = 1.0
    mu0 = 0.0
    sigma0 = sigma * np.sqrt(w / (1 - w))
    D = 2.5  # Observed data
    alpha = 0.05

    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

    print(f"Model: w={w}, mu0={mu0}, sigma={sigma}, sigma0={sigma0:.2f}")
    print(f"Data: D={D}, mu_n={mu_n:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Theta range to scan
    thetas = np.linspace(-2, 5, 200)

    # ===================
    # Panel A: Prior residual |delta(theta)|
    # ===================
    ax1 = axes[0]

    # Compute prior residual: |delta(theta)| = |theta - mu0| / sigma0
    delta_theta = np.abs(thetas - mu0) / sigma0

    ax1.plot(thetas, delta_theta, color=COLORS['tilted'], linewidth=2.5)

    # Mark key locations
    ax1.axvline(x=mu0, color=COLORS['prior_mean'], linestyle='-', linewidth=2, alpha=0.8,
                label=f'Prior mean $\\mu_0={mu0}$')
    ax1.axvline(x=D, color=COLORS['mle'], linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Observed $D={D}$')

    # Mark delta at key points
    delta_at_D = np.abs(D - mu0) / sigma0
    delta_at_mu_n = np.abs(mu_n - mu0) / sigma0
    ax1.scatter([D], [delta_at_D], color=COLORS['mle'], s=100, zorder=5)
    ax1.scatter([mu_n], [delta_at_mu_n], color=COLORS['waldo'], s=80, zorder=5,
                marker='s', label=f'Posterior mean $\\mu_n={mu_n:.1f}$')
    ax1.scatter([mu0], [0], color=COLORS['prior_mean'], s=100, zorder=5)

    # Shade regions for intuition
    ax1.axhspan(0, 1, alpha=0.1, color='green', label='Near prior: trust it')
    ax1.axhspan(2, max(delta_theta), alpha=0.1, color='red', label='Far from prior: ignore it')

    ax1.set_xlabel(r'Hypothesized parameter $\theta$', fontsize=12)
    ax1.set_ylabel(r'Prior residual $|\delta(\theta)| = |\theta - \mu_0|/\sigma_0$', fontsize=12)
    ax1.set_title('(A) Distance from Prior Mean\n'
                  'How much does each hypothesis conflict with the prior?',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 5)
    ax1.set_ylim(0, max(delta_theta) * 1.05)

    # ===================
    # Panel B: eta*(theta) for each theta
    # ===================
    ax2 = axes[1]

    # Compute optimal eta for each theta
    # We need to convert |delta(theta)| to the |Delta| scale that optimal_eta_mlp expects
    # |Delta| = (1-w)|mu0 - theta|/sigma = (1-w) * sigma0/sigma * |delta(theta)|
    # For w=0.5, sigma=sigma0=1: |Delta| = 0.5 * |delta(theta)|
    delta_scaled = (1 - w) * np.abs(thetas - mu0) / sigma
    eta_star_theta = np.array([optimal_eta_mlp(d, w=w, alpha=alpha) for d in delta_scaled])

    ax2.plot(thetas, eta_star_theta, color=COLORS['tilted'], linewidth=2.5,
             label=r'Optimal $\eta^*(\theta)$')

    # Reference lines
    ax2.axhline(y=0, color=COLORS['waldo'], linestyle='--', linewidth=1.5, alpha=0.5,
                label=r'WALDO ($\eta=0$)')
    ax2.axhline(y=1, color=COLORS['wald'], linestyle='--', linewidth=1.5, alpha=0.5,
                label=r'Wald ($\eta=1$)')

    # Mark key locations
    ax2.axvline(x=mu0, color=COLORS['prior_mean'], linestyle='-', linewidth=2, alpha=0.8)
    ax2.axvline(x=D, color=COLORS['mle'], linestyle='--', linewidth=1.5, alpha=0.7)

    # Shade oversharpening region
    ax2.fill_between(thetas, -1.1, eta_star_theta,
                     where=eta_star_theta < 0,
                     alpha=0.2, color='purple', label=r'Oversharpening ($\eta < 0$)')

    # Add annotation arrows connecting concepts
    eta_at_mu0 = optimal_eta_mlp(0.0, w=w, alpha=alpha)
    eta_at_D = optimal_eta_mlp(delta_scaled[np.argmin(np.abs(thetas - D))], w=w, alpha=alpha)

    ax2.annotate('Near prior:\noversharpening',
                 xy=(mu0, eta_at_mu0), xytext=(mu0 + 0.8, eta_at_mu0 - 0.4),
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.annotate('Far from prior:\napproach Wald',
                 xy=(4, eta_star_theta[np.argmin(np.abs(thetas - 4))]),
                 xytext=(3, 0.5),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel(r'Hypothesized parameter $\theta$', fontsize=12)
    ax2.set_ylabel(r'Optimal tilting $\eta^*(\theta)$', fontsize=12)
    ax2.set_title(r'(B) Optimal Tilt Adapts to $\theta$' + '\n'
                  r'$\eta^*$ is low near $\mu_0$, high far from $\mu_0$',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 5)
    ax2.set_ylim(-1.1, 1.1)

    fig.suptitle('Dynamic Tilting: The Optimal Tilt Depends on the Hypothesis',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_7_dynamic_tilting", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.7b: The Envelope Construction
# =============================================================================

def figure_1_7b_envelope_construction(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.7b: The Envelope Construction for Dynamic Tilting.

    Shows how the dynamic p-value function is constructed as an envelope
    of tilted p-value curves, each evaluated at its locally optimal eta.
    """
    print("\n" + "="*60)
    print("Figure 1.7b: The Envelope Construction")
    print("="*60)

    # Model parameters
    w = 0.5
    sigma = 1.0
    mu0 = 0.0
    sigma0 = sigma * np.sqrt(w / (1 - w))
    D = 3.0  # Moderate conflict to show clear envelope
    alpha = 0.05

    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
    delta_D = np.abs((1 - w) * (mu0 - D) / sigma)

    print(f"  Model: w={w}, D={D}, |Delta|={delta_D:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Theta range
    thetas = np.linspace(-1, 6, 200)

    # ===================
    # Panel A: The envelope concept
    # ===================

    # Show p-value curves for several representative theta values
    sample_thetas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    cmap = plt.cm.viridis

    envelope_thetas = []
    envelope_pvals = []
    envelope_etas = []

    for i, theta_sample in enumerate(sample_thetas):
        # Compute |Delta| at this theta
        delta_at_theta = np.abs((1 - w) * (mu0 - theta_sample) / sigma)
        eta_star = optimal_eta_mlp(delta_at_theta, w=w, alpha=alpha)

        # Plot p-value curve for this eta (faded)
        pvals = [tilted_pvalue(theta, D, mu0, sigma, sigma0, eta_star) for theta in thetas]
        color = cmap(i / len(sample_thetas))
        ax1.plot(thetas, pvals, color=color, linewidth=1.2, alpha=0.4)

        # Mark the evaluation point - THIS forms the envelope
        p_at_sample = tilted_pvalue(theta_sample, D, mu0, sigma, sigma0, eta_star)
        ax1.scatter([theta_sample], [p_at_sample], color=color, s=120, zorder=5,
                   edgecolor='black', linewidth=2)

        envelope_thetas.append(theta_sample)
        envelope_pvals.append(p_at_sample)
        envelope_etas.append(eta_star)

    # Draw the envelope line connecting the points
    ax1.plot(envelope_thetas, envelope_pvals, color='black', linewidth=3, linestyle='-',
             marker='o', markersize=0, label='Dynamic p-value (envelope)', zorder=4)

    # Reference line
    ax1.axhline(y=alpha, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'$\\alpha={alpha}$')

    # Add annotation explaining the construction
    ax1.annotate('At each $\\theta$:\nuse $\\eta^*(\\theta)$,\nevaluate $p_{\\eta^*}(\\theta)$',
                 xy=(2.0, envelope_pvals[2]), xytext=(3.5, 0.7),
                 fontsize=10, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.8),
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax1.set_xlabel(r'Hypothesized parameter $\theta$', fontsize=12)
    ax1.set_ylabel('p-value', fontsize=12)
    ax1.set_title('(A) The Envelope Construction\n'
                  'Each point uses locally optimal $\\eta^*(\\theta)$',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(0, 1.05)

    # ===================
    # Panel B: eta* values used at each point
    # ===================

    # Show eta* as function of theta
    theta_fine = np.linspace(-1, 6, 200)
    delta_fine = np.abs((1 - w) * (mu0 - theta_fine) / sigma)
    eta_fine = np.array([optimal_eta_mlp(d, w=w, alpha=alpha) for d in delta_fine])

    ax2.plot(theta_fine, eta_fine, color=COLORS['tilted'], linewidth=2.5,
             label=r'$\eta^*(\theta)$')

    # Mark the sample points
    for i, (th, eta) in enumerate(zip(envelope_thetas, envelope_etas)):
        color = cmap(i / len(sample_thetas))
        ax2.scatter([th], [eta], color=color, s=120, zorder=5,
                   edgecolor='black', linewidth=2)

    # Reference lines
    ax2.axhline(y=0, color=COLORS['waldo'], linestyle='--', linewidth=1.5, alpha=0.5,
                label=r'WALDO ($\eta=0$)')
    ax2.axhline(y=1, color=COLORS['wald'], linestyle='--', linewidth=1.5, alpha=0.5,
                label=r'Wald ($\eta=1$)')

    # Shade regions
    ax2.fill_between(theta_fine, 0, eta_fine, where=eta_fine < 0,
                     alpha=0.2, color='blue', label='Oversharpening')
    ax2.fill_between(theta_fine, 0, eta_fine, where=eta_fine > 0,
                     alpha=0.2, color='red', label='Toward Wald')

    ax2.set_xlabel(r'Hypothesized parameter $\theta$', fontsize=12)
    ax2.set_ylabel(r'Optimal tilt $\eta^*(\theta)$', fontsize=12)
    ax2.set_title('(B) The Optimal Tilt at Each Point\n'
                  r'Colors match points in Panel A',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 6)
    ax2.set_ylim(-0.5, 1.1)

    fig.suptitle(f'Constructing the Dynamic P-value Function (D={D}, $|\\Delta|$={delta_D:.2f})',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_7b_envelope_construction", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.8: Constructing the Dynamic P-value and CI
# =============================================================================

def figure_1_8_dynamic_construction(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.8: Dynamic Tilting Across Conflict Levels.

    Three rows showing data-prior agreement, mild disagreement, and major disagreement.
    """
    print("\n" + "="*60)
    print("Figure 1.8: Dynamic Tilting Across Conflict Levels")
    print("="*60)

    # Model parameters (fixed)
    w = 0.5
    sigma = 1.0
    mu0 = 0.0
    sigma0 = sigma * np.sqrt(w / (1 - w))
    alpha = 0.05

    # Three conflict levels
    scenarios = [
        {"D": 0.5, "label": "Agreement", "xlim": (-3, 4)},
        {"D": 2.5, "label": "Mild Disagreement", "xlim": (-2, 6)},
        {"D": 5.0, "label": "Major Disagreement", "xlim": (-1, 9)},
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for row, (ax, scenario) in enumerate(zip(axes, scenarios)):
        D = scenario["D"]
        label = scenario["label"]
        xlim = scenario["xlim"]

        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        delta = np.abs((1 - w) * (mu0 - D) / sigma)

        print(f"  Row {row+1}: {label} - D={D}, |Delta|={delta:.2f}")

        # Theta range for this scenario
        thetas = np.linspace(xlim[0], xlim[1], 300)

        # Compute p-values
        waldo_pvals = [tilted_pvalue(theta, D, mu0, sigma, sigma0, 0.0) for theta in thetas]
        wald_pvals = [tilted_pvalue(theta, D, mu0, sigma, sigma0, 1.0) for theta in thetas]
        dynamic_pvals = [dynamic_tilted_pvalue(theta, D, mu0, sigma, sigma0, alpha) for theta in thetas]

        # Plot p-value curves
        ax.plot(thetas, wald_pvals, color=COLORS['wald'], linewidth=2, alpha=0.7,
                linestyle='--', label=r'Wald ($\eta=1$)')
        ax.plot(thetas, waldo_pvals, color=COLORS['waldo'], linewidth=2, alpha=0.7,
                linestyle='--', label=r'WALDO ($\eta=0$)')
        ax.plot(thetas, dynamic_pvals, color=COLORS['tilted'], linewidth=2.5,
                label=r'Dynamic ($\eta^*(\theta)$)')

        # Reference line
        ax.axhline(y=alpha, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

        # Compute CIs
        ci_wald = (D - 1.96*sigma, D + 1.96*sigma)
        width_wald = ci_wald[1] - ci_wald[0]

        from frasian.waldo import confidence_interval as waldo_ci
        ci_waldo = waldo_ci(D, mu0, sigma, sigma0, alpha)
        width_waldo = ci_waldo[1] - ci_waldo[0]

        regions_dynamic, width_dynamic = dynamic_tilted_ci(D, mu0, sigma, sigma0, alpha)
        n_regions = len(regions_dynamic)

        # Shade dynamic CI regions
        for region in regions_dynamic:
            ax.axvspan(region[0], region[1], alpha=0.15, color=COLORS['tilted'])

        # CI markers
        ax.axvline(x=ci_wald[0], color=COLORS['wald'], linestyle=':', alpha=0.4, linewidth=1.5)
        ax.axvline(x=ci_wald[1], color=COLORS['wald'], linestyle=':', alpha=0.4, linewidth=1.5)
        ax.axvline(x=ci_waldo[0], color=COLORS['waldo'], linestyle=':', alpha=0.4, linewidth=1.5)
        ax.axvline(x=ci_waldo[1], color=COLORS['waldo'], linestyle=':', alpha=0.4, linewidth=1.5)

        # Width annotations with arrows
        y_positions = [0.18, 0.30, 0.42]

        # Dynamic CI width
        y_dyn = y_positions[0]
        all_bounds = [b for region in regions_dynamic for b in region]
        ax.annotate('', xy=(min(all_bounds), y_dyn), xytext=(max(all_bounds), y_dyn),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['tilted'], lw=1.5))
        region_note = f' ({n_regions} regions)' if n_regions > 1 else ''
        ax.text((min(all_bounds) + max(all_bounds))/2, y_dyn + 0.03,
                f'Dynamic: {width_dynamic:.2f}{region_note}', ha='center', fontsize=9,
                color=COLORS['tilted'], fontweight='bold')

        # Wald CI width
        y_wald = y_positions[1]
        ax.annotate('', xy=(ci_wald[0], y_wald), xytext=(ci_wald[1], y_wald),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['wald'], lw=1.5))
        ax.text((ci_wald[0] + ci_wald[1])/2, y_wald + 0.03,
                f'Wald: {width_wald:.2f}', ha='center', fontsize=9,
                color=COLORS['wald'], fontweight='bold')

        # WALDO CI width
        y_waldo = y_positions[2]
        ax.annotate('', xy=(ci_waldo[0], y_waldo), xytext=(ci_waldo[1], y_waldo),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['waldo'], lw=1.5))
        ax.text((ci_waldo[0] + ci_waldo[1])/2, y_waldo + 0.03,
                f'WALDO: {width_waldo:.2f}', ha='center', fontsize=9,
                color=COLORS['waldo'], fontweight='bold')

        # Info box
        info_text = f'$|\\Delta| = {delta:.2f}$\n$D = {D}$'
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # Row label
        ax.text(0.02, 0.98, f'({chr(65+row)}) {label}', transform=ax.transAxes,
                fontsize=12, fontweight='bold', verticalalignment='top')

        ax.set_ylabel('p-value', fontsize=11)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if row == 2:
            ax.set_xlabel(r'Hypothesized parameter $\theta$', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Dynamic Tilting Across Prior-Data Conflict Levels',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_8_dynamic_construction", "theory")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate core theory figures")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--fast", action="store_true", help="Use fewer MC samples")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (1.1-1.8)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    fast = args.fast

    print("="*60)
    print("CORE THEORY FIGURES")
    print("="*60)

    figures_to_generate = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8']
    if args.figure:
        figures_to_generate = [args.figure]

    if '1.1' in figures_to_generate:
        figure_1_1_posterior_mean_dist(save=save, show=show, fast=fast)

    if '1.2' in figures_to_generate:
        n_samples_fig = 2000 if fast else None
        figure_1_2_waldo_statistic_dist(save=save, show=show, n_samples=n_samples_fig)

    if '1.3' in figures_to_generate:
        figure_1_3_pvalue_function(save=save, show=show)

    if '1.4' in figures_to_generate:
        figure_1_4_what_is_tilting(save=save, show=show)

    if '1.5' in figures_to_generate:
        figure_1_5_why_tilt(save=save, show=show, fast=fast)

    if '1.6' in figures_to_generate:
        figure_1_6_optimal_tilt(save=save, show=show, fast=fast)

    if '1.7' in figures_to_generate:
        figure_1_7_dynamic_tilting(save=save, show=show)

    if '1.7b' in figures_to_generate:
        figure_1_7b_envelope_construction(save=save, show=show)

    if '1.8' in figures_to_generate:
        figure_1_8_dynamic_construction(save=save, show=show)

    print("\n" + "="*60)
    print("DONE - Core theory figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
