#!/usr/bin/env python3
"""
Core Theory Figures (Category 1)

Generates figures 1.1-1.3:
- Figure 1.1: Posterior Mean Distribution (Theorem 1) - uses MC
- Figure 1.2: WALDO Statistic Distribution (Theorem 2) - uses MC
- Figure 1.3: The P-value Function (Theorem 3) - formula-based

Usage:
    python scripts/plot_theory.py [--no-save] [--show]
    python scripts/plot_theory.py --fast  # Reduced MC samples
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
)
from frasian.figure_style import (
    COLORS, FIGSIZE, setup_style, save_figure,
)
from frasian.simulations import (
    load_raw_simulation,
    raw_simulation_exists,
    generate_all_raw_simulations,
    DEFAULT_CONFIG,
    FAST_CONFIG,
    compute_sigma0_from_w,
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
    Generate Figure 1.1: Posterior Mean Distribution (Theorem 1).

    Validates that mu_n - theta ~ N(b(theta), v) under true theta.
    Uses raw D samples from simulation infrastructure.
    """
    print("\n" + "="*60)
    print("Figure 1.1: Posterior Mean Distribution (Theorem 1)")
    print("="*60)

    # Load raw simulation data
    data, metadata = load_distribution_data(fast=fast)
    D_samples = data["D_samples"]  # [n_theta, n_samples]
    theta_values = data["theta_values"]
    mu0 = metadata["mu0"]
    sigma = metadata["sigma"]
    w = metadata["w"]
    sigma0 = compute_sigma0_from_w(w, sigma)

    n_theta, n_samples = D_samples.shape
    print(f"Loaded {n_samples} D samples for {n_theta} theta values: {theta_values}")

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["panel_2x2"])

    # Helper to compute posterior means from D samples
    def compute_posterior_means(D_arr):
        mu_n_arr = []
        for D in D_arr:
            mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
            mu_n_arr.append(mu_n)
        return np.array(mu_n_arr)

    # Panel A: QQ-plot at theta = mu0 (zero bias) - use theta closest to 0
    ax1 = axes[0, 0]
    idx_zero = np.argmin(np.abs(theta_values - mu0))
    theta_true = theta_values[idx_zero]
    D_row = D_samples[idx_zero, :]
    mu_n_samples = compute_posterior_means(D_row)
    deviations = mu_n_samples - theta_true

    b_theta, v = posterior_mean_distribution_params(theta_true, mu0, sigma, sigma0)
    theoretical_std = np.sqrt(v)

    stats.probplot((deviations - b_theta) / theoretical_std, dist="norm", plot=ax1)
    ax1.set_title(f'QQ-Plot at theta = {theta_true} (theta = mu_0)', fontsize=11)
    ax1.get_lines()[0].set_color(COLORS['waldo'])
    ax1.get_lines()[1].set_color('black')

    # Panel B: Histogram with theoretical overlay at theta != mu0
    ax2 = axes[0, 1]
    idx_away = np.argmin(np.abs(theta_values - 2.0))  # Try to get theta ~ 2
    theta_true = theta_values[idx_away]
    D_row = D_samples[idx_away, :]
    mu_n_samples = compute_posterior_means(D_row)
    deviations = mu_n_samples - theta_true

    b_theta, v = posterior_mean_distribution_params(theta_true, mu0, sigma, sigma0)
    theoretical_std = np.sqrt(v)

    ax2.hist(deviations, bins=50, density=True, alpha=0.7, color=COLORS['waldo'],
             edgecolor='black', linewidth=0.5, label='Simulated')

    x_range = np.linspace(deviations.min(), deviations.max(), 100)
    theoretical_pdf = stats.norm.pdf(x_range, loc=b_theta, scale=theoretical_std)
    ax2.plot(x_range, theoretical_pdf, 'r-', linewidth=2,
             label=f'N({b_theta:.2f}, {v:.3f})')

    ax2.axvline(x=b_theta, color='red', linestyle='--', alpha=0.7, label=f'b(theta)={b_theta:.2f}')
    ax2.axvline(x=0, color='black', linestyle=':', alpha=0.5, label='Zero')

    ax2.set_xlabel(r'$\mu_n - \theta$', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'Distribution at theta = {theta_true}', fontsize=11)
    ax2.legend(fontsize=8)

    # Panel C: Bias function b(theta) = (1-w)(mu0 - theta) - formula-based
    ax3 = axes[1, 0]
    theta_range = np.linspace(-3, 5, 100)
    biases = [bias(t, mu0, w) for t in theta_range]

    ax3.plot(theta_range, biases, color=COLORS['waldo'], linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=mu0, color='gray', linestyle=':', alpha=0.5, label=f'mu_0 = {mu0}')

    ax3.fill_between(theta_range, 0, biases,
                     where=np.array(biases) > 0, alpha=0.3, color='green',
                     label='Positive bias (toward mu_0)')
    ax3.fill_between(theta_range, 0, biases,
                     where=np.array(biases) < 0, alpha=0.3, color='red',
                     label='Negative bias (toward mu_0)')

    ax3.set_xlabel(r'$\theta$', fontsize=11)
    ax3.set_ylabel(r'Bias $b(\theta) = (1-w)(\mu_0 - \theta)$', fontsize=11)
    ax3.set_title('Bias Function (Theorem 1)', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel D: Distribution shift with different theta values
    ax4 = axes[1, 1]
    thetas = [0, 1, 2, 3]
    colors_list = plt.cm.viridis(np.linspace(0.2, 0.8, len(thetas)))

    for theta_true, color in zip(thetas, colors_list):
        mu_n_samples = simulate_posterior_means(theta_true, mu0, sigma, sigma0, min(n_mc, 2000), rng)
        deviations = mu_n_samples - theta_true
        ax4.hist(deviations, bins=30, density=True, alpha=0.4, color=color,
                 label=f'theta = {theta_true}')

        b_theta, v = posterior_mean_distribution_params(theta_true, mu0, sigma, sigma0)
        ax4.axvline(x=b_theta, color=color, linestyle='--', linewidth=1.5)

    ax4.set_xlabel(r'$\mu_n - \theta$', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Distribution Shifts with theta (dashed = b(theta))', fontsize=11)
    ax4.legend(fontsize=8)

    fig.suptitle('Theorem 1: Posterior Mean Distribution\n'
                 r'$\mu_n - \theta \sim N(b(\theta), v)$ where $b(\theta) = (1-w)(\mu_0 - \theta)$',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_1_posterior_mean_dist", "theory")

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
) -> plt.Figure:
    """
    Generate Figure 1.2: WALDO Statistic Distribution (Theorem 2).

    Validates that tau_WALDO ~ w * chi^2_1(lambda(theta)).
    """
    print("\n" + "="*60)
    print("Figure 1.2: WALDO Statistic Distribution (Theorem 2)")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)
    n_mc = n_samples or MC_CONFIG["n_samples"]
    rng = np.random.default_rng(MC_CONFIG["seed"])

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["panel_2x2"])

    # Panel A: Central case (theta = mu0, lambda = 0)
    ax1 = axes[0, 0]
    theta_true = mu0
    lambda_true = noncentrality(theta_true, mu0, w, sigma, sigma0)

    D_samples = rng.normal(theta_true, sigma, n_mc)
    tau_samples = []
    for D in D_samples:
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)
        tau = waldo_statistic(mu_n, sigma_n, theta_true)
        tau_samples.append(tau)
    tau_samples = np.array(tau_samples)

    # Scale to chi-squared
    scaled_tau = tau_samples / w

    ax1.hist(scaled_tau, bins=50, density=True, alpha=0.7, color=COLORS['waldo'],
             edgecolor='black', linewidth=0.5, label='Simulated tau/w')

    x_range = np.linspace(0, np.percentile(scaled_tau, 99), 100)
    theoretical_pdf = stats.chi2.pdf(x_range, df=1)
    ax1.plot(x_range, theoretical_pdf, 'r-', linewidth=2, label=r'$\chi^2_1$ (central)')

    ax1.set_xlabel(r'$\tau_{WALDO} / w$', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'Central Case: theta = mu_0 (lambda = {lambda_true:.2f})', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 10)

    # Panel B: Non-central case (theta != mu0)
    ax2 = axes[0, 1]
    theta_true = 2.0
    lambda_true = noncentrality(theta_true, mu0, w, sigma, sigma0)

    D_samples = rng.normal(theta_true, sigma, n_mc)
    tau_samples = []
    for D in D_samples:
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)
        tau = waldo_statistic(mu_n, sigma_n, theta_true)
        tau_samples.append(tau)
    tau_samples = np.array(tau_samples)

    scaled_tau = tau_samples / w

    ax2.hist(scaled_tau, bins=50, density=True, alpha=0.7, color=COLORS['waldo'],
             edgecolor='black', linewidth=0.5, label='Simulated tau/w')

    x_range = np.linspace(0, np.percentile(scaled_tau, 99), 100)
    theoretical_pdf = stats.ncx2.pdf(x_range, df=1, nc=lambda_true)
    ax2.plot(x_range, theoretical_pdf, 'r-', linewidth=2,
             label=f'$\\chi^2_1({lambda_true:.1f})$')

    ax2.set_xlabel(r'$\tau_{WALDO} / w$', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'Non-central: theta = {theta_true} (lambda = {lambda_true:.1f})', fontsize=11)
    ax2.legend(fontsize=8)

    # Panel C: Non-centrality as function of theta
    ax3 = axes[1, 0]
    theta_range = np.linspace(-3, 5, 100)
    lambdas = [noncentrality(t, mu0, w, sigma, sigma0) for t in theta_range]

    ax3.plot(theta_range, lambdas, color=COLORS['waldo'], linewidth=2)
    ax3.axvline(x=mu0, color='gray', linestyle=':', alpha=0.5, label=f'mu_0 = {mu0}')

    # Mark key points
    for theta in [0, 2, 3]:
        lam = noncentrality(theta, mu0, w, sigma, sigma0)
        ax3.scatter([theta], [lam], color='red', s=60, zorder=5)
        ax3.annotate(f'({theta}, {lam:.1f})', xy=(theta, lam),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax3.set_xlabel(r'$\theta$', fontsize=11)
    ax3.set_ylabel(r'Non-centrality $\lambda(\theta) = \delta(\theta)^2 / w$', fontsize=11)
    ax3.set_title('Non-centrality Parameter (Theorem 2)', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel D: Mean/variance comparison
    ax4 = axes[1, 1]
    thetas = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    sim_means = []
    sim_vars = []
    theo_means = []
    theo_vars = []

    for theta_true in thetas:
        lambda_true = noncentrality(theta_true, mu0, w, sigma, sigma0)

        D_samples = rng.normal(theta_true, sigma, min(n_mc, 2000))
        tau_samples = []
        for D in D_samples:
            mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)
            tau = waldo_statistic(mu_n, sigma_n, theta_true)
            tau_samples.append(tau)

        scaled_tau = np.array(tau_samples) / w

        sim_means.append(np.mean(scaled_tau))
        sim_vars.append(np.var(scaled_tau))

        # Theoretical: E[chi2(1,lambda)] = 1 + lambda, Var = 2(1 + 2*lambda)
        theo_means.append(1 + lambda_true)
        theo_vars.append(2 * (1 + 2 * lambda_true))

    ax4.scatter(thetas, sim_means, color=COLORS['waldo'], s=60, label='Simulated mean')
    ax4.plot(thetas, theo_means, 'r--', linewidth=2, label='Theoretical mean')

    ax4.set_xlabel(r'$\theta$', fontsize=11)
    ax4.set_ylabel(r'E[$\tau/w$]', fontsize=11)
    ax4.set_title('Mean Comparison: Simulated vs Theoretical', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Theorem 2: WALDO Statistic Distribution\n'
                 r'$\tau_{WALDO} \sim w \cdot \chi^2_1(\lambda(\theta))$',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_2_waldo_statistic_dist", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 1.3: The P-value Function (Theorem 3)
# =============================================================================

def figure_1_3_pvalue_function(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 1.3: The P-value Function (Theorem 3).

    Shows p(theta) = Phi(b-a) + Phi(-a-b) formula.
    """
    print("\n" + "="*60)
    print("Figure 1.3: The P-value Function (Theorem 3)")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Three scenarios: Delta = 0, 1.5, 3
    scenarios = [
        (0, 0, "No Conflict"),
        (3, -1.5, "Mild Conflict"),
        (5, -2.5, "Severe Conflict"),
    ]

    for ax, (D, delta_approx, title) in zip(axes, scenarios):
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)
        delta = scaled_conflict(D, mu0, w, sigma)

        theta_range = np.linspace(mu_n - 4*sigma, D + 2*sigma, 200)
        pvals = [pvalue(t, mu_n, mu0, w, sigma) for t in theta_range]

        # Also get components
        component1 = []
        component2 = []
        for theta in theta_range:
            a, b = pvalue_components(theta, mu_n, mu0, w, sigma)
            component1.append(stats.norm.cdf(b - a))
            component2.append(stats.norm.cdf(-a - b))

        # Main p-value curve
        ax.plot(theta_range, pvals, color=COLORS['waldo'], linewidth=2.5,
                label=r'$p(\theta) = \Phi(b-a) + \Phi(-a-b)$')

        # Components (lighter)
        ax.plot(theta_range, component1, '--', color='green', linewidth=1, alpha=0.7,
                label=r'$\Phi(b-a)$')
        ax.plot(theta_range, component2, '--', color='orange', linewidth=1, alpha=0.7,
                label=r'$\Phi(-a-b)$')

        # Reference lines
        ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=mu_n, color='black', linestyle='-', linewidth=1.5,
                   label=f'Mode $\\mu_n$={mu_n:.1f}')
        ax.axvline(x=D, color=COLORS['mle'], linestyle=':', linewidth=1.5,
                   label=f'MLE D={D:.1f}')

        ax.set_xlabel(r'$\theta$', fontsize=11)
        ax.set_ylabel('p-value', fontsize=11)
        ax.set_title(f'{title}\nD={D}, $\\Delta$={delta:.1f}', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Theorem 3: P-value Function\n'
                 r'$p(\theta) = \Phi(b-a) + \Phi(-a-b)$ where $a = |u|$, $b = (1-w)(\mu_0-\theta)/(w\sigma)$',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_1_3_pvalue_function", "theory")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate core theory figures")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--fast", action="store_true", help="Use fewer MC samples")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (1.1, 1.2, 1.3)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    n_samples = 2000 if args.fast else None

    print("="*60)
    print("CORE THEORY FIGURES")
    print("="*60)

    figures_to_generate = ['1.1', '1.2', '1.3']
    if args.figure:
        figures_to_generate = [args.figure]

    if '1.1' in figures_to_generate:
        figure_1_1_posterior_mean_dist(save=save, show=show, n_samples=n_samples)

    if '1.2' in figures_to_generate:
        figure_1_2_waldo_statistic_dist(save=save, show=show, n_samples=n_samples)

    if '1.3' in figures_to_generate:
        figure_1_3_pvalue_function(save=save, show=show)

    print("\n" + "="*60)
    print("DONE - Core theory figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
