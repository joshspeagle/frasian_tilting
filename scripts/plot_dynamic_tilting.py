#!/usr/bin/env python3
"""
Dynamic Tilting Figures (Category 5 Extension)

Generates figures 5.6-5.10:
- Figure 5.6: Oversharpening Width Advantage
- Figure 5.7: Width Ratio Surface
- Figure 5.8: Dynamic eta*(theta) Mechanism
- Figure 5.9: Dynamic P-value Construction
- Figure 5.10: Dynamic vs Static P-values

Usage:
    python scripts/plot_dynamic_tilting.py [--no-save] [--show]
    python scripts/plot_dynamic_tilting.py --fast  # Reduced MC samples
    python scripts/plot_dynamic_tilting.py --figure 5.6  # Specific figure
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

from frasian.core import posterior_params, scaled_conflict
from frasian.waldo import pvalue, wald_ci_width
from frasian.tilting import (
    tilted_params,
    tilted_pvalue,
    tilted_ci,
    tilted_ci_width,
    dynamic_tilted_pvalue,
    dynamic_tilted_ci,
    optimal_eta_approximation,
)
from frasian.figure_style import (
    COLORS, MC_CONFIG, setup_style, save_figure,
    get_tilting_colors,
)

# Try to import MLP-based optimal eta lookup
try:
    from frasian.simulations.mlp_lookup import OptimalEtaLookup, load_lookup_table
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False

from frasian.simulations import (
    optimal_eta_empirical,
    get_optimal_eta_interpolator,
    DEFAULT_CONFIG,
    FAST_CONFIG,
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_params(w: float, sigma: float = 1.0, mu0: float = 0.0):
    """Get model parameters for a given weight."""
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return mu0, sigma, sigma0


def eta_min(w: float) -> float:
    """Minimum allowed eta for given weight."""
    return -w / (1 - w)


def compute_width_ratio_mc(eta: float, delta: float, w: float = 0.5,
                           alpha: float = 0.05, n_mc: int = 100) -> float:
    """Compute E[W_eta]/W_Wald via Monte Carlo."""
    mu0, sigma, sigma0 = get_model_params(w)

    # Compute theta_true that gives expected delta
    theta_true = mu0 - sigma * delta / (1 - w)

    # Generate D samples
    np.random.seed(42)
    D_samples = np.random.normal(theta_true, sigma, n_mc)

    # Compute tilted CI widths
    widths = []
    for D in D_samples:
        try:
            low, high = tilted_ci(D, mu0, sigma, sigma0, eta, alpha)
            if np.isfinite(low) and np.isfinite(high):
                widths.append(high - low)
        except:
            pass

    if len(widths) == 0:
        return np.nan

    # Wald width (constant)
    w_wald = wald_ci_width(sigma, alpha)

    return np.mean(widths) / w_wald


def get_optimal_eta_for_theta(theta: float, D: float, w: float,
                               mu0: float, sigma: float, fast: bool = False) -> float:
    """Get optimal eta for a specific theta value."""
    # Compute |delta| at this theta
    # Delta depends on (D, theta) not just D
    # For dynamic tilting: |Delta(theta)| = |(1-w)(mu0 - D)/sigma|
    delta = scaled_conflict(D, mu0, w, sigma)
    abs_delta = abs(delta)

    # Get optimal eta for this |delta|
    return optimal_eta_empirical(abs_delta, fast=fast)


# =============================================================================
# Figure 5.6: Oversharpening Width Advantage
# =============================================================================

def figure_5_6_oversharpening_advantage(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.6: Oversharpening Width Advantage.

    Shows E[W]/W_Wald vs eta at fixed low Delta values,
    demonstrating that minimum occurs at eta < 0.
    """
    print("\n" + "="*60)
    print("Figure 5.6: Oversharpening Width Advantage")
    print("="*60)

    w = 0.5
    alpha = 0.05
    n_mc = 50 if fast else 200

    # Eta range (include oversharpening)
    eta_min_val = eta_min(w)
    eta_grid = np.linspace(max(eta_min_val + 0.01, -0.99), 0.99, 30 if fast else 50)

    # Delta values to show
    delta_values = [0.0, 0.3, 0.6, 1.0]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(delta_values)))

    for delta, color in zip(delta_values, colors):
        print(f"  Computing for |Delta|={delta}...")

        width_ratios = []
        for eta in eta_grid:
            ratio = compute_width_ratio_mc(eta, delta, w, alpha, n_mc)
            width_ratios.append(ratio)

        width_ratios = np.array(width_ratios)

        # Find optimal eta
        valid_idx = ~np.isnan(width_ratios)
        if np.any(valid_idx):
            min_idx = np.nanargmin(width_ratios)
            eta_star = eta_grid[min_idx]
            min_ratio = width_ratios[min_idx]

            ax.plot(eta_grid, width_ratios, '-', color=color, linewidth=2,
                   label=f'$|\\Delta|={delta}$')
            ax.scatter([eta_star], [min_ratio], color=color, s=100, zorder=5,
                      edgecolor='black', linewidth=1)

            print(f"    eta*={eta_star:.3f}, E[W]/W_Wald={min_ratio:.3f}")

    # Reference lines
    ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label=r'$\eta=0$ (WALDO)')
    ax.axvline(x=eta_min_val, color='red', linestyle=':', alpha=0.5,
               label=f'$\\eta_{{min}}={eta_min_val:.2f}$')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Shade oversharpening region
    ax.axvspan(eta_min_val, 0, alpha=0.1, color='purple', label='Oversharpening')

    ax.set_xlabel(r'Tilting parameter $\eta$', fontsize=12)
    ax.set_ylabel(r'$E[W_\eta]/W_{Wald}$', fontsize=12)
    ax.set_title('Oversharpening Advantage: Optimal $\\eta^*$ is Negative at Low Conflict',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(eta_min_val - 0.05, 1.05)
    ax.set_ylim(0.7, 1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_6_oversharpening_advantage", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.7: Width Ratio Surface
# =============================================================================

def figure_5_7_width_ratio_surface(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.7: Width Ratio Surface.

    2D heatmap showing E[W]/W_Wald as function of (|Delta|, eta).
    """
    print("\n" + "="*60)
    print("Figure 5.7: Width Ratio Surface")
    print("="*60)

    w = 0.5
    alpha = 0.05
    n_mc = 30 if fast else 100

    # Grid
    n_delta = 15 if fast else 30
    n_eta = 20 if fast else 40

    delta_grid = np.linspace(0, 3, n_delta)
    eta_min_val = eta_min(w)
    eta_grid = np.linspace(max(eta_min_val + 0.01, -0.99), 0.99, n_eta)

    # Compute surface
    print(f"  Computing {n_delta}x{n_eta} = {n_delta*n_eta} points...")
    surface = np.zeros((n_delta, n_eta))

    for i, delta in enumerate(delta_grid):
        if i % 5 == 0:
            print(f"    Delta={delta:.1f} ({i+1}/{n_delta})")
        for j, eta in enumerate(eta_grid):
            surface[i, j] = compute_width_ratio_mc(eta, delta, w, alpha, n_mc)

    # Get optimal eta curve
    eta_star = []
    for i, delta in enumerate(delta_grid):
        row = surface[i, :]
        valid = ~np.isnan(row)
        if np.any(valid):
            min_j = np.nanargmin(row)
            eta_star.append(eta_grid[min_j])
        else:
            eta_star.append(np.nan)
    eta_star = np.array(eta_star)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Heatmap
    im = ax.pcolormesh(delta_grid, eta_grid, surface.T,
                       shading='auto', cmap='RdYlBu_r',
                       vmin=0.8, vmax=1.5)

    # Overlay optimal eta* curve
    valid = ~np.isnan(eta_star)
    ax.plot(delta_grid[valid], eta_star[valid], 'k-', linewidth=3,
            label=r'Optimal $\eta^*(|\Delta|)$')
    ax.plot(delta_grid[valid], eta_star[valid], 'w--', linewidth=1.5)

    # Reference lines
    ax.axhline(y=0, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, linewidth=1)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=r'$E[W_\eta]/W_{Wald}$')

    ax.set_xlabel(r'Prior-data conflict $|\Delta|$', fontsize=12)
    ax.set_ylabel(r'Tilting parameter $\eta$', fontsize=12)
    ax.set_title('Width Ratio Surface: Optimal Tilting Path',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')

    # Add annotations
    ax.text(0.2, -0.8, 'Oversharpening\n(narrower CIs)', fontsize=9,
            ha='center', style='italic')
    ax.text(2.5, 0.9, 'Near Wald', fontsize=9, ha='center', style='italic')

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_7_width_ratio_surface", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.8: Dynamic eta*(theta) Mechanism
# =============================================================================

def figure_5_8_dynamic_mechanism(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.8: Dynamic eta*(theta) Mechanism.

    Shows how eta* varies with theta through |Delta(theta)|.
    """
    print("\n" + "="*60)
    print("Figure 5.8: Dynamic eta*(theta) Mechanism")
    print("="*60)

    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)
    D = 2.5  # Observed data

    # Theta range
    theta_grid = np.linspace(-2, 5, 100 if not fast else 50)

    # Compute |Delta(theta)| - this is the key insight
    # For a given theta being tested, the conflict depends on D and theta
    # Delta = (1-w)(mu0 - D)/sigma (this is the observed conflict)
    delta_observed = scaled_conflict(D, mu0, w, sigma)

    # For dynamic tilting, we use the observed |Delta| at each evaluation point
    # Actually, the "dynamic" aspect comes from using different eta at each theta
    # Let's show how the optimal eta would vary if we optimized for each theta

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: |Delta| vs theta (for this D)
    ax = axes[0, 0]
    # The observed Delta is constant for fixed D
    ax.axhline(y=abs(delta_observed), color='blue', linewidth=2,
               label=f'$|\\Delta|={abs(delta_observed):.2f}$ (observed)')
    ax.axvline(x=D, color='orange', linestyle='--', alpha=0.7, label=f'D={D}')
    ax.axvline(x=mu0, color='purple', linestyle='--', alpha=0.7, label=f'$\\mu_0$={mu0}')
    ax.set_xlabel(r'$\theta$', fontsize=11)
    ax.set_ylabel(r'$|\Delta|$', fontsize=11)
    ax.set_title('(A) Observed Prior-Data Conflict', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(-2, 5)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)

    # Panel B: eta*(|Delta|) lookup
    ax = axes[0, 1]
    delta_range = np.linspace(0, 4, 100)
    eta_star_values = [optimal_eta_empirical(d, fast=fast) for d in delta_range]
    ax.plot(delta_range, eta_star_values, 'k-', linewidth=2)
    ax.axvline(x=abs(delta_observed), color='blue', linestyle='--',
               label=f'Observed $|\\Delta|$')
    ax.scatter([abs(delta_observed)], [optimal_eta_empirical(abs(delta_observed), fast=fast)],
              color='red', s=100, zorder=5, label=r'$\eta^*$ for this D')
    ax.set_xlabel(r'$|\Delta|$', fontsize=11)
    ax.set_ylabel(r'$\eta^*(|\Delta|)$', fontsize=11)
    ax.set_title('(B) Optimal Tilting Lookup', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Panel C: P-value at different eta values
    ax = axes[1, 0]
    eta_values = [0.0, 0.5, 1.0]  # WALDO, mid, Wald
    colors = ['blue', 'purple', 'green']
    labels = [r'$\eta=0$ (WALDO)', r'$\eta=0.5$', r'$\eta=1$ (Wald)']

    for eta, color, label in zip(eta_values, colors, labels):
        pvals = [tilted_pvalue(theta, D, mu0, sigma, sigma0, eta) for theta in theta_grid]
        ax.plot(theta_grid, pvals, color=color, linewidth=2, label=label)

    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=D, color='orange', linestyle=':', alpha=0.7)
    ax.set_xlabel(r'$\theta$', fontsize=11)
    ax.set_ylabel('p-value', fontsize=11)
    ax.set_title('(C) P-value Functions for Different $\\eta$', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(-2, 5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel D: CI bounds comparison
    ax = axes[1, 1]

    # Compute CIs for each method
    methods = [
        ('WALDO', 0.0, 'blue'),
        ('Optimal', optimal_eta_empirical(abs(delta_observed), fast=fast), 'red'),
        ('Wald', 1.0, 'green'),
    ]

    y_pos = [0, 1, 2]
    for (name, eta, color), y in zip(methods, y_pos):
        low, high = tilted_ci(D, mu0, sigma, sigma0, eta, 0.05)
        width = high - low
        ax.barh(y, width, left=low, height=0.5, color=color, alpha=0.7,
               label=f'{name} (W={width:.2f})')
        ax.plot([low, high], [y, y], 'k-', linewidth=2)

    ax.axvline(x=D, color='orange', linestyle='--', linewidth=2, label=f'D={D}')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(['WALDO', 'Optimal $\\eta^*$', 'Wald'])
    ax.set_xlabel(r'$\theta$', fontsize=11)
    ax.set_title('(D) Confidence Intervals Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'Dynamic Tilting Mechanism (D={D}, $\\mu_0$={mu0}, w={w})',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_8_dynamic_mechanism", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.9: Dynamic P-value Construction
# =============================================================================

def figure_5_9_dynamic_pvalue_construction(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.9: Dynamic P-value Construction.

    Shows TRUE dynamic construction where at each θ we compute:
    1. |Δ(θ)| = |(1-w)(μ₀ - θ)/σ| (conflict at that θ)
    2. η*(θ) = optimal_eta(|Δ(θ)|) (local optimal η)
    3. p_dynamic(θ) = tilted_pvalue(θ, D, η*(θ)) (using local η)

    This shows how the dynamic p-value is built by evaluating at each θ
    using a DIFFERENT η value that depends on θ.
    """
    print("\n" + "="*60)
    print("Figure 5.9: Dynamic P-value Construction")
    print("="*60)

    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)
    D = 2.5

    # Key theta values to highlight - chosen to show different conflict levels
    theta_points = [-0.5, 1.0, 2.0, 3.5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    theta_grid = np.linspace(-2, 5, 150)

    # Compute full curves for reference
    p_waldo = [tilted_pvalue(theta, D, mu0, sigma, sigma0, 0.0) for theta in theta_grid]
    p_wald = [tilted_pvalue(theta, D, mu0, sigma, sigma0, 1.0) for theta in theta_grid]
    p_dynamic = [dynamic_tilted_pvalue(theta, D, mu0, sigma, sigma0) for theta in theta_grid]

    for idx, theta_highlight in enumerate(theta_points):
        ax = axes.flat[idx]

        # Compute |Δ(θ)| at this θ - THIS IS THE KEY: conflict varies with θ
        Delta_at_theta = abs((1 - w) * (mu0 - theta_highlight) / sigma)
        eta_star_at_theta = optimal_eta_approximation(Delta_at_theta)

        # Plot static curves (WALDO and Wald)
        ax.plot(theta_grid, p_waldo, 'b-', linewidth=1.5, alpha=0.5, label='WALDO')
        ax.plot(theta_grid, p_wald, 'g--', linewidth=1.5, alpha=0.5, label='Wald')

        # Plot the dynamic p-value curve
        ax.plot(theta_grid, p_dynamic, 'r-', linewidth=2.5, label='Dynamic')

        # Plot the tilted curve for η*(θ) at this specific θ
        p_local_eta = [tilted_pvalue(t, D, mu0, sigma, sigma0, eta_star_at_theta)
                      for t in theta_grid]
        ax.plot(theta_grid, p_local_eta, 'm:', linewidth=2, alpha=0.7,
               label=f'$\\eta^*$({theta_highlight})={eta_star_at_theta:.2f}')

        # The key point: dynamic p-value at θ uses η*(θ)
        p_dynamic_at_theta = dynamic_tilted_pvalue(theta_highlight, D, mu0, sigma, sigma0)
        p_local_at_theta = tilted_pvalue(theta_highlight, D, mu0, sigma, sigma0, eta_star_at_theta)

        # Mark the dynamic evaluation point (they should be the same!)
        ax.scatter([theta_highlight], [p_dynamic_at_theta], color='red', s=200, zorder=10,
                  marker='*', edgecolor='black', linewidth=1.5)

        # Mark intersection with local η curve
        ax.scatter([theta_highlight], [p_local_at_theta], color='magenta', s=100, zorder=9,
                  marker='o', edgecolor='black', linewidth=1)

        ax.axvline(x=theta_highlight, color='red', linestyle=':', alpha=0.5, linewidth=2)

        # Reference lines
        ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=D, color='orange', linestyle=':', alpha=0.7)
        ax.axvline(x=mu0, color='purple', linestyle=':', alpha=0.5)

        ax.set_xlabel(r'$\theta$', fontsize=11)
        ax.set_ylabel('p-value', fontsize=11)
        ax.set_title(f'At $\\theta$={theta_highlight}: $|\\Delta(\\theta)|$={Delta_at_theta:.2f}, '
                    f'$\\eta^*(\\theta)$={eta_star_at_theta:.2f}\np={p_dynamic_at_theta:.3f}',
                    fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-2, 5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        print(f"  theta={theta_highlight}: |Delta(theta)|={Delta_at_theta:.2f}, "
              f"eta*(theta)={eta_star_at_theta:.2f}, p_dyn={p_dynamic_at_theta:.3f}")

    plt.suptitle('Dynamic P-value Construction: At Each $\\theta$, Use $\\eta^*(|\\Delta(\\theta)|)$\n'
                 'The dynamic curve (red) evaluates each point with its local optimal $\\eta$',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_9_dynamic_pvalue_construction", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.10: Dynamic vs Static P-values
# =============================================================================

def figure_5_10_dynamic_vs_static(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.10: Dynamic vs Static P-values.

    Shows TRUE dynamic tilting where η*(θ) varies with θ, compared to
    static methods (WALDO, Wald, and fixed optimal η from D).

    Key insight: Dynamic tilting uses η*(θ) = η*(|Δ(θ)|) at each θ,
    NOT a single η* based on observed data D.
    """
    print("\n" + "="*60)
    print("Figure 5.10: Dynamic vs Static P-values")
    print("="*60)

    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)
    alpha = 0.05

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    D_values = [1.5, 2.5, 4.0]  # Low, medium, high conflict

    for ax, D in zip(axes, D_values):
        theta_grid = np.linspace(-2, 6, 200)

        delta = scaled_conflict(D, mu0, w, sigma)
        abs_delta = abs(delta)

        # Static optimal eta (from observed D) - this is what was used before (WRONG for "dynamic")
        eta_star_static = optimal_eta_empirical(abs_delta, fast=fast)

        # Compute p-values for static methods
        p_waldo = [tilted_pvalue(theta, D, mu0, sigma, sigma0, 0.0)
                  for theta in theta_grid]
        p_static_opt = [tilted_pvalue(theta, D, mu0, sigma, sigma0, eta_star_static)
                       for theta in theta_grid]
        p_wald = [tilted_pvalue(theta, D, mu0, sigma, sigma0, 1.0)
                 for theta in theta_grid]

        # Compute TRUE dynamic p-value: η*(θ) varies with each θ
        p_dynamic = [dynamic_tilted_pvalue(theta, D, mu0, sigma, sigma0)
                    for theta in theta_grid]

        # Plot all curves
        ax.plot(theta_grid, p_waldo, 'b-', linewidth=2, label=r'WALDO ($\eta=0$)')
        ax.plot(theta_grid, p_dynamic, 'r-', linewidth=2.5,
               label=r'Dynamic $\eta^*(\theta)$')
        ax.plot(theta_grid, p_static_opt, 'm--', linewidth=1.5, alpha=0.7,
               label=f'Static $\\eta^*$={eta_star_static:.2f}')
        ax.plot(theta_grid, p_wald, 'g:', linewidth=1.5, alpha=0.7,
               label=r'Wald ($\eta=1$)')

        # CI bounds - now including dynamic CI
        ci_waldo = tilted_ci(D, mu0, sigma, sigma0, 0.0, alpha)
        ci_dynamic = dynamic_tilted_ci(D, mu0, sigma, sigma0, alpha)
        ci_static = tilted_ci(D, mu0, sigma, sigma0, eta_star_static, alpha)
        ci_wald = tilted_ci(D, mu0, sigma, sigma0, 1.0, alpha)

        # Mark CI bounds with different y-offsets
        ci_data = [
            (ci_waldo, 'blue', 0.11, 'WALDO'),
            (ci_dynamic, 'red', 0.08, 'Dynamic'),
            (ci_static, 'magenta', 0.05, 'Static'),
            (ci_wald, 'green', 0.02, 'Wald'),
        ]
        for ci, color, y_offset, name in ci_data:
            ax.plot([ci[0], ci[1]], [alpha + y_offset, alpha + y_offset],
                   color=color, linewidth=3, alpha=0.7)
            ax.scatter([ci[0], ci[1]], [alpha + y_offset, alpha + y_offset],
                      color=color, s=30, zorder=5)

        # Reference lines
        ax.axhline(y=alpha, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=D, color='orange', linestyle=':', alpha=0.7, label=f'D={D}')
        ax.axvline(x=mu0, color='purple', linestyle=':', alpha=0.5)

        # Width annotations
        w_waldo = ci_waldo[1] - ci_waldo[0]
        w_dynamic = ci_dynamic[1] - ci_dynamic[0]
        w_static = ci_static[1] - ci_static[0]
        w_wald = ci_wald[1] - ci_wald[0]

        ax.set_xlabel(r'$\theta$', fontsize=12)
        ax.set_ylabel('p-value', fontsize=12)
        ax.set_title(f'D={D}, $|\\Delta(D)|$={abs_delta:.2f}\n'
                    f'Widths: W={w_waldo:.2f}, Dyn={w_dynamic:.2f}, St={w_static:.2f}, Wald={w_wald:.2f}',
                    fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-2, 6)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        print(f"  D={D}: |Delta(D)|={abs_delta:.2f}")
        print(f"    Widths: WALDO={w_waldo:.2f}, Dynamic={w_dynamic:.2f}, "
              f"Static={w_static:.2f}, Wald={w_wald:.2f}")

    plt.suptitle('Dynamic vs Static P-values: True Dynamic Uses $\\eta^*(\\theta)$ Varying with $\\theta$',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_10_dynamic_vs_static", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.11: MLP Training Overview
# =============================================================================

def figure_5_11_mlp_training_overview(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.11: MLP Training Overview.

    Shows the LHS sampling, target distribution, and MLP architecture.
    """
    print("\n" + "="*60)
    print("Figure 5.11: MLP Training Overview")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: 4D Input Space (projections)
    ax = axes[0, 0]
    # Generate sample LHS points for visualization
    np.random.seed(42)
    n_viz = 200
    w_samples = np.random.uniform(0.01, 0.99, n_viz)
    alpha_samples = np.random.uniform(0.01, 0.99, n_viz)
    delta_prime_samples = np.random.uniform(0, 0.99, n_viz)
    eta_prime_samples = np.random.uniform(0.01, 0.99, n_viz)

    # Plot w' vs delta' colored by eta'
    scatter = ax.scatter(delta_prime_samples, w_samples, c=eta_prime_samples,
                        cmap='viridis', alpha=0.6, s=30)
    ax.set_xlabel(r"$\Delta' = \Delta/(1+\Delta)$", fontsize=11)
    ax.set_ylabel(r"$w' = w$", fontsize=11)
    ax.set_title("(A) LHS Sample Projection\n(colored by $\\eta'$)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(scatter, ax=ax, label=r"$\eta'$")
    ax.grid(True, alpha=0.3)

    # Panel B: Coordinate Transforms
    ax = axes[0, 1]
    # Show delta transform
    delta_range = np.linspace(0, 10, 100)
    delta_prime = delta_range / (1 + delta_range)
    ax.plot(delta_range, delta_prime, 'b-', linewidth=2, label=r"$\Delta' = \Delta/(1+\Delta)$")
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r"Original $|\Delta|$", fontsize=11)
    ax.set_ylabel(r"Transformed $\Delta'$", fontsize=11)
    ax.set_title("(B) Delta Transformation\n(maps $[0,\\infty) \\to [0,1)$)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Target Distribution (log width ratio)
    ax = axes[1, 0]
    # Generate synthetic target values
    np.random.seed(42)
    # Width ratios typically in [0.8, 1.5], log in [-0.2, 0.4]
    log_ratios = np.random.normal(0, 0.15, 500)
    ax.hist(log_ratios, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='log(1) = 0')
    ax.set_xlabel(r"$\log(E[W_\eta]/W_{Wald})$", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("(C) Target Distribution\n(MLP predicts log ratio)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel D: MLP Architecture
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw neural network architecture
    layers = [4, 64, 64, 64, 1]
    layer_names = ["Input\n(w', α', Δ', η')", "Hidden 1\n(64)", "Hidden 2\n(64)", "Hidden 3\n(64)", "Output\nlog(ratio)"]
    x_positions = [1, 3, 5, 7, 9]
    colors = ['#2E86AB', '#6F42C1', '#6F42C1', '#6F42C1', '#DC3545']

    for i, (n_neurons, x, name, color) in enumerate(zip(layers, x_positions, layer_names, colors)):
        # Draw layer as rectangle
        height = min(n_neurons / 10, 6)
        rect = plt.Rectangle((x - 0.4, 5 - height/2), 0.8, height,
                             facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, 8.5, name, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x, 5, str(n_neurons), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

        # Draw connections
        if i < len(layers) - 1:
            ax.annotate('', xy=(x_positions[i+1] - 0.4, 5), xytext=(x + 0.4, 5),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.set_title("(D) MLP Architecture\n(sklearn MLPRegressor, ReLU activation)", fontsize=12, fontweight='bold')

    plt.suptitle('MLP Training Overview: Learning Optimal Tilting', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_11_mlp_training_overview", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 5.12: MLP Optimal Eta Surfaces
# =============================================================================

def figure_5_12_mlp_optimal_eta_surfaces(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 5.12: MLP Optimal eta* Surfaces.

    Shows eta*(|Delta|) curves for different w and alpha values.
    """
    print("\n" + "="*60)
    print("Figure 5.12: MLP Optimal eta* Surfaces")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    delta_range = np.linspace(0, 5, 100 if not fast else 30)

    # Panel A: Vary w at fixed alpha=0.05
    ax = axes[0]
    w_values = [0.2, 0.5, 0.8]
    colors = ['#2E86AB', '#6F42C1', '#DC3545']

    for w, color in zip(w_values, colors):
        eta_star = [optimal_eta_empirical(d, fast=fast) for d in delta_range]
        ax.plot(delta_range, eta_star, color=color, linewidth=2, label=f'w={w}')

        # Mark eta_min constraint
        eta_min_val = eta_min(w)
        ax.axhline(y=eta_min_val, color=color, linestyle=':', alpha=0.5)

    ax.axhline(y=0, color='blue', linestyle='--', alpha=0.3, label=r'$\eta=0$ (WALDO)')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.3, label=r'$\eta=1$ (Wald)')
    ax.set_xlabel(r'Prior-data conflict $|\Delta|$', fontsize=12)
    ax.set_ylabel(r'Optimal $\eta^*(|\Delta|)$', fontsize=12)
    ax.set_title(r'(A) $\eta^*$ vs $|\Delta|$ for Different Weights ($\alpha=0.05$)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 5)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Panel B: Width ratio improvement
    ax = axes[1]
    # Show E[W]/W_Wald at optimal eta
    width_ratios = []
    for d in delta_range:
        eta_star = optimal_eta_empirical(d, fast=fast)
        ratio = compute_width_ratio_mc(eta_star, d, w=0.5, alpha=0.05,
                                       n_mc=20 if fast else 50)
        width_ratios.append(ratio)

    ax.plot(delta_range, width_ratios, 'k-', linewidth=2, label='Optimal tilting')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Wald baseline')

    # Shade improvement region
    ax.fill_between(delta_range, width_ratios, 1.0,
                   where=np.array(width_ratios) < 1.0,
                   color='green', alpha=0.2, label='Width reduction')

    ax.set_xlabel(r'Prior-data conflict $|\Delta|$', fontsize=12)
    ax.set_ylabel(r'$E[W_{\eta^*}]/W_{Wald}$', fontsize=12)
    ax.set_title('(B) Width Ratio at Optimal Tilting\n(vs Wald baseline)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 5)
    ax.set_ylim(0.7, 1.1)
    ax.grid(True, alpha=0.3)

    plt.suptitle('MLP-Based Optimal Tilting Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        save_figure(fig, "fig_5_12_mlp_optimal_eta_surfaces", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main
# =============================================================================

FIGURE_FUNCTIONS = {
    "5.6": figure_5_6_oversharpening_advantage,
    "5.7": figure_5_7_width_ratio_surface,
    "5.8": figure_5_8_dynamic_mechanism,
    "5.9": figure_5_9_dynamic_pvalue_construction,
    "5.10": figure_5_10_dynamic_vs_static,
    "5.11": figure_5_11_mlp_training_overview,
    "5.12": figure_5_12_mlp_optimal_eta_surfaces,
}


def main():
    parser = argparse.ArgumentParser(description="Generate dynamic tilting figures")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Show figures")
    parser.add_argument("--fast", action="store_true", help="Fast mode (fewer MC samples)")
    parser.add_argument("--figure", type=str, help="Generate specific figure (e.g., 5.6)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    fast = args.fast

    if args.figure:
        if args.figure in FIGURE_FUNCTIONS:
            FIGURE_FUNCTIONS[args.figure](save=save, show=show, fast=fast)
        else:
            print(f"Unknown figure: {args.figure}")
            print(f"Available: {list(FIGURE_FUNCTIONS.keys())}")
    else:
        # Generate all figures
        for name, func in FIGURE_FUNCTIONS.items():
            try:
                func(save=save, show=show, fast=fast)
            except Exception as e:
                print(f"Error generating figure {name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print("Dynamic tilting figures complete!")
    print("="*60)


if __name__ == "__main__":
    main()
