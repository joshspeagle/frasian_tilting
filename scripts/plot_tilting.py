#!/usr/bin/env python3
"""
Tilting Methodology Figures (Category 3)

Generates figures 3.1-3.2:
- Figure 3.1: Two-Stage MLP Architecture
- Figure 3.2: MLP Optimal Eta Surfaces

Usage:
    python scripts/plot_tilting.py [--no-save] [--show]
    python scripts/plot_tilting.py --fast  # Reduced MC samples for quick testing
    python scripts/plot_tilting.py --figure 3.1  # Specific figure
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from frasian.core import posterior_params, scaled_conflict
from frasian.waldo import pvalue, wald_ci_width, confidence_interval_width
from frasian.waldo import pvalue as waldo_pvalue
from frasian.tilting import (
    tilted_params,
    tilted_pvalue,
    tilted_ci,
    tilted_ci_width,
    dynamic_tilted_pvalue,
    dynamic_tilted_ci,
    dynamic_tilted_pvalue_batch,
    optimal_eta_mlp,
)
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
)
from scipy.stats import norm


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


def eta_min(w: float) -> float:
    """Minimum allowed eta for given weight."""
    return -w / (1 - w)


# =============================================================================
# Figure 3.1: Two-Stage MLP Architecture (expanded from 5.11)
# =============================================================================

def figure_3_1_mlp_architecture(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 3.1: Two-Stage MLP Architecture.

    Shows both Stage 1 (Width Ratio MLP) and Stage 2 (Monotonic η* MLP)
    architectures, coordinate transformations, and monotonicity verification.
    """
    print("\n" + "="*60)
    print("Figure 3.1: Two-Stage MLP Architecture")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # =========================================================================
    # Panel A: Stage 1 - Width Ratio MLP
    # =========================================================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw neural network architecture
    layers = [4, 64, 64, 64, 1]
    layer_names = ["Input\n(w, α, Δ', η')", "Hidden\n(64)", "Hidden\n(64)", "Hidden\n(64)", "Output\nlog(W/W_Wald)"]
    x_positions = [1, 2.8, 4.6, 6.4, 8.2]
    colors = ['#2E86AB', '#6F42C1', '#6F42C1', '#6F42C1', '#DC3545']

    for i, (n_neurons, x, name, color) in enumerate(zip(layers, x_positions, layer_names, colors)):
        height = min(n_neurons / 10, 5)
        rect = FancyBboxPatch((x - 0.35, 5 - height/2), 0.7, height,
                              boxstyle="round,pad=0.05", facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, 8, name, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, 5, str(n_neurons), ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        if i < len(layers) - 1:
            ax.annotate('', xy=(x_positions[i+1] - 0.4, 5), xytext=(x + 0.4, 5),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # GELU label
    ax.text(5.5, 2.5, 'GELU activations', ha='center', fontsize=9, style='italic')

    ax.set_title('(A) Stage 1: Width Ratio MLP\nPredicts log(E[W_η]/W_Wald) for any (w, α, Δ, η)',
                fontsize=11, fontweight='bold')

    # =========================================================================
    # Panel B: Stage 2 - Monotonic η* MLP
    # =========================================================================
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Shared pathway (w, α)
    ax.add_patch(FancyBboxPatch((0.5, 6.5), 1.2, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#2E86AB', alpha=0.7, edgecolor='black'))
    ax.text(1.1, 7.25, '(w, α)', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Monotonic pathway (Δ')
    ax.add_patch(FancyBboxPatch((0.5, 3), 1.2, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#28A745', alpha=0.7, edgecolor='black'))
    ax.text(1.1, 3.75, "Δ'", ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # Hidden layers - shared
    ax.add_patch(FancyBboxPatch((2.5, 6.2), 1, 2, boxstyle="round,pad=0.05",
                                facecolor='#6F42C1', alpha=0.7, edgecolor='black'))
    ax.text(3, 7.2, '64×2\nGELU', ha='center', va='center', fontsize=8, color='white')

    # Hidden layers - monotonic
    ax.add_patch(FancyBboxPatch((2.5, 2.7), 1, 2, boxstyle="round,pad=0.05",
                                facecolor='#28A745', alpha=0.7, edgecolor='black'))
    ax.text(3, 3.7, '64×2\n|W|+ReLU', ha='center', va='center', fontsize=8, color='white')

    # Combination layer
    ax.add_patch(FancyBboxPatch((5, 4.5), 1.5, 2, boxstyle="round,pad=0.05",
                                facecolor='#FD7E14', alpha=0.7, edgecolor='black'))
    ax.text(5.75, 5.5, 'base +\nscale×mono', ha='center', va='center', fontsize=8, color='white')

    # Output
    ax.add_patch(FancyBboxPatch((7.5, 4.5), 1.3, 2, boxstyle="round,pad=0.05",
                                facecolor='#DC3545', alpha=0.7, edgecolor='black'))
    ax.text(8.15, 5.5, 'η\'*\n∈[0,1]', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Arrows
    ax.annotate('', xy=(2.4, 7.2), xytext=(1.75, 7.25), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(2.4, 3.7), xytext=(1.75, 3.75), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(4.9, 5.5), xytext=(3.6, 7.2), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(4.9, 5.5), xytext=(3.6, 3.7), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(7.4, 5.5), xytext=(6.6, 5.5), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Annotations
    ax.text(3, 1.5, 'Positive weights\nguarantee ∂η*/∂Δ ≥ 0', ha='center', fontsize=9,
           style='italic', color='#28A745')

    ax.set_title("(B) Stage 2: Monotonic η* MLP\nGuarantees ∂η*/∂|Δ| ≥ 0 architecturally",
                fontsize=11, fontweight='bold')

    # =========================================================================
    # Panel C: Coordinate Transformations
    # =========================================================================
    ax = axes[1, 0]

    # Show delta transform
    delta_range = np.linspace(0, 10, 100)
    delta_prime = delta_range / (1 + delta_range)

    ax.plot(delta_range, delta_prime, 'b-', linewidth=2.5, label=r"$\Delta' = \Delta/(1+\Delta)$")
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.3)

    # Mark key points
    ax.scatter([0, 1, 5], [0, 0.5, 5/6], color='red', s=80, zorder=5)
    ax.annotate('(0, 0)', (0.2, 0.05), fontsize=9)
    ax.annotate('(1, 0.5)', (1.2, 0.45), fontsize=9)
    ax.annotate('(5, 0.83)', (5.2, 0.78), fontsize=9)

    ax.set_xlabel(r"Original conflict $|\Delta|$", fontsize=11)
    ax.set_ylabel(r"Transformed $\Delta'$", fontsize=11)
    ax.set_title("(C) Coordinate Transformation\nMaps $[0,\\infty) \\to [0,1)$ for neural network input",
                fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel D: Monotonicity Verification
    # =========================================================================
    ax = axes[1, 1]

    delta_grid = np.linspace(0, 5, 100 if not fast else 30)
    w_values = [0.2, 0.5, 0.8]
    colors = ['#2E86AB', '#6F42C1', '#DC3545']
    linestyles = ['-', '--', ':']

    for w, color, ls in zip(w_values, colors, linestyles):
        # Use MLP predictions
        eta_star = [optimal_eta_mlp(d, w=w, alpha=0.05) for d in delta_grid]
        ax.plot(delta_grid, eta_star, color=color, linewidth=2, linestyle=ls, label=f'w={w}')

        # Mark eta_min constraint
        eta_min_val = eta_min(w)
        ax.axhline(y=eta_min_val, color=color, linestyle=':', alpha=0.3, linewidth=1)

    ax.axhline(y=0, color='blue', linestyle='--', alpha=0.3, label=r'$\eta=0$ (WALDO)')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.3, label=r'$\eta=1$ (Wald)')

    # Shade oversharpening region
    ax.axhspan(-1.1, 0, alpha=0.05, color='purple')
    ax.text(4.5, -0.5, 'Oversharpening\n(η < 0)', fontsize=9, ha='center', style='italic', color='purple')

    ax.set_xlabel(r'Prior-data conflict $|\Delta|$', fontsize=11)
    ax.set_ylabel(r'Optimal $\eta^*(|\Delta|)$ from MLP', fontsize=11)
    ax.set_title('(D) Monotonicity Verification\nAll curves monotonically increasing (by construction)',
                fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 5)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Two-Stage Neural Network for Optimal Tilting', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        save_figure(fig, "fig_3_1_mlp_architecture", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 3.2: MLP η* Surfaces (from 5.12, fixed)
# =============================================================================

def figure_3_2_mlp_surfaces(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 3.2: MLP Optimal η* Surfaces.

    Shows η*(|Δ|) curves for different w values, and width improvement.
    Uses optimal_eta_mlp() for predictions.
    """
    print("\n" + "="*60)
    print("Figure 3.2: MLP Optimal eta* Surfaces")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    delta_range = np.linspace(0, 5, 100 if not fast else 30)
    alpha = 0.05

    # =========================================================================
    # Panel A: η* vs |Δ| for different weights
    # =========================================================================
    ax = axes[0]
    w_values = [0.2, 0.5, 0.8]
    colors = ['#2E86AB', '#6F42C1', '#DC3545']

    for w, color in zip(w_values, colors):
        # Use MLP predictions (not empirical!)
        eta_star = [optimal_eta_mlp(d, w=w, alpha=alpha) for d in delta_range]
        ax.plot(delta_range, eta_star, color=color, linewidth=2.5, label=f'w={w}')

        # Mark eta_min constraint
        eta_min_val = eta_min(w)
        ax.axhline(y=eta_min_val, color=color, linestyle=':', alpha=0.4, linewidth=1)

    ax.axhline(y=0, color='blue', linestyle='--', alpha=0.3, linewidth=1.5, label=r'$\eta=0$ (WALDO)')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.3, linewidth=1.5, label=r'$\eta=1$ (Wald)')

    # Shade regions
    ax.axhspan(-1.1, 0, alpha=0.05, color='purple')
    ax.text(0.3, -0.5, 'Oversharpening', fontsize=9, style='italic', color='purple')

    ax.set_xlabel(r'Prior-data conflict $|\Delta|$', fontsize=12)
    ax.set_ylabel(r'Optimal $\eta^*(|\Delta|)$', fontsize=12)
    ax.set_title(r'(A) MLP-Predicted $\eta^*$ vs $|\Delta|$ ($\alpha=0.05$)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel B: Width ratio improvement at w=0.5
    # =========================================================================
    ax = axes[1]
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)
    n_mc = 30 if fast else 100

    print("  Computing width ratios...")
    width_ratios = []
    width_ratios_se = []

    for d in delta_range:
        # Compute theta that gives this expected delta
        theta_true = mu0 - sigma * d / (1 - w)

        # Get optimal eta from MLP
        eta_star = optimal_eta_mlp(d, w=w, alpha=alpha)

        # Monte Carlo for width at this eta
        np.random.seed(42)
        D_samples = np.random.normal(theta_true, sigma, n_mc)

        widths = []
        for D in D_samples:
            try:
                low, high = tilted_ci(D, mu0, sigma, sigma0, eta_star, alpha)
                if np.isfinite(low) and np.isfinite(high):
                    widths.append(high - low)
            except:
                pass

        if len(widths) > 0:
            w_wald = wald_ci_width(sigma, alpha)
            ratio = np.mean(widths) / w_wald
            width_ratios.append(ratio)
        else:
            width_ratios.append(np.nan)

    width_ratios = np.array(width_ratios)

    ax.plot(delta_range, width_ratios, 'k-', linewidth=2.5, label='Optimal tilting (MLP)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Wald baseline')

    # Shade improvement region
    ax.fill_between(delta_range, width_ratios, 1.0,
                   where=np.array(width_ratios) < 1.0,
                   color='green', alpha=0.2, label='Width reduction')

    # Annotate max improvement
    min_ratio = np.nanmin(width_ratios)
    min_idx = np.nanargmin(width_ratios)
    ax.scatter([delta_range[min_idx]], [min_ratio], color='red', s=100, zorder=5)
    ax.annotate(f'Max gain: {(1-min_ratio)*100:.0f}%',
               xy=(delta_range[min_idx], min_ratio),
               xytext=(delta_range[min_idx] + 0.5, min_ratio + 0.05),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel(r'Prior-data conflict $|\Delta|$', fontsize=12)
    ax.set_ylabel(r'$E[W_{\eta^*}]/W_{Wald}$', fontsize=12)
    ax.set_title('(B) Width Ratio at Optimal Tilting (w=0.5)\nMLP provides efficiency gains across all conflict levels',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0.7, 1.1)
    ax.grid(True, alpha=0.3)

    plt.suptitle('MLP-Based Optimal Tilting Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        save_figure(fig, "fig_3_2_mlp_surfaces", "tilting")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

FIGURE_FUNCTIONS = {
    "3.1": figure_3_1_mlp_architecture,
    "3.2": figure_3_2_mlp_surfaces,
}


def main():
    parser = argparse.ArgumentParser(description="Generate tilting figures (3.1-3.2)")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (reduced MC samples)")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (3.1-3.2)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    fast = args.fast

    print("="*60)
    print("TILTING VALIDATION & METHODOLOGY FIGURES")
    if fast:
        print("(Fast mode - reduced MC samples)")
    print("="*60)

    figures_to_generate = list(FIGURE_FUNCTIONS.keys())
    if args.figure:
        figures_to_generate = [args.figure]

    for name in figures_to_generate:
        if name in FIGURE_FUNCTIONS:
            try:
                FIGURE_FUNCTIONS[name](save=save, show=show, fast=fast)
            except Exception as e:
                print(f"Error generating figure {name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Unknown figure: {name}")
            print(f"Available: {list(FIGURE_FUNCTIONS.keys())}")

    print("\n" + "="*60)
    print("DONE - Tilting figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
