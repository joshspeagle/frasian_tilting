#!/usr/bin/env python3
"""
Summary and Comparison Figures (Category 7)

Generates figures 7.1-7.3:
- Figure 7.1: Method Comparison Summary (2x3 panel)
- Figure 7.2: Decision Flowchart
- Figure 7.3: Unified Framework Schematic

Usage:
    python scripts/plot_summary.py [--fast] [--no-save] [--show]
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from scipy import stats
from tqdm import tqdm

from frasian.core import posterior_params, scaled_conflict, prior_residual
from frasian.waldo import (
    pvalue, confidence_interval, wald_ci, posterior_ci,
    confidence_interval_width, wald_ci_width, posterior_ci_width
)
from frasian.tilting import tilted_ci
from frasian.simulations import optimal_eta_empirical
from frasian.figure_style import (
    COLORS, FIGSIZE, MC_CONFIG, setup_style, save_figure,
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


def compute_coverage(theta_true, mu0, sigma, sigma0, n_reps=1000, alpha=0.05):
    """Compute coverage rates for all methods."""
    wald_covers = 0
    post_covers = 0
    waldo_covers = 0

    for _ in range(n_reps):
        D = np.random.normal(theta_true, sigma)

        # Wald CI
        wald_lo, wald_hi = wald_ci(D, sigma, alpha)
        if wald_lo <= theta_true <= wald_hi:
            wald_covers += 1

        # Posterior CI
        post_lo, post_hi = posterior_ci(D, mu0, sigma, sigma0, alpha)
        if post_lo <= theta_true <= post_hi:
            post_covers += 1

        # WALDO CI
        waldo_lo, waldo_hi = confidence_interval(D, mu0, sigma, sigma0, alpha)
        if waldo_lo <= theta_true <= waldo_hi:
            waldo_covers += 1

    return wald_covers/n_reps, post_covers/n_reps, waldo_covers/n_reps


# =============================================================================
# Figure 7.1: Method Comparison Summary
# =============================================================================

def figure_7_1_method_comparison(
    save: bool = True,
    show: bool = False,
    fast: bool = False,
) -> plt.Figure:
    """
    Generate Figure 7.1: Method Comparison Summary (2x3 panel).

    Row 1: Coverage for Wald, Posterior, WALDO
    Row 2: Width for same three methods
    Columns: Low, Medium, High conflict scenarios.
    """
    print("\n" + "="*60)
    print("Figure 7.1: Method Comparison Summary")
    print("="*60)

    n_reps = 500 if fast else MC_CONFIG['n_coverage']

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Conflict scenarios
    scenarios = [
        ('Low Conflict', 0, [-1, 0, 1]),
        ('Medium Conflict', -1.5, [-1, 1, 3]),
        ('High Conflict', -3, [0, 3, 6]),
    ]

    for col, (scenario_name, delta_center, theta_values) in enumerate(scenarios):
        print(f"\n{scenario_name} (Delta ~ {delta_center}):")

        # Coverage plot (top row)
        ax_cov = axes[0, col]

        methods = ['Wald', 'Posterior', 'WALDO']
        colors_list = [COLORS['wald'], COLORS['posterior'], COLORS['waldo']]

        coverage_data = {m: [] for m in methods}

        for theta in tqdm(theta_values, desc=f"Coverage {scenario_name}", leave=False):
            wald_cov, post_cov, waldo_cov = compute_coverage(
                theta, mu0, sigma, sigma0, n_reps=n_reps
            )
            coverage_data['Wald'].append(wald_cov * 100)
            coverage_data['Posterior'].append(post_cov * 100)
            coverage_data['WALDO'].append(waldo_cov * 100)

        x = np.arange(len(theta_values))
        width_bar = 0.25

        for i, (method, color) in enumerate(zip(methods, colors_list)):
            bars = ax_cov.bar(x + i*width_bar, coverage_data[method], width_bar,
                             label=method, color=color, alpha=0.8)
            # Add value labels
            for bar, val in zip(bars, coverage_data[method]):
                ax_cov.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                              xytext=(0, 2), textcoords='offset points',
                              ha='center', va='bottom', fontsize=8)

        ax_cov.axhline(y=95, color='black', linestyle='--', linewidth=1, label='95% target')
        ax_cov.axhspan(93.5, 96.5, alpha=0.2, color='green')

        ax_cov.set_xlabel(r'True $\theta$', fontsize=11)
        ax_cov.set_ylabel('Coverage %', fontsize=11)
        ax_cov.set_title(f'{scenario_name}', fontsize=12, fontweight='bold')
        ax_cov.set_xticks(x + width_bar)
        ax_cov.set_xticklabels([f'{t}' for t in theta_values])
        ax_cov.set_ylim(0, 110)
        if col == 0:
            ax_cov.legend(fontsize=8, loc='lower left')

        # Width plot (bottom row)
        ax_width = axes[1, col]

        width_data = {m: [] for m in methods}

        for theta in theta_values:
            D = theta  # Use D = theta for width comparison
            delta = abs(scaled_conflict(D, mu0, w, sigma))

            # Compute widths
            wald_w = wald_ci_width(sigma)
            post_w = posterior_ci_width(sigma, sigma0)
            waldo_w = confidence_interval_width(D, mu0, sigma, sigma0)

            width_data['Wald'].append(wald_w)
            width_data['Posterior'].append(post_w)
            width_data['WALDO'].append(waldo_w)

        for i, (method, color) in enumerate(zip(methods, colors_list)):
            bars = ax_width.bar(x + i*width_bar, width_data[method], width_bar,
                               label=method, color=color, alpha=0.8)
            for bar, val in zip(bars, width_data[method]):
                ax_width.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                                xytext=(0, 2), textcoords='offset points',
                                ha='center', va='bottom', fontsize=8)

        ax_width.set_xlabel('D value', fontsize=11)
        ax_width.set_ylabel('CI Width', fontsize=11)
        ax_width.set_xticks(x + width_bar)
        ax_width.set_xticklabels([f'{t}' for t in theta_values])
        if col == 0:
            ax_width.legend(fontsize=8, loc='upper left')

    # Row labels
    axes[0, 0].set_ylabel('Coverage %\n(target: 95%)', fontsize=11)
    axes[1, 0].set_ylabel('CI Width\n(narrower = better)', fontsize=11)

    fig.suptitle('Method Comparison Across Conflict Levels\n'
                 'WALDO maintains 95% coverage; trades width for validity',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, "fig_7_1_method_comparison", "summary")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 7.2: Decision Flowchart
# =============================================================================

def figure_7_2_decision_flowchart(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 7.2: Decision Flowchart.

    Practical guidance for method selection.
    """
    print("\n" + "="*60)
    print("Figure 7.2: Decision Flowchart")
    print("="*60)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Box style helper
    def draw_box(x, y, w, h, text, color, fontsize=10, textcolor='black'):
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
               color=textcolor, fontweight='bold', wrap=True)

    # Decision diamond
    def draw_diamond(x, y, size, text, fontsize=9):
        diamond = plt.Polygon([(x, y+size), (x+size, y), (x, y-size), (x-size, y)],
                             facecolor='lightyellow', edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
               fontweight='bold', wrap=True)

    # Arrow helper
    def draw_arrow(x1, y1, x2, y2, label='', label_pos=0.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        if label:
            mx, my = x1 + (x2-x1)*label_pos, y1 + (y2-y1)*label_pos
            ax.text(mx + 0.3, my, label, fontsize=9, fontweight='bold')

    # Start
    draw_box(6, 9.2, 3, 0.7, 'START:\nHave prior + data', 'lightgray')

    # Decision 1
    draw_diamond(6, 7.5, 0.9, 'Is prior-data\nconflict severe?\n(|Δ| > 2?)', fontsize=9)

    draw_arrow(6, 9.2 - 0.35, 6, 7.5 + 0.9)

    # Branch: No conflict
    draw_box(2.5, 5.5, 2.8, 1.2, 'Use Posterior\n• Narrowest CIs\n• Correct coverage\nnear prior', COLORS['posterior'], 9, 'white')
    draw_arrow(6 - 0.9, 7.5, 2.5 + 1.4, 5.5 + 0.6, 'No')

    # Branch: Yes conflict
    draw_diamond(9, 5.5, 0.8, 'Need simple\ncomputation?', fontsize=9)
    draw_arrow(6 + 0.9, 7.5, 9 - 0.8, 5.5 + 0.8, 'Yes')

    # Simple computation: Wald
    draw_box(11, 3.5, 2, 1.2, 'Use Wald\n• Ignores prior\n• Always 95%\n• Fixed width', COLORS['wald'], 9, 'white')
    draw_arrow(9 + 0.8, 5.5, 11 - 1, 3.5 + 0.6, 'Yes')

    # Need prior info
    draw_diamond(6, 3.5, 0.8, 'Want optimal\nefficiency?', fontsize=9)
    draw_arrow(9 - 0.8, 5.5, 6 + 0.8, 3.5 + 0.8, 'No')

    # WALDO
    draw_box(3, 1.5, 2.8, 1.2, 'Use WALDO\n• Uses prior info\n• Always 95%\n• Adaptive width', COLORS['waldo'], 9, 'white')
    draw_arrow(6 - 0.8, 3.5, 3 + 1.4, 1.5 + 0.6, 'No')

    # Tilted
    draw_box(9, 1.5, 3, 1.2, 'Use Tilted(η*)\n• Optimal tilting\n• Always 95%\n• Narrower on avg', COLORS['tilted'], 9, 'white')
    draw_arrow(6 + 0.8, 3.5, 9 - 1.5, 1.5 + 0.6, 'Yes')

    # Summary box
    summary_text = ('Key Trade-offs:\n'
                   '• Posterior: Efficient but wrong coverage when |Δ| large\n'
                   '• Wald: Always correct but ignores prior (widest)\n'
                   '• WALDO: Correct + uses prior (intermediate width)\n'
                   '• Tilted: Correct + optimal (narrowest on average)')
    ax.text(6, 0.3, summary_text, ha='center', va='bottom', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_title('Method Selection Flowchart', fontsize=14, fontweight='bold', y=1.0)

    if save:
        save_figure(fig, "fig_7_2_decision_flowchart", "summary")

    if show:
        plt.show()

    return fig


# =============================================================================
# Figure 7.3: Unified Framework Schematic
# =============================================================================

def figure_7_3_unified_framework(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Generate Figure 7.3: Unified Framework Schematic.

    Shows how all pieces fit together.
    """
    print("\n" + "="*60)
    print("Figure 7.3: Unified Framework Schematic")
    print("="*60)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Box helper
    def draw_box(x, y, w, h, text, color, fontsize=10, alpha=1.0):
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=alpha)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
               fontweight='bold', wrap=True)

    # Arrow with label
    def draw_arrow(x1, y1, x2, y2, label='', curved=False, color='black'):
        if curved:
            style = "arc3,rad=0.3"
        else:
            style = "arc3,rad=0"
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                  connectionstyle=style))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.3, label, fontsize=9, ha='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # === Input layer ===
    draw_box(3, 9, 2.5, 1, 'Prior\nθ ~ N(μ₀, σ₀²)', 'lightblue', 10)
    draw_box(7, 9, 2.5, 1, 'Likelihood\nD|θ ~ N(θ, σ²)', 'lightgreen', 10)

    # === Posterior ===
    draw_box(5, 7, 3, 1, 'Posterior\nθ|D ~ N(μₙ, σₙ²)', COLORS['posterior'], 10)
    draw_arrow(3, 8.5, 5, 7.5)
    draw_arrow(7, 8.5, 5, 7.5)

    # === Key quantities ===
    draw_box(10, 7, 2.5, 1.5, 'Weight\nw = σ₀²/(σ²+σ₀²)\n\nConflict\nΔ = (1-w)(μ₀-D)/σ', 'lightyellow', 9)
    draw_arrow(5 + 1.5, 7, 10 - 1.25, 7, 'computes')

    # === WALDO branch ===
    draw_box(3, 4.5, 3, 1.2, 'WALDO Statistic\nτ ~ w·χ²₁(λ(θ))', COLORS['waldo'], 10)
    draw_arrow(5, 6.5, 3, 5.1)
    ax.text(3.5, 5.8, 'Thm 2', fontsize=8)

    # Non-centrality
    draw_box(3, 2.5, 3, 1, 'Non-centrality\nλ(θ) = δ(θ)²/w', 'lightcyan', 10)
    draw_arrow(3, 3.9, 3, 3, 'depends on')

    # === P-value branch ===
    draw_box(7, 4.5, 3.5, 1.2, 'P-value Function\np(θ) = Φ(b-a) + Φ(-a-b)', 'plum', 10)
    draw_arrow(5, 6.5, 7, 5.1)
    ax.text(6.5, 5.8, 'Thm 3', fontsize=8)

    # === Tilting branch ===
    draw_box(11, 4.5, 2.5, 1, 'Tilting η ∈ [0,1]\nη=0: WALDO\nη=1: Wald', COLORS['tilted'], 9)
    draw_arrow(7 + 1.75, 4.5, 11 - 1.25, 4.5, 'extends')
    ax.text(9.5, 4.8, 'Thm 6-8', fontsize=8)

    # === CI outputs ===
    draw_box(3, 0.8, 2.5, 1, 'WALDO CI\n95% coverage', COLORS['waldo'], 10)
    draw_box(7, 0.8, 2.5, 1, 'Tilted CI\n95% + narrower', COLORS['tilted'], 10)
    draw_box(11, 0.8, 2.5, 1, 'Wald CI\n95% (no prior)', COLORS['wald'], 10)

    draw_arrow(3, 1.9, 3, 1.3)
    draw_arrow(7, 3.9, 7, 1.3)
    draw_arrow(11, 4, 11, 1.3)

    # === Key insight box ===
    insight_text = ('Unified Framework:\n'
                   '• Posterior mean μₙ is mode of confidence distribution (Thm 4)\n'
                   '• Non-centrality λ captures prior influence on coverage\n'
                   '• Tilting η controls interpolation between Bayesian & frequentist\n'
                   '• Optimal η*(|Δ|) achieves correct coverage + efficiency')

    ax.text(7, -0.5, insight_text, ha='center', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='black'))

    ax.set_title('The Frasian Inference Framework: Unified View',
                fontsize=14, fontweight='bold', y=0.98)

    if save:
        save_figure(fig, "fig_7_3_unified_framework", "summary")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate summary figures")
    parser.add_argument("--fast", action="store_true", help="Use fewer MC samples")
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--figure", type=str, help="Generate only specific figure (7.1, 7.2, 7.3)")
    args = parser.parse_args()

    setup_style()

    save = not args.no_save
    show = args.show
    fast = args.fast

    print("="*60)
    print("SUMMARY AND COMPARISON FIGURES")
    print("="*60)

    figures_to_generate = ['7.1', '7.2', '7.3']
    if args.figure:
        figures_to_generate = [args.figure]

    if '7.1' in figures_to_generate:
        figure_7_1_method_comparison(save=save, show=show, fast=fast)

    if '7.2' in figures_to_generate:
        figure_7_2_decision_flowchart(save=save, show=show)

    if '7.3' in figures_to_generate:
        figure_7_3_unified_framework(save=save, show=show)

    print("\n" + "="*60)
    print("DONE - Summary figures generated")
    print("="*60)


if __name__ == "__main__":
    main()
