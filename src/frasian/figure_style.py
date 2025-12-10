"""
Shared styling configuration for publication-quality figures.

Provides consistent colors, fonts, and figure dimensions across all
visualization scripts for the Frasian inference framework.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

# =============================================================================
# Color Palette (Colorblind-friendly)
# =============================================================================

COLORS = {
    # Primary method colors
    "waldo": "#2E86AB",      # Blue - WALDO
    "posterior": "#28A745",   # Green - Posterior
    "wald": "#DC3545",        # Red - Wald
    "tilted": "#6F42C1",      # Purple - Tilted

    # Secondary colors
    "mle": "#FD7E14",         # Orange - MLE/Data
    "prior_mean": "#17A2B8",  # Cyan - Prior mean
    "mode": "#20C997",        # Teal - Mode
    "mean": "#E83E8C",        # Pink - Mean

    # Neutral colors
    "grid": "#CCCCCC",        # Light gray for grid
    "annotation": "#495057",  # Dark gray for annotations
    "ci_fill": "#2E86AB",     # Same as WALDO for CI shading

    # Coverage scale
    "coverage_good": "#28A745",   # Green - correct coverage
    "coverage_ok": "#FFC107",     # Yellow - marginal
    "coverage_bad": "#DC3545",    # Red - severe undercoverage
}

# Tilting gradient (from WALDO blue to Wald red)
def get_tilting_colors(n_etas: int) -> list:
    """Get color gradient for tilting parameter values."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "tilting", [COLORS["waldo"], COLORS["tilted"], COLORS["wald"]]
    )
    return [cmap(i / (n_etas - 1)) for i in range(n_etas)]


# =============================================================================
# Figure Dimensions
# =============================================================================

FIGSIZE = {
    "single": (6, 4),       # Single column
    "double": (12, 4),      # Double column (wide)
    "square": (6, 6),       # Square
    "tall": (6, 8),         # Tall single column
    "panel_2x2": (10, 8),   # 2x2 panel figure
    "panel_1x3": (12, 4),   # 1x3 panel figure
    "panel_2x3": (12, 8),   # 2x3 panel figure
}

DPI = {
    "screen": 100,
    "print": 300,
    "publication": 600,
}


# =============================================================================
# Monte Carlo Settings
# =============================================================================

MC_CONFIG = {
    "n_samples": 10_000,      # Default MC samples
    "n_coverage": 5_000,      # Coverage simulation replicates
    "seed": 42,               # Base random seed
    "n_theta_grid": 50,       # Points for theta grid
    "n_delta_grid": 100,      # Points for Delta grid
}


# =============================================================================
# Style Configuration
# =============================================================================

def setup_style(use_latex: bool = False) -> None:
    """
    Configure matplotlib for publication-quality figures.

    Parameters
    ----------
    use_latex : bool
        If True, use LaTeX for text rendering (requires LaTeX installation)
    """
    # Reset to defaults first
    plt.rcdefaults()

    # Font settings
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        })

    # Font sizes
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    # Line and marker settings
    plt.rcParams.update({
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
    })

    # Grid settings
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })

    # Legend settings
    plt.rcParams.update({
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })

    # Figure settings
    plt.rcParams.update({
        "figure.dpi": DPI["screen"],
        "savefig.dpi": DPI["print"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def get_output_dir(category: str) -> Path:
    """Get the output directory for a figure category."""
    base = Path(__file__).parent.parent.parent / "output" / "figures"
    return base / category


def save_figure(
    fig: plt.Figure,
    name: str,
    category: str,
    formats: Tuple[str, ...] = ("png", "pdf"),
    dpi: Optional[int] = None,
) -> None:
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    name : str
        Base filename (without extension)
    category : str
        Figure category subdirectory
    formats : tuple
        Output formats to generate
    dpi : int, optional
        Override default DPI
    """
    output_dir = get_output_dir(category)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dpi = dpi or DPI["print"]

    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved: {filepath}")


# =============================================================================
# Common Plot Elements
# =============================================================================

def add_reference_lines(
    ax: plt.Axes,
    alpha: float = 0.05,
    show_alpha: bool = True,
    show_nominal_coverage: bool = False,
) -> None:
    """Add common reference lines to a plot."""
    if show_alpha:
        ax.axhline(y=alpha, color=COLORS["annotation"], linestyle="--",
                   linewidth=1, alpha=0.7, label=f"$\\alpha = {alpha}$")

    if show_nominal_coverage:
        ax.axhline(y=0.95, color=COLORS["coverage_good"], linestyle="--",
                   linewidth=1, alpha=0.7, label="95% nominal")
        ax.axhspan(0.935, 0.965, alpha=0.1, color=COLORS["coverage_good"],
                   label="Acceptable range")


def add_method_legend(ax: plt.Axes, loc: str = "best") -> None:
    """Add a legend with method colors."""
    ax.legend(loc=loc, framealpha=0.9)


def format_theta_axis(ax: plt.Axes, label: str = r"$\theta$") -> None:
    """Format theta axis with proper label."""
    ax.set_xlabel(label)


def format_pvalue_axis(ax: plt.Axes) -> None:
    """Format p-value axis."""
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 1.05)


def format_coverage_axis(ax: plt.Axes) -> None:
    """Format coverage axis."""
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)


def format_width_axis(ax: plt.Axes) -> None:
    """Format CI width axis."""
    ax.set_ylabel("CI Width")


# =============================================================================
# Coverage Heatmap Utilities
# =============================================================================

def get_coverage_cmap():
    """Get colormap for coverage heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap

    # Red (bad) -> Yellow (marginal) -> Green (good)
    colors = [
        (0.0, COLORS["coverage_bad"]),
        (0.5, COLORS["coverage_ok"]),
        (1.0, COLORS["coverage_good"]),
    ]

    # Create custom colormap centered at 95%
    return LinearSegmentedColormap.from_list(
        "coverage",
        [(0.0, "#DC3545"), (0.475, "#FFC107"), (0.5, "#28A745"),
         (0.525, "#28A745"), (1.0, "#28A745")]
    )


def coverage_to_color(coverage: float, target: float = 0.95) -> str:
    """Map coverage value to color."""
    if coverage < target - 0.10:
        return COLORS["coverage_bad"]
    elif coverage < target - 0.015:
        return COLORS["coverage_ok"]
    else:
        return COLORS["coverage_good"]


# =============================================================================
# Annotation Utilities
# =============================================================================

def annotate_point(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    offset: Tuple[float, float] = (10, 10),
    **kwargs
) -> None:
    """Add annotation to a point."""
    ax.annotate(
        text, (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=9,
        color=COLORS["annotation"],
        **kwargs
    )


def add_vertical_marker(
    ax: plt.Axes,
    x: float,
    label: str,
    color: str,
    linestyle: str = ":",
    **kwargs
) -> None:
    """Add a labeled vertical line marker."""
    ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=1,
               alpha=0.7, label=label, **kwargs)


# =============================================================================
# Initialize style on import
# =============================================================================

setup_style(use_latex=False)
