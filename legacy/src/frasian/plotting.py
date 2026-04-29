"""
Visualization utilities for Frasian inference.

Provides functions for plotting:
- P-value functions
- Confidence intervals
- Coverage curves
- CI width comparisons
- Tilting parameter relationships
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

from .core import posterior_params, weight, scaled_conflict
from .waldo import (
    pvalue,
    confidence_interval,
    confidence_interval_width,
    wald_ci_width,
    posterior_ci_width,
)
from .tilting import (
    tilted_pvalue,
    tilted_ci_width,
    optimal_eta_approximation,
)


def plot_pvalue_function(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    theta_range: Optional[Tuple[float, float]] = None,
    n_points: int = 200,
    ax: Optional[plt.Axes] = None,
    show_ci: bool = True,
    alpha: float = 0.05,
) -> plt.Axes:
    """
    Plot the WALDO p-value function.

    Parameters
    ----------
    D : float
        Observed data
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    theta_range : tuple, optional
        (min, max) for theta axis
    n_points : int
        Number of points to plot
    ax : Axes, optional
        Matplotlib axes to plot on
    show_ci : bool
        Whether to show confidence interval
    alpha : float
        Significance level for CI

    Returns
    -------
    ax : Axes
        The matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    if theta_range is None:
        theta_range = (mu_n - 4 * sigma, mu_n + 4 * sigma)

    thetas = np.linspace(theta_range[0], theta_range[1], n_points)
    pvals = np.array([pvalue(theta, mu_n, mu0, w, sigma) for theta in thetas])

    ax.plot(thetas, pvals, 'b-', linewidth=2, label='p(θ)')
    ax.axhline(y=alpha, color='r', linestyle='--', label=f'α = {alpha}')
    ax.axvline(x=mu_n, color='g', linestyle=':', label=f'Mode = μₙ = {mu_n:.2f}')
    ax.axvline(x=D, color='orange', linestyle=':', alpha=0.7, label=f'MLE = D = {D:.2f}')

    if show_ci:
        lower, upper = confidence_interval(D, mu0, sigma, sigma0, alpha)
        ax.axvspan(lower, upper, alpha=0.2, color='blue', label=f'{100*(1-alpha):.0f}% CI')

    ax.set_xlabel('θ')
    ax.set_ylabel('p-value')
    ax.set_title('WALDO P-value Function')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    return ax


def plot_ci_comparison(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.05,
) -> plt.Axes:
    """
    Plot comparison of WALDO, Wald, and Posterior CIs.

    Parameters
    ----------
    D : float
        Observed data
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    ax : Axes, optional
        Matplotlib axes to plot on
    alpha : float
        Significance level

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    from scipy import stats

    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)
    z = stats.norm.ppf(1 - alpha / 2)

    # WALDO CI
    waldo_lower, waldo_upper = confidence_interval(D, mu0, sigma, sigma0, alpha)

    # Wald CI
    wald_lower, wald_upper = D - z * sigma, D + z * sigma

    # Posterior CI
    post_lower, post_upper = mu_n - z * sigma_n, mu_n + z * sigma_n

    methods = ['Posterior', 'WALDO', 'Wald']
    lowers = [post_lower, waldo_lower, wald_lower]
    uppers = [post_upper, waldo_upper, wald_upper]
    colors = ['green', 'blue', 'red']

    y_positions = [0, 1, 2]

    for y, name, lo, hi, color in zip(y_positions, methods, lowers, uppers, colors):
        ax.plot([lo, hi], [y, y], color=color, linewidth=8, solid_capstyle='butt', label=name)
        ax.plot([lo, hi], [y, y], 'k|', markersize=15)

    ax.axvline(x=D, color='orange', linestyle='--', label=f'D = {D:.2f}')
    ax.axvline(x=mu_n, color='purple', linestyle=':', label=f'μₙ = {mu_n:.2f}')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods)
    ax.set_xlabel('θ')
    ax.set_title(f'Confidence Interval Comparison (D={D:.1f}, w={w:.2f})')
    ax.legend(loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)

    return ax


def plot_ci_width_vs_conflict(
    w: float = 0.5,
    sigma: float = 1.0,
    delta_range: Tuple[float, float] = (-5, 5),
    n_points: int = 50,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.05,
) -> plt.Axes:
    """
    Plot CI width as a function of prior-data conflict.

    Parameters
    ----------
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation
    delta_range : tuple
        Range of Delta values
    n_points : int
        Number of points
    ax : Axes, optional
        Matplotlib axes
    alpha : float
        Significance level

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Compute sigma0 from w
    sigma0 = sigma * np.sqrt(w / (1 - w))
    mu0 = 0.0

    deltas = np.linspace(delta_range[0], delta_range[1], n_points)

    # Wald width (constant)
    w_wald = wald_ci_width(sigma, alpha)

    # Posterior width (constant)
    w_post = posterior_ci_width(sigma, sigma0, alpha)

    # WALDO widths
    waldo_widths = []
    for delta in deltas:
        D = mu0 - sigma * delta / (1 - w)
        w_waldo = confidence_interval_width(D, mu0, sigma, sigma0, alpha)
        waldo_widths.append(w_waldo)

    ax.plot(deltas, waldo_widths, 'b-', linewidth=2, label='WALDO')
    ax.axhline(y=w_wald, color='r', linestyle='--', label=f'Wald ({w_wald:.2f})')
    ax.axhline(y=w_post, color='g', linestyle=':', label=f'Posterior ({w_post:.2f})')

    ax.set_xlabel('Δ (prior-data conflict)')
    ax.set_ylabel('CI Width')
    ax.set_title(f'CI Width vs Prior-Data Conflict (w={w:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_optimal_tilting(
    delta_range: Tuple[float, float] = (0, 5),
    n_points: int = 50,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot optimal tilting parameter as a function of |Delta|.

    Parameters
    ----------
    delta_range : tuple
        Range of |Delta| values
    n_points : int
        Number of points
    ax : Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    deltas = np.linspace(delta_range[0], delta_range[1], n_points)
    etas = [optimal_eta_approximation(d) for d in deltas]

    ax.plot(deltas, etas, 'b-', linewidth=2)

    # Mark key points from the document
    key_deltas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    key_etas = [0.23, 0.52, 0.77, 0.94, 0.97, 0.99]

    ax.scatter(key_deltas, key_etas, color='red', s=50, zorder=5, label='Reference values')

    ax.set_xlabel('|Δ| (prior-data conflict)')
    ax.set_ylabel('η* (optimal tilting)')
    ax.set_title('Optimal Tilting Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    return ax


def plot_tilted_pvalue_comparison(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    etas: List[float] = [0.0, 0.5, 1.0],
    theta_range: Optional[Tuple[float, float]] = None,
    n_points: int = 200,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot tilted p-value functions for different eta values.

    Parameters
    ----------
    D : float
        Observed data
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    etas : list
        List of tilting parameters to plot
    theta_range : tuple, optional
        (min, max) for theta axis
    n_points : int
        Number of points
    ax : Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

    if theta_range is None:
        theta_range = (D - 4 * sigma, D + 4 * sigma)

    thetas = np.linspace(theta_range[0], theta_range[1], n_points)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(etas)))

    for eta, color in zip(etas, colors):
        pvals = [tilted_pvalue(theta, D, mu0, sigma, sigma0, eta) for theta in thetas]
        label = f'η={eta:.1f}' if eta < 1 else 'η=1 (Wald)'
        if eta == 0:
            label = 'η=0 (WALDO)'
        ax.plot(thetas, pvals, color=color, linewidth=2, label=label)

    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α=0.05')
    ax.axvline(x=D, color='orange', linestyle=':', alpha=0.7, label=f'D={D:.1f}')

    ax.set_xlabel('θ')
    ax.set_ylabel('p-value')
    ax.set_title('Tilted P-value Functions')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    return ax


def create_summary_figure(
    D: float = 2.0,
    mu0: float = 0.0,
    sigma: float = 1.0,
    sigma0: float = 1.0,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Create a summary figure with multiple panels.

    Parameters
    ----------
    D : float
        Observed data
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: P-value function
    plot_pvalue_function(D, mu0, sigma, sigma0, ax=axes[0, 0])

    # Panel 2: CI comparison
    plot_ci_comparison(D, mu0, sigma, sigma0, ax=axes[0, 1])

    # Panel 3: CI width vs conflict
    w = weight(sigma, sigma0)
    plot_ci_width_vs_conflict(w=w, sigma=sigma, ax=axes[1, 0])

    # Panel 4: Optimal tilting
    plot_optimal_tilting(ax=axes[1, 1])

    fig.suptitle(f'Frasian Inference Summary (D={D}, μ₀={mu0}, σ={sigma}, σ₀={sigma0})',
                 fontsize=14)
    plt.tight_layout()

    return fig
