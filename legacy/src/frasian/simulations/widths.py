"""
CI Width Simulation Experiments

Simulates CI widths across different conflict levels and tilting parameters.
"""

from typing import Literal, Optional

import numpy as np
from tqdm import tqdm

from ..core import posterior_params, scaled_conflict, weight
from ..waldo import confidence_interval_width, wald_ci_width, posterior_ci_width
from ..tilting import tilted_ci, optimal_eta_approximation
from ..uncertainty import mean_se, mean_ci


MethodType = Literal["waldo", "tilted", "wald", "posterior"]


def get_model_params(w: float, sigma: float = 1.0, mu0: float = 0.0):
    """Get model parameters for a given weight."""
    if w <= 0 or w >= 1:
        raise ValueError(f"w must be in (0, 1), got {w}")
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return mu0, sigma, sigma0


def data_for_conflict(delta: float, mu0: float, w: float, sigma: float) -> float:
    """Compute D value that produces a given Delta conflict."""
    # Delta = (1-w)(mu0 - D) / sigma
    # => D = mu0 - sigma * Delta / (1 - w)
    return mu0 - sigma * delta / (1 - w)


def simulate_width_samples(
    delta: float,
    method: MethodType,
    n_samples: int,
    eta: Optional[float] = None,
    w: float = 0.5,
    alpha: float = 0.05,
    sigma: float = 1.0,
    mu0: float = 0.0,
    seed: Optional[int] = None
) -> dict:
    """Simulate CI widths for a given conflict level.

    Note: We sample D from its true distribution (centered at theta_true)
    where theta_true is chosen to produce the specified Delta conflict.

    Args:
        delta: Prior-data conflict level (Delta)
        method: CI method - "waldo", "tilted", "wald", "posterior"
        n_samples: Number of width samples to generate
        eta: Tilting parameter (required for "tilted" method, uses optimal if None)
        w: Prior weight
        alpha: Significance level for CIs
        sigma: Likelihood standard deviation
        mu0: Prior mean
        seed: Random seed

    Returns:
        Dictionary with:
            - "widths": Array of CI widths
            - "mean_width": Mean width
            - "se": Standard error of mean width
            - "ci_low", "ci_high": 95% CI for mean width
    """
    rng = np.random.default_rng(seed)

    mu0, sigma, sigma0 = get_model_params(w, sigma, mu0)

    # For tilted method with optimal eta
    if method == "tilted" and eta is None:
        eta = optimal_eta_approximation(abs(delta))

    widths = np.zeros(n_samples)

    # theta_true that produces the given Delta
    # Note: We simulate D from theta_true, but the observed Delta depends on D
    # For simplicity, we use theta = D that would give the target Delta
    # This means we're sampling D near values that produce approximately that conflict
    D_target = data_for_conflict(delta, mu0, w, sigma)

    for i in range(n_samples):
        # Sample D from its true distribution centered at D_target
        # (This simulates the sampling variability of D)
        D = rng.normal(D_target, sigma)

        # Compute CI width
        if method == "wald":
            widths[i] = wald_ci_width(sigma, alpha)
        elif method == "posterior":
            widths[i] = posterior_ci_width(sigma, sigma0, alpha)
        elif method == "waldo":
            widths[i] = confidence_interval_width(D, mu0, sigma, sigma0, alpha)
        elif method == "tilted":
            lo, hi = tilted_ci(D, mu0, sigma, sigma0, eta, alpha)
            widths[i] = hi - lo
        else:
            raise ValueError(f"Unknown method: {method}")

    # Compute summary statistics
    mean_width = np.mean(widths)
    se = mean_se(widths)
    ci_lo, ci_hi = mean_ci(widths, alpha=0.05)

    return {
        "widths": widths,
        "mean_width": mean_width,
        "se": se,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "delta": delta,
        "method": method,
        "eta": eta,
        "n_samples": n_samples,
    }


def simulate_width_grid(
    delta_grid: np.ndarray,
    methods: list[MethodType],
    n_samples: int,
    eta_values: Optional[list[float]] = None,
    w: float = 0.5,
    alpha: float = 0.05,
    sigma: float = 1.0,
    mu0: float = 0.0,
    seed: int = 42,
    show_progress: bool = True
) -> dict[str, np.ndarray]:
    """Simulate CI widths across a grid of conflict levels.

    For "tilted" method, simulates with each eta in eta_values.
    Also includes "tilted_optimal" using the optimal eta for each Delta.

    Args:
        delta_grid: Array of Delta values to test
        methods: List of methods (excluding "tilted" which is handled specially)
        n_samples: Samples per cell
        eta_values: List of eta values for tilted method (default: [0, 0.25, 0.5, 0.75, 1])
        w: Prior weight
        alpha: Significance level
        sigma: Likelihood standard deviation
        mu0: Prior mean
        seed: Base random seed
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with:
            - "delta_grid": Array of Delta values
            - "mean_widths": Dict mapping method -> array of mean widths
            - "se": Dict mapping method -> array of SEs
            - "widths": Dict mapping method -> 2D array [n_delta, n_samples]
    """
    if eta_values is None:
        eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    delta_grid = np.asarray(delta_grid)
    n_delta = len(delta_grid)

    # Build list of (method, eta) pairs to simulate
    method_configs = []
    for m in methods:
        if m == "tilted":
            # Add fixed eta values
            for eta in eta_values:
                method_configs.append((f"tilted_eta{eta:.2f}", m, eta))
            # Add optimal eta
            method_configs.append(("tilted_optimal", m, None))
        else:
            method_configs.append((m, m, None))

    # Initialize output
    mean_widths = {name: np.zeros(n_delta) for name, _, _ in method_configs}
    se_widths = {name: np.zeros(n_delta) for name, _, _ in method_configs}
    all_widths = {name: np.zeros((n_delta, n_samples)) for name, _, _ in method_configs}

    # Progress tracking
    total_cells = n_delta * len(method_configs)
    pbar = tqdm(total=total_cells, desc="Width simulation", disable=not show_progress)

    cell = 0
    for i, delta in enumerate(delta_grid):
        for name, method, eta in method_configs:
            result = simulate_width_samples(
                delta=delta,
                method=method,
                n_samples=n_samples,
                eta=eta,
                w=w,
                alpha=alpha,
                sigma=sigma,
                mu0=mu0,
                seed=seed + cell * 1000
            )

            mean_widths[name][i] = result["mean_width"]
            se_widths[name][i] = result["se"]
            all_widths[name][i, :] = result["widths"]

            cell += 1
            pbar.update(1)

    pbar.close()

    return {
        "delta_grid": delta_grid,
        "mean_widths": mean_widths,
        "se": se_widths,
        "widths": all_widths,
        "methods": [name for name, _, _ in method_configs],
        "n_samples": n_samples,
        "w": w,
        "alpha": alpha,
    }


def run_standard_width_simulation(
    n_samples: int = 5000,
    seed: int = 42,
    show_progress: bool = True
) -> dict[str, np.ndarray]:
    """Run width simulation with standard parameters.

    Standard configuration:
        - delta: [0, 5] in 51 points
        - methods: [waldo, tilted]
        - eta_values: [0, 0.25, 0.5, 0.75, 1.0]
        - w: 0.5

    Args:
        n_samples: Samples per cell
        seed: Random seed
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with width results
    """
    return simulate_width_grid(
        delta_grid=np.linspace(0, 5, 51),
        methods=["waldo", "wald", "posterior", "tilted"],
        n_samples=n_samples,
        eta_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        w=0.5,
        alpha=0.05,
        sigma=1.0,
        mu0=0.0,
        seed=seed,
        show_progress=show_progress
    )
