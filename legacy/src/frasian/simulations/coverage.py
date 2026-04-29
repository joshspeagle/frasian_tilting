"""
Coverage Simulation Experiments

Simulates coverage rates for Wald, Posterior, and WALDO confidence intervals
across a grid of parameters (theta, w, method).

Saves raw binary coverage indicators for robust uncertainty quantification.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from tqdm import tqdm

from ..core import posterior_params, weight
from ..waldo import confidence_interval, wald_ci, posterior_ci
from ..uncertainty import proportion_se, proportion_ci as proportion_confidence_interval


MethodType = Literal["wald", "posterior", "waldo"]


@dataclass
class CoverageResult:
    """Result for a single coverage cell."""
    theta: float
    w: float
    method: str
    coverage: float
    se: float
    ci_low: float
    ci_high: float
    n_reps: int
    # Raw binary indicators (1 = covered, 0 = not covered)
    indicators: Optional[np.ndarray] = None


def get_model_params(w: float, sigma: float = 1.0, mu0: float = 0.0):
    """Get model parameters (mu0, sigma, sigma0) for a given weight w."""
    # w = sigma0^2 / (sigma^2 + sigma0^2)
    # => sigma0^2 = w * sigma^2 / (1 - w)
    # => sigma0 = sigma * sqrt(w / (1 - w))
    if w <= 0 or w >= 1:
        raise ValueError(f"w must be in (0, 1), got {w}")
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return mu0, sigma, sigma0


def simulate_coverage_cell(
    theta: float,
    w: float,
    method: MethodType,
    n_reps: int,
    alpha: float = 0.05,
    sigma: float = 1.0,
    mu0: float = 0.0,
    seed: Optional[int] = None,
    return_indicators: bool = True
) -> CoverageResult:
    """Simulate coverage rate for a single (theta, w, method) cell.

    Args:
        theta: True parameter value
        w: Prior weight (determines sigma0)
        method: CI method - "wald", "posterior", or "waldo"
        n_reps: Number of MC replicates
        alpha: Significance level for CI (default 0.05)
        sigma: Likelihood standard deviation (default 1.0)
        mu0: Prior mean (default 0.0)
        seed: Random seed for reproducibility
        return_indicators: If True, return raw binary indicators

    Returns:
        CoverageResult with coverage rate, uncertainty, and optionally raw indicators
    """
    rng = np.random.default_rng(seed)

    _, sigma, sigma0 = get_model_params(w, sigma, mu0)

    # Store raw binary indicators
    indicators = np.zeros(n_reps, dtype=np.int8)

    for i in range(n_reps):
        # Simulate data
        D = rng.normal(theta, sigma)

        # Compute CI based on method
        if method == "wald":
            ci_low, ci_high = wald_ci(D, sigma, alpha)
        elif method == "posterior":
            ci_low, ci_high = posterior_ci(D, mu0, sigma, sigma0, alpha)
        elif method == "waldo":
            ci_low, ci_high = confidence_interval(D, mu0, sigma, sigma0, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Check coverage
        indicators[i] = 1 if (ci_low <= theta <= ci_high) else 0

    # Compute coverage and uncertainty
    coverage = np.mean(indicators)
    se = proportion_se(coverage, n_reps)
    ci_lo, ci_hi = proportion_confidence_interval(coverage, n_reps, alpha=0.05, method="wilson")

    return CoverageResult(
        theta=theta,
        w=w,
        method=method,
        coverage=coverage,
        se=se,
        ci_low=ci_lo,
        ci_high=ci_hi,
        n_reps=n_reps,
        indicators=indicators if return_indicators else None
    )


def simulate_coverage_grid(
    theta_grid: np.ndarray,
    w_values: list[float],
    methods: list[MethodType],
    n_reps: int,
    alpha: float = 0.05,
    sigma: float = 1.0,
    mu0: float = 0.0,
    seed: int = 42,
    show_progress: bool = True,
    save_raw: bool = True
) -> dict[str, np.ndarray]:
    """Simulate coverage across a grid of parameters.

    Args:
        theta_grid: Array of theta values to test
        w_values: List of prior weights to test
        methods: List of CI methods to test
        n_reps: Number of MC replicates per cell
        alpha: Significance level for CIs
        sigma: Likelihood standard deviation
        mu0: Prior mean
        seed: Base random seed
        show_progress: Whether to show progress bar
        save_raw: If True, save raw binary indicators (larger file, enables bootstrap)

    Returns:
        Dictionary with arrays:
            - "theta_grid": 1D array of theta values
            - "w_values": 1D array of w values
            - "methods": List of method names
            - "coverage": 3D array [n_theta, n_w, n_methods]
            - "se": 3D array matching coverage
            - "indicators": 4D array [n_theta, n_w, n_methods, n_reps] (if save_raw=True)
    """
    theta_grid = np.asarray(theta_grid)
    n_theta = len(theta_grid)
    n_w = len(w_values)
    n_methods = len(methods)

    # Initialize output arrays
    coverage = np.zeros((n_theta, n_w, n_methods))
    se = np.zeros((n_theta, n_w, n_methods))

    # Raw indicators: 4D array for bootstrapping
    if save_raw:
        indicators = np.zeros((n_theta, n_w, n_methods, n_reps), dtype=np.int8)
    else:
        indicators = None

    # Create method index mapping
    method_idx = {m: i for i, m in enumerate(methods)}

    # Total cells for progress
    total_cells = n_theta * n_w * n_methods
    pbar = tqdm(total=total_cells, desc="Coverage simulation", unit="cell", disable=not show_progress)

    # Simulate each cell
    cell = 0
    for i, theta in enumerate(theta_grid):
        for j, w in enumerate(w_values):
            for method in methods:
                k = method_idx[method]

                # Use different seed for each cell for reproducibility
                cell_seed = seed + cell * 1000

                result = simulate_coverage_cell(
                    theta=theta,
                    w=w,
                    method=method,
                    n_reps=n_reps,
                    alpha=alpha,
                    sigma=sigma,
                    mu0=mu0,
                    seed=cell_seed,
                    return_indicators=save_raw
                )

                coverage[i, j, k] = result.coverage
                se[i, j, k] = result.se

                if save_raw and result.indicators is not None:
                    indicators[i, j, k, :] = result.indicators

                cell += 1
                pbar.update(1)

    pbar.close()

    result_dict = {
        "theta_grid": theta_grid,
        "w_values": np.array(w_values),
        "methods": methods,
        "coverage": coverage,
        "se": se,
    }

    if save_raw:
        result_dict["indicators"] = indicators

    return result_dict


def bootstrap_coverage_ci(
    indicators: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for coverage rate.

    Args:
        indicators: 1D array of binary coverage indicators
        n_boot: Number of bootstrap replicates
        alpha: Significance level for CI
        seed: Random seed

    Returns:
        Tuple of (coverage, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    n = len(indicators)

    # Point estimate
    coverage = np.mean(indicators)

    # Bootstrap
    boot_coverages = np.zeros(n_boot)
    for b in range(n_boot):
        boot_sample = rng.choice(indicators, size=n, replace=True)
        boot_coverages[b] = np.mean(boot_sample)

    # Percentile CI
    ci_low = np.percentile(boot_coverages, 100 * alpha / 2)
    ci_high = np.percentile(boot_coverages, 100 * (1 - alpha / 2))

    return coverage, ci_low, ci_high


def run_standard_coverage_simulation(
    n_reps: int = 10000,
    seed: int = 42,
    show_progress: bool = True,
    save_raw: bool = True
) -> dict[str, np.ndarray]:
    """Run coverage simulation with standard parameters.

    Standard configuration:
        - theta: [-4, 6] in 21 points
        - w: [0.2, 0.5, 0.8]
        - methods: [wald, posterior, waldo]
        - sigma: 1.0, mu0: 0.0

    Args:
        n_reps: Number of MC replicates per cell
        seed: Random seed
        show_progress: Whether to show progress bar
        save_raw: If True, save raw indicators for bootstrap

    Returns:
        Dictionary with coverage results
    """
    return simulate_coverage_grid(
        theta_grid=np.linspace(-4, 6, 21),
        w_values=[0.2, 0.5, 0.8],
        methods=["wald", "posterior", "waldo"],
        n_reps=n_reps,
        alpha=0.05,
        sigma=1.0,
        mu0=0.0,
        seed=seed,
        show_progress=show_progress,
        save_raw=save_raw
    )
