"""
Processing Functions for Raw D Samples

This module provides functions to compute derived quantities (CIs, coverage,
widths) from raw D samples. All computations are parameterized by (method, alpha, eta)
so the same raw data can be reprocessed with different parameters.

Key functions:
- compute_ci: Compute CI bounds for a single D value
- compute_coverage_indicators: Check which CIs cover theta_true
- compute_width_samples: Compute CI widths for an array of D samples
- compute_posterior_samples: Compute (mu_n - theta) and tau_WALDO for distribution validation
"""

from typing import Optional, Literal
import numpy as np
from scipy import stats

from ..core import posterior_params, scaled_conflict, weight
from ..waldo import confidence_interval, wald_ci, posterior_ci
from ..tilting import tilted_ci, optimal_eta_approximation, dynamic_tilted_ci


# ==============================================================================
# Core CI Computation
# ==============================================================================

MethodType = Literal["wald", "posterior", "waldo", "tilted", "tilted_optimal", "dynamic"]


def compute_ci(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    method: MethodType,
    alpha: float = 0.05,
    eta: Optional[float] = None,
) -> tuple[float, float]:
    """Compute confidence interval bounds for a single D value.

    Args:
        D: Observed data point
        mu0: Prior mean
        sigma: Likelihood standard deviation
        sigma0: Prior standard deviation
        method: CI method - one of:
            - "wald": Wald interval (ignores prior)
            - "posterior": Posterior credible interval
            - "waldo": WALDO confidence interval
            - "tilted": Tilted CI with fixed eta
            - "tilted_optimal": Tilted CI with optimal eta*(|Delta|) from observed D (STATIC)
            - "dynamic": True dynamic tilting where eta*(theta) varies with theta
        alpha: Significance level (default 0.05 for 95% CI)
        eta: Tilting parameter for "tilted" method (required if method="tilted")

    Returns:
        Tuple of (ci_lower, ci_upper)

    Raises:
        ValueError: If method is unknown or eta is missing for tilted method
    """
    if method == "wald":
        return wald_ci(D, sigma, alpha)

    elif method == "posterior":
        return posterior_ci(D, mu0, sigma, sigma0, alpha)

    elif method == "waldo":
        return confidence_interval(D, mu0, sigma, sigma0, alpha)

    elif method == "tilted":
        if eta is None:
            raise ValueError("eta parameter required for 'tilted' method")
        return tilted_ci(D, mu0, sigma, sigma0, eta, alpha)

    elif method == "tilted_optimal":
        # Compute optimal eta from realized Delta (STATIC optimization)
        w = weight(sigma, sigma0)
        Delta = scaled_conflict(D, mu0, w, sigma)
        eta_star = optimal_eta_approximation(abs(Delta))
        return tilted_ci(D, mu0, sigma, sigma0, eta_star, alpha)

    elif method == "dynamic":
        # TRUE dynamic tilting: eta*(theta) varies as you scan across theta
        # Uses dynamic_tilted_ci which evaluates each theta with its local optimal eta
        return dynamic_tilted_ci(D, mu0, sigma, sigma0, alpha)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_sigma0_from_w(sigma: float, w: float) -> float:
    """Compute sigma0 given sigma and weight w.

    w = sigma0^2 / (sigma^2 + sigma0^2)
    => sigma0 = sigma * sqrt(w / (1-w))
    """
    return sigma * np.sqrt(w / (1 - w))


# ==============================================================================
# Coverage Computation
# ==============================================================================

def compute_coverage_indicators(
    D_samples: np.ndarray,
    theta_true: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    method: MethodType,
    alpha: float = 0.05,
    eta: Optional[float] = None,
) -> np.ndarray:
    """Compute coverage indicators for an array of D samples.

    For each D sample, computes the CI and checks if it covers theta_true.

    Args:
        D_samples: 1D array of D values
        theta_true: True parameter value
        mu0: Prior mean
        sigma: Likelihood standard deviation
        sigma0: Prior standard deviation
        method: CI method
        alpha: Significance level
        eta: Tilting parameter (for "tilted" method)

    Returns:
        Boolean array where True means the CI covered theta_true
    """
    D_samples = np.asarray(D_samples).ravel()
    n = len(D_samples)
    covers = np.zeros(n, dtype=bool)

    for i, D in enumerate(D_samples):
        ci_low, ci_high = compute_ci(D, mu0, sigma, sigma0, method, alpha, eta)
        covers[i] = (ci_low <= theta_true <= ci_high)

    return covers


def compute_coverage_rate(
    D_samples: np.ndarray,
    theta_true: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    method: MethodType,
    alpha: float = 0.05,
    eta: Optional[float] = None,
) -> tuple[float, float]:
    """Compute coverage rate and standard error.

    Args:
        D_samples: 1D array of D values
        theta_true: True parameter value
        mu0: Prior mean
        sigma: Likelihood standard deviation
        sigma0: Prior standard deviation
        method: CI method
        alpha: Significance level
        eta: Tilting parameter (for "tilted" method)

    Returns:
        Tuple of (coverage_rate, standard_error)
    """
    indicators = compute_coverage_indicators(
        D_samples, theta_true, mu0, sigma, sigma0, method, alpha, eta
    )
    n = len(indicators)
    p = indicators.mean()
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
    return float(p), float(se)


# ==============================================================================
# Width Computation
# ==============================================================================

def compute_width_samples(
    D_samples: np.ndarray,
    mu0: float,
    sigma: float,
    sigma0: float,
    method: MethodType,
    alpha: float = 0.05,
    eta: Optional[float] = None,
) -> np.ndarray:
    """Compute CI widths for an array of D samples.

    Args:
        D_samples: 1D array of D values
        mu0: Prior mean
        sigma: Likelihood standard deviation
        sigma0: Prior standard deviation
        method: CI method
        alpha: Significance level
        eta: Tilting parameter (for "tilted" method)

    Returns:
        Array of CI widths (same length as D_samples)
    """
    D_samples = np.asarray(D_samples).ravel()
    n = len(D_samples)
    widths = np.zeros(n)

    for i, D in enumerate(D_samples):
        ci_low, ci_high = compute_ci(D, mu0, sigma, sigma0, method, alpha, eta)
        widths[i] = ci_high - ci_low

    return widths


def compute_mean_width(
    D_samples: np.ndarray,
    mu0: float,
    sigma: float,
    sigma0: float,
    method: MethodType,
    alpha: float = 0.05,
    eta: Optional[float] = None,
) -> tuple[float, float]:
    """Compute mean CI width and standard error.

    Args:
        D_samples: 1D array of D values
        mu0: Prior mean
        sigma: Likelihood standard deviation
        sigma0: Prior standard deviation
        method: CI method
        alpha: Significance level
        eta: Tilting parameter (for "tilted" method)

    Returns:
        Tuple of (mean_width, standard_error)
    """
    widths = compute_width_samples(D_samples, mu0, sigma, sigma0, method, alpha, eta)
    n = len(widths)
    mean = widths.mean()
    se = widths.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return float(mean), float(se)


# ==============================================================================
# Distribution Validation (Posterior Mean and WALDO Statistic)
# ==============================================================================

def compute_posterior_samples(
    D_samples: np.ndarray,
    theta_true: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> dict[str, np.ndarray]:
    """Compute posterior mean errors and WALDO statistics from D samples.

    For distribution validation experiments (Theorems 1-2).

    Args:
        D_samples: 1D array of D values
        theta_true: True parameter value
        mu0: Prior mean
        sigma: Likelihood standard deviation
        sigma0: Prior standard deviation

    Returns:
        Dictionary with:
            - "posterior_mean_errors": array of (mu_n - theta_true)
            - "waldo_statistics": array of tau_WALDO values
            - "scaled_waldo": array of tau_WALDO / w (should be chi2(1, lambda))
    """
    D_samples = np.asarray(D_samples).ravel()
    n = len(D_samples)

    # Compute posterior parameters for each D
    errors = np.zeros(n)
    tau_values = np.zeros(n)

    for i, D in enumerate(D_samples):
        mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)
        errors[i] = mu_n - theta_true
        tau_values[i] = ((mu_n - theta_true) / sigma_n) ** 2

    # Get w (constant for all samples)
    _, _, w = posterior_params(D_samples[0], mu0, sigma, sigma0)

    return {
        "posterior_mean_errors": errors,
        "waldo_statistics": tau_values,
        "scaled_waldo": tau_values / w,
        "w": w,
    }


def compute_delta_from_D(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> float:
    """Compute scaled conflict Delta from realized D.

    Delta = (1-w)(mu0 - D) / sigma
    """
    w = weight(sigma, sigma0)
    return scaled_conflict(D, mu0, w, sigma)


def compute_delta_samples(
    D_samples: np.ndarray,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> np.ndarray:
    """Compute Delta for each D sample."""
    D_samples = np.asarray(D_samples).ravel()
    w = weight(sigma, sigma0)
    return np.array([scaled_conflict(D, mu0, w, sigma) for D in D_samples])


# ==============================================================================
# Bootstrap Utilities
# ==============================================================================

def bootstrap_proportion(
    indicators: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float, float, float]:
    """Bootstrap confidence interval for a proportion.

    Args:
        indicators: Boolean array (coverage indicators)
        n_boot: Number of bootstrap resamples
        alpha: Significance level for bootstrap CI
        seed: Random seed

    Returns:
        Tuple of (point_estimate, se, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    indicators = np.asarray(indicators).ravel()
    n = len(indicators)

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    point = indicators.mean()

    # Bootstrap
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        boot_sample = rng.choice(indicators, size=n, replace=True)
        boot_means[b] = boot_sample.mean()

    se = boot_means.std()
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return float(point), float(se), float(ci_low), float(ci_high)


def bootstrap_mean(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float, float, float]:
    """Bootstrap confidence interval for a mean.

    Args:
        values: Array of values
        n_boot: Number of bootstrap resamples
        alpha: Significance level for bootstrap CI
        seed: Random seed

    Returns:
        Tuple of (point_estimate, se, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values).ravel()
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    point = values.mean()

    # Bootstrap
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        boot_sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = boot_sample.mean()

    se = boot_means.std()
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return float(point), float(se), float(ci_low), float(ci_high)


def bootstrap_statistic(
    samples: np.ndarray,
    statistic_fn=np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float, float, float]:
    """Bootstrap confidence interval for any statistic.

    Args:
        samples: Array of samples
        statistic_fn: Function to compute statistic (default: np.mean)
        n_boot: Number of bootstrap resamples
        alpha: Significance level for bootstrap CI
        seed: Random seed

    Returns:
        Tuple of (point_estimate, se, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples).ravel()
    n = len(samples)

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    point = statistic_fn(samples)

    # Bootstrap
    boot_stats = np.zeros(n_boot)
    for b in range(n_boot):
        boot_sample = rng.choice(samples, size=n, replace=True)
        boot_stats[b] = statistic_fn(boot_sample)

    se = boot_stats.std()
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return float(point), float(se), float(ci_low), float(ci_high)


# ==============================================================================
# High-Level Processing Functions (for use with cached results)
# ==============================================================================

def process_coverage_grid(
    D_samples: np.ndarray,
    theta_grid: np.ndarray,
    w_values: np.ndarray,
    method: MethodType,
    alpha: float = 0.05,
    eta: Optional[float] = None,
    mu0: float = 0.0,
    sigma: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Process coverage for a full grid of (theta, w) values.

    Args:
        D_samples: 3D array [n_theta, n_w, n_reps] of D samples
        theta_grid: Array of theta values
        w_values: Array of w values
        method: CI method
        alpha: Significance level
        eta: Tilting parameter (for "tilted" method)
        mu0: Prior mean
        sigma: Likelihood standard deviation
        verbose: Print progress

    Returns:
        Dictionary with:
            - "coverage_rates": 2D array [n_theta, n_w]
            - "coverage_se": 2D array [n_theta, n_w]
            - "indicators": 3D array [n_theta, n_w, n_reps] of bool
    """
    n_theta = len(theta_grid)
    n_w = len(w_values)
    n_reps = D_samples.shape[2]

    coverage_rates = np.zeros((n_theta, n_w))
    coverage_se = np.zeros((n_theta, n_w))
    indicators = np.zeros((n_theta, n_w, n_reps), dtype=bool)

    for i, theta in enumerate(theta_grid):
        for j, w in enumerate(w_values):
            sigma0 = compute_sigma0_from_w(sigma, w)
            D_cell = D_samples[i, j, :]

            ind = compute_coverage_indicators(
                D_cell, theta, mu0, sigma, sigma0, method, alpha, eta
            )
            indicators[i, j, :] = ind
            coverage_rates[i, j] = ind.mean()
            n = len(ind)
            p = coverage_rates[i, j]
            coverage_se[i, j] = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0

            if verbose:
                print(f"  theta={theta:.1f}, w={w:.1f}: {100*p:.1f}% +/- {100*coverage_se[i,j]:.2f}%")

    return {
        "coverage_rates": coverage_rates,
        "coverage_se": coverage_se,
        "indicators": indicators,
        "method": method,
        "alpha": alpha,
        "eta": eta,
    }


def process_width_grid(
    D_samples: np.ndarray,
    theta_grid: np.ndarray,
    method: MethodType,
    alpha: float = 0.05,
    eta: Optional[float] = None,
    mu0: float = 0.0,
    sigma: float = 1.0,
    w: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Process CI widths for a grid of theta values.

    Args:
        D_samples: 2D array [n_theta, n_samples] of D samples
        theta_grid: Array of theta values
        method: CI method
        alpha: Significance level
        eta: Tilting parameter (for "tilted" method)
        mu0: Prior mean
        sigma: Likelihood standard deviation
        w: Prior weight
        verbose: Print progress

    Returns:
        Dictionary with:
            - "mean_widths": 1D array [n_theta]
            - "width_se": 1D array [n_theta]
            - "width_samples": 2D array [n_theta, n_samples]
            - "delta_samples": 2D array [n_theta, n_samples] - realized Delta values
    """
    n_theta = len(theta_grid)
    n_samples = D_samples.shape[1]
    sigma0 = compute_sigma0_from_w(sigma, w)

    mean_widths = np.zeros(n_theta)
    width_se = np.zeros(n_theta)
    width_samples = np.zeros((n_theta, n_samples))
    delta_samples = np.zeros((n_theta, n_samples))

    for i, theta in enumerate(theta_grid):
        D_row = D_samples[i, :]

        # Compute widths
        widths = compute_width_samples(D_row, mu0, sigma, sigma0, method, alpha, eta)
        width_samples[i, :] = widths
        mean_widths[i] = widths.mean()
        width_se[i] = widths.std(ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0.0

        # Compute Delta for each D
        delta_samples[i, :] = compute_delta_samples(D_row, mu0, sigma, sigma0)

        if verbose:
            print(f"  theta={theta:.1f}: mean_width={mean_widths[i]:.3f} +/- {width_se[i]:.4f}")

    return {
        "mean_widths": mean_widths,
        "width_se": width_se,
        "width_samples": width_samples,
        "delta_samples": delta_samples,
        "method": method,
        "alpha": alpha,
        "eta": eta,
    }
