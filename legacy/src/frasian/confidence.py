"""
Confidence distribution functions for the WALDO framework.

This module provides functions for computing confidence distributions (CDs)
and their estimators (mean, mode) using the Schweder-Hjort methodology.

Key result: The confidence density is derived from the p-value derivative:
    c(θ) = (1/2) |dp/dθ|

For WALDO, this yields a 50-50 Gaussian mixture:
    c_WALDO(θ) = 0.5 × N(D, σ²) + 0.5 × N(μ*, σ*²)
where:
    μ* = (wD + 2(1-w)μ₀) / (2-w)
    σ* = wσ / (2-w)
"""

import numpy as np
from scipy import stats
from scipy import optimize
from typing import Union, Tuple, Dict, Optional, Callable

from .core import posterior_params, weight, scaled_conflict
from .waldo import pvalue, pvalue_components

ArrayLike = Union[float, np.ndarray]


# =============================================================================
# Wald Confidence Distribution (Simple Normal)
# =============================================================================

def wald_cd_density(
    theta: ArrayLike,
    D: float,
    sigma: float,
) -> ArrayLike:
    """
    Compute the Wald confidence distribution density.

    The Wald CD is simply N(D, σ²), centered at the MLE.

    Parameters
    ----------
    theta : float or array
        Parameter value(s)
    D : float
        Observed data (MLE)
    sigma : float
        Likelihood standard deviation

    Returns
    -------
    density : float or array
        CD density at theta
    """
    return stats.norm.pdf(theta, loc=D, scale=sigma)


def wald_cd_mean(D: float) -> float:
    """
    Compute the mean of the Wald CD.

    The Wald CD mean equals D (the MLE).

    Parameters
    ----------
    D : float
        Observed data (MLE)

    Returns
    -------
    mean : float
        CD mean (= D)
    """
    return D


def wald_cd_mode(D: float) -> float:
    """
    Compute the mode of the Wald CD.

    The Wald CD mode equals D (the MLE).

    Parameters
    ----------
    D : float
        Observed data (MLE)

    Returns
    -------
    mode : float
        CD mode (= D)
    """
    return D


# =============================================================================
# WALDO Confidence Distribution (Gaussian Mixture)
# =============================================================================

def waldo_cd_params(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> Dict[str, float]:
    """
    Compute the parameters of the WALDO confidence distribution.

    The WALDO CD is a 50-50 mixture of two Gaussians:
        0.5 × N(D, σ²) + 0.5 × N(μ*, σ*²)

    Parameters
    ----------
    D : float
        Observed data (MLE)
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation

    Returns
    -------
    params : dict
        Dictionary containing:
        - mu_n: posterior mean
        - w: weight on data
        - mu_star: second component mean
        - sigma_star: second component std
        - component1_mean: D (first component)
        - component1_std: sigma
        - component2_mean: mu_star
        - component2_std: sigma_star
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    # Second component parameters
    mu_star = (w * D + 2 * (1 - w) * mu0) / (2 - w)
    sigma_star = w * sigma / (2 - w)

    return {
        'mu_n': mu_n,
        'w': w,
        'mu_star': mu_star,
        'sigma_star': sigma_star,
        'component1_mean': D,
        'component1_std': sigma,
        'component2_mean': mu_star,
        'component2_std': sigma_star,
    }


def waldo_cd_density(
    theta: ArrayLike,
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> ArrayLike:
    """
    Compute the WALDO confidence distribution density.

    The WALDO CD is a 50-50 Gaussian mixture:
        c(θ) = 0.5 × N(θ | D, σ²) + 0.5 × N(θ | μ*, σ*²)

    Parameters
    ----------
    theta : float or array
        Parameter value(s)
    D : float
        Observed data (MLE)
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation

    Returns
    -------
    density : float or array
        CD density at theta
    """
    params = waldo_cd_params(D, mu0, sigma, sigma0)

    # 50-50 mixture
    component1 = stats.norm.pdf(theta, loc=params['component1_mean'],
                                scale=params['component1_std'])
    component2 = stats.norm.pdf(theta, loc=params['component2_mean'],
                                scale=params['component2_std'])

    return 0.5 * component1 + 0.5 * component2


def waldo_cd_mean(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> float:
    """
    Compute the mean of the WALDO confidence distribution.

    Closed form: E[θ] = (μ_n + (1-w)D) / (2-w)

    Equivalently: E[θ] = 0.5 × D + 0.5 × μ*

    Parameters
    ----------
    D : float
        Observed data (MLE)
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation

    Returns
    -------
    mean : float
        CD mean
    """
    mu_n, _, w = posterior_params(D, mu0, sigma, sigma0)
    return (mu_n + (1 - w) * D) / (2 - w)


def waldo_cd_mode(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> float:
    """
    Compute the mode of the WALDO confidence distribution.

    The WALDO CD mode equals μ_n (the posterior mean).

    Note: This is also where the p-value is maximized (Theorem 4).

    Parameters
    ----------
    D : float
        Observed data (MLE)
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation

    Returns
    -------
    mode : float
        CD mode (= μ_n)
    """
    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
    return mu_n


# =============================================================================
# Numerical CD from P-value (Schweder-Hjort Method)
# =============================================================================

def cd_from_pvalue(
    theta_grid: np.ndarray,
    p_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence distribution from a p-value function.

    Uses the Schweder-Hjort methodology:
        c(θ) = (1/2) |dp/dθ|
        C(θ) = p(θ)/2 for θ ≤ mode, else 1 - p(θ)/2

    Parameters
    ----------
    theta_grid : ndarray
        Grid of θ values
    p_values : ndarray
        P-values at each grid point

    Returns
    -------
    cd_density : ndarray
        Confidence density at each grid point
    cd_cdf : ndarray
        Confidence CDF at each grid point
    """
    # Numerical derivative
    dtheta = theta_grid[1] - theta_grid[0]
    dp_dtheta = np.gradient(p_values, dtheta)

    # Confidence density
    cd_density = 0.5 * np.abs(dp_dtheta)

    # Find mode (where p = 1)
    mode_idx = np.argmax(p_values)

    # Confidence CDF
    cd_cdf = np.where(
        np.arange(len(theta_grid)) <= mode_idx,
        p_values / 2,
        1 - p_values / 2
    )

    return cd_density, cd_cdf


def cd_mean_numerical(
    theta_grid: np.ndarray,
    cd_density: np.ndarray,
) -> float:
    """
    Compute the mean of a confidence distribution numerically.

    Parameters
    ----------
    theta_grid : ndarray
        Grid of θ values
    cd_density : ndarray
        CD density at each grid point

    Returns
    -------
    mean : float
        CD mean
    """
    return np.trapezoid(theta_grid * cd_density, theta_grid)


def cd_mode_numerical(
    theta_grid: np.ndarray,
    cd_density: np.ndarray,
) -> float:
    """
    Compute the mode of a confidence distribution numerically.

    Parameters
    ----------
    theta_grid : ndarray
        Grid of θ values
    cd_density : ndarray
        CD density at each grid point

    Returns
    -------
    mode : float
        CD mode
    """
    return theta_grid[np.argmax(cd_density)]


def cd_quantile(
    theta_grid: np.ndarray,
    cd_cdf: np.ndarray,
    q: float,
) -> float:
    """
    Compute a quantile of the confidence distribution.

    Parameters
    ----------
    theta_grid : ndarray
        Grid of θ values
    cd_cdf : ndarray
        CD CDF at each grid point
    q : float
        Quantile (0 to 1)

    Returns
    -------
    theta_q : float
        θ value at the q-th quantile
    """
    idx = np.searchsorted(cd_cdf, q)
    if idx >= len(theta_grid):
        return theta_grid[-1]
    if idx == 0:
        return theta_grid[0]
    return theta_grid[idx]


def cd_variance_numerical(
    theta_grid: np.ndarray,
    cd_density: np.ndarray,
) -> float:
    """
    Compute the variance of a confidence distribution numerically.

    Parameters
    ----------
    theta_grid : ndarray
        Grid of θ values
    cd_density : ndarray
        CD density at each grid point

    Returns
    -------
    variance : float
        CD variance
    """
    mean = cd_mean_numerical(theta_grid, cd_density)
    return np.trapezoid((theta_grid - mean)**2 * cd_density, theta_grid)


# =============================================================================
# Dynamic WALDO Confidence Distribution
# =============================================================================

def dynamic_cd_density(
    theta_grid: np.ndarray,
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    smooth: bool = True,
    smooth_window: int = 15,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the Dynamic WALDO confidence distribution density.

    This uses the dynamic tilted p-value function (which adapts η locally)
    and applies the Schweder-Hjort differentiation method.

    Parameters
    ----------
    theta_grid : ndarray
        Grid of θ values
    D : float
        Observed data (MLE)
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    smooth : bool
        Whether to apply Savitzky-Golay smoothing to reduce noise
    smooth_window : int
        Window size for smoothing (must be odd)
    normalize : bool
        Whether to normalize the density to integrate to 1

    Returns
    -------
    density : ndarray
        CD density at each grid point
    """
    from .tilting import dynamic_tilted_pvalue
    from scipy.signal import savgol_filter

    # Compute dynamic p-values
    p_values = np.array([
        dynamic_tilted_pvalue(theta, D, mu0, sigma, sigma0)
        for theta in theta_grid
    ])

    # Apply Schweder-Hjort method
    cd_density, _ = cd_from_pvalue(theta_grid, p_values)

    # Smooth the density if requested
    if smooth and len(cd_density) > smooth_window:
        # Ensure window is odd and smaller than data
        window = min(smooth_window, len(cd_density) - 1)
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            cd_density = savgol_filter(cd_density, window, polyorder=2)
            # Ensure non-negative
            cd_density = np.maximum(cd_density, 0)

    # Normalize if requested
    if normalize:
        integral = np.trapezoid(cd_density, theta_grid)
        if integral > 1e-10:
            cd_density = cd_density / integral

    return cd_density


def dynamic_cd_mean(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    n_grid: int = 500,
    n_sigma: float = 5.0,
) -> float:
    """
    Compute the mean of the Dynamic WALDO CD numerically.

    Parameters
    ----------
    D : float
        Observed data (MLE)
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    n_grid : int
        Number of grid points
    n_sigma : float
        Number of standard deviations for grid extent

    Returns
    -------
    mean : float
        CD mean
    """
    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

    # Create grid covering both mu_n and D
    center = (mu_n + D) / 2
    extent = max(abs(D - mu_n), sigma) * n_sigma
    theta_grid = np.linspace(center - extent, center + extent, n_grid)

    # Compute normalized density
    cd_density = dynamic_cd_density(theta_grid, D, mu0, sigma, sigma0, normalize=True)

    return cd_mean_numerical(theta_grid, cd_density)


def dynamic_cd_mode(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    n_grid: int = 500,
    n_sigma: float = 5.0,
) -> float:
    """
    Compute the mode of the Dynamic WALDO CD numerically.

    Parameters
    ----------
    D : float
        Observed data (MLE)
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    n_grid : int
        Number of grid points
    n_sigma : float
        Number of standard deviations for grid extent

    Returns
    -------
    mode : float
        CD mode
    """
    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

    # Create grid
    center = (mu_n + D) / 2
    extent = max(abs(D - mu_n), sigma) * n_sigma
    theta_grid = np.linspace(center - extent, center + extent, n_grid)

    # Compute density with extra smoothing for mode finding
    cd_density = dynamic_cd_density(theta_grid, D, mu0, sigma, sigma0,
                                     smooth=True, smooth_window=21, normalize=True)

    return cd_mode_numerical(theta_grid, cd_density)


# =============================================================================
# Legacy Functions (Kept for Compatibility)
# =============================================================================

def pvalue_mode(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> float:
    """
    Compute the mode of the WALDO confidence distribution.

    From Theorem 4: The mode equals the posterior mean mu_n.

    This is equivalent to waldo_cd_mode() and is kept for compatibility.

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

    Returns
    -------
    mode : float
        Mode of the confidence distribution (= posterior mean)
    """
    return waldo_cd_mode(D, mu0, sigma, sigma0)


def pvalue_at_mode(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> float:
    """
    Compute p-value at the mode (should be 1).

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

    Returns
    -------
    p : float
        P-value at the mode (should be 1.0)
    """
    mu_n, _, w = posterior_params(D, mu0, sigma, sigma0)
    return pvalue(mu_n, mu_n, mu0, w, sigma)


def verify_mode_is_max(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    n_check: int = 100,
) -> Tuple[bool, float, float]:
    """
    Verify that the mode (mu_n) maximizes the p-value function.

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
    n_check : int
        Number of points to check

    Returns
    -------
    is_max : bool
        True if mu_n is the maximum
    p_at_mode : float
        P-value at the mode
    max_p_found : float
        Maximum p-value found in search
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    # Check p-value at mode
    p_at_mode = pvalue(mu_n, mu_n, mu0, w, sigma)

    # Search nearby for higher values
    thetas = np.linspace(mu_n - 5 * sigma, mu_n + 5 * sigma, n_check)
    p_values = np.array([pvalue(theta, mu_n, mu0, w, sigma) for theta in thetas])
    max_p_found = np.max(p_values)

    return np.isclose(p_at_mode, max_p_found, atol=1e-6), p_at_mode, max_p_found
