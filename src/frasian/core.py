"""
Core mathematical functions for the conjugate Normal model.

This module provides the fundamental computations for the Frasian inference framework:
- Posterior parameter calculations
- Coordinate transformations
- Bias and variance under the true parameter
"""

import numpy as np
from typing import Tuple, Union

# Type alias for scalar or array
ArrayLike = Union[float, np.ndarray]


def posterior_params(
    D: ArrayLike,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> Tuple[ArrayLike, float, float]:
    """
    Compute posterior parameters for the conjugate Normal model.

    Prior: theta ~ N(mu0, sigma0^2)
    Likelihood: D | theta ~ N(theta, sigma^2)
    Posterior: theta | D ~ N(mu_n, sigma_n^2)

    Parameters
    ----------
    D : float or array
        Observed data
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation

    Returns
    -------
    mu_n : float or array
        Posterior mean(s)
    sigma_n : float
        Posterior standard deviation
    w : float
        Weight on data (0 < w < 1)
    """
    # Weight placed on data relative to prior
    w = sigma0**2 / (sigma**2 + sigma0**2)

    # Posterior mean: weighted average of data and prior mean
    mu_n = w * D + (1 - w) * mu0

    # Posterior standard deviation
    sigma_n = np.sqrt(w) * sigma

    return mu_n, sigma_n, w


def weight(sigma: float, sigma0: float) -> float:
    """
    Compute the weight placed on data vs prior.

    Parameters
    ----------
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation

    Returns
    -------
    w : float
        Weight on data: w = sigma0^2 / (sigma^2 + sigma0^2)
    """
    return sigma0**2 / (sigma**2 + sigma0**2)


def standardized_coord(
    theta: ArrayLike,
    mu_n: ArrayLike,
    w: float,
    sigma: float,
) -> ArrayLike:
    """
    Transform theta to standardized coordinate u.

    u = (theta - mu_n) / (w * sigma)

    Parameters
    ----------
    theta : float or array
        Parameter value(s) to transform
    mu_n : float or array
        Posterior mean(s)
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation

    Returns
    -------
    u : float or array
        Standardized coordinate(s)
    """
    return (theta - mu_n) / (w * sigma)


def inverse_standardized_coord(
    u: ArrayLike,
    mu_n: ArrayLike,
    w: float,
    sigma: float,
) -> ArrayLike:
    """
    Transform standardized coordinate u back to theta.

    theta = mu_n + w * sigma * u

    Parameters
    ----------
    u : float or array
        Standardized coordinate(s)
    mu_n : float or array
        Posterior mean(s)
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation

    Returns
    -------
    theta : float or array
        Parameter value(s)
    """
    return mu_n + w * sigma * u


def scaled_conflict(
    D: ArrayLike,
    mu0: float,
    w: float,
    sigma: float,
) -> ArrayLike:
    """
    Compute the scaled prior-data conflict Delta.

    Delta = (1 - w) * (mu0 - D) / sigma

    Positive Delta: prior mean > data (D below prior expectation)
    Negative Delta: prior mean < data (D above prior expectation)

    Parameters
    ----------
    D : float or array
        Observed data
    mu0 : float
        Prior mean
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation

    Returns
    -------
    Delta : float or array
        Scaled prior-data conflict
    """
    return (1 - w) * (mu0 - D) / sigma


def prior_residual(
    theta: ArrayLike,
    mu0: float,
    sigma0: float,
) -> ArrayLike:
    """
    Compute the prior residual delta(theta).

    delta(theta) = (theta - mu0) / sigma0

    This measures how far theta is from the prior mean in prior standard deviations.

    Parameters
    ----------
    theta : float or array
        Parameter value(s)
    mu0 : float
        Prior mean
    sigma0 : float
        Prior standard deviation

    Returns
    -------
    delta : float or array
        Prior residual(s)
    """
    return (theta - mu0) / sigma0


def bias(
    theta: ArrayLike,
    mu0: float,
    w: float,
) -> ArrayLike:
    """
    Compute the bias b(theta) of the posterior mean under true theta.

    b(theta) = E[mu_n - theta | theta] = (1 - w) * (mu0 - theta)

    Parameters
    ----------
    theta : float or array
        True parameter value(s)
    mu0 : float
        Prior mean
    w : float
        Weight on data

    Returns
    -------
    b : float or array
        Bias of posterior mean
    """
    return (1 - w) * (mu0 - theta)


def variance(w: float, sigma: float) -> float:
    """
    Compute the variance v of (mu_n - theta) under any true theta.

    v = Var[mu_n - theta | theta] = w^2 * sigma^2

    Note: This is the variance due to sampling variability in D.

    Parameters
    ----------
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation

    Returns
    -------
    v : float
        Variance of (mu_n - theta)
    """
    return w**2 * sigma**2


def posterior_mean_distribution_params(
    theta: ArrayLike,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> Tuple[ArrayLike, float]:
    """
    Get the distribution parameters of (mu_n - theta) under true theta.

    From Theorem 1: (mu_n - theta) | theta ~ N(b(theta), v)

    Parameters
    ----------
    theta : float or array
        True parameter value(s)
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation

    Returns
    -------
    mean : float or array
        Mean b(theta) = (1-w)(mu0 - theta)
    var : float
        Variance v = w^2 * sigma^2
    """
    w = weight(sigma, sigma0)
    b = bias(theta, mu0, w)
    v = variance(w, sigma)
    return b, v


def delta_scaled(
    theta: ArrayLike,
    mu0: float,
    w: float,
    sigma: float,
) -> ArrayLike:
    """
    Compute delta(theta) = (1-w)(mu0 - theta) / (sqrt(w) * sigma).

    This is the standardized bias, used in the non-central chi-squared parameter.
    From Theorem 2 proof: delta(theta) = b(theta) / sigma_n

    Parameters
    ----------
    theta : float or array
        Parameter value(s)
    mu0 : float
        Prior mean
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation

    Returns
    -------
    delta : float or array
        Standardized bias
    """
    sigma_n = np.sqrt(w) * sigma
    b = bias(theta, mu0, w)
    return b / sigma_n
