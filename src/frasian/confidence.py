"""
Confidence distribution functions for the WALDO framework.

This module provides functions for computing point estimators and properties
of the WALDO confidence distribution, including mode, mean, and sampling.
"""

import numpy as np
from scipy import stats
from scipy import integrate
from scipy import optimize
from typing import Union, Tuple, Optional

from .core import posterior_params, weight, scaled_conflict
from .waldo import pvalue, pvalue_components

ArrayLike = Union[float, np.ndarray]


def pvalue_mode(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> float:
    """
    Compute the mode of the WALDO confidence distribution.

    From Theorem 4: The mode equals the posterior mean mu_n.

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
    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
    return mu_n


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


def numerical_mode(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    bracket_mult: float = 5.0,
) -> float:
    """
    Find the mode numerically by maximizing the p-value function.

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
    bracket_mult : float
        Multiplier for search bracket

    Returns
    -------
    mode : float
        Numerically determined mode
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    # Negative p-value for minimization
    def neg_pvalue(theta):
        return -pvalue(theta, mu_n, mu0, w, sigma)

    # Search around mu_n
    result = optimize.minimize_scalar(
        neg_pvalue,
        bounds=(mu_n - bracket_mult * sigma, mu_n + bracket_mult * sigma),
        method='bounded'
    )

    return result.x


def pvalue_normalizing_constant(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    limit: float = 10.0,
) -> float:
    """
    Compute the normalizing constant for the p-value function.

    Z = integral of p(theta) over theta

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
    limit : float
        Integration limit in standard deviations

    Returns
    -------
    Z : float
        Normalizing constant
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    def integrand(theta):
        return pvalue(theta, mu_n, mu0, w, sigma)

    # Integrate over a wide range
    result, _ = integrate.quad(
        integrand,
        mu_n - limit * sigma,
        mu_n + limit * sigma,
        limit=100
    )

    return result


def pvalue_mean_numerical(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    limit: float = 10.0,
) -> float:
    """
    Compute the mean of the normalized p-value function numerically.

    E[theta] = (1/Z) * integral of theta * p(theta) over theta

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
    limit : float
        Integration limit in standard deviations

    Returns
    -------
    mean : float
        Mean of the confidence distribution
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    def p_func(theta):
        return pvalue(theta, mu_n, mu0, w, sigma)

    def theta_p_func(theta):
        return theta * pvalue(theta, mu_n, mu0, w, sigma)

    # Compute normalizing constant
    Z, _ = integrate.quad(p_func, mu_n - limit * sigma, mu_n + limit * sigma, limit=100)

    # Compute first moment
    moment1, _ = integrate.quad(theta_p_func, mu_n - limit * sigma, mu_n + limit * sigma, limit=100)

    return moment1 / Z


def pvalue_mean_closed_form(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> float:
    """
    Compute the mean using the closed-form expression from Theorem 5.

    E[theta] = mu_n + sigma * (w-1)/(2-w) * [(Delta^2 + 1)*E + 2*Delta*psi] / [Delta*E + 2*psi]

    where:
        Delta = (1-w)(mu0 - D) / sigma
        E = 2*Phi(Delta) - 1
        psi = phi(Delta)

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
    mean : float
        Mean of the confidence distribution
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)
    Delta = scaled_conflict(D, mu0, w, sigma)

    # Components of the formula
    E = 2 * stats.norm.cdf(Delta) - 1  # = erf(Delta/sqrt(2)) for reference
    psi = stats.norm.pdf(Delta)

    # Handle the case Delta = 0 to avoid division issues
    if np.abs(Delta) < 1e-10:
        # When Delta = 0, E = 0 and the formula simplifies
        # numerator = (0 + 1) * 0 + 2 * 0 * phi(0) = 0
        # denominator = 0 * 0 + 2 * phi(0) = 2 * phi(0) > 0
        # So the correction is 0, mean = mu_n
        return mu_n

    # Full formula
    numerator = (Delta**2 + 1) * E + 2 * Delta * psi
    denominator = Delta * E + 2 * psi

    correction = sigma * (w - 1) / (2 - w) * numerator / denominator

    return mu_n + correction


def pvalue_mean(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    method: str = 'closed_form',
) -> float:
    """
    Compute the mean of the WALDO confidence distribution.

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
    method : str
        'closed_form' or 'numerical'

    Returns
    -------
    mean : float
        Mean of the confidence distribution
    """
    if method == 'closed_form':
        return pvalue_mean_closed_form(D, mu0, sigma, sigma0)
    elif method == 'numerical':
        return pvalue_mean_numerical(D, mu0, sigma, sigma0)
    else:
        raise ValueError(f"Unknown method: {method}")


def sample_confidence_dist(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
    method: str = 'rejection',
) -> np.ndarray:
    """
    Sample from the normalized p-value function (confidence distribution).

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
    n_samples : int
        Number of samples to generate
    rng : Generator, optional
        Random number generator
    method : str
        'rejection' for rejection sampling

    Returns
    -------
    samples : ndarray
        Samples from the confidence distribution
    """
    if rng is None:
        rng = np.random.default_rng()

    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    if method == 'rejection':
        # Use rejection sampling with a uniform proposal over a wide range
        samples = []
        proposal_width = 10 * sigma

        # Maximum p-value is 1 (at the mode)
        max_p = 1.0

        while len(samples) < n_samples:
            # Propose from uniform
            n_batch = 2 * (n_samples - len(samples))
            proposals = rng.uniform(
                mu_n - proposal_width,
                mu_n + proposal_width,
                n_batch
            )

            # Compute acceptance probabilities
            p_vals = np.array([pvalue(theta, mu_n, mu0, w, sigma) for theta in proposals])

            # Accept/reject
            u = rng.uniform(0, max_p, n_batch)
            accepted = proposals[u < p_vals]
            samples.extend(accepted)

        return np.array(samples[:n_samples])
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def mean_between_mode_and_mle(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> Tuple[bool, float, float, float]:
    """
    Check that the mean lies between the mode (mu_n) and the MLE (D).

    From Section 5.2.1:
    - When D > mu0: mu_n < E[theta] < D
    - When D < mu0: D < E[theta] < mu_n
    - When D = mu0: E[theta] = mu_n

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
    is_between : bool
        True if mean is between mode and MLE
    mode : float
        Mode (= mu_n)
    mean : float
        Mean of confidence distribution
    mle : float
        MLE (= D)
    """
    mode = pvalue_mode(D, mu0, sigma, sigma0)
    mean = pvalue_mean(D, mu0, sigma, sigma0)
    mle = D

    if np.isclose(D, mu0, atol=1e-10):
        # No prior-data conflict: mean should equal mode
        is_between = np.isclose(mean, mode, atol=0.01 * sigma)
    elif D > mu0:
        # Mean should be between mode (= mu_n < D) and MLE (= D)
        is_between = (mode <= mean <= mle) or np.isclose(mean, mode, atol=0.01 * sigma)
    else:  # D < mu0
        # Mean should be between MLE (= D) and mode (= mu_n > D)
        is_between = (mle <= mean <= mode) or np.isclose(mean, mode, atol=0.01 * sigma)

    return is_between, mode, mean, mle


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


def pvalue_wald_symmetric(
    theta: ArrayLike,
    mu_n: float,
    sigma_n: float,
) -> ArrayLike:
    """
    Compute the symmetric (Wald) p-value for comparison.

    p(theta) = 2 * Phi(-|mu_n - theta| / sigma_n)

    This is what you get when b = 0 (no prior bias).

    Parameters
    ----------
    theta : float or array
        Parameter value(s) to test
    mu_n : float
        Point estimate
    sigma_n : float
        Standard error

    Returns
    -------
    p : float or array
        Wald p-value(s)
    """
    a = np.abs(mu_n - theta) / sigma_n
    return 2 * stats.norm.cdf(-a)
