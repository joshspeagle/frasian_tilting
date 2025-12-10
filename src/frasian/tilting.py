"""
Tilted posterior framework for adaptive Bayesian-frequentist inference.

This module implements the tilted posterior q(theta; eta) that interpolates
between WALDO (eta=0) and Wald (eta=1):

    q(theta; eta) ∝ L(theta) * pi(theta)^{1-eta}

Key results:
- Theorem 6: Closed-form tilted posterior parameters mu_eta, sigma_eta^2
- Theorem 7: Non-centrality reduction lambda_eta = (1-eta)^2 * lambda_0
- Theorem 8: Tilted p-value formula
- Section 10: Optimal tilting eta*(|Delta|)
"""

import numpy as np
from scipy import stats
from scipy import optimize
from typing import Tuple, Union, Optional

from .core import posterior_params, weight, scaled_conflict, bias
from .waldo import pvalue, pvalue_components, noncentrality

ArrayLike = Union[float, np.ndarray]


def tilted_params(
    D: ArrayLike,
    mu0: float,
    sigma: float,
    sigma0: float,
    eta: float,
) -> Tuple[ArrayLike, float, float]:
    """
    Compute tilted posterior parameters from Theorem 6.

    mu_eta = [w*D + (1-eta)*(1-w)*mu0] / [1 - eta*(1-w)]
    sigma_eta^2 = w*sigma^2 / [1 - eta*(1-w)]

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
    eta : float
        Tilting parameter (0 = WALDO, 1 = Wald)

    Returns
    -------
    mu_eta : float or array
        Tilted posterior mean
    sigma_eta : float
        Tilted posterior standard deviation
    w_eta : float
        Effective weight for tilted distribution
    """
    w = weight(sigma, sigma0)

    # Denominator in the tilted formulas
    denom = 1 - eta * (1 - w)

    # Tilted posterior mean
    mu_eta = (w * D + (1 - eta) * (1 - w) * mu0) / denom

    # Tilted posterior variance
    sigma_eta_sq = w * sigma**2 / denom
    sigma_eta = np.sqrt(sigma_eta_sq)

    # Effective weight (useful for some computations)
    w_eta = w / denom

    return mu_eta, sigma_eta, w_eta


def tilted_noncentrality(
    lambda0: ArrayLike,
    eta: float,
) -> ArrayLike:
    """
    Compute the tilted non-centrality parameter from Theorem 7.

    lambda_eta = (1 - eta)^2 * lambda_0

    Parameters
    ----------
    lambda0 : float or array
        Non-centrality parameter at eta=0 (WALDO)
    eta : float
        Tilting parameter

    Returns
    -------
    lambda_eta : float or array
        Tilted non-centrality parameter
    """
    return (1 - eta)**2 * lambda0


def tilted_pvalue_components(
    theta: ArrayLike,
    mu_eta: ArrayLike,
    mu0: float,
    w: float,
    sigma: float,
    eta: float,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the a and b components for the tilted p-value formula (Theorem 8).

    The p-value formula p(theta) = Phi(b-a) + Phi(-a-b) requires:
    a_eta = |mu_eta - theta| / (sqrt(w_eta) * sigma_eta)
    b_eta = bias_eta(theta) / (sqrt(w_eta) * sigma_eta)

    where:
    - w_eta = w / [1 - eta*(1-w)]
    - sigma_eta = sqrt(w * sigma^2 / [1 - eta*(1-w)])
    - bias_eta(theta) = (1-eta)(1-w)(mu0 - theta) / [1 - eta*(1-w)]

    Parameters
    ----------
    theta : float or array
        Parameter value(s) being tested
    mu_eta : float or array
        Tilted posterior mean
    mu0 : float
        Prior mean
    w : float
        Original weight on data
    sigma : float
        Likelihood standard deviation
    eta : float
        Tilting parameter

    Returns
    -------
    a_eta : float or array
        Distance component
    b_eta : float or array
        Bias component
    """
    denom = 1 - eta * (1 - w)

    # Tilted effective weight
    w_eta = w / denom

    # Tilted posterior std
    sigma_eta = np.sqrt(w * sigma**2 / denom)

    # The normalizing factor for a and b
    # This should be sqrt(w_eta) * sigma_eta = sqrt(w/denom) * sqrt(w*sigma^2/denom)
    # = sqrt(w^2 * sigma^2 / denom^2) = w * sigma / denom
    norm_factor = w * sigma / denom

    # a_eta = |mu_eta - theta| / norm_factor
    a_eta = np.abs(mu_eta - theta) / norm_factor

    # bias_eta = (1-eta)(1-w)(mu0 - theta) / denom
    bias_eta = (1 - eta) * (1 - w) * (mu0 - theta) / denom

    # b_eta = bias_eta / norm_factor
    b_eta = bias_eta / norm_factor

    return a_eta, b_eta


def tilted_pvalue(
    theta: ArrayLike,
    D: ArrayLike,
    mu0: float,
    sigma: float,
    sigma0: float,
    eta: float,
) -> ArrayLike:
    """
    Compute the tilted p-value at theta (Theorem 8).

    p_eta(theta) = Phi(b_eta - a_eta) + Phi(-a_eta - b_eta)

    Parameters
    ----------
    theta : float or array
        Parameter value(s) being tested
    D : float or array
        Observed data
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    eta : float
        Tilting parameter

    Returns
    -------
    p : float or array
        Tilted p-value
    """
    w = weight(sigma, sigma0)
    mu_eta, sigma_eta, _ = tilted_params(D, mu0, sigma, sigma0, eta)

    a_eta, b_eta = tilted_pvalue_components(theta, mu_eta, mu0, w, sigma, eta)

    return stats.norm.cdf(b_eta - a_eta) + stats.norm.cdf(-a_eta - b_eta)


def tilted_ci(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    eta: float,
    alpha: float = 0.05,
    bracket_mult: float = 10.0,
) -> Tuple[float, float]:
    """
    Compute the tilted confidence interval.

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
    eta : float
        Tilting parameter
    alpha : float
        Significance level
    bracket_mult : float
        Multiplier for search bracket

    Returns
    -------
    lower : float
        Lower CI bound
    upper : float
        Upper CI bound
    """
    mu_eta, sigma_eta, _ = tilted_params(D, mu0, sigma, sigma0, eta)

    def f(theta):
        return tilted_pvalue(theta, D, mu0, sigma, sigma0, eta) - alpha

    bracket = bracket_mult * sigma

    # Find lower bound
    try:
        lower = optimize.brentq(f, mu_eta - bracket, mu_eta)
    except ValueError:
        lower = optimize.brentq(f, mu_eta - 2 * bracket, mu_eta)

    # Find upper bound
    try:
        upper = optimize.brentq(f, mu_eta, mu_eta + bracket)
    except ValueError:
        upper = optimize.brentq(f, mu_eta, mu_eta + 2 * bracket)

    return lower, upper


def tilted_ci_width(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    eta: float,
    alpha: float = 0.05,
) -> float:
    """Compute width of tilted CI."""
    lower, upper = tilted_ci(D, mu0, sigma, sigma0, eta, alpha)
    return upper - lower


def optimal_eta_numerical(
    abs_Delta: float,
    w: float = 0.5,
    sigma: float = 1.0,
    alpha: float = 0.05,
    n_sims: int = 1000,
    eta_grid: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Find optimal tilting parameter by minimizing expected CI width.

    Parameters
    ----------
    abs_Delta : float
        Absolute value of scaled prior-data conflict
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation
    alpha : float
        Significance level
    n_sims : int
        Number of simulations for width estimation
    eta_grid : array, optional
        Grid of eta values to search over
    rng : Generator, optional
        Random number generator

    Returns
    -------
    eta_star : float
        Optimal tilting parameter
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if eta_grid is None:
        eta_grid = np.linspace(0, 1, 21)

    # Compute sigma0 from w
    sigma0 = sigma * np.sqrt(w / (1 - w))
    mu0 = 0.0

    # Choose theta to achieve the target |Delta|
    # Delta = (1-w)(mu0 - D) / sigma, so for theta=D, Delta = (1-w)(mu0 - theta)/sigma
    # We want |Delta(theta)| = abs_Delta, so theta = mu0 - sigma * abs_Delta / (1-w) or + version
    theta_true = mu0 + sigma * abs_Delta / (1 - w)

    best_eta = 0.0
    best_width = np.inf

    for eta in eta_grid:
        # Simulate expected CI width
        D_samples = rng.normal(theta_true, sigma, n_sims)
        widths = []

        for D in D_samples:
            try:
                width = tilted_ci_width(D, mu0, sigma, sigma0, eta, alpha)
                widths.append(width)
            except (ValueError, RuntimeError):
                continue

        if len(widths) > 0.5 * n_sims:  # Need enough successful simulations
            mean_width = np.mean(widths)
            if mean_width < best_width:
                best_width = mean_width
                best_eta = eta

    return best_eta


def optimal_eta_approximation(
    abs_Delta: float,
    c: float = 0.18,
    power: float = 1.7,
) -> float:
    """
    Approximate optimal eta using the power-law formula from Section 10.3.

    1 - eta*(|Delta|) ≈ c / |Delta|^power

    Parameters
    ----------
    abs_Delta : float
        Absolute value of scaled prior-data conflict
    c : float
        Coefficient (default 0.18 from document)
    power : float
        Exponent (default 1.7 from document)

    Returns
    -------
    eta_star : float
        Approximate optimal tilting parameter
    """
    if abs_Delta < 0.01:
        # At zero conflict, the formula gives eta ≈ 0.23 based on document
        return 0.23

    # 1 - eta* ≈ c / |Delta|^power
    one_minus_eta = c / abs_Delta**power

    # Clamp to [0, 1]
    eta_star = max(0.0, min(1.0, 1 - one_minus_eta))

    return eta_star


def dynamic_tilted_pvalue(
    theta: ArrayLike,
    D: ArrayLike,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> ArrayLike:
    """
    Compute p-value with dynamic tilting based on |Delta(theta)|.

    Uses eta = eta*(|Delta(theta)|) where eta* is from the approximation formula.

    Parameters
    ----------
    theta : float or array
        Parameter value(s) being tested
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
    p : float or array
        Dynamically-tilted p-value
    """
    w = weight(sigma, sigma0)

    # Handle scalar vs array
    theta = np.atleast_1d(theta)
    D_scalar = np.isscalar(D)
    D = np.atleast_1d(D)

    pvals = []
    for th in theta:
        # Compute |Delta| at this theta
        Delta_theta = np.abs((1 - w) * (mu0 - th) / sigma)

        # Get optimal eta for this |Delta|
        eta_star = optimal_eta_approximation(Delta_theta)

        # Compute tilted p-value
        p = tilted_pvalue(th, D, mu0, sigma, sigma0, eta_star)
        pvals.append(p)

    result = np.array(pvals)

    if len(theta) == 1:
        return float(result.item()) if D_scalar else result
    return result


def dynamic_tilted_ci(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    alpha: float = 0.05,
    bracket_mult: float = 10.0,
) -> Tuple[float, float]:
    """
    Compute confidence interval with dynamic tilting.

    Uses eta*(|Delta(theta)|) which varies across the interval.

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
    alpha : float
        Significance level
    bracket_mult : float
        Multiplier for search bracket

    Returns
    -------
    lower : float
        Lower CI bound
    upper : float
        Upper CI bound
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    def f(theta):
        return dynamic_tilted_pvalue(theta, D, mu0, sigma, sigma0) - alpha

    bracket = bracket_mult * sigma

    # Find lower bound
    try:
        lower = optimize.brentq(f, mu_n - bracket, mu_n)
    except ValueError:
        lower = optimize.brentq(f, mu_n - 2 * bracket, mu_n)

    # Find upper bound
    try:
        upper = optimize.brentq(f, mu_n, mu_n + bracket)
    except ValueError:
        upper = optimize.brentq(f, mu_n, mu_n + 2 * bracket)

    return lower, upper


def tilted_mode(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    eta: float,
) -> float:
    """
    Compute the mode of the tilted p-value function.

    For fixed eta, the mode is mu_eta (the tilted posterior mean).

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
    eta : float
        Tilting parameter

    Returns
    -------
    mode : float
        Mode of tilted p-value function (= mu_eta)
    """
    mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta)
    return mu_eta


def dynamic_tilted_mode(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Find the mode of the dynamically-tilted p-value function.

    From Theorem 10: The mode theta* satisfies the fixed-point equation
    theta* = mu_{eta*(theta*)}

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
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    mode : float
        Mode of dynamically-tilted p-value function
    """
    w = weight(sigma, sigma0)
    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)

    # Fixed-point iteration starting from mu_n
    theta = mu_n

    for _ in range(max_iter):
        # Compute |Delta| at current theta
        Delta_theta = np.abs((1 - w) * (mu0 - theta) / sigma)

        # Get eta* for this |Delta|
        eta_star = optimal_eta_approximation(Delta_theta)

        # Compute mu_eta at this eta*
        mu_eta, _, _ = tilted_params(D, mu0, sigma, sigma0, eta_star)

        # Update theta
        theta_new = mu_eta

        if np.abs(theta_new - theta) < tol:
            return theta_new

        theta = theta_new

    return theta  # Return last value if not converged
