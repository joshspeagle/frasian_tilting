"""
WALDO-specific computations.

This module implements the WALDO (Weighted Accurate Likelihood-free inference
via Diagnostic Orderings) test statistic, p-value function, and confidence intervals.
"""

import numpy as np
from scipy import stats
from scipy import optimize
from typing import Tuple, Union, Optional

from .core import (
    posterior_params,
    weight,
    bias,
    variance,
    delta_scaled,
)

ArrayLike = Union[float, np.ndarray]


def waldo_statistic(
    mu_n: ArrayLike,
    sigma_n: float,
    theta: ArrayLike,
) -> ArrayLike:
    """
    Compute the WALDO test statistic.

    tau_WALDO = (mu_n - theta)^2 / sigma_n^2

    Parameters
    ----------
    mu_n : float or array
        Posterior mean(s)
    sigma_n : float
        Posterior standard deviation
    theta : float or array
        Parameter value(s) being tested

    Returns
    -------
    tau : float or array
        WALDO test statistic
    """
    return (mu_n - theta)**2 / sigma_n**2


def noncentrality(
    theta: ArrayLike,
    mu0: float,
    w: float,
    sigma: float,
    sigma0: float,
) -> ArrayLike:
    """
    Compute the non-centrality parameter lambda(theta) for the WALDO statistic.

    From Theorem 2: lambda(theta) = delta(theta)^2 / w
    where delta(theta) = (1-w)(mu0 - theta) / (sqrt(w) * sigma)

    Equivalently: lambda(theta) = (1-w)^2 * (mu0 - theta)^2 / (w^2 * sigma^2)

    Parameters
    ----------
    theta : float or array
        True parameter value(s)
    mu0 : float
        Prior mean
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation (not used directly but kept for API consistency)

    Returns
    -------
    lambda_ : float or array
        Non-centrality parameter
    """
    delta = delta_scaled(theta, mu0, w, sigma)
    return delta**2 / w


def noncentrality_from_prior_residual(
    delta_prior: ArrayLike,
    w: float,
) -> ArrayLike:
    """
    Compute non-centrality from the prior residual.

    Alternative formula: lambda(theta) = delta_prior(theta)^2 * kappa^2
    where delta_prior = (theta - mu0) / sigma0 and kappa = (1-w) * sigma / (w * sigma0)

    Parameters
    ----------
    delta_prior : float or array
        Prior residual (theta - mu0) / sigma0
    w : float
        Weight on data

    Returns
    -------
    lambda_ : float or array
        Non-centrality parameter
    """
    # From document: lambda = delta_prior^2 / w when using the scaled delta
    # Need to be careful about which delta we're using
    return delta_prior**2 * (1 - w)**2 / w


def pvalue_components(
    theta: ArrayLike,
    mu_n: ArrayLike,
    mu0: float,
    w: float,
    sigma: float,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the a and b components of the p-value formula.

    a(theta) = |mu_n - theta| / (w * sigma) >= 0
    b(theta) = (1 - w) * (mu0 - theta) / (w * sigma)

    Parameters
    ----------
    theta : float or array
        Parameter value(s) being tested
    mu_n : float or array
        Posterior mean(s)
    mu0 : float
        Prior mean
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation

    Returns
    -------
    a : float or array
        Absolute standardized distance from posterior mean
    b : float or array
        Standardized bias term
    """
    a = np.abs(mu_n - theta) / (w * sigma)
    b = (1 - w) * (mu0 - theta) / (w * sigma)
    return a, b


def pvalue(
    theta: ArrayLike,
    mu_n: ArrayLike,
    mu0: float,
    w: float,
    sigma: float,
) -> ArrayLike:
    """
    Compute the WALDO p-value at theta.

    From Theorem 3: p(theta) = Phi(b - a) + Phi(-a - b)

    Parameters
    ----------
    theta : float or array
        Parameter value(s) being tested
    mu_n : float or array
        Posterior mean(s)
    mu0 : float
        Prior mean
    w : float
        Weight on data
    sigma : float
        Likelihood standard deviation

    Returns
    -------
    p : float or array
        P-value(s)
    """
    a, b = pvalue_components(theta, mu_n, mu0, w, sigma)
    return stats.norm.cdf(b - a) + stats.norm.cdf(-a - b)


def pvalue_from_data(
    theta: ArrayLike,
    D: ArrayLike,
    mu0: float,
    sigma: float,
    sigma0: float,
) -> ArrayLike:
    """
    Compute the WALDO p-value from raw data.

    Convenience function that first computes posterior parameters.

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
        P-value(s)
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)
    return pvalue(theta, mu_n, mu0, w, sigma)


def critical_value(
    theta: ArrayLike,
    w: float,
    alpha: float = 0.05,
) -> ArrayLike:
    """
    Compute the location-dependent critical value for WALDO.

    The critical value C_{theta, alpha} satisfies:
    P(tau_WALDO > C_{theta, alpha} | theta) = alpha

    Since tau_WALDO | theta ~ w * chi^2_1(lambda(theta)), we have:
    C_{theta, alpha} = w * chi^2_{1, lambda(theta)}(1 - alpha)

    Note: This function computes the critical value for the *central* chi-squared
    when lambda = 0, which corresponds to testing at theta = mu0.

    For the general case with non-zero lambda, use critical_value_noncentrality.

    Parameters
    ----------
    theta : float or array
        Parameter value(s)
    w : float
        Weight on data
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    c : float or array
        Critical value(s)
    """
    # For central chi-squared (lambda = 0):
    return w * stats.chi2.ppf(1 - alpha, df=1)


def critical_value_noncentral(
    lambda_: ArrayLike,
    w: float,
    alpha: float = 0.05,
) -> ArrayLike:
    """
    Compute the critical value for non-central chi-squared distribution.

    Parameters
    ----------
    lambda_ : float or array
        Non-centrality parameter(s)
    w : float
        Weight on data (scale factor)
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    c : float or array
        Critical value(s)
    """
    # tau ~ w * ncx2(df=1, nc=lambda)
    # P(tau > c) = alpha => P(tau/w > c/w) = alpha
    # c/w = ncx2.ppf(1-alpha, df=1, nc=lambda)
    lambda_ = np.atleast_1d(lambda_)
    c = np.array([
        w * stats.ncx2.ppf(1 - alpha, df=1, nc=lam)
        for lam in lambda_
    ])
    return c.squeeze() if c.size == 1 else c


def confidence_interval(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    alpha: float = 0.05,
    bracket_mult: float = 10.0,
) -> Tuple[float, float]:
    """
    Compute the WALDO confidence interval for theta.

    The CI is the set {theta : p(theta) >= alpha}.

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
        Significance level (default 0.05)
    bracket_mult : float
        Multiplier for search bracket (default 10)

    Returns
    -------
    lower : float
        Lower CI bound
    upper : float
        Upper CI bound
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

    # Search bracket: extend far from posterior mean
    bracket_width = bracket_mult * sigma

    # Define function to find roots of p(theta) - alpha = 0
    def f(theta):
        return pvalue(theta, mu_n, mu0, w, sigma) - alpha

    # Find lower bound (search below mu_n)
    try:
        lower = optimize.brentq(f, mu_n - bracket_width, mu_n)
    except ValueError:
        # If no root found, extend bracket
        lower = optimize.brentq(f, mu_n - 2 * bracket_width, mu_n)

    # Find upper bound (search above mu_n)
    try:
        upper = optimize.brentq(f, mu_n, mu_n + bracket_width)
    except ValueError:
        upper = optimize.brentq(f, mu_n, mu_n + 2 * bracket_width)

    return lower, upper


def confidence_interval_width(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute the width of the WALDO confidence interval.

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
        Significance level (default 0.05)

    Returns
    -------
    width : float
        CI width (upper - lower)
    """
    lower, upper = confidence_interval(D, mu0, sigma, sigma0, alpha)
    return upper - lower


def wald_ci(
    D: float,
    sigma: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute the Wald (pure likelihood) confidence interval.

    CI: D +/- z_{1-alpha/2} * sigma

    Parameters
    ----------
    D : float
        Observed data (MLE)
    sigma : float
        Likelihood standard deviation
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    lower : float
        Lower CI bound
    upper : float
        Upper CI bound
    """
    z = stats.norm.ppf(1 - alpha / 2)
    half_width = z * sigma
    return D - half_width, D + half_width


def wald_ci_width(sigma: float, alpha: float = 0.05) -> float:
    """
    Compute the width of the Wald confidence interval.

    Width = 2 * z_{1-alpha/2} * sigma

    Parameters
    ----------
    sigma : float
        Likelihood standard deviation
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    width : float
        CI width
    """
    z = stats.norm.ppf(1 - alpha / 2)
    return 2 * z * sigma


def posterior_ci(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute the posterior credible interval.

    CI: mu_n +/- z_{1-alpha/2} * sigma_n

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
        Significance level (default 0.05)

    Returns
    -------
    lower : float
        Lower CI bound
    upper : float
        Upper CI bound
    """
    mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)
    z = stats.norm.ppf(1 - alpha / 2)
    half_width = z * sigma_n
    return mu_n - half_width, mu_n + half_width


def posterior_ci_width(
    sigma: float,
    sigma0: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute the width of the posterior credible interval.

    Width = 2 * z_{1-alpha/2} * sqrt(w) * sigma

    Parameters
    ----------
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    width : float
        CI width
    """
    w = weight(sigma, sigma0)
    sigma_n = np.sqrt(w) * sigma
    z = stats.norm.ppf(1 - alpha / 2)
    return 2 * z * sigma_n


def ci_asymmetry(
    D: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute the asymmetry of the WALDO CI.

    Asymmetry = (upper - mu_n) - (mu_n - lower)

    Positive asymmetry: CI extends further above mu_n (toward D when D > mu_n)
    Negative asymmetry: CI extends further below mu_n

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
        Significance level (default 0.05)

    Returns
    -------
    asymmetry : float
        CI asymmetry measure
    """
    mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
    lower, upper = confidence_interval(D, mu0, sigma, sigma0, alpha)
    return (upper - mu_n) - (mu_n - lower)
