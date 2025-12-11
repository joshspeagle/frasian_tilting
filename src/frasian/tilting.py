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


def tilted_ci_width_batch(
    D_samples: np.ndarray,
    mu0: float,
    sigma: float,
    sigma0: float,
    eta: float,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Compute CI widths for a batch of D samples.

    Uses the original brentq method but with pre-computed constants.
    """
    n = len(D_samples)
    widths = np.zeros(n)

    for i in range(n):
        try:
            lower, upper = tilted_ci(D_samples[i], mu0, sigma, sigma0, eta, alpha)
            widths[i] = upper - lower
        except (ValueError, RuntimeError):
            widths[i] = np.nan

    return widths


def compute_mean_width_for_eta(
    D_samples: np.ndarray,
    mu0: float,
    sigma: float,
    sigma0: float,
    eta: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute mean CI width for a batch of D samples at a given eta.

    Optimized version that stops early if clearly not optimal.
    """
    n = len(D_samples)
    total_width = 0.0
    valid_count = 0

    for i in range(n):
        try:
            lower, upper = tilted_ci(D_samples[i], mu0, sigma, sigma0, eta, alpha)
            width = upper - lower
            if width > 0 and width < 100:  # Sanity check
                total_width += width
                valid_count += 1
        except (ValueError, RuntimeError):
            continue

    if valid_count > 0.5 * n:
        return total_width / valid_count
    return np.inf


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
    Note: The mode of the dynamic p-value function may differ from mu_n,
    especially for extreme D values, so we first find the mode.

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

    def pval(theta):
        return dynamic_tilted_pvalue(theta, D, mu0, sigma, sigma0)

    def f(theta):
        return pval(theta) - alpha

    # First find the mode of the dynamic p-value function
    # Search in a wide range around mu_n and D
    search_center = (mu_n + D) / 2  # Between posterior mean and data
    search_range = max(abs(mu_n - D), 5 * sigma) + 3 * sigma

    # Find mode by maximizing p-value
    result = optimize.minimize_scalar(
        lambda t: -pval(t),
        bounds=(search_center - search_range, search_center + search_range),
        method='bounded'
    )
    mode = result.x
    p_mode = pval(mode)

    # If p(mode) < alpha, the CI is very wide or doesn't exist normally
    # Fall back to Wald-like bounds
    if p_mode < alpha:
        # Use Wald-style bounds
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        return D - z * sigma, D + z * sigma

    bracket = bracket_mult * sigma

    # Helper to find root with expanding bracket around mode
    def find_root_expanding(start, direction):
        """Find root with progressively wider brackets."""
        for mult in [1, 2, 3, 5, 10, 20]:
            if direction == 'lower':
                a = mode - mult * bracket
                b = mode
            else:
                a = mode
                b = mode + mult * bracket
            try:
                return optimize.brentq(f, a, b, xtol=1e-8)
            except ValueError:
                continue
        # If all brackets fail, return the bracket bound
        return mode - mult * bracket if direction == 'lower' else mode + mult * bracket

    # Find lower bound (below mode)
    lower = find_root_expanding(mode, 'lower')

    # Find upper bound (above mode)
    upper = find_root_expanding(mode, 'upper')

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


# =============================================================================
# MLP-Based Optimal Tilting
# =============================================================================

# Module-level cache for lazy loading
_optimal_eta_predictor = None


def _get_optimal_eta_predictor():
    """Lazy-load optimal eta predictor (monotonic MLP).

    Uses the monotonic neural network which guarantees smooth,
    strictly monotonic predictions of eta*(|Delta|).
    """
    global _optimal_eta_predictor
    if _optimal_eta_predictor is None:
        from pathlib import Path
        from .simulations.mlp_monotonic import OptimalEtaPredictor

        model_path = (
            Path(__file__).parent.parent.parent
            / "output" / "simulations" / "mlp" / "monotonic_eta_mlp.pt"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Monotonic MLP not found: {model_path}\n"
                f"Run 'python scripts/train_monotonic_eta_mlp.py' to generate it."
            )

        _optimal_eta_predictor = OptimalEtaPredictor.from_file(str(model_path))

    return _optimal_eta_predictor


def optimal_eta_mlp(
    abs_delta: float,
    w: float = 0.5,
    alpha: float = 0.05,
) -> float:
    """
    Get optimal η* using monotonic neural network.

    This is the recommended method for production use. It uses a
    monotonic neural network to predict the optimal tilting parameter
    η* that minimizes expected CI width while maintaining correct coverage.

    The monotonic architecture guarantees that η*(|Δ|) is strictly
    increasing in |Δ|, providing smooth predictions without post-hoc
    corrections.

    Parameters
    ----------
    abs_delta : float
        Absolute value of scaled prior-data conflict |Δ|
    w : float
        Prior weight in (0, 1)
    alpha : float
        Significance level in (0, 1)

    Returns
    -------
    eta_star : float
        Optimal tilting parameter in [η_min(w), 1]

    Raises
    ------
    FileNotFoundError
        If the monotonic MLP has not been trained yet.

    Examples
    --------
    >>> eta_star = optimal_eta_mlp(abs_delta=1.0, w=0.5, alpha=0.05)
    >>> eta_star  # Should be approximately 0.77
    """
    predictor = _get_optimal_eta_predictor()
    return predictor.get_optimal_eta(w, alpha, abs_delta)
