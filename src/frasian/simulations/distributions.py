"""
Distribution Validation Simulation Experiments

Generates samples of posterior means and WALDO statistics for validating
the theoretical distributions (Theorems 1 and 2).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import posterior_params, bias, variance, weight
from ..waldo import waldo_statistic, noncentrality


@dataclass
class DistributionSamples:
    """Container for distribution validation samples."""
    theta: float
    posterior_mean_errors: np.ndarray  # (mu_n - theta) samples
    waldo_statistics: np.ndarray       # tau samples
    n_samples: int
    seed: int

    # Model parameters used
    mu0: float
    sigma: float
    sigma0: float
    w: float

    # Theoretical values for comparison
    theoretical_bias: float      # b(theta) = (1-w)(mu0 - theta)
    theoretical_variance: float  # v = w^2 * sigma^2
    theoretical_noncentrality: float  # lambda(theta)


def simulate_distribution_samples(
    theta: float,
    n_samples: int,
    mu0: float = 0.0,
    sigma: float = 1.0,
    sigma0: float = 1.0,
    seed: Optional[int] = None
) -> DistributionSamples:
    """Simulate samples of posterior mean errors and WALDO statistics.

    For validating:
    - Theorem 1: (mu_n - theta) ~ N(b(theta), v)
    - Theorem 2: tau_WALDO ~ w * chi^2_1(lambda(theta))

    Args:
        theta: True parameter value
        n_samples: Number of samples to generate
        mu0: Prior mean
        sigma: Likelihood standard deviation
        sigma0: Prior standard deviation
        seed: Random seed for reproducibility

    Returns:
        DistributionSamples containing raw samples and theoretical values
    """
    rng = np.random.default_rng(seed)

    # Compute weight
    w = weight(sigma, sigma0)

    # Compute theoretical values
    b_theta = bias(theta, mu0, w)  # (1-w)(mu0 - theta)
    v = variance(w, sigma)  # w^2 * sigma^2
    lambda_theta = noncentrality(theta, mu0, w, sigma, sigma0)

    # Generate samples
    posterior_mean_errors = np.zeros(n_samples)
    waldo_stats = np.zeros(n_samples)

    for i in range(n_samples):
        # Simulate data
        D = rng.normal(theta, sigma)

        # Compute posterior mean
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)

        # Posterior mean error
        posterior_mean_errors[i] = mu_n - theta

        # WALDO statistic
        waldo_stats[i] = waldo_statistic(mu_n, sigma_n, theta)

    return DistributionSamples(
        theta=theta,
        posterior_mean_errors=posterior_mean_errors,
        waldo_statistics=waldo_stats,
        n_samples=n_samples,
        seed=seed if seed is not None else -1,
        mu0=mu0,
        sigma=sigma,
        sigma0=sigma0,
        w=w,
        theoretical_bias=b_theta,
        theoretical_variance=v,
        theoretical_noncentrality=lambda_theta
    )


def simulate_multiple_theta(
    theta_values: list[float],
    n_samples: int,
    mu0: float = 0.0,
    sigma: float = 1.0,
    sigma0: float = 1.0,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """Simulate distribution samples for multiple theta values.

    Args:
        theta_values: List of theta values to simulate
        n_samples: Samples per theta
        mu0: Prior mean
        sigma: Likelihood standard deviation
        sigma0: Prior standard deviation
        seed: Base random seed

    Returns:
        Dictionary with:
            - "theta_values": Array of theta values
            - "posterior_mean_errors": 2D array [n_theta, n_samples]
            - "waldo_statistics": 2D array [n_theta, n_samples]
            - "theoretical_bias": Array of b(theta) values
            - "theoretical_variance": Scalar v
            - "theoretical_noncentrality": Array of lambda(theta) values
            - "w": Weight parameter
    """
    n_theta = len(theta_values)

    # Output arrays
    posterior_mean_errors = np.zeros((n_theta, n_samples))
    waldo_statistics = np.zeros((n_theta, n_samples))
    theoretical_bias = np.zeros(n_theta)
    theoretical_noncentrality = np.zeros(n_theta)

    for i, theta in enumerate(theta_values):
        samples = simulate_distribution_samples(
            theta=theta,
            n_samples=n_samples,
            mu0=mu0,
            sigma=sigma,
            sigma0=sigma0,
            seed=seed + i * 1000
        )

        posterior_mean_errors[i, :] = samples.posterior_mean_errors
        waldo_statistics[i, :] = samples.waldo_statistics
        theoretical_bias[i] = samples.theoretical_bias
        theoretical_noncentrality[i] = samples.theoretical_noncentrality

    # Get w and v from first sample (same for all theta)
    w = weight(sigma, sigma0)
    v = variance(w, sigma)

    return {
        "theta_values": np.array(theta_values),
        "posterior_mean_errors": posterior_mean_errors,
        "waldo_statistics": waldo_statistics,
        "theoretical_bias": theoretical_bias,
        "theoretical_variance": v,
        "theoretical_noncentrality": theoretical_noncentrality,
        "w": w,
        "mu0": mu0,
        "sigma": sigma,
        "sigma0": sigma0,
        "n_samples": n_samples,
    }


def run_standard_distribution_simulation(
    n_samples: int = 10000,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """Run distribution validation with standard parameters.

    Standard configuration:
        - theta: [0, 1, 2, 3]
        - mu0: 0, sigma: 1, sigma0: 1 (w = 0.5)

    Args:
        n_samples: Samples per theta value
        seed: Random seed

    Returns:
        Dictionary with distribution samples
    """
    return simulate_multiple_theta(
        theta_values=[0.0, 1.0, 2.0, 3.0],
        n_samples=n_samples,
        mu0=0.0,
        sigma=1.0,
        sigma0=1.0,
        seed=seed
    )
