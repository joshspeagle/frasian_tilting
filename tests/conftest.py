"""
Pytest fixtures and configuration for Frasian inference tests.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Set random seed for reproducibility
RANDOM_SEED = 42


@dataclass
class ModelParams:
    """Container for conjugate Normal model parameters."""
    mu0: float  # Prior mean
    sigma: float  # Likelihood std
    sigma0: float  # Prior std
    w: float  # Weight on data
    sigma_n: float  # Posterior std

    @classmethod
    def from_basic(cls, mu0: float, sigma: float, sigma0: float) -> "ModelParams":
        """Create from basic parameters, computing derived quantities."""
        w = sigma0**2 / (sigma**2 + sigma0**2)
        sigma_n = np.sqrt(w) * sigma
        return cls(mu0=mu0, sigma=sigma, sigma0=sigma0, w=w, sigma_n=sigma_n)


@dataclass
class TestConfig:
    """Configuration for Monte Carlo tests."""
    n_samples: int = 10000  # Default MC samples
    n_coverage: int = 5000  # Samples for coverage tests
    seed: int = RANDOM_SEED
    ks_threshold: float = 0.01  # KS test p-value threshold
    coverage_tol: float = 0.015  # Coverage tolerance (±1.5%)
    moment_rtol: float = 0.05  # Relative tolerance for moments (5%)
    point_atol: float = 0.01  # Absolute tolerance for point estimates


# Standard test configurations
@pytest.fixture
def config():
    """Standard test configuration."""
    return TestConfig()


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return RANDOM_SEED


@pytest.fixture
def rng(seed):
    """Numpy random generator."""
    return np.random.default_rng(seed)


# Standard model parameter sets
@pytest.fixture
def balanced_model():
    """Balanced model with w = 0.5 (equal weight on prior and data)."""
    return ModelParams.from_basic(mu0=0.0, sigma=1.0, sigma0=1.0)


@pytest.fixture
def informative_prior():
    """Informative prior model with w = 0.2 (strong prior)."""
    return ModelParams.from_basic(mu0=0.0, sigma=1.0, sigma0=0.5)


@pytest.fixture
def weak_prior():
    """Weak prior model with w = 0.8 (weak prior, data dominates)."""
    return ModelParams.from_basic(mu0=0.0, sigma=1.0, sigma0=2.0)


@pytest.fixture(params=[0.2, 0.5, 0.8])
def w_values(request):
    """Parametrized weight values."""
    return request.param


@pytest.fixture(params=[0.5, 1.0, 2.0])
def sigma_values(request):
    """Parametrized sigma values."""
    return request.param


@pytest.fixture(params=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
def conflict_values(request):
    """Parametrized |Delta| values for prior-data conflict."""
    return request.param


@pytest.fixture(params=[-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
def theta_true_values(request):
    """Parametrized true theta values for coverage tests."""
    return request.param


@pytest.fixture(params=[0.0, 0.25, 0.5, 0.75, 1.0])
def eta_values(request):
    """Parametrized tilting parameter values."""
    return request.param


# Helper functions for tests
def simulate_data(
    theta_true: float,
    sigma: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate data D from N(theta_true, sigma^2)."""
    return rng.normal(theta_true, sigma, n_samples)


def compute_posterior_means(
    D: np.ndarray,
    mu0: float,
    w: float,
) -> np.ndarray:
    """Compute posterior means for array of data."""
    return w * D + (1 - w) * mu0


def compute_waldo_stats(
    mu_n: np.ndarray,
    theta: float,
    sigma_n: float,
) -> np.ndarray:
    """Compute WALDO statistics for array of posterior means."""
    return (mu_n - theta)**2 / sigma_n**2


def get_model_from_w(w: float, sigma: float = 1.0, mu0: float = 0.0) -> ModelParams:
    """Create ModelParams from weight w."""
    # w = sigma0^2 / (sigma^2 + sigma0^2)
    # Solve for sigma0: sigma0 = sigma * sqrt(w / (1 - w))
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return ModelParams.from_basic(mu0=mu0, sigma=sigma, sigma0=sigma0)


def theta_for_conflict(
    Delta: float,
    D: float,
    mu0: float,
    w: float,
    sigma: float,
) -> float:
    """Compute theta value that gives a specific scaled conflict Delta.

    Note: Delta = (1-w)(mu0 - D)/sigma is determined by D and model params.
    This function finds theta such that if we were testing at theta,
    the implied |Delta(theta)| matches the target.

    For testing, we often want to set D to achieve a target Delta.
    """
    # Delta = (1-w)(mu0 - D) / sigma
    # D = mu0 - sigma * Delta / (1-w)
    return mu0 - sigma * Delta / (1 - w)


def data_for_conflict(
    Delta: float,
    mu0: float,
    w: float,
    sigma: float,
) -> float:
    """Compute data value D that gives a specific scaled conflict Delta.

    Delta = (1-w)(mu0 - D) / sigma
    """
    return mu0 - sigma * Delta / (1 - w)


# Pytest configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "tier1: Tier 1 foundational tests")
    config.addinivalue_line("markers", "tier2: Tier 2 p-value tests")
    config.addinivalue_line("markers", "tier3: Tier 3 coverage tests")
    config.addinivalue_line("markers", "tier4: Tier 4 tilting tests")
    config.addinivalue_line("markers", "tier5: Tier 5 integration tests")
