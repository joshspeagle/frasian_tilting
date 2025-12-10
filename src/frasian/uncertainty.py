"""
Uncertainty Quantification Utilities

Provides functions for computing standard errors, confidence intervals,
and bootstrap estimates for Monte Carlo simulation results.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import stats


@dataclass
class MCResult:
    """Container for Monte Carlo estimate with uncertainty.

    Attributes:
        value: Point estimate (mean or proportion)
        se: Standard error
        ci_low: Lower bound of 95% confidence interval
        ci_high: Upper bound of 95% confidence interval
        n_samples: Number of MC replicates used
        raw_samples: Optional array of raw samples for custom analysis
    """
    value: float
    se: float
    ci_low: float
    ci_high: float
    n_samples: int
    raw_samples: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"MCResult({self.value:.4f} +/- {self.se:.4f}, 95% CI: [{self.ci_low:.4f}, {self.ci_high:.4f}], n={self.n_samples})"

    def format(self, decimals: int = 2) -> str:
        """Format as 'value +/- se'."""
        return f"{self.value:.{decimals}f} +/- {self.se:.{decimals}f}"

    def format_percent(self, decimals: int = 1) -> str:
        """Format as 'value% +/- se%' for proportions."""
        return f"{self.value*100:.{decimals}f}% +/- {self.se*100:.{decimals}f}%"


# =============================================================================
# Standard Error Functions
# =============================================================================

def proportion_se(p: float, n: int) -> float:
    """Compute standard error for a proportion.

    SE = sqrt(p * (1 - p) / n)

    Args:
        p: Estimated proportion (between 0 and 1)
        n: Sample size

    Returns:
        Standard error of the proportion
    """
    if n <= 0:
        raise ValueError("Sample size must be positive")
    # Clip p to avoid numerical issues at boundaries
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.sqrt(p * (1 - p) / n)


def mean_se(samples: np.ndarray) -> float:
    """Compute standard error of the mean.

    SE = std(samples) / sqrt(n)

    Args:
        samples: Array of samples

    Returns:
        Standard error of the mean
    """
    samples = np.asarray(samples)
    n = len(samples)
    if n <= 1:
        return np.nan
    return np.std(samples, ddof=1) / np.sqrt(n)


# =============================================================================
# Confidence Interval Functions
# =============================================================================

def proportion_ci(
    p: float,
    n: int,
    alpha: float = 0.05,
    method: str = "wilson"
) -> tuple[float, float]:
    """Compute confidence interval for a proportion.

    Args:
        p: Estimated proportion
        n: Sample size
        alpha: Significance level (default 0.05 for 95% CI)
        method: CI method - "wilson" (default) or "normal"

    Returns:
        Tuple of (lower, upper) bounds
    """
    if method == "normal":
        se = proportion_se(p, n)
        z = stats.norm.ppf(1 - alpha / 2)
        return (p - z * se, p + z * se)

    elif method == "wilson":
        # Wilson score interval - better for proportions near 0 or 1
        z = stats.norm.ppf(1 - alpha / 2)
        z2 = z ** 2
        denom = 1 + z2 / n
        center = (p + z2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom
        return (center - margin, center + margin)

    else:
        raise ValueError(f"Unknown method: {method}")


def mean_ci(
    samples: np.ndarray,
    alpha: float = 0.05,
    method: str = "normal"
) -> tuple[float, float]:
    """Compute confidence interval for the mean.

    Args:
        samples: Array of samples
        alpha: Significance level (default 0.05 for 95% CI)
        method: CI method - "normal" (default) or "t"

    Returns:
        Tuple of (lower, upper) bounds
    """
    samples = np.asarray(samples)
    n = len(samples)
    mean = np.mean(samples)
    se = mean_se(samples)

    if method == "normal":
        z = stats.norm.ppf(1 - alpha / 2)
        return (mean - z * se, mean + z * se)

    elif method == "t":
        t = stats.t.ppf(1 - alpha / 2, df=n - 1)
        return (mean - t * se, mean + t * se)

    else:
        raise ValueError(f"Unknown method: {method}")


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: Optional[int] = None
) -> tuple[float, float]:
    """Compute bootstrap confidence interval.

    Args:
        samples: Array of samples
        statistic: Function to compute statistic of interest (default: mean)
        alpha: Significance level (default 0.05 for 95% CI)
        n_boot: Number of bootstrap replicates
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower, upper) bounds using percentile method
    """
    samples = np.asarray(samples)
    n = len(samples)

    rng = np.random.default_rng(seed)

    boot_stats = np.zeros(n_boot)
    for i in range(n_boot):
        boot_sample = rng.choice(samples, size=n, replace=True)
        boot_stats[i] = statistic(boot_sample)

    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return (lower, upper)


# =============================================================================
# Convenience Functions
# =============================================================================

def proportion_result(
    successes: int,
    n: int,
    alpha: float = 0.05
) -> MCResult:
    """Create MCResult for a proportion from count data.

    Args:
        successes: Number of successes
        n: Total trials
        alpha: Significance level for CI

    Returns:
        MCResult with proportion, SE, and CI
    """
    p = successes / n
    se = proportion_se(p, n)
    ci_low, ci_high = proportion_ci(p, n, alpha)

    return MCResult(
        value=p,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        n_samples=n,
        raw_samples=None
    )


def mean_result(
    samples: np.ndarray,
    alpha: float = 0.05,
    keep_samples: bool = False
) -> MCResult:
    """Create MCResult for a mean from sample data.

    Args:
        samples: Array of samples
        alpha: Significance level for CI
        keep_samples: Whether to store raw samples in result

    Returns:
        MCResult with mean, SE, and CI
    """
    samples = np.asarray(samples)
    mean = np.mean(samples)
    se = mean_se(samples)
    ci_low, ci_high = mean_ci(samples, alpha)

    return MCResult(
        value=mean,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        n_samples=len(samples),
        raw_samples=samples.copy() if keep_samples else None
    )


def format_with_uncertainty(
    value: float,
    se: float,
    decimals: int = 1,
    percent: bool = False
) -> str:
    """Format a value with its uncertainty.

    Args:
        value: Point estimate
        se: Standard error
        decimals: Number of decimal places
        percent: If True, multiply by 100 and add % symbol

    Returns:
        Formatted string like "95.2 +/- 0.3" or "95.2% +/- 0.3%"
    """
    if percent:
        return f"{value*100:.{decimals}f}% +/- {se*100:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f} +/- {se:.{decimals}f}"


# =============================================================================
# Array-wise Operations (for grids)
# =============================================================================

def proportion_se_array(p: np.ndarray, n: int) -> np.ndarray:
    """Compute standard errors for an array of proportions.

    Args:
        p: Array of proportions
        n: Sample size (same for all)

    Returns:
        Array of standard errors
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.sqrt(p * (1 - p) / n)


def wilson_ci_array(
    p: np.ndarray,
    n: int,
    alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Wilson confidence intervals for array of proportions.

    Args:
        p: Array of proportions
        n: Sample size (same for all)
        alpha: Significance level

    Returns:
        Tuple of (lower, upper) arrays
    """
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z ** 2
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom
    return (center - margin, center + margin)
