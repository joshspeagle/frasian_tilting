"""
Frasian Inference Framework

Numerical experiments validating the connection between WALDO,
Fraser's higher-order likelihood inference, and confidence distributions.
"""

from .core import (
    posterior_params,
    standardized_coord,
    scaled_conflict,
    prior_residual,
    bias,
    variance,
)

from .waldo import (
    waldo_statistic,
    noncentrality,
    pvalue,
    pvalue_components,
    confidence_interval,
    critical_value,
)

from .confidence import (
    pvalue_mode,
    pvalue_mean,
    sample_confidence_dist,
    mean_between_mode_and_mle,
)

from .tilting import (
    tilted_params,
    tilted_noncentrality,
    tilted_pvalue,
    tilted_ci,
    optimal_eta_approximation,
    dynamic_tilted_pvalue,
    dynamic_tilted_ci,
    dynamic_tilted_mode,
)

__version__ = "0.1.0"

__all__ = [
    # Core functions
    "posterior_params",
    "standardized_coord",
    "scaled_conflict",
    "prior_residual",
    "bias",
    "variance",
    # WALDO functions
    "waldo_statistic",
    "noncentrality",
    "pvalue",
    "pvalue_components",
    "confidence_interval",
    "critical_value",
    # Confidence distribution
    "pvalue_mode",
    "pvalue_mean",
    "sample_confidence_dist",
    "mean_between_mode_and_mle",
]
