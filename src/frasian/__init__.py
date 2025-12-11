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
    # Wald CD
    wald_cd_density,
    wald_cd_mean,
    wald_cd_mode,
    # WALDO CD
    waldo_cd_params,
    waldo_cd_density,
    waldo_cd_mean,
    waldo_cd_mode,
    # Numerical CD
    cd_from_pvalue,
    cd_mean_numerical,
    cd_mode_numerical,
    cd_quantile,
    cd_variance_numerical,
    # Dynamic CD
    dynamic_cd_density,
    dynamic_cd_mean,
    dynamic_cd_mode,
    # Legacy
    pvalue_mode,
    pvalue_at_mode,
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
    # Confidence distributions
    "wald_cd_density",
    "wald_cd_mean",
    "wald_cd_mode",
    "waldo_cd_params",
    "waldo_cd_density",
    "waldo_cd_mean",
    "waldo_cd_mode",
    "cd_from_pvalue",
    "cd_mean_numerical",
    "cd_mode_numerical",
    "cd_quantile",
    "cd_variance_numerical",
    "dynamic_cd_density",
    "dynamic_cd_mean",
    "dynamic_cd_mode",
    "pvalue_mode",
    "pvalue_at_mode",
]
