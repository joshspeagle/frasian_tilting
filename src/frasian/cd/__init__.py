"""Confidence distributions.

Public surface:
  - `ConfidenceDistribution` — protocol (the type hint).
  - `GridConfidenceDistribution` — the concrete grid-based container
    (pdf-primary; cdf is always monotone, derived as ∫pdf).
  - `CDValidityIssue` — record returned by `validate()`.
  - `build_cd_from_pvalue` — universal CD constructor: evaluates
    `tilting.pvalue` on a fine θ-grid, FD-derives the density.
  - `wasserstein_1`, `wasserstein_2`, `total_variation` — distance
    functions on CDs.

Closed-form helpers (`from_closed_form.wald_cd`, `waldo_cd`,
`tilted_waldo_cd`) are test fixtures; they are accessible by name but
not re-exported here.
"""

from .base import ConfidenceDistribution
from .distances import (total_variation, wasserstein_1,
                          wasserstein_1_gaussian_shift,
                          wasserstein_1_gaussian_zero_mean_scale,
                          wasserstein_2, wasserstein_2_gaussian)
from .from_pvalue import build_cd_from_pvalue
from .grid import CDValidityIssue, GridConfidenceDistribution

__all__ = [
    "ConfidenceDistribution",
    "GridConfidenceDistribution",
    "CDValidityIssue",
    "build_cd_from_pvalue",
    "wasserstein_1",
    "wasserstein_2",
    "total_variation",
    "wasserstein_1_gaussian_shift",
    "wasserstein_1_gaussian_zero_mean_scale",
    "wasserstein_2_gaussian",
]
