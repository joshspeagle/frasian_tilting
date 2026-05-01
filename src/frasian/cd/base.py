"""ConfidenceDistribution protocol.

A `ConfidenceDistribution` is a Schweder–Hjort-style frequentist
distribution over the parameter, derived from a `(Model, TestStatistic,
data)` triple — and optionally tilted by a `(TiltingScheme, η)` choice.

The framework's concrete implementation is `cd.grid.GridConfidenceDistribution`,
built by `cd.from_pvalue.build_cd_from_pvalue(...)` from a fine θ-grid
evaluation of `tilting.pvalue(θ, …, statistic)` and a finite-difference
density. `cd.from_closed_form` provides Wald/WALDO closed-form CDs as
test fixtures only.

Distances on CDs (`cd.distances.wasserstein_1`, `wasserstein_2`,
`total_variation`) act on the *derived* probability CDF, which is
always monotone because the framework stores `pdf_values ≥ 0` as the
primitive and computes `cdf` as its cumulative integral. The optional
`signed_confidence` field (the inversion-based C(θ) curve) can be
non-monotone when the underlying p-value is multimodal — that is the
diagnostic surface the smoothness experiment uses to flag pathologies,
not the object distance metrics operate on.

References:
- Schweder, T. & Hjort, N. L. *Confidence, Likelihood, Probability:
  Statistical Inference with Confidence Distributions*. Cambridge, 2016.
- Singh, K., Xie, M. & Strawderman, W. E. "Combining information from
  independent sources through confidence distributions." *Annals of
  Statistics* 33 (2005): 159–183.
- Xie, M. & Singh, K. "Confidence Distribution, the Frequentist
  Distribution Estimator of a Parameter: A Review." *International
  Statistical Review* 81 (2013): 3–39.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class ConfidenceDistribution(Protocol):
    """A single CD over a 1D parameter.

    The `cdf` returned by implementations is the *probability CDF*
    derived from a non-negative `pdf_values` primitive, so it is
    monotone non-decreasing by construction. Implementations may also
    expose a `signed_confidence` curve (the inversion-based C(θ)) as
    auxiliary diagnostic data; that quantity may be non-monotone when
    the underlying p-value is multimodal.

    Invariants any implementation must satisfy
    (tests/properties/test_cd_invariants.py):
        - `pdf(theta) >= 0` everywhere on `theta_grid`.
        - `pdf` integrates to 1 over `theta_grid` (Z-normalised at
          construction).
        - `cdf` is monotone non-decreasing on `theta_grid`.
        - `cdf(theta_grid[0]) ≈ 0`, `cdf(theta_grid[-1]) ≈ 1`
          (within the truncation tolerance of a finite grid).
        - `interval(alpha)` matches `(quantile(α/2), quantile(1−α/2))`.
        - Under H0 (θ_true = data-generating parameter), and on
          single-region cells, `cdf(theta_true)` is Uniform[0, 1]
          across replicates (Singh–Xie–Strawderman 2005, Defn 2.1).
    """

    name: str
    theta_grid: NDArray[np.float64]

    def pdf(self, theta: ArrayLike) -> NDArray[np.float64]: ...
    def cdf(self, theta: ArrayLike) -> NDArray[np.float64]: ...
    def quantile(self, q: ArrayLike) -> NDArray[np.float64]: ...
    def interval(self, alpha: float) -> tuple[float, float]: ...
    def mean(self) -> float: ...
    def median(self) -> float: ...
    def mode(self) -> float: ...
