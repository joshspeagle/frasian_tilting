"""ConfidenceDistribution protocol and CDFamily.

A `ConfidenceDistribution` is a Schweder–Hjort-style frequentist distribution
over the parameter, derived from a (Model, TestStatistic, data) triple — and
optionally tilted by a (TiltingScheme, η) choice. The single `build_cd`
factory in `cd.factory` dispatches to one of three constructors:

  - `cd.from_pvalue`        — Schweder–Hjort: c(θ) = ½ |dp/dθ|
  - `cd.from_closed_form`   — registered analytic CDs for specific (model, statistic) pairs
  - `cd.from_tilted`        — uses the tilted-posterior CDF as the CD

A `CDFamily` is a parametric family of CDs (e.g. one per η on a tilting path)
held separately, so the single-distribution `ConfidenceDistribution` does not
need to pretend to also be a family.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class ConfidenceDistribution(Protocol):
    """A single CD over a 1D parameter.

    Invariants any implementation must satisfy
    (tests/properties/test_cd_invariants.py):
        - `cdf` is monotone non-decreasing on `theta_grid`.
        - `cdf(-inf) ≈ 0`, `cdf(+inf) ≈ 1`.
        - `interval(alpha)` matches `(quantile(α/2), quantile(1−α/2))`.
        - Under H0, `cdf(theta_true)` is Uniform[0, 1] across replicates.
    """

    name: str
    theta_grid: NDArray[np.float64]

    def pdf(self, theta: ArrayLike) -> NDArray[np.float64]: ...
    def cdf(self, theta: ArrayLike) -> NDArray[np.float64]: ...
    def quantile(self, q: ArrayLike) -> NDArray[np.float64]: ...
    def interval(self, alpha: float) -> tuple[float, float]: ...
    def mean(self) -> float: ...
    def mode(self) -> float: ...


@runtime_checkable
class CDFamily(Protocol):
    """A parametric family of CDs (e.g. one per η along a tilting path)."""

    name: str

    def __iter__(self) -> Iterator[ConfidenceDistribution]: ...
    def __len__(self) -> int: ...
    def at(self, parameter: float) -> ConfidenceDistribution: ...
