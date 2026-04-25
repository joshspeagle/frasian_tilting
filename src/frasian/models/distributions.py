"""Concrete distributions used by `NormalNormalModel`.

`NormalDistribution` is the working horse for prior, posterior, and tilted
posterior. `GaussianLikelihood` is the model's likelihood-of-data view.
Both are dataclasses and conform to the protocols in `models.base`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class NormalDistribution:
    """1D Normal(loc, scale) with the `Distribution` protocol surface."""

    loc: float
    scale: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.loc):
            raise ValueError(f"loc must be finite, got {self.loc!r}")
        if not (np.isfinite(self.scale) and self.scale > 0):
            raise ValueError(f"scale must be positive and finite, got {self.scale!r}")

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.norm.pdf(x, loc=self.loc, scale=self.scale),
                          dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.norm.logpdf(x, loc=self.loc, scale=self.scale),
                          dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.norm.cdf(x, loc=self.loc, scale=self.scale),
                          dtype=np.float64)

    def quantile(self, q: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.norm.ppf(q, loc=self.loc, scale=self.scale),
                          dtype=np.float64)

    def mean(self) -> float:
        return float(self.loc)

    def var(self) -> float:
        return float(self.scale ** 2)

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        return rng.normal(loc=self.loc, scale=self.scale, size=n)


@dataclass(frozen=True)
class GaussianLikelihood:
    """Likelihood for D ~ N(theta, sigma^2) — Normal-Normal model.

    The `D` and `sigma` attributes are public because conjugate-aware tilting
    schemes and test statistics need them; they form the model-specific
    surface that lets `PowerLawTilting` recognize a Normal-Normal context.
    """

    D: float
    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.D):
            raise ValueError(f"D must be finite, got {self.D!r}")
        if not (np.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError(f"sigma must be positive and finite, got {self.sigma!r}")

    def __call__(self, theta: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self.loglik(theta))

    def loglik(self, theta: ArrayLike) -> NDArray[np.float64]:
        z = (np.asarray(theta, dtype=np.float64) - self.D) / self.sigma
        return -0.5 * z * z - 0.5 * np.log(2.0 * np.pi * self.sigma ** 2)
