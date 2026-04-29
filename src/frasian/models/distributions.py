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
class BetaDistribution:
    """Beta(alpha, beta) distribution conforming to the Distribution protocol.

    The Beta is the conjugate prior for Bernoulli, so it pops up as both
    prior and posterior under the BernoulliModel. Both shape parameters
    must be strictly positive.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.alpha) and self.alpha > 0):
            raise ValueError(f"alpha must be positive, got {self.alpha!r}")
        if not (np.isfinite(self.beta) and self.beta > 0):
            raise ValueError(f"beta must be positive, got {self.beta!r}")

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.beta.pdf(x, self.alpha, self.beta),
                          dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.beta.logpdf(x, self.alpha, self.beta),
                          dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.beta.cdf(x, self.alpha, self.beta),
                          dtype=np.float64)

    def quantile(self, q: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.beta.ppf(q, self.alpha, self.beta),
                          dtype=np.float64)

    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def var(self) -> float:
        ab = self.alpha + self.beta
        return float(self.alpha * self.beta / (ab ** 2 * (ab + 1.0)))

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        return rng.beta(self.alpha, self.beta, size=n)


@dataclass(frozen=True)
class BernoulliLikelihood:
    """Likelihood for `data` ~ Bernoulli(theta) — a vector of 0/1 outcomes.

    Stores the sufficient statistic (n_success, n_total) rather than the
    raw data array; this is what BernoulliModel.posterior consumes.
    """

    n_success: int
    n_total: int

    def __post_init__(self) -> None:
        if self.n_total <= 0:
            raise ValueError(f"n_total must be positive, got {self.n_total!r}")
        if not (0 <= self.n_success <= self.n_total):
            raise ValueError(
                f"n_success ({self.n_success}) outside [0, {self.n_total}]"
            )

    def __call__(self, theta: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self.loglik(theta))

    def loglik(self, theta: ArrayLike) -> NDArray[np.float64]:
        theta_arr = np.asarray(theta, dtype=np.float64)
        # Guard against log(0) at the support boundary by clipping.
        eps = 1e-300
        return np.asarray(
            self.n_success * np.log(np.clip(theta_arr, eps, 1.0))
            + (self.n_total - self.n_success)
            * np.log(np.clip(1.0 - theta_arr, eps, 1.0)),
            dtype=np.float64,
        )


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
