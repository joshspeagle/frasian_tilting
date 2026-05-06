"""Concrete distributions used by the framework's models.

`NormalDistribution` is the working horse for prior, posterior, and tilted
posterior under `NormalNormalModel`. `BetaDistribution` plays the same
role for `BernoulliModel`. Both are dataclasses and conform to the
protocols in `models.base`. All density/log-density/cdf paths return
`jax.Array` so they remain JAX-traceable for autodiff (Fisher info,
learned-η training, etc.); random sampling still consumes a numpy
`Generator` to keep RNG state at the I/O boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from scipy import stats as _sp_stats

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


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

    def pdf(self, x: ArrayLike) -> jax.Array:
        return jsp_stats.norm.pdf(jnp.asarray(x), loc=self.loc, scale=self.scale)

    def logpdf(self, x: ArrayLike) -> jax.Array:
        return jsp_stats.norm.logpdf(jnp.asarray(x), loc=self.loc, scale=self.scale)

    def cdf(self, x: ArrayLike) -> jax.Array:
        return jsp_stats.norm.cdf(jnp.asarray(x), loc=self.loc, scale=self.scale)

    def quantile(self, q: ArrayLike) -> jax.Array:
        return jsp_stats.norm.ppf(jnp.asarray(q), loc=self.loc, scale=self.scale)

    def mean(self) -> float:
        return float(self.loc)

    def var(self) -> float:
        return float(self.scale**2)

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        return rng.normal(loc=self.loc, scale=self.scale, size=n)

    def fingerprint(self) -> tuple:
        return ("normal", float(self.loc), float(self.scale))


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

    def pdf(self, x: ArrayLike) -> jax.Array:
        return jsp_stats.beta.pdf(jnp.asarray(x), self.alpha, self.beta)

    def logpdf(self, x: ArrayLike) -> jax.Array:
        return jsp_stats.beta.logpdf(jnp.asarray(x), self.alpha, self.beta)

    def cdf(self, x: ArrayLike) -> jax.Array:
        return jsp_stats.beta.cdf(jnp.asarray(x), self.alpha, self.beta)

    def quantile(self, q: ArrayLike) -> jax.Array:
        # scipy: jax.scipy.stats.beta has no `ppf`; fall back to numpy/scipy.
        # This is non-differentiable, but quantile only appears at the
        # CI-inversion / sampling boundary, never on a learned-η loss path.
        result = _sp_stats.beta.ppf(np.asarray(q, dtype=np.float64), self.alpha, self.beta)
        return jnp.asarray(result)

    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def var(self) -> float:
        ab = self.alpha + self.beta
        return float(self.alpha * self.beta / (ab**2 * (ab + 1.0)))

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        return rng.beta(self.alpha, self.beta, size=n)

    def fingerprint(self) -> tuple:
        return ("beta", float(self.alpha), float(self.beta))


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
            raise ValueError(f"n_success ({self.n_success}) outside [0, {self.n_total}]")

    def __call__(self, theta: ArrayLike) -> jax.Array:
        return jnp.exp(self.loglik(theta))

    def loglik(self, theta: ArrayLike) -> jax.Array:
        theta_arr = jnp.asarray(theta)
        # Guard against log(0) at the support boundary by clipping.
        eps = 1e-300
        return self.n_success * jnp.log(jnp.clip(theta_arr, eps, 1.0)) + (
            self.n_total - self.n_success
        ) * jnp.log(jnp.clip(1.0 - theta_arr, eps, 1.0))


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

    def __call__(self, theta: ArrayLike) -> jax.Array:
        return jnp.exp(self.loglik(theta))

    def loglik(self, theta: ArrayLike) -> jax.Array:
        z = (jnp.asarray(theta) - self.D) / self.sigma
        return -0.5 * z * z - 0.5 * jnp.log(2.0 * jnp.pi * self.sigma**2)
