"""The 1D conjugate Normal-Normal model.

  Prior:       theta ~ N(mu0, sigma0^2)
  Likelihood:  D | theta ~ N(theta, sigma^2)
  Posterior:   theta | D ~ N(mu_n, sigma_n^2)

Closed-form posterior:
  w        = sigma0^2 / (sigma^2 + sigma0^2)         (data weight)
  mu_n     = w * D + (1 - w) * mu0
  sigma_n  = sqrt(w) * sigma

The legacy module `legacy/src/frasian/core.py` had these as free functions; the
ports below keep the *math* identical (the regression tests in
`tests/regression/test_normal_normal.py` pin this byte-for-byte) but expose them
through the `Model` protocol so the rest of the framework never has to know we
are in the Normal-Normal special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from .._registry import register_model
from .base import Prior
from .distributions import GaussianLikelihood, NormalDistribution


def weight(sigma: float, sigma0: float) -> float:
    """w = sigma0^2 / (sigma^2 + sigma0^2). Verbatim port from legacy core.py."""
    return sigma0**2 / (sigma**2 + sigma0**2)


def posterior_params(
    D: ArrayLike, mu0: float, sigma: float, sigma0: float
) -> tuple[NDArray[np.float64], float, float]:
    """Return (mu_n, sigma_n, w). Verbatim port from legacy core.py."""
    w = weight(sigma, sigma0)
    mu_n = np.asarray(w * np.asarray(D, dtype=np.float64) + (1.0 - w) * mu0, dtype=np.float64)
    sigma_n = float(np.sqrt(w) * sigma)
    return mu_n, sigma_n, w


def scaled_conflict(D: ArrayLike, mu0: float, w: float, sigma: float) -> NDArray[np.float64]:
    """Delta = (1 - w) * (mu0 - D) / sigma. Port from legacy core.py."""
    return np.asarray((1.0 - w) * (mu0 - np.asarray(D, dtype=np.float64)) / sigma, dtype=np.float64)


def prior_residual(theta: ArrayLike, mu0: float, sigma0: float) -> NDArray[np.float64]:
    """delta(theta) = (theta - mu0) / sigma0. Port from legacy core.py."""
    return np.asarray((np.asarray(theta, dtype=np.float64) - mu0) / sigma0, dtype=np.float64)


def noncentrality(theta: ArrayLike, mu0: float, w: float, sigma: float) -> NDArray[np.float64]:
    """lambda(theta) = (1-w)^2 * (mu0 - theta)^2 / (w^2 * sigma^2).

    From Theorem 2 in the legacy derivations: lambda(theta) = delta^2 / w
    where delta = (1-w)(mu0 - theta) / (sqrt(w) * sigma).
    """
    delta_num = (1.0 - w) * (mu0 - np.asarray(theta, dtype=np.float64))
    delta_den = np.sqrt(w) * sigma
    delta = delta_num / delta_den
    return np.asarray(delta * delta / w, dtype=np.float64)


@register_model(name="normal_normal", brief="docs/methods/normal_normal.md")
@dataclass(frozen=True)
class NormalNormalModel:
    """The 1D conjugate Normal-Normal model.

    Public attributes (`name`, `param_dim`, `sigma`) form the Normal-specific
    surface that conjugate-aware tilting schemes and test statistics rely on.
    `name` and `param_dim` are class-level constants (not constructor kwargs)
    so they cannot be overridden — that would let an instance silently lie
    about its identity past the fingerprint check.
    """

    sigma: float

    name: ClassVar[str] = "normal_normal"
    param_dim: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not (np.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError(f"sigma must be positive and finite, got {self.sigma!r}")

    def fingerprint(self) -> tuple:
        return ("normal_normal", float(self.sigma))

    # ----- Model protocol -----

    def sample_data(self, theta: ArrayLike, rng: Generator, n: int) -> NDArray[np.float64]:
        return rng.normal(loc=float(np.asarray(theta)), scale=self.sigma, size=n)

    def likelihood(self, data: NDArray[np.float64]) -> GaussianLikelihood:
        # The model treats `data` as a single sufficient statistic D = mean(data)
        # with variance sigma^2 / n. The framework currently keeps n=1 (D is
        # the observation itself); generalising to n>1 via the sufficient-stat
        # path is a future extension.
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        return GaussianLikelihood(D=D, sigma=self.sigma)

    def posterior(self, data: NDArray[np.float64], prior: Prior) -> NormalDistribution:
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "NormalNormalModel.posterior currently requires a "
                "NormalDistribution prior; non-conjugate priors are a future extension."
            )
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, sigma_n, _ = posterior_params(D, prior.loc, self.sigma, prior.scale)
        return NormalDistribution(loc=float(mu_n), scale=sigma_n)

    def mle(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            np.atleast_1d(np.asarray(data, dtype=np.float64)).mean(), dtype=np.float64
        )

    def fisher_information(self, theta: ArrayLike) -> NDArray[np.float64]:
        # I(theta) = 1 / sigma^2 for a Normal location family.
        out = np.full_like(np.asarray(theta, dtype=np.float64), 1.0 / self.sigma**2)
        return out

    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    # ----- Conjugate-Normal-specific surface (public, used by PowerLawTilting etc.) -----

    def weight(self, prior: NormalDistribution) -> float:
        """Frasian data weight w = sigma0^2 / (sigma^2 + sigma0^2)."""
        return weight(self.sigma, prior.scale)
