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

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._registry import register_model
from .base import Prior
from .distributions import GaussianLikelihood, NormalDistribution

_FORCE_X64 = _x64


def weight(sigma: float, sigma0: float) -> float:
    """w = sigma0^2 / (sigma^2 + sigma0^2). Verbatim port from legacy core.py."""
    return sigma0**2 / (sigma**2 + sigma0**2)


def posterior_params(
    D: ArrayLike, mu0: float, sigma: float, sigma0: float
) -> tuple[jax.Array, float, float]:
    """Return (mu_n, sigma_n, w). Verbatim port from legacy core.py."""
    w = weight(sigma, sigma0)
    mu_n = w * jnp.asarray(D) + (1.0 - w) * mu0
    sigma_n = float(np.sqrt(w) * sigma)
    return mu_n, sigma_n, w


def scaled_conflict(D: ArrayLike, mu0: float, w: float, sigma: float) -> jax.Array:
    """Delta = (1 - w) * (mu0 - D) / sigma. Port from legacy core.py."""
    return (1.0 - w) * (mu0 - jnp.asarray(D)) / sigma


def prior_residual(theta: ArrayLike, mu0: float, sigma0: float) -> jax.Array:
    """delta(theta) = (theta - mu0) / sigma0. Port from legacy core.py."""
    return (jnp.asarray(theta) - mu0) / sigma0


def noncentrality(theta: ArrayLike, mu0: float, w: float, sigma: float) -> jax.Array:
    """lambda(theta) = (1-w)^2 * (mu0 - theta)^2 / (w^2 * sigma^2).

    From Theorem 2 in the legacy derivations: lambda(theta) = delta^2 / w
    where delta = (1-w)(mu0 - theta) / (sqrt(w) * sigma).
    """
    delta_num = (1.0 - w) * (mu0 - jnp.asarray(theta))
    delta_den = jnp.sqrt(w) * sigma
    delta = delta_num / delta_den
    return delta * delta / w


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

    def mle(self, data: NDArray[np.float64]) -> jax.Array:
        return jnp.asarray(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())

    def fisher_information(self, theta: ArrayLike) -> jax.Array:
        # I(theta) = 1 / sigma^2 for a Normal location family.
        return jnp.full_like(jnp.asarray(theta, dtype=jnp.float64), 1.0 / self.sigma**2)

    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)

    # ----- Conjugate-Normal-specific surface (public, used by PowerLawTilting etc.) -----

    def weight(self, prior: NormalDistribution) -> float:
        """Frasian data weight w = sigma0^2 / (sigma^2 + sigma0^2)."""
        return weight(self.sigma, prior.scale)
