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

    # ----- Batched / vectorised hot-path overrides (Model protocol opt-ins) -----

    def sample_data_batch(
        self, theta: float, rng: Generator, n_mc: int, n_obs: int
    ) -> NDArray[np.float64]:
        """Draw `n_mc` independent N(theta, sigma^2) datasets, each n_obs obs.

        Single `rng.normal` call returning shape `(n_mc, n_obs)`. Replaces
        the default `n_mc`-iteration Python loop in
        `frasian.models.base.default_sample_data_batch` for the
        Normal-Normal sandbox; the speedup vs the loop is ~50–200x at
        n_mc=2000 because `rng.normal` is a single C-level draw.
        """
        return rng.normal(
            loc=float(np.asarray(theta)),
            scale=self.sigma,
            size=(int(n_mc), int(n_obs)),
        )

    def batch_loglik_grid(
        self, data_batch: NDArray[np.float64], theta_grid: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Vectorised Gaussian log-likelihood across rows × grid.

        For row i, the likelihood treats `D_i = mean(data_batch[i])` as
        a single scalar observation under N(theta, sigma^2). The full
        formula (matching `GaussianLikelihood.loglik`) is:

            loglik(theta; D_i) = -0.5 * ((theta - D_i) / sigma)^2
                                  - 0.5 * log(2*pi*sigma^2)

        Returns shape `(n_mc, n_grid)`. Replaces the per-row
        `GaussianLikelihood(D=...).loglik(...)` Python loop in the
        default fallback; ~50-200x speedup on Normal-Normal MC paths.
        """
        arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
        grid = np.asarray(theta_grid, dtype=np.float64)
        sigma = float(self.sigma)
        D_means = arr.mean(axis=-1)  # (n_mc,) — n=1 sufficient stat
        diff = grid[None, :] - D_means[:, None]  # (n_mc, n_grid)
        z = diff / sigma
        const = -0.5 * np.log(2.0 * np.pi * sigma**2)
        return -0.5 * z * z + const

    def posterior_moments_batch(
        self, data_batch: NDArray[np.float64], prior: Prior
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorised posterior moments across rows of `data_batch`.

        Closed-form Normal-Normal posterior:
            mu_n_i = w * mean(data_batch[i]) + (1 - w) * mu0
            sigma_n^2 = w * sigma^2  (constant, theta-/data-independent)

        Returns `(mu_arr, var_arr)` each shape `(n_mc,)`. The variance
        is constant across i so `var_arr` is materialised as a constant
        array — keeps the consumer-side broadcast trivial without
        leaking the conjugate structure.
        """
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "NormalNormalModel.posterior_moments_batch currently requires "
                "a NormalDistribution prior."
            )
        arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
        D_means = arr.mean(axis=-1)  # shape (n_mc,)
        sigma2 = float(self.sigma) ** 2
        sigma0_2 = float(prior.scale) ** 2
        w = sigma0_2 / (sigma2 + sigma0_2)
        mu_n = w * D_means + (1.0 - w) * float(prior.loc)
        sigma_n_sq = w * sigma2
        var_n = np.full_like(mu_n, sigma_n_sq)
        return mu_n, var_n

    def posterior_quantile_batch(
        self,
        data_batch: NDArray[np.float64],
        prior: Prior,
        u_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Vectorised Gaussian posterior quantile across rows × u-grid.

        For row i the posterior is N(mu_n_i, sigma_n^2) with
        sigma_n^2 = w * sigma^2 (constant across i — depends only on
        prior + model, not data). The quantile is closed form:
            F_post,i^{-1}(u) = mu_n_i + sigma_n * Phi^{-1}(u)
        Returns shape `(n_mc, n_u)`.
        """
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "NormalNormalModel.posterior_quantile_batch currently requires "
                "a NormalDistribution prior."
            )
        from scipy.special import ndtri  # scipy: standard-Normal quantile

        arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
        u_arr = np.asarray(u_grid, dtype=np.float64)
        D_means = arr.mean(axis=-1)
        sigma2 = float(self.sigma) ** 2
        sigma0_2 = float(prior.scale) ** 2
        w = sigma0_2 / (sigma2 + sigma0_2)
        mu_n = w * D_means + (1.0 - w) * float(prior.loc)  # (n_mc,)
        sigma_n = float(np.sqrt(w) * self.sigma)
        z = ndtri(u_arr)  # (n_u,)
        return mu_n[:, None] + sigma_n * z[None, :]

    # ----- Conjugate-Normal-specific surface (public, used by PowerLawTilting etc.) -----

    def weight(self, prior: NormalDistribution) -> float:
        """Frasian data weight w = sigma0^2 / (sigma^2 + sigma0^2)."""
        return weight(self.sigma, prior.scale)
