"""The Bernoulli model with Beta-conjugate prior.

  Prior:       theta ~ Beta(alpha_0, beta_0)
  Likelihood:  X_i | theta ~ Bernoulli(theta), iid for i = 1..n
  Posterior:   theta | X ~ Beta(alpha_0 + k, beta_0 + n - k)

with `k = sum(X_i)` the success count and `n` the trial count.

Included as the framework's second `Model` implementation. The first
purpose is architectural: prove the Model protocol generalises beyond
Normal-Normal. The second is to make explicit which downstream methods
are Normal-only — `WaldStatistic`, `WaldoStatistic`, and
`PowerLawTilting` all raise `NotImplementedError` when paired with a
`BernoulliModel`, by design. Implementing non-Normal versions of those
methods is research, not refactoring.
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
from .distributions import BernoulliLikelihood, BetaDistribution

_FORCE_X64 = _x64


@register_model(name="bernoulli", brief="docs/methods/bernoulli.md")
@dataclass(frozen=True)
class BernoulliModel:
    """Bernoulli location parameter with conjugate Beta prior.

    `name` and `param_dim` are class-level constants (ClassVars) so
    they are not constructor kwargs; that prevents an instance from
    silently lying about its identity past the fingerprint check.
    """

    name: ClassVar[str] = "bernoulli"
    param_dim: ClassVar[int] = 1

    def fingerprint(self) -> tuple:
        return ("bernoulli",)

    # ----- Model protocol -----

    def sample_data(self, theta: ArrayLike, rng: Generator, n: int) -> NDArray[np.float64]:
        theta_f = float(np.asarray(theta))
        if not (0.0 <= theta_f <= 1.0):
            raise ValueError(f"theta must lie in [0, 1], got {theta_f!r}")
        return rng.binomial(1, theta_f, size=n).astype(np.float64)

    def likelihood(self, data: NDArray[np.float64]) -> BernoulliLikelihood:
        arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        n_total = int(arr.size)
        n_success = int(arr.sum())
        return BernoulliLikelihood(n_success=n_success, n_total=n_total)

    def posterior(self, data: NDArray[np.float64], prior: Prior) -> BetaDistribution:
        if not isinstance(prior, BetaDistribution):
            raise NotImplementedError(
                "BernoulliModel.posterior currently requires a "
                "BetaDistribution prior; non-conjugate priors are a future extension."
            )
        arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        n_total = int(arr.size)
        n_success = int(arr.sum())
        return BetaDistribution(
            alpha=prior.alpha + n_success,
            beta=prior.beta + (n_total - n_success),
        )

    def mle(self, data: NDArray[np.float64]) -> jax.Array:
        arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        return jnp.asarray(arr.mean())

    def fisher_information(self, theta: ArrayLike) -> jax.Array:
        theta_arr = jnp.asarray(theta)
        # I(theta) = 1 / (theta * (1 - theta)) for Bernoulli.
        # Guard against the boundary singularity.
        eps = 1e-300
        denom = jnp.clip(theta_arr * (1.0 - theta_arr), eps, None)
        return 1.0 / denom

    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)

    # ----- Batched / vectorised hot-path overrides (Model protocol opt-ins) -----

    def sample_data_batch(
        self, theta: float, rng: Generator, n_mc: int, n_obs: int
    ) -> NDArray[np.float64]:
        """Draw `n_mc` independent Bernoulli(theta) datasets, n_obs each.

        Single `rng.binomial` call returning shape `(n_mc, n_obs)`.
        Replaces the default per-row Python loop in
        `frasian.models.base.default_sample_data_batch`.
        """
        theta_f = float(np.asarray(theta))
        if not (0.0 <= theta_f <= 1.0):
            raise ValueError(f"theta must lie in [0, 1], got {theta_f!r}")
        return rng.binomial(1, theta_f, size=(int(n_mc), int(n_obs))).astype(np.float64)

    def batch_loglik_grid(
        self, data_batch: NDArray[np.float64], theta_grid: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Vectorised Bernoulli log-likelihood across rows × grid.

        For row i, the likelihood is parameterised by `(k_i, n_obs)`
        with `k_i = sum(data_batch[i])`. The full formula (matching
        `BernoulliLikelihood.loglik`, omitting the n_obs / k_i
        combinatorial constant which cancels in tilted normalisation) is:

            loglik(theta; k_i, n_obs)
              = k_i * log(theta) + (n_obs - k_i) * log(1 - theta)

        Returns shape `(n_mc, n_grid)`. The clip-eps boundary handling
        mirrors the scalar `BernoulliLikelihood.loglik`.
        """
        arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
        grid = np.asarray(theta_grid, dtype=np.float64)
        n_obs = arr.shape[-1]
        k = arr.sum(axis=-1)  # (n_mc,)
        eps = 1e-300
        log_theta = np.log(np.clip(grid, eps, 1.0))  # (n_grid,)
        log_1m = np.log(np.clip(1.0 - grid, eps, 1.0))  # (n_grid,)
        return k[:, None] * log_theta[None, :] + (n_obs - k)[:, None] * log_1m[None, :]

    def sample_data_batch_at_thetas(
        self,
        theta_arr: NDArray[np.float64],
        rng: Generator,
        n_data: int,
    ) -> NDArray[np.float64]:
        """Vectorised per-θ Bernoulli sampling.

        Returns shape `(n_theta, n_data)`. Single `rng.binomial` call
        with broadcast probability — ~50-100× faster than the default
        per-θ loop.
        """
        arr = np.asarray(theta_arr, dtype=np.float64)
        if not np.all((arr >= 0.0) & (arr <= 1.0)):
            bad = float(arr[(arr < 0.0) | (arr > 1.0)][0])
            raise ValueError(
                f"theta_arr must lie in [0, 1] elementwise; got {bad!r}"
            )
        n_theta = int(arr.size)
        n_data = int(n_data)
        prob = np.broadcast_to(arr[:, None], (n_theta, n_data))
        return rng.binomial(1, prob).astype(np.float64)

    def posterior_quantile_batch(
        self,
        data_batch: NDArray[np.float64],
        prior: Prior,
        u_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Vectorised Beta posterior quantile across rows × u-grid.

        For row i: posterior = Beta(prior.alpha + k_i, prior.beta + n - k_i).
        scipy.special.betaincinv broadcasts over `(alpha, beta, u)` so a
        single call returns the full `(n_mc, n_u)` matrix.
        """
        if not isinstance(prior, BetaDistribution):
            raise NotImplementedError(
                "BernoulliModel.posterior_quantile_batch currently requires a "
                "BetaDistribution prior."
            )
        from scipy.special import betaincinv  # scipy: beta inverse-incomplete

        arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
        u_arr = np.asarray(u_grid, dtype=np.float64)
        n_obs = arr.shape[-1]
        k = arr.sum(axis=-1)
        alpha = prior.alpha + k  # (n_mc,)
        beta = prior.beta + (n_obs - k)
        return betaincinv(alpha[:, None], beta[:, None], u_arr[None, :])

    def posterior_moments_batch(
        self, data_batch: NDArray[np.float64], prior: Prior
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorised Beta-conjugate posterior moments over rows.

        For each row i:
            k_i = sum(data_batch[i]),  n = data_batch.shape[1]
            alpha_i = prior.alpha + k_i,  beta_i = prior.beta + n - k_i
            mu_i  = alpha_i / (alpha_i + beta_i)
            var_i = alpha_i * beta_i / ((alpha_i + beta_i)^2 * (alpha_i + beta_i + 1))

        Returns `(mu_arr, var_arr)` each shape `(n_mc,)`.
        """
        if not isinstance(prior, BetaDistribution):
            raise NotImplementedError(
                "BernoulliModel.posterior_moments_batch currently requires a "
                "BetaDistribution prior."
            )
        arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
        n_obs = arr.shape[-1]
        k = arr.sum(axis=-1)  # shape (n_mc,)
        alpha = prior.alpha + k
        beta = prior.beta + (n_obs - k)
        ab = alpha + beta
        mu = alpha / ab
        var = (alpha * beta) / (ab * ab * (ab + 1.0))
        return mu.astype(np.float64), var.astype(np.float64)
