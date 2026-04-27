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

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from .._registry import register_model
from .base import Model, Prior
from .distributions import BernoulliLikelihood, BetaDistribution


@register_model(name="bernoulli", brief="docs/methods/bernoulli.md")
@dataclass(frozen=True)
class BernoulliModel:
    """Bernoulli location parameter with conjugate Beta prior."""

    name: str = "bernoulli"
    param_dim: int = 1

    # ----- Model protocol -----

    def sample_data(self, theta: ArrayLike, rng: Generator, n: int
                    ) -> NDArray[np.float64]:
        theta_f = float(np.asarray(theta))
        if not (0.0 <= theta_f <= 1.0):
            raise ValueError(f"theta must lie in [0, 1], got {theta_f!r}")
        return rng.binomial(1, theta_f, size=n).astype(np.float64)

    def likelihood(self, data: NDArray[np.float64]) -> BernoulliLikelihood:
        arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        n_total = int(arr.size)
        n_success = int(arr.sum())
        return BernoulliLikelihood(n_success=n_success, n_total=n_total)

    def posterior(self, data: NDArray[np.float64], prior: Prior
                  ) -> BetaDistribution:
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

    def mle(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
        return np.asarray(arr.mean(), dtype=np.float64)

    def fisher_information(self, theta: ArrayLike) -> NDArray[np.float64]:
        theta_arr = np.asarray(theta, dtype=np.float64)
        # I(theta) = 1 / (theta * (1 - theta)) for Bernoulli.
        # Guard against the boundary singularity.
        eps = 1e-300
        denom = np.clip(theta_arr * (1.0 - theta_arr), eps, None)
        return np.asarray(1.0 / denom, dtype=np.float64)

    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)
