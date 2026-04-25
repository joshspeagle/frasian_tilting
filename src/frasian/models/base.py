"""Model / Prior / Posterior / Likelihood protocols.

The conjugate Normal-Normal special case lives in `frasian.models.normal_normal`
and is reachable only through these protocols. Nothing in `frasian.tilting`,
`frasian.statistics`, `frasian.cd`, or `frasian.experiments` may import that
module directly — that discipline is what keeps the framework extensible.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class Distribution(Protocol):
    """Common surface for any 1D distribution (prior, posterior, tilted)."""

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def cdf(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def quantile(self, q: ArrayLike) -> NDArray[np.float64]: ...
    def mean(self) -> float: ...
    def var(self) -> float: ...
    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]: ...


@runtime_checkable
class Prior(Distribution, Protocol):
    """A prior distribution over the parameter."""


@runtime_checkable
class Posterior(Distribution, Protocol):
    """A posterior or tilted posterior distribution over the parameter."""


@runtime_checkable
class Likelihood(Protocol):
    """A likelihood evaluable at parameter values given fixed data."""

    def __call__(self, theta: ArrayLike) -> NDArray[np.float64]: ...
    def loglik(self, theta: ArrayLike) -> NDArray[np.float64]: ...


@runtime_checkable
class Model(Protocol):
    """A generative probabilistic model.

    Implementations specify how data is sampled given a parameter, how the
    likelihood is evaluated, and how a posterior is constructed under a given
    prior. The 1D conjugate-Normal case sets `param_dim = 1`.

    Invariants any implementation must satisfy (verified in
    tests/properties/test_model_invariants.py):
        - `posterior(data, prior).mean()` lies between `prior.mean()` and the MLE.
        - `posterior(data, prior).var()` <= `prior.var()` for informative data.
        - `quantile(cdf(x)) == x` for all returned distributions, atol depends on tail.
        - `mle(sample_data(theta, ...))` is consistent under increasing n.
    """

    name: str
    param_dim: int

    def sample_data(self, theta: ArrayLike, rng: Generator, n: int
                    ) -> NDArray[np.float64]: ...

    def likelihood(self, data: NDArray[np.float64]) -> Likelihood: ...

    def posterior(self, data: NDArray[np.float64], prior: Prior) -> Posterior: ...

    def mle(self, data: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def fisher_information(self, theta: ArrayLike) -> NDArray[np.float64]: ...

    def support(self) -> tuple[float, float]: ...
