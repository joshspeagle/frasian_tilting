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
    """A prior distribution over the parameter.

    Priors must expose a `fingerprint()` -> hashable tuple identifying
    the prior up to its parameters. The learned-η selector compares
    fingerprints between training-time and inference-time to refuse
    cross-experiment checkpoint use.
    """

    def fingerprint(self) -> tuple: ...


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

    Invariants any implementation must satisfy (verified per-model in
    `tests/properties/test_<model_name>_invariants.py` — currently
    `test_normal_distribution.py` for normal_normal and
    `test_bernoulli_invariants.py` for bernoulli):
        - `posterior(data, prior).mean()` lies between `prior.mean()` and
          `mle(data)` (the precision-weighted average of the two).
        - `posterior(data, prior).var() -> 0` as `n_obs -> infinity`
          (asymptotic concentration). Note: posterior variance is *not*
          monotonically below prior variance for every finite n — when
          the prior strongly opposes the data the posterior may
          transiently widen (verified for Bernoulli; the original draft
          of this docstring overstated the property).
        - `quantile(cdf(x)) == x` for all returned distributions
          (round-trip; atol depends on tail).
        - `mle(sample_data(theta, ...))` is consistent under increasing n.
    """

    @property
    def name(self) -> str: ...

    @property
    def param_dim(self) -> int: ...

    def fingerprint(self) -> tuple: ...

    def sample_data(self, theta: ArrayLike, rng: Generator, n: int) -> NDArray[np.float64]: ...

    def likelihood(self, data: NDArray[np.float64]) -> Likelihood: ...

    def posterior(self, data: NDArray[np.float64], prior: Prior) -> Posterior: ...

    def mle(self, data: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def fisher_information(self, theta: ArrayLike) -> NDArray[np.float64]: ...

    def support(self) -> tuple[float, float]: ...
