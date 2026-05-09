"""Model / Prior / Posterior / Likelihood protocols.

The conjugate Normal-Normal special case lives in `frasian.models.normal_normal`
and is reachable only through these protocols. Nothing in `frasian.tilting`,
`frasian.statistics`, `frasian.cd`, or `frasian.experiments` may import that
module directly — that discipline is what keeps the framework extensible.

Distributions/likelihoods return `jax.Array` (the framework runs in
`jax_enable_x64` mode so the default dtype is `float64`). Random
sampling still consumes a numpy `Generator` because it sits at the
I/O boundary; that contract is intentional and not a JAX-port oversight.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
import numpy as np


@runtime_checkable
class Distribution(Protocol):
    """Common surface for any 1D distribution (prior, posterior, tilted)."""

    def pdf(self, x: ArrayLike) -> jax.Array: ...
    def logpdf(self, x: ArrayLike) -> jax.Array: ...
    def cdf(self, x: ArrayLike) -> jax.Array: ...
    def quantile(self, q: ArrayLike) -> jax.Array: ...
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

    def __call__(self, theta: ArrayLike) -> jax.Array: ...
    def loglik(self, theta: ArrayLike) -> jax.Array: ...


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

    def mle(self, data: NDArray[np.float64]) -> jax.Array: ...

    def fisher_information(self, theta: ArrayLike) -> jax.Array: ...

    def support(self) -> tuple[float, float]: ...

    # ----- Batched / vectorised hot-path helpers (optional) -----
    #
    # These are NOT required by the protocol — `runtime_checkable` only
    # checks attribute presence at the cost of false negatives, so we
    # surface them here as documentation rather than `... ` stubs.
    # Helper functions `default_sample_data_batch` and
    # `default_posterior_moments_batch` below provide loop-based fallbacks
    # that work for any model conforming to the *scalar* surface; the
    # `Generic-MC` consumers (WaldoStatistic._generic_mc_reference,
    # power_law._generic_tilted_pvalue, future LRT/SR statistics) use
    # `getattr(model, "posterior_moments_batch", default_posterior_moments_batch)`
    # so models that override these get the fast path automatically.
    #
    # Recommended signatures for overrides:
    #
    #   def sample_data_batch(self, theta: float, rng: Generator,
    #                         n_mc: int, n_obs: int) -> NDArray[np.float64]:
    #       """Returns shape (n_mc, n_obs); rows i.i.d. under H_0:theta."""
    #
    #   def posterior_moments_batch(self, data_batch: NDArray[np.float64],
    #                               prior: Prior) -> tuple[NDArray, NDArray]:
    #       """`data_batch` shape (n_mc, n_obs); returns
    #       `(mu_arr, var_arr)` each shape (n_mc,)."""
    #
    # `NormalNormalModel` and `BernoulliModel` provide closed-form
    # vectorised overrides; non-conjugate / future models can defer.


def default_sample_data_batch(
    model: Model, theta: float, rng: Generator, n_mc: int, n_obs: int
) -> NDArray[np.float64]:
    """Loop-based fallback for `Model.sample_data_batch`.

    Calls `model.sample_data(theta, rng, n_obs)` n_mc times and stacks.
    O(n_mc) Python-loop overhead; override on the model for speed when
    `numpy.random` natively supports batched sampling at the relevant
    distribution.
    """
    out = np.empty((int(n_mc), int(n_obs)), dtype=np.float64)
    for i in range(int(n_mc)):
        out[i, :] = np.asarray(model.sample_data(theta, rng, int(n_obs)), dtype=np.float64)
    return out


def default_posterior_moments_batch(
    model: Model, data_batch: NDArray[np.float64], prior: Prior
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Loop-based fallback for `Model.posterior_moments_batch`.

    For each row of `data_batch`, calls `model.posterior(row, prior)`
    and reads `.mean() / .var()`. O(n_mc) Python-loop overhead with
    posterior-object construction per row; override for closed-form
    conjugate models for ~100x speedup.
    """
    arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
    n_mc = arr.shape[0]
    mu = np.empty(n_mc, dtype=np.float64)
    var = np.empty(n_mc, dtype=np.float64)
    for i in range(n_mc):
        post = model.posterior(arr[i], prior)
        mu[i] = float(np.asarray(post.mean()))
        var[i] = float(np.asarray(post.var()))
    return mu, var


def sample_data_batch(
    model: Model, theta: float, rng: Generator, n_mc: int, n_obs: int
) -> NDArray[np.float64]:
    """Dispatch helper: use the model's `sample_data_batch` if it
    exposes one, else the loop-based default. Centralised so the MC
    consumers don't all have to repeat the duck-typing dance.
    """
    fn = getattr(model, "sample_data_batch", None)
    if callable(fn):
        return np.asarray(fn(theta, rng, n_mc, n_obs), dtype=np.float64)
    return default_sample_data_batch(model, theta, rng, n_mc, n_obs)


def posterior_moments_batch(
    model: Model, data_batch: NDArray[np.float64], prior: Prior
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Dispatch helper for `posterior_moments_batch` (see above)."""
    fn = getattr(model, "posterior_moments_batch", None)
    if callable(fn):
        mu, var = fn(data_batch, prior)
        return np.asarray(mu, dtype=np.float64), np.asarray(var, dtype=np.float64)
    return default_posterior_moments_batch(model, data_batch, prior)


def default_batch_loglik_grid(
    model: Model,
    data_batch: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Loop-based fallback for `Model.batch_loglik_grid`.

    For each row of `data_batch`, builds a likelihood object via
    `model.likelihood(row)` and evaluates `loglik(theta_grid)`. Returns
    shape `(n_mc, n_grid)`. O(n_mc) overhead with object construction
    per row.
    """
    arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
    n_mc = arr.shape[0]
    n_grid = int(np.asarray(theta_grid).size)
    out = np.empty((n_mc, n_grid), dtype=np.float64)
    for i in range(n_mc):
        lik = model.likelihood(arr[i])
        out[i, :] = np.asarray(lik.loglik(theta_grid), dtype=np.float64)
    return out


def batch_loglik_grid(
    model: Model,
    data_batch: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Dispatch helper for `batch_loglik_grid`.

    Returns `loglik_batch` of shape `(n_mc, n_grid)` where
    `loglik_batch[i, j] = loglik(theta_grid[j])` evaluated under the
    likelihood implied by `data_batch[i]`. The hot inner kernel of
    the generic-MC tilted-pvalue path (`power_law._generic_tilted_pvalue`,
    `ot._generic_tilted_pvalue_ot`); models implementing a vectorised
    override get ~50–200x speedup on Normal-Normal-sized problems.

    Recommended override signature:

        def batch_loglik_grid(self, data_batch, theta_grid) -> NDArray:
            ...
    """
    fn = getattr(model, "batch_loglik_grid", None)
    if callable(fn):
        return np.asarray(fn(data_batch, theta_grid), dtype=np.float64)
    return default_batch_loglik_grid(model, data_batch, theta_grid)


def default_posterior_quantile_batch(
    model: Model,
    data_batch: NDArray[np.float64],
    prior: Prior,
    u_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Loop-based fallback for `Model.posterior_quantile_batch`.

    For each row, build the posterior via `model.posterior(row, prior)`
    and call `posterior.quantile(u_grid)`. Returns shape `(n_mc, n_u)`.
    O(n_mc) Python overhead with one Distribution object per row.
    """
    arr = np.atleast_2d(np.asarray(data_batch, dtype=np.float64))
    u_arr = np.asarray(u_grid, dtype=np.float64)
    n_mc = arr.shape[0]
    n_u = int(u_arr.size)
    out = np.empty((n_mc, n_u), dtype=np.float64)
    for i in range(n_mc):
        post = model.posterior(arr[i], prior)
        out[i, :] = np.asarray(post.quantile(u_arr), dtype=np.float64)
    return out


def default_sample_data_batch_at_thetas(
    model: Model,
    theta_arr: NDArray[np.float64],
    rng: Generator,
    n_data: int,
) -> NDArray[np.float64]:
    """Loop-based fallback for `Model.sample_data_batch_at_thetas`.

    For each `theta_arr[i]`, draws `n_data` observations from
    `model.sample_data(theta_arr[i], rng, n_data)`. Returns shape
    `(n_theta, n_data)`. The Python `for` loop here is the slow
    path — vectorised models (NN: single rng.normal call shifted per
    theta; Bernoulli: single rng.binomial call with broadcast
    probability) override this to remove the per-theta dispatch.
    """
    n_theta = int(np.asarray(theta_arr).size)
    n_data = int(n_data)
    out = np.empty((n_theta, n_data), dtype=np.float64)
    arr = np.asarray(theta_arr, dtype=np.float64)
    for i in range(n_theta):
        out[i, :] = np.asarray(
            model.sample_data(float(arr[i]), rng, n_data), dtype=np.float64
        )
    return out


def sample_data_batch_at_thetas(
    model: Model,
    theta_arr: NDArray[np.float64],
    rng: Generator,
    n_data: int,
) -> NDArray[np.float64]:
    """Dispatch helper for `sample_data_batch_at_thetas`.

    Returns shape `(n_theta, n_data)` of MC draws under H_0:theta_arr[i]
    per row. Used by the learned-η training loop (one draw per batch
    element) and the dynamic-η + force_generic CI path. Models with a
    natively vectorised override (NN, Bernoulli) skip the per-theta
    Python loop in `default_sample_data_batch_at_thetas`.
    """
    fn = getattr(model, "sample_data_batch_at_thetas", None)
    if callable(fn):
        return np.asarray(fn(theta_arr, rng, n_data), dtype=np.float64)
    return default_sample_data_batch_at_thetas(model, theta_arr, rng, n_data)


def posterior_quantile_batch(
    model: Model,
    data_batch: NDArray[np.float64],
    prior: Prior,
    u_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Dispatch helper for `posterior_quantile_batch`.

    Returns `quantile_batch[i, j] = F_post,i^{-1}(u_grid[j])` for the
    posterior implied by `data_batch[i]`. The hot inner kernel of OT-
    tilted MC (the W2 quantile mixture needs per-row posterior inverse-
    CDF at Gauss-Legendre nodes). Models implementing a vectorised
    override get the fast path; conjugate models get closed-form
    speedup of ~50-200x at n_mc=200, n_u=64.

    Recommended override signature:

        def posterior_quantile_batch(self, data_batch, prior, u_grid)
                -> NDArray[shape=(n_mc, n_u)]:
            ...
    """
    fn = getattr(model, "posterior_quantile_batch", None)
    if callable(fn):
        return np.asarray(fn(data_batch, prior, u_grid), dtype=np.float64)
    return default_posterior_quantile_batch(model, data_batch, prior, u_grid)
