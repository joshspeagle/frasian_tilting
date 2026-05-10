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
from typing import Any, ClassVar

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

    # ----- Phase G hyperparam protocol -----

    hyperparam_dim: ClassVar[int] = 2

    @classmethod
    def hyperparam_names(cls) -> tuple[str, ...]:
        return ("loc", "scale")

    def hyperparams(self) -> NDArray[np.float64]:
        return np.array([self.loc, self.scale], dtype=np.float64)

    @classmethod
    def from_hyperparams(cls, hp: NDArray[np.float64]) -> "NormalDistribution":
        arr = np.asarray(hp, dtype=np.float64)
        if arr.shape != (2,):
            raise ValueError(
                f"NormalDistribution.from_hyperparams expects length 2 "
                f"vector [loc, scale]; got shape {arr.shape!r}."
            )
        return cls(loc=float(arr[0]), scale=float(arr[1]))


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

    # ----- Phase G hyperparam protocol -----

    hyperparam_dim: ClassVar[int] = 2

    @classmethod
    def hyperparam_names(cls) -> tuple[str, ...]:
        return ("alpha", "beta")

    def hyperparams(self) -> NDArray[np.float64]:
        return np.array([self.alpha, self.beta], dtype=np.float64)

    @classmethod
    def from_hyperparams(cls, hp: NDArray[np.float64]) -> "BetaDistribution":
        arr = np.asarray(hp, dtype=np.float64)
        if arr.shape != (2,):
            raise ValueError(
                f"BetaDistribution.from_hyperparams expects length 2 "
                f"vector [alpha, beta]; got shape {arr.shape!r}."
            )
        return cls(alpha=float(arr[0]), beta=float(arr[1]))


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


@dataclass(frozen=True)
class GaussianMixtureDistribution:
    """Two-component Gaussian mixture conforming to the `Distribution` protocol.

    Used by `MixtureTilting.tilt(...)` to wrap the linear-density
    interpolation `(1-eta) * N(mu_n, sigma_n^2) + eta * N(D, sigma^2)`.
    May be bimodal when components are far apart relative to their widths
    (Behboodian 1970: bimodal iff `|mu_0 - mu_1| > 2 * min(sigma_0, sigma_1)`).

    Closed-form `mean`, `var`, `pdf`, `logpdf`, `cdf`. Numerical
    `quantile` via brentq on the (monotone) CDF. `sample` via
    categorical pick + Gaussian draw.

    The class is hard-wired to n=2 (the only shape used by the framework:
    posterior + likelihood). Generic n-component mixtures are out of scope.
    """

    weights: tuple[float, float]
    means: tuple[float, float]
    scales: tuple[float, float]
    n_components: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if len(self.weights) != 2 or len(self.means) != 2 or len(self.scales) != 2:
            raise ValueError(
                f"GaussianMixtureDistribution: expected 2 components; "
                f"got weights={self.weights!r}, means={self.means!r}, "
                f"scales={self.scales!r}."
            )
        w0, w1 = float(self.weights[0]), float(self.weights[1])
        if not (np.isfinite(w0) and np.isfinite(w1)):
            raise ValueError(f"weights must be finite, got {self.weights!r}")
        if w0 < 0.0 or w1 < 0.0:
            raise ValueError(f"weights must be non-negative, got {self.weights!r}")
        if not np.isclose(w0 + w1, 1.0, atol=1e-10):
            raise ValueError(
                f"weights must sum to 1, got {self.weights!r} (sum={w0+w1})"
            )
        for s in self.scales:
            if not (np.isfinite(s) and float(s) > 0.0):
                raise ValueError(
                    f"scale must be positive and finite, got {self.scales!r}"
                )
        for m in self.means:
            if not np.isfinite(m):
                raise ValueError(f"mean must be finite, got {self.means!r}")

    def pdf(self, x: ArrayLike) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        c0 = jsp_stats.norm.pdf(x_arr, loc=self.means[0], scale=self.scales[0])
        c1 = jsp_stats.norm.pdf(x_arr, loc=self.means[1], scale=self.scales[1])
        return self.weights[0] * c0 + self.weights[1] * c1

    def logpdf(self, x: ArrayLike) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        log_c0 = jsp_stats.norm.logpdf(x_arr, loc=self.means[0], scale=self.scales[0])
        log_c1 = jsp_stats.norm.logpdf(x_arr, loc=self.means[1], scale=self.scales[1])
        log_w0 = jnp.log(jnp.asarray(self.weights[0]))
        log_w1 = jnp.log(jnp.asarray(self.weights[1]))
        a = log_w0 + log_c0
        b = log_w1 + log_c1
        m = jnp.maximum(a, b)
        return m + jnp.log(jnp.exp(a - m) + jnp.exp(b - m))

    def cdf(self, x: ArrayLike) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        c0 = jsp_stats.norm.cdf(x_arr, loc=self.means[0], scale=self.scales[0])
        c1 = jsp_stats.norm.cdf(x_arr, loc=self.means[1], scale=self.scales[1])
        return self.weights[0] * c0 + self.weights[1] * c1

    def quantile(self, q: ArrayLike) -> jax.Array:
        """Inverse CDF via brentq. The mixture CDF is strictly increasing
        wherever both component pdfs are non-zero (i.e., everywhere on R
        for Gaussian endpoints), so a unique inverse exists.
        """
        from scipy import optimize
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        out = np.empty_like(q_arr)
        # Bracket: union of component +/-20 sigma ranges.
        lo = float(min(self.means) - 20.0 * max(self.scales))
        hi = float(max(self.means) + 20.0 * max(self.scales))
        for i, qi in enumerate(q_arr):
            qi_f = float(qi)
            if qi_f <= 0.0:
                out[i] = -np.inf
                continue
            if qi_f >= 1.0:
                out[i] = np.inf
                continue

            def f(x: float, qi_f: float = qi_f) -> float:
                return float(self.cdf(jnp.asarray(x))) - qi_f

            out[i] = float(optimize.brentq(f, lo, hi, xtol=1e-12))
        result = out if q_arr.size > 1 else np.asarray(float(out[0]))
        return jnp.asarray(result)

    def mean(self) -> float:
        w0, w1 = float(self.weights[0]), float(self.weights[1])
        m0, m1 = float(self.means[0]), float(self.means[1])
        return w0 * m0 + w1 * m1

    def var(self) -> float:
        # Total variance: E[Var|Z] + Var[E[X|Z]]
        # = sum_k w_k sigma_k^2 + sum_k w_k (mu_k - E[X])^2
        w = (float(self.weights[0]), float(self.weights[1]))
        m = (float(self.means[0]), float(self.means[1]))
        s = (float(self.scales[0]), float(self.scales[1]))
        mean_total = w[0] * m[0] + w[1] * m[1]
        within = w[0] * s[0] ** 2 + w[1] * s[1] ** 2
        between = w[0] * (m[0] - mean_total) ** 2 + w[1] * (m[1] - mean_total) ** 2
        return within + between

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        comp = rng.choice(
            2, size=int(n),
            p=[float(self.weights[0]), float(self.weights[1])],
        )
        x = np.where(
            comp == 0,
            rng.normal(loc=float(self.means[0]), scale=float(self.scales[0]), size=int(n)),
            rng.normal(loc=float(self.means[1]), scale=float(self.scales[1]), size=int(n)),
        )
        return x.astype(np.float64)


@dataclass(frozen=True)
class MixtureDistribution:
    """Two-component mixture of arbitrary `Distribution`-protocol endpoints.

    Generic counterpart to `GaussianMixtureDistribution`. Used by
    `MixtureTilting._generic_tilt_mixture` when the likelihood-as-
    distribution is not Gaussian (e.g., on Bernoulli + Beta, where it
    is a `GridDistribution`).

    Closed-form `pdf`, `logpdf`, `cdf`, `mean`. `var`, `quantile`,
    `sample` are numerical (Gauss-Legendre / brentq / categorical pick).
    """

    weights: tuple[float, float]
    components: tuple[Any, Any]
    n_components: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if len(self.weights) != 2 or len(self.components) != 2:
            raise ValueError(
                f"MixtureDistribution: expected 2 components; "
                f"got weights={self.weights!r}, "
                f"components={[type(c).__name__ for c in self.components]}"
            )
        w0, w1 = float(self.weights[0]), float(self.weights[1])
        if w0 < 0.0 or w1 < 0.0:
            raise ValueError(f"weights must be non-negative, got {self.weights!r}")
        if not np.isclose(w0 + w1, 1.0, atol=1e-10):
            raise ValueError(f"weights must sum to 1, got {self.weights!r}")

    def pdf(self, x: ArrayLike) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        c0 = jnp.asarray(self.components[0].pdf(x_arr))
        c1 = jnp.asarray(self.components[1].pdf(x_arr))
        return self.weights[0] * c0 + self.weights[1] * c1

    def logpdf(self, x: ArrayLike) -> jax.Array:
        return jnp.log(jnp.maximum(self.pdf(x), 1e-300))

    def cdf(self, x: ArrayLike) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        c0 = jnp.asarray(self.components[0].cdf(x_arr))
        c1 = jnp.asarray(self.components[1].cdf(x_arr))
        return self.weights[0] * c0 + self.weights[1] * c1

    def quantile(self, q: ArrayLike) -> jax.Array:
        from scipy import optimize
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        out = np.empty_like(q_arr)

        def _safe_q(comp, u):
            v = float(np.asarray(comp.quantile(np.asarray(u))))
            if not np.isfinite(v):
                return float(np.sign(u - 0.5)) * 1e6
            return v

        c_lo_a = _safe_q(self.components[0], 1e-5)
        c_lo_b = _safe_q(self.components[1], 1e-5)
        c_hi_a = _safe_q(self.components[0], 1.0 - 1e-5)
        c_hi_b = _safe_q(self.components[1], 1.0 - 1e-5)
        lo = float(min(c_lo_a, c_lo_b))
        hi = float(max(c_hi_a, c_hi_b))
        for i, qi in enumerate(q_arr):
            qi_f = float(qi)
            if qi_f <= 0.0:
                out[i] = lo
                continue
            if qi_f >= 1.0:
                out[i] = hi
                continue

            def f(x: float, qi_f: float = qi_f) -> float:
                return float(self.cdf(jnp.asarray(x))) - qi_f

            out[i] = float(optimize.brentq(f, lo, hi, xtol=1e-10))
        result = out if q_arr.size > 1 else np.asarray(float(out[0]))
        return jnp.asarray(result)

    def mean(self) -> float:
        m0 = float(self.components[0].mean())
        m1 = float(self.components[1].mean())
        return float(self.weights[0]) * m0 + float(self.weights[1]) * m1

    def var(self) -> float:
        # Numerical via 64-point Gauss-Legendre on each component's quantile,
        # then total variance = within + between.
        nodes, weights_gl = np.polynomial.legendre.leggauss(64)
        u01 = 0.5 * (nodes + 1.0)
        w01 = 0.5 * weights_gl

        def _comp_moments(comp) -> tuple[float, float]:
            x = np.asarray(comp.quantile(jnp.asarray(u01)), dtype=np.float64)
            m = float(np.sum(w01 * x))
            m2 = float(np.sum(w01 * x * x))
            return m, max(m2 - m * m, 0.0)

        m0, v0 = _comp_moments(self.components[0])
        m1, v1 = _comp_moments(self.components[1])
        w0, w1 = float(self.weights[0]), float(self.weights[1])
        mean_total = w0 * m0 + w1 * m1
        within = w0 * v0 + w1 * v1
        between = w0 * (m0 - mean_total) ** 2 + w1 * (m1 - mean_total) ** 2
        return within + between

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        comp_idx = rng.choice(
            2, size=int(n),
            p=[float(self.weights[0]), float(self.weights[1])],
        )
        x = np.empty(int(n), dtype=np.float64)
        n0 = int(np.sum(comp_idx == 0))
        n1 = int(n) - n0
        if n0 > 0:
            x[comp_idx == 0] = self.components[0].sample(rng, n0)
        if n1 > 0:
            x[comp_idx == 1] = self.components[1].sample(rng, n1)
        return x
