"""Concrete distributions used by `NormalNormalModel`.

`NormalDistribution` is the working horse for prior, posterior, and tilted
posterior. `GaussianLikelihood` is the model's likelihood-of-data view.
Both are dataclasses and conform to the protocols in `models.base`.

`MixtureDistribution` is the 2-component Gaussian mixture used by
`MixtureTilting` (the m-geodesic). It is *not* in the Normal family
generally — its pdf is `(1-eta)·N(m1, s1) + eta·N(m2, s2)` and is
bimodal beyond the Behboodian threshold. See `docs/methods/mixture.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats


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

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.norm.pdf(x, loc=self.loc, scale=self.scale), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.norm.logpdf(x, loc=self.loc, scale=self.scale), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.norm.cdf(x, loc=self.loc, scale=self.scale), dtype=np.float64)

    def quantile(self, q: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.norm.ppf(q, loc=self.loc, scale=self.scale), dtype=np.float64)

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

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.beta.pdf(x, self.alpha, self.beta), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.beta.logpdf(x, self.alpha, self.beta), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.beta.cdf(x, self.alpha, self.beta), dtype=np.float64)

    def quantile(self, q: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(stats.beta.ppf(q, self.alpha, self.beta), dtype=np.float64)

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

    def __call__(self, theta: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self.loglik(theta))

    def loglik(self, theta: ArrayLike) -> NDArray[np.float64]:
        theta_arr = np.asarray(theta, dtype=np.float64)
        # Guard against log(0) at the support boundary by clipping.
        eps = 1e-300
        return np.asarray(
            self.n_success * np.log(np.clip(theta_arr, eps, 1.0))
            + (self.n_total - self.n_success) * np.log(np.clip(1.0 - theta_arr, eps, 1.0)),
            dtype=np.float64,
        )


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

    def __call__(self, theta: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self.loglik(theta))

    def loglik(self, theta: ArrayLike) -> NDArray[np.float64]:
        z = (np.asarray(theta, dtype=np.float64) - self.D) / self.sigma
        return -0.5 * z * z - 0.5 * np.log(2.0 * np.pi * self.sigma**2)


@dataclass(frozen=True)
class MixtureDistribution:
    """Two-component Gaussian mixture used by `MixtureTilting` (the m-geodesic).

    The m-geodesic between the WALDO posterior `N(mu_n, sigma_n^2)` and the
    likelihood-as-Gaussian `N(D, sigma^2)` at parameter ``eta in [0, 1]`` is

        p_eta(theta) = (1 - eta) * N(theta; m1, s1^2) + eta * N(theta; m2, s2^2)

    with ``(m1, s1) = (mu_n, sigma_n)`` and ``(m2, s2) = (D, sigma)``. This
    is *not* in the Normal family; the dataclass conforms to the
    `Distribution` protocol surface (pdf / logpdf / cdf / quantile / mean /
    var / sample / fingerprint) so the tilted output can be consumed by the
    framework's existing CD constructor and CI inversion paths.

    Closed-form for everything except `quantile`, which requires numerical
    inversion via brentq on `cdf(theta) - q`. Mean and variance use
    Step 3 of `audit/tier2/mixture_derivation.md`:

        mean = (1 - eta) * m1 + eta * m2
        var  = (1 - eta) * s1^2 + eta * s2^2 + eta * (1 - eta) * (m1 - m2)^2

    The dataclass is K=2-only on purpose: the framework's tilted-pvalue
    closed form (Phi-pairs per component) is hard-coded for two components,
    so generalising to K>2 would silently mismatch. A K-component
    extension is a separate refactor.
    """

    weights: tuple[float, float]
    means: tuple[float, float]
    sigmas: tuple[float, float]

    def __post_init__(self) -> None:
        w0, w1 = self.weights
        if not (np.isfinite(w0) and np.isfinite(w1)):
            raise ValueError(f"weights must be finite, got {self.weights!r}")
        if not (w0 >= 0.0 and w1 >= 0.0):
            raise ValueError(f"weights must be non-negative, got {self.weights!r}")
        if not np.isclose(w0 + w1, 1.0, atol=1e-12):
            raise ValueError(f"weights must sum to 1, got {self.weights!r} (sum={w0+w1!r})")
        for m in self.means:
            if not np.isfinite(m):
                raise ValueError(f"means must be finite, got {self.means!r}")
        for s in self.sigmas:
            if not (np.isfinite(s) and s > 0):
                raise ValueError(f"sigmas must be positive and finite, got {self.sigmas!r}")

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        w0, w1 = self.weights
        m0, m1 = self.means
        s0, s1 = self.sigmas
        p0 = stats.norm.pdf(x_arr, loc=m0, scale=s0)
        p1 = stats.norm.pdf(x_arr, loc=m1, scale=s1)
        return np.asarray(w0 * p0 + w1 * p1, dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        # log-sum-exp over the two components for numerical stability.
        x_arr = np.asarray(x, dtype=np.float64)
        w0, w1 = self.weights
        m0, m1 = self.means
        s0, s1 = self.sigmas
        # log(w_k) may be -inf if a weight is exactly zero — mask those out
        # so log-sum-exp handles them as missing components.
        log_w0 = -np.inf if w0 == 0.0 else np.log(w0)
        log_w1 = -np.inf if w1 == 0.0 else np.log(w1)
        a = log_w0 + stats.norm.logpdf(x_arr, loc=m0, scale=s0)
        b = log_w1 + stats.norm.logpdf(x_arr, loc=m1, scale=s1)
        return np.asarray(np.logaddexp(a, b), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        w0, w1 = self.weights
        m0, m1 = self.means
        s0, s1 = self.sigmas
        c0 = stats.norm.cdf(x_arr, loc=m0, scale=s0)
        c1 = stats.norm.cdf(x_arr, loc=m1, scale=s1)
        return np.asarray(w0 * c0 + w1 * c1, dtype=np.float64)

    def quantile(self, q: ArrayLike) -> NDArray[np.float64]:
        """Numerical inversion of `cdf` via brentq.

        Bracket: between the most extreme component quantile bounds. Bisection
        in u-space is robust because the mixture CDF is monotone (a convex
        combination of monotone CDFs).
        """
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        out = np.empty_like(q_arr)
        # Cheap bracket: 1e-12 / 1-1e-12 quantile of each component, take
        # the more extreme on each side. Bisection is monotone-CDF-safe.
        eps = 1e-12
        m0, m1 = self.means
        s0, s1 = self.sigmas
        # Use the wider component quantiles as the search bracket.
        bracket_lo = float(min(stats.norm.ppf(eps, loc=m0, scale=s0),
                                stats.norm.ppf(eps, loc=m1, scale=s1)))
        bracket_hi = float(max(stats.norm.ppf(1.0 - eps, loc=m0, scale=s0),
                                stats.norm.ppf(1.0 - eps, loc=m1, scale=s1)))
        for i, qi in enumerate(q_arr):
            qi_f = float(qi)
            if qi_f <= 0.0:
                out[i] = -np.inf
                continue
            if qi_f >= 1.0:
                out[i] = np.inf
                continue

            def f(theta: float, _q: float = qi_f) -> float:
                return float(self.cdf(np.asarray(theta))) - _q

            try:
                out[i] = optimize.brentq(f, bracket_lo, bracket_hi, xtol=1e-10)
            except ValueError:
                # Defensive: if the bracket fails, fall back to component-q
                # midpoint. Should not happen in practice on the canonical
                # sandbox.
                out[i] = 0.5 * (
                    stats.norm.ppf(qi_f, loc=m0, scale=s0)
                    + stats.norm.ppf(qi_f, loc=m1, scale=s1)
                )
        return out if q_arr.size > 1 else np.asarray(float(out[0]))

    def mean(self) -> float:
        w0, w1 = self.weights
        m0, m1 = self.means
        return float(w0 * m0 + w1 * m1)

    def var(self) -> float:
        """Mixture variance: weighted within-component + between-component.

        For weights `(1-eta, eta)`, components `(m_k, s_k)`:
            var = (1-eta) s_n^2 + eta s^2 + eta (1-eta) (m_n - m_2)^2
        """
        w0, w1 = self.weights
        m0, m1 = self.means
        s0, s1 = self.sigmas
        mu = w0 * m0 + w1 * m1
        # E[X^2] = sum_k w_k (m_k^2 + s_k^2)
        ex2 = w0 * (m0 * m0 + s0 * s0) + w1 * (m1 * m1 + s1 * s1)
        return float(max(ex2 - mu * mu, 0.0))

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        """Draw `n` i.i.d. mixture samples: draw a component, then sample it."""
        w0, _w1 = self.weights
        m0, m1 = self.means
        s0, s1 = self.sigmas
        # Component indicator: 0 with prob w0, 1 with prob w1.
        u = rng.uniform(0.0, 1.0, size=n)
        which = (u >= w0).astype(np.int64)
        z = rng.standard_normal(n)
        means_arr = np.where(which == 0, m0, m1)
        sigmas_arr = np.where(which == 0, s0, s1)
        return np.asarray(means_arr + sigmas_arr * z, dtype=np.float64)

    def fingerprint(self) -> tuple:
        return (
            "mixture_2_normal",
            tuple(float(w) for w in self.weights),
            tuple(float(m) for m in self.means),
            tuple(float(s) for s in self.sigmas),
        )
