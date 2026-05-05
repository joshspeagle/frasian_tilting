"""1D Wasserstein-2 geodesic between two arbitrary Distributions.

The W2 geodesic between two 1D distributions p and q at parameter
`t in [0, 1]` is the distribution whose quantile function is the
linear interpolation of the endpoint quantiles:

    F_t^{-1}(u) = (1 - t) * F_p^{-1}(u) + t * F_q^{-1}(u),  u in [0, 1].

Equivalently: the law of the random variable
`(1 - t) * F_p^{-1}(U) + t * F_q^{-1}(U)` for `U ~ Uniform[0, 1]`. The
geodesic is well-defined for any two endpoints exposing `quantile`,
which is in the `Distribution` protocol — so this is the **general**
1D OT path. The Gaussian-endpoint case collapses to the closed-form
linear interpolation in `(mu, sigma)` that `OTTilting.tilt` uses as a
fast path.

Why a wrapper rather than materialising a Gaussian: when endpoints are
non-Gaussian (e.g. Beta, Bernoulli) the W2 path is generically not in
any closed-form family, so it must be represented by its quantile
function. PDF and CDF are then derived numerically (CDF by 1D root-find
on the quantile, PDF by reciprocal of the quantile derivative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from scipy import optimize


@dataclass(frozen=True)
class QuantileMixturePath:
    """W2 geodesic between two Distributions, evaluated at `t in [0, 1]`.

    Conforms to the `Distribution` protocol. Closed-form for `quantile`,
    `mean`, and `sample`; numerical for `cdf`, `pdf`, `var` (via
    quadrature on a fixed `u`-grid).
    """

    p: Any
    q: Any
    t: float

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.t) <= 1.0):
            raise ValueError(
                f"t must lie in [0, 1], got {self.t!r}."
            )

    # ----- Closed-form pieces -----

    def quantile(self, u: ArrayLike) -> NDArray[np.float64]:
        u_arr = np.asarray(u, dtype=np.float64)
        return np.asarray(
            (1.0 - self.t) * np.asarray(self.p.quantile(u_arr), dtype=np.float64)
            + self.t * np.asarray(self.q.quantile(u_arr), dtype=np.float64),
            dtype=np.float64,
        )

    def mean(self) -> float:
        return float((1.0 - self.t) * self.p.mean() + self.t * self.q.mean())

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        u = rng.uniform(0.0, 1.0, size=n)
        return self.quantile(u)

    # ----- Numerical pieces (root-find / quadrature on a u-grid) -----

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """F_t(x) by 1D root-find on F_t^{-1}(u) = x.

        The inversion uses brentq on `u in (0, 1)`. We bracket via the
        endpoints' CDFs: at `u = 0` and `u = 1` the quantile is
        `-inf` and `+inf`, so the bracket is safe in principle. We
        clip to `(eps, 1-eps)` for numerical stability.
        """
        eps = 1e-12
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        out = np.empty_like(x_arr)
        for i, xi in enumerate(x_arr):
            f = lambda u, xi=xi: float(self.quantile(np.asarray(u))) - float(xi)
            try:
                u_star = optimize.brentq(f, eps, 1.0 - eps, xtol=1e-10)
            except ValueError:
                # x outside the support of the path — return exact 0.0 / 1.0.
                # f(eps) > 0 means quantile(eps) > xi, i.e. xi is below support.
                out[i] = 0.0 if f(eps) > 0.0 else 1.0
                continue
            out[i] = u_star
        return out if x_arr.size > 1 else np.asarray(float(out[0]))

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """f_t(x) = 1 / (d/du F_t^{-1}(u))|_{u = F_t(x)}.

        By chain rule: dF_t^{-1}/du = (1 - t) / f_p(F_p^{-1}(u))
        + t / f_q(F_q^{-1}(u)). Evaluated at u = F_t(x).
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        u = self.cdf(x_arr)
        u_arr = np.atleast_1d(u)
        # u == 0.0 or u == 1.0 from cdf() means x lies outside the support;
        # return exact 0.0 density there rather than relying on the chain-rule
        # quotient (which can underflow to a tiny positive number).
        outside = (u_arr <= 0.0) | (u_arr >= 1.0)
        xp = np.asarray(self.p.quantile(u_arr), dtype=np.float64)
        xq = np.asarray(self.q.quantile(u_arr), dtype=np.float64)
        fp = np.asarray(self.p.pdf(xp), dtype=np.float64)
        fq = np.asarray(self.q.pdf(xq), dtype=np.float64)
        # Guard against division by zero at endpoints — return 0 density there.
        denom = np.where(fp > 0, (1.0 - self.t) / np.where(fp > 0, fp, 1.0), 0.0) \
              + np.where(fq > 0, self.t / np.where(fq > 0, fq, 1.0), 0.0)
        out = np.where(denom > 0, 1.0 / np.where(denom > 0, denom, 1.0), 0.0)
        out = np.where(outside, 0.0, out)
        return out if x_arr.size > 1 else np.asarray(float(out[0]))

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.log(self.pdf(x))

    def var(self) -> float:
        """Var = E[X^2] - E[X]^2 via Gauss-Legendre on the quantile."""
        # 64-point fixed Gauss-Legendre on (0, 1) — sufficient for the
        # smooth Gaussian / Beta endpoints we exercise.
        u, w = np.polynomial.legendre.leggauss(64)
        u01 = 0.5 * (u + 1.0)
        w01 = 0.5 * w
        x = self.quantile(u01)
        m1 = float(np.sum(w01 * x))
        m2 = float(np.sum(w01 * x * x))
        return max(m2 - m1 * m1, 0.0)
