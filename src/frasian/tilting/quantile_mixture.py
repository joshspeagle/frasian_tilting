"""1D Wasserstein-2 geodesic / displacement line between two Distributions.

The W2 geodesic between two 1D distributions p and q at parameter
`t in [0, 1]` is the distribution whose quantile function is the
linear interpolation of the endpoint quantiles:

    F_t^{-1}(u) = (1 - t) * F_p^{-1}(u) + t * F_q^{-1}(u),  u in [0, 1].

Equivalently: the law of the random variable
`(1 - t) * F_p^{-1}(U) + t * F_q^{-1}(U)` for `U ~ Uniform[0, 1]`. The
geodesic *segment* `t in [0, 1]` is always a valid distribution: the
quantile function is non-decreasing in u as a non-negative convex
combination of two non-decreasing functions.

Extrapolation outside `[0, 1]` traces the W2 displacement line and
**may or may not** be a valid distribution: the linear combination is
guaranteed monotone in u only as long as `(1-t) f_p^{-1}'(u) + t
f_q^{-1}'(u) >= 0` for all u (where `f_p^{-1}'` is the derivative of
the quantile, i.e. the inverse density). For two Gaussians this
reduces to `sigma_t = (1-t) sigma_p + t sigma_q >= 0`; for general
endpoints it can fail at some u. We therefore validate monotonicity
on a u-grid at construction time when `t` is outside `[0, 1]`, and
raise a clear `ValueError` if the resulting quantile would not be a
valid distribution. This preserves the OT extrapolation goal (audit
P0-4 — admissible `eta in (-w/(1-w), inf)` for the closed-form
Gaussian WALDO pvalue) while guarding non-Gaussian endpoints from
silently producing nonsense.

JAX seam (`docs/jax_style.md`):
- `quantile`, `mean`, `var`, `sample` are bulk kernels that return
  `jax.Array` / `float` and use `jnp` internally. They are
  autodiff-clean for Phase 4's W2-baseline experiments.
- `cdf`, `pdf` are numpy-eager: each scalar `x` triggers one
  `scipy.optimize.brentq` root-find, whose closure body must convert
  the JAX `quantile` return to a Python float per iteration. This
  matches the principle that scalar Python control flow stays numpy.
  Output is converted to `jax.Array` once at the boundary so the
  Distribution protocol is honoured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

# scipy: brentq has no JAX equivalent we want yet; the cdf/pdf inner
# loops live at the public-distribution boundary and run on numpy/scipy.
from scipy import optimize

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


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
        # Audit P0-review #2: extrapolation outside [0, 1] is allowed
        # (the OT W2 displacement line — see audit P0-4), BUT we must
        # verify the resulting law is still a valid distribution. The
        # linear combination of quantile functions is guaranteed monotone
        # only inside the segment; outside, it can decrease at some u
        # and the downstream `pdf` / `var` / `sample` would silently
        # produce nonsense. We probe a coarse u-grid here so callers
        # who extrapolate get a clean error at construction rather than
        # a corrupted distribution at first downstream use.
        t = float(self.t)
        if not np.isfinite(t):
            raise ValueError(f"t must be finite, got {self.t!r}.")
        if t < 0.0 or t > 1.0:
            self._validate_extrapolation_monotone(t)

    def _validate_extrapolation_monotone(self, t: float) -> None:
        """Probe the linear-combo quantile on a u-grid; raise if non-monotone.

        Gaussian-Gaussian endpoints reduce to the closed-form check
        `sigma_t = (1-t) sigma_p + t sigma_q >= 0`. For general
        endpoints we sample on a 65-point u-grid (interior only — the
        exact endpoints u=0 / u=1 may yield infinite quantiles for
        unbounded supports). A small tolerance absorbs floating-point
        noise on flat segments; anything below it is a true monotonicity
        violation.
        """
        # Cheap closed-form fast path: both endpoints Gaussian.
        from ..models.distributions import NormalDistribution
        if isinstance(self.p, NormalDistribution) and isinstance(self.q, NormalDistribution):
            sigma_t = (1.0 - t) * float(self.p.scale) + t * float(self.q.scale)
            if sigma_t <= 0.0:
                raise ValueError(
                    f"QuantileMixturePath at t={t!r} produces a degenerate "
                    f"or reversed Gaussian (sigma_t={sigma_t!r} <= 0); the "
                    f"law is not a valid distribution. "
                    f"Endpoints: sigma_p={self.p.scale!r}, sigma_q={self.q.scale!r}."
                )
            return

        # General-1D probe: sample the quantile, check monotonicity.
        u_grid = np.linspace(1.0e-4, 1.0 - 1.0e-4, 65)
        try:
            qp = np.asarray(self.p.quantile(u_grid), dtype=np.float64)
            qq = np.asarray(self.q.quantile(u_grid), dtype=np.float64)
        except Exception as exc:
            raise ValueError(
                f"QuantileMixturePath: cannot validate t={t!r} (outside "
                f"[0, 1]) because endpoint quantile evaluation failed: "
                f"{exc!r}."
            ) from exc
        qt = (1.0 - t) * qp + t * qq
        if not (np.all(np.isfinite(qt))):
            raise ValueError(
                f"QuantileMixturePath at t={t!r} produces non-finite "
                f"quantile values; the resulting law is not a valid "
                f"distribution."
            )
        diffs = np.diff(qt)
        scale = float(np.max(np.abs(qt))) + 1.0
        # Tolerance: absorb fp noise on flat segments but flag genuine
        # decreases. 1e-9 * scale is below MC noise but above fp.
        if np.any(diffs < -1.0e-9 * scale):
            bad = int(np.argmin(diffs))
            raise ValueError(
                f"QuantileMixturePath at t={t!r} (outside [0, 1]) "
                f"produces a non-monotone quantile function — "
                f"the resulting law is not a valid distribution. "
                f"(1-t)*F_p^{{-1}}(u) + t*F_q^{{-1}}(u) decreases near "
                f"u={u_grid[bad+1]:.4g} (from {qt[bad]:.4g} to "
                f"{qt[bad+1]:.4g}). Extrapolation is admissible only "
                f"when the endpoint inverse-density slopes balance: "
                f"(1-t)/f_p(F_p^{{-1}}(u)) + t/f_q(F_q^{{-1}}(u)) >= 0 "
                f"for every u."
            )

    # ----- Closed-form pieces -----

    def quantile(self, u: ArrayLike) -> jax.Array:
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        return (1.0 - self.t) * jnp.asarray(self.p.quantile(u_arr)) + self.t * jnp.asarray(
            self.q.quantile(u_arr)
        )

    def mean(self) -> float:
        return float((1.0 - self.t) * self.p.mean() + self.t * self.q.mean())

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        u = rng.uniform(0.0, 1.0, size=n)
        # Sampling lives at the I/O boundary: numpy in, numpy out.
        return np.asarray(self.quantile(u), dtype=np.float64)

    # ----- Numerical pieces (root-find / quadrature on a u-grid) -----

    def cdf(self, x: ArrayLike) -> jax.Array:
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
            xi_f = float(xi)

            def f(u: float, xi_f: float = xi_f) -> float:
                # quantile() returns a jax.Array; float(jax_0d_array) is
                # one device dispatch (~5 us), acceptable per iteration.
                return float(self.quantile(u)) - xi_f

            try:
                u_star = float(optimize.brentq(f, eps, 1.0 - eps, xtol=1e-10))
            except ValueError:
                # x outside the support of the path — return exact 0.0 / 1.0.
                # f(eps) > 0 means quantile(eps) > xi, i.e. xi is below support.
                # Defensive: if quantile(eps) is non-finite (pathological
                # endpoint distribution), the `f_eps > 0` test would silently
                # fall through to 1.0 (the wrong tail). Refuse explicitly so
                # the failure surfaces.
                f_eps = f(eps)
                if not np.isfinite(f_eps):
                    raise ValueError(
                        f"QuantileMixturePath.cdf cannot decide tail for "
                        f"x={xi_f!r}: f(eps={eps!r})={f_eps!r} is non-finite. "
                        f"Endpoint quantile() likely returned NaN/Inf at "
                        f"the support boundary."
                    ) from None
                u_star = 0.0 if f_eps > 0.0 else 1.0
            out[i] = u_star
        result = out if x_arr.size > 1 else np.asarray(float(out[0]))
        return jnp.asarray(result)

    def _outside_support_mask(self, x_arr: NDArray[np.float64]) -> NDArray[np.bool_]:
        """True at indices where `x_arr[i]` lies outside the path's support.

        Symmetric with `cdf`'s boundary detector: an x lies outside the
        path's support iff `quantile(eps) > x` (below the lower endpoint)
        or `quantile(1-eps) < x` (above the upper endpoint). Computing
        this directly — instead of inferring from `cdf` returning exact
        0.0 / 1.0 — keeps `pdf` robust against future refactors that
        might change `cdf`'s exact-boundary return value.
        """
        eps = 1e-12
        q_lo = float(self.quantile(eps))
        q_hi = float(self.quantile(1.0 - eps))
        return (x_arr < q_lo) | (x_arr > q_hi)

    def pdf(self, x: ArrayLike) -> jax.Array:
        """f_t(x) = 1 / (d/du F_t^{-1}(u))|_{u = F_t(x)}.

        By chain rule: dF_t^{-1}/du = (1 - t) / f_p(F_p^{-1}(u))
        + t / f_q(F_q^{-1}(u)). Evaluated at u = F_t(x).
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        u = np.asarray(self.cdf(x_arr), dtype=np.float64)
        u_arr = np.atleast_1d(u)
        # Compute the outside-support mask symmetrically with `cdf`'s own
        # boundary detector (compare x to the path's endpoint quantiles)
        # rather than inferring it from `u_arr <= 0.0 | u_arr >= 1.0`.
        outside = self._outside_support_mask(x_arr)
        # Endpoint distributions are JAX-ported; convert to numpy at the
        # boundary since the chain-rule arithmetic below is numpy-eager.
        xp = np.asarray(self.p.quantile(u_arr), dtype=np.float64)
        xq = np.asarray(self.q.quantile(u_arr), dtype=np.float64)
        fp = np.asarray(self.p.pdf(xp), dtype=np.float64)
        fq = np.asarray(self.q.pdf(xq), dtype=np.float64)
        # Guard against division by zero at endpoints — return 0 density there.
        denom = np.where(fp > 0, (1.0 - self.t) / np.where(fp > 0, fp, 1.0), 0.0) + np.where(
            fq > 0, self.t / np.where(fq > 0, fq, 1.0), 0.0
        )
        out = np.where(denom > 0, 1.0 / np.where(denom > 0, denom, 1.0), 0.0)
        out = np.where(outside, 0.0, out)
        result = out if x_arr.size > 1 else np.asarray(float(out[0]))
        return jnp.asarray(result)

    def logpdf(self, x: ArrayLike) -> jax.Array:
        return jnp.log(self.pdf(x))

    def var(self) -> float:
        """Var = E[X^2] - E[X]^2 via Gauss-Legendre on the quantile."""
        # 64-point fixed Gauss-Legendre on (0, 1) — sufficient for the
        # smooth Gaussian / Beta endpoints we exercise.
        # scipy: numpy.polynomial.legendre has no JAX equivalent; the
        # nodes/weights are constants computed once per call.
        u, w = np.polynomial.legendre.leggauss(64)
        u01 = jnp.asarray(0.5 * (u + 1.0))
        w01 = jnp.asarray(0.5 * w)
        x = self.quantile(u01)
        m1 = float(jnp.sum(w01 * x))
        m2 = float(jnp.sum(w01 * x * x))
        return max(m2 - m1 * m1, 0.0)
