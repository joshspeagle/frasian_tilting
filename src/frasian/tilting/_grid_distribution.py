"""GridDistribution: a numerical 1D Distribution stored on a theta-grid.

Used by `PowerLawTilting._generic_tilt` and `OTTilting._generic_tilt` as
the model-agnostic numerical tilted distribution. Conforms to the
`Distribution` protocol from `frasian.models.base`.

Distinct from `frasian.cd.grid.GridConfidenceDistribution` — that class
is purpose-built for confidence distributions (with `signed_confidence`
provenance, `interval`, validity-check semantics) and lacks the `mean`
/ `var` / `sample` surface the Distribution protocol demands. The two
classes share the linear-interpolation grid pattern but have different
roles in the framework.

JAX seam (`docs/jax_style.md`):
- `pdf` / `logpdf` / `cdf` / `quantile` return `jax.Array` and use
  `jnp.interp` so they remain autodiff-clean for Phase 4's learned-eta
  loss when called from inside `@jax.jit`.
- `mean` / `var` are float-returning aggregates computed via
  `jnp.trapezoid` then materialised at the boundary.
- `sample(rng, n)` consumes a numpy `Generator` and returns a numpy
  array — sampling lives at the I/O boundary by framework convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


@dataclass(frozen=True)
class GridDistribution:
    """1D Distribution stored on a theta-grid via its pdf.

    Parameters
    ----------
    theta_grid : ndarray, shape (n,), strictly increasing
        Support grid for the distribution. Outside this range the pdf
        is treated as 0 and the cdf as the boundary value (consistent
        with truncating the distribution to the grid).
    pdf_values : ndarray, shape (n,), non-negative
        Density at each grid point. Constructors should normalise so
        that `trapezoid(pdf_values, theta_grid) == 1`; the class itself
        does NOT re-normalise (that would hide upstream construction
        bugs).
    metadata : dict, optional
        Free-form. `_generic_tilt` records `(eta, scheme, model.fingerprint,
        prior.fingerprint)` for debugging / cache invalidation.
    """

    theta_grid: NDArray[np.float64]
    pdf_values: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.theta_grid.ndim != 1:
            raise ValueError(
                f"theta_grid must be 1-D; got shape {self.theta_grid.shape}"
            )
        if self.theta_grid.size != self.pdf_values.size:
            raise ValueError(
                f"theta_grid (size {self.theta_grid.size}) and pdf_values "
                f"(size {self.pdf_values.size}) must have equal length."
            )
        if self.theta_grid.size < 2:
            raise ValueError(
                f"theta_grid must have at least 2 points; got "
                f"{self.theta_grid.size}"
            )

    @cached_property
    def cdf_values(self) -> jax.Array:
        """Cumulative trapezoidal integral of pdf_values. Monotone non-decreasing."""
        theta = jnp.asarray(self.theta_grid, dtype=jnp.float64)
        pdf = jnp.asarray(self.pdf_values, dtype=jnp.float64)
        dx = jnp.diff(theta)
        mid = 0.5 * (pdf[:-1] + pdf[1:])
        increments = mid * dx
        return jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.float64), jnp.cumsum(increments)]
        )

    # ----- Distribution protocol surface -----

    def pdf(self, x: ArrayLike) -> jax.Array:
        """pdf interpolated at `x` (linear; 0 outside the grid)."""
        return jnp.interp(
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(self.theta_grid, dtype=jnp.float64),
            jnp.asarray(self.pdf_values, dtype=jnp.float64),
            left=0.0,
            right=0.0,
        )

    def logpdf(self, x: ArrayLike) -> jax.Array:
        """log pdf at `x`. Returns -inf where the pdf is 0.

        The clipping at `1e-300` prevents `-inf` propagating through
        downstream arithmetic; consumers that need a hard `-inf` outside
        the support should mask explicitly using `pdf(x) == 0`.
        """
        return jnp.log(jnp.maximum(self.pdf(x), 1e-300))

    def cdf(self, x: ArrayLike) -> jax.Array:
        """cdf interpolated at `x` (linear; clamped to [0, total_mass])."""
        cdf_vals = self.cdf_values
        return jnp.interp(
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(self.theta_grid, dtype=jnp.float64),
            cdf_vals,
            left=0.0,
            right=float(cdf_vals[-1]),
        )

    def quantile(self, q: ArrayLike) -> jax.Array:
        """Inverse-cdf at probability `q ∈ [0, 1]` via linear interpolation.

        Well-defined whenever `cdf_values` is monotone non-decreasing,
        which the trapezoidal cumulant guarantees for non-negative pdf.
        """
        return jnp.interp(
            jnp.asarray(q, dtype=jnp.float64),
            self.cdf_values,
            jnp.asarray(self.theta_grid, dtype=jnp.float64),
            left=float(self.theta_grid[0]),
            right=float(self.theta_grid[-1]),
        )

    def mean(self) -> float:
        """E[θ] via numerical integration of θ * pdf(θ)."""
        theta = jnp.asarray(self.theta_grid, dtype=jnp.float64)
        pdf = jnp.asarray(self.pdf_values, dtype=jnp.float64)
        return float(jnp.trapezoid(theta * pdf, theta))

    def var(self) -> float:
        """Var[θ] = E[θ^2] - (E[θ])^2 via numerical integration."""
        theta = jnp.asarray(self.theta_grid, dtype=jnp.float64)
        pdf = jnp.asarray(self.pdf_values, dtype=jnp.float64)
        m1 = float(jnp.trapezoid(theta * pdf, theta))
        m2 = float(jnp.trapezoid(theta * theta * pdf, theta))
        return max(m2 - m1 * m1, 0.0)

    def sample(self, rng: Generator, n: int) -> NDArray[np.float64]:
        """Inverse-CDF sampling: U ~ Uniform[0, 1] → quantile(U).

        Sampling lives at the I/O boundary; numpy `Generator` in,
        numpy array out (consistent with `NormalDistribution.sample` etc.).
        """
        u = rng.uniform(0.0, 1.0, size=n)
        return np.asarray(self.quantile(u), dtype=np.float64)


def grid_distribution_from_log_density(
    theta_grid: NDArray[np.float64],
    log_density_values: NDArray[np.float64] | jax.Array,
    *,
    metadata: dict[str, Any] | None = None,
) -> GridDistribution:
    """Build a `GridDistribution` from an unnormalised log-density.

    Common ingest path for `_generic_tilt` implementations: compute
    `log p_unnorm(theta_i)` on a grid, then normalise here. The
    log-sum-exp + max-subtraction idiom keeps the exponential well-
    conditioned even when the log-density has -O(100) tails (e.g.
    a tilted Gaussian far from the mode).

    Parameters
    ----------
    theta_grid : ndarray, (n,)
    log_density_values : ndarray or jax.Array, (n,)
        Log of an unnormalised density on the grid; will be normalised
        to integrate to 1 via trapezoidal quadrature.
    metadata : optional
        Forwarded to the resulting `GridDistribution`.
    """
    log_d = jnp.asarray(log_density_values, dtype=jnp.float64)
    # Subtract the max for numerical stability before exp.
    pdf_unnorm = jnp.exp(log_d - jnp.max(log_d))
    theta = jnp.asarray(theta_grid, dtype=jnp.float64)
    Z = jnp.trapezoid(pdf_unnorm, theta)
    if not bool(jnp.isfinite(Z)) or float(Z) <= 0.0:
        raise ValueError(
            f"GridDistribution.from_log_density: trapezoidal Z={float(Z)!r} "
            f"is non-finite or non-positive; check for NaN log-density "
            f"or a degenerate grid."
        )
    pdf = pdf_unnorm / Z
    return GridDistribution(
        theta_grid=np.asarray(theta_grid, dtype=np.float64),
        pdf_values=np.asarray(pdf, dtype=np.float64),
        metadata=metadata or {},
    )
