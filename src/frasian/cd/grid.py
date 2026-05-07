"""GridConfidenceDistribution — the framework's concrete CD.

A confidence distribution stored as a grid of (θ, pdf) values. The pdf
is the **primary primitive**: by construction it is non-negative and
integrates to 1, so the cumulative integral (cdf) is always monotone
non-decreasing — even when the underlying p-value function is multimodal
(e.g. Dyn-WALDO under conflict). This means W₁ and W₂ act on the *real*
probability distribution implied by the density, not a flattened
rearrangement.

The optional `signed_confidence` field stores the inversion-based curve
C(θ) — when the constructor has access to it (e.g. the p-value-based
constructor in `cd.from_pvalue`). For unimodal p, signed_confidence
matches the cumulative cdf. For multimodal p (Dyn-WALDO at conflict),
signed_confidence is non-monotone — this is the *diagnostic*
quantity that reveals the smoothness pathology. It is never used by
distance metrics.

References
----------
Schweder, T. and Hjort, N. L. (2002). "Confidence and Likelihood."
*Scandinavian Journal of Statistics* 29: 309–332. — defines the CD
density `c(θ) = ½|dp/dθ|` we use as the pdf primitive.

Singh, K., Xie, M., and Strawderman, W. E. (2005). "Combining
information from independent sources through confidence distributions."
*Annals of Statistics* 33: 159–183. — formalises the CD-validity
property `cdf(θ_true) ~ U[0, 1]` under H0.

Singh, K., Xie, M., and Strawderman, W. E. (2007). "Confidence
distribution (CD) — distribution estimator of a parameter." *IMS
Lecture Notes — Monograph Series* 54: 132–150. — CD-mean, CD-median,
CD-mode point estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


@dataclass(frozen=True)
class CDValidityIssue:
    """One flagged property failure on a `GridConfidenceDistribution`.

    Issues are non-fatal — the CD is still usable; the issue list is
    a diagnostic surface (typically inspected by tests, the CD experiment,
    or the illustration script).
    """

    code: str  # short tag, e.g. "non-monotone-signed-confidence"
    message: str  # human-readable description
    severity: str = "warning"  # "warning" | "error"


@dataclass(frozen=True)
class GridConfidenceDistribution:
    """A 1D confidence distribution stored on a θ-grid via its pdf.

    Parameters
    ----------
    name : str
        Identifier for the cell that produced this CD (e.g.
        "(identity, waldo)@D=1.5").
    theta_grid : ndarray, shape (n,), strictly increasing
        Support grid for the CD.
    pdf_values : ndarray, shape (n,), non-negative
        The CD density. Must integrate to ≈ 1 over `theta_grid`
        (constructors enforce this via Z-normalisation). The cdf is
        always derived as the cumulative integral of `pdf_values`.
    signed_confidence : ndarray, shape (n,), optional
        The inversion-based confidence curve C(θ) — possibly non-monotone
        for multimodal p-values. Stored as auxiliary diagnostic data;
        not used by distance metrics or quantile/interval queries.
    metadata : dict
        Free-form. Constructors record provenance (cell name, D, w, …).
    """

    name: str
    theta_grid: NDArray[np.float64]
    pdf_values: NDArray[np.float64]
    signed_confidence: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----- Derived quantities -----

    @cached_property
    def cdf_values(self) -> jax.Array:
        """Cumulative trapezoidal integral of `pdf_values`. Always monotone."""
        if self.theta_grid.size != self.pdf_values.size:
            raise ValueError(
                f"theta_grid (size {self.theta_grid.size}) and pdf_values "
                f"(size {self.pdf_values.size}) must have equal length."
            )
        theta = jnp.asarray(self.theta_grid, dtype=jnp.float64)
        pdf = jnp.asarray(self.pdf_values, dtype=jnp.float64)
        # Cumulative trapezoid: starts at 0, ends at total mass.
        if theta.size < 2:
            return jnp.zeros_like(theta)
        dx = jnp.diff(theta)
        mid = 0.5 * (pdf[:-1] + pdf[1:])
        increments = mid * dx
        return jnp.concatenate([jnp.zeros((1,), dtype=jnp.float64), jnp.cumsum(increments)])

    # ----- Probability-density queries -----

    def pdf(self, theta: ArrayLike) -> jax.Array:
        """pdf interpolated at `theta` (linear)."""
        return jnp.interp(
            jnp.asarray(theta, dtype=jnp.float64),
            jnp.asarray(self.theta_grid, dtype=jnp.float64),
            jnp.asarray(self.pdf_values, dtype=jnp.float64),
            left=0.0,
            right=0.0,
        )

    def cdf(self, theta: ArrayLike) -> jax.Array:
        """cdf interpolated at `theta` (linear). Always monotone."""
        cdf_vals = self.cdf_values
        return jnp.interp(
            jnp.asarray(theta, dtype=jnp.float64),
            jnp.asarray(self.theta_grid, dtype=jnp.float64),
            cdf_vals,
            left=0.0,
            right=float(cdf_vals[-1]),
        )

    def quantile(self, q: ArrayLike) -> jax.Array:
        """Inverse-cdf at probability `q ∈ [0, 1]`, linear interpolation.

        Always well-defined since `cdf_values` is monotone non-decreasing
        (the pdf-primary design).
        """
        q_arr = jnp.asarray(q, dtype=jnp.float64)
        # Strictly monotonise to handle plateaus where cdf is flat (zero
        # density). jnp.interp treats ties by returning the leftmost match,
        # which is the correct inverse-cdf in that case.
        return jnp.interp(
            q_arr,
            self.cdf_values,
            jnp.asarray(self.theta_grid, dtype=jnp.float64),
            left=float(self.theta_grid[0]),
            right=float(self.theta_grid[-1]),
        )

    def interval(self, alpha: float) -> tuple[float, float]:
        """Equal-tailed (1−α) interval: (quantile(α/2), quantile(1−α/2))."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")
        lo = float(self.quantile(alpha / 2.0))
        hi = float(self.quantile(1.0 - alpha / 2.0))
        return (lo, hi)

    # ----- Point estimators (Singh-Xie-Strawderman 2007) -----

    def mean(self) -> float:
        """CD-mean: ∫θ pdf(θ) dθ via trapezoidal integration."""
        theta = jnp.asarray(self.theta_grid, dtype=jnp.float64)
        pdf = jnp.asarray(self.pdf_values, dtype=jnp.float64)
        return float(jnp.trapezoid(theta * pdf, theta))

    def median(self) -> float:
        """CD-median: quantile(0.5). Reparametrisation-invariant."""
        return float(self.quantile(0.5))

    def mode(self) -> float:
        """CD-mode: argmax of pdf. Returns the leftmost peak when ties."""
        idx = int(jnp.argmax(jnp.asarray(self.pdf_values)))
        return float(self.theta_grid[idx])

    def secondary_modes(self, *, prominence_frac: float = 0.1) -> list[float]:
        """Local maxima other than the global mode whose prominence
        (peak height − adjacent valley) exceeds `prominence_frac` of the
        global maximum. Useful for detecting multimodal CDs."""
        pdf = self.pdf_values
        if pdf.size < 3:
            return []
        # Identify all interior local maxima.
        is_peak = (pdf[1:-1] > pdf[:-2]) & (pdf[1:-1] > pdf[2:])
        peak_idx = np.where(is_peak)[0] + 1
        if peak_idx.size == 0:
            return []
        global_max = float(pdf.max())
        if global_max <= 0:
            return []
        threshold = prominence_frac * global_max
        global_idx = int(np.argmax(pdf))
        out: list[float] = []
        for i in peak_idx:
            if i == global_idx:
                continue
            # Approximate prominence: peak value minus the lower of the
            # adjacent interior minima (or grid endpoint).
            left_min = float(pdf[:i].min()) if i > 0 else 0.0
            right_min = float(pdf[i:].min()) if i < pdf.size else 0.0
            valley = max(left_min, right_min)
            if (pdf[i] - valley) >= threshold:
                out.append(float(self.theta_grid[i]))
        return out

    # ----- Validity / diagnostics -----

    def is_monotone_inversion(self) -> bool:
        """True iff `signed_confidence` is monotone non-decreasing,
        or absent (no inversion data)."""
        if self.signed_confidence is None:
            return True
        return bool(np.all(np.diff(self.signed_confidence) >= -1e-12))

    def validate(self) -> list[CDValidityIssue]:
        """Return a list of flagged validity issues. Non-fatal."""
        issues: list[CDValidityIssue] = []

        # Shape consistency.
        if self.theta_grid.size != self.pdf_values.size:
            issues.append(
                CDValidityIssue(
                    code="shape-mismatch",
                    message=(
                        f"theta_grid (size {self.theta_grid.size}) and "
                        f"pdf_values (size {self.pdf_values.size}) "
                        f"must match"
                    ),
                    severity="error",
                )
            )
            return issues

        # pdf non-negativity.
        min_pdf = float(self.pdf_values.min())
        if min_pdf < -1e-9:
            issues.append(
                CDValidityIssue(
                    code="negative-pdf",
                    message=f"pdf has negative values; min = {min_pdf:.3e}",
                    severity="error",
                )
            )

        # Integration mass.
        mass = float(
            jnp.trapezoid(
                jnp.asarray(self.pdf_values, dtype=jnp.float64),
                jnp.asarray(self.theta_grid, dtype=jnp.float64),
            )
        )
        if abs(mass - 1.0) > 1e-2:
            issues.append(
                CDValidityIssue(
                    code="mass-not-unit",
                    message=f"pdf integrates to {mass:.4f}, not ≈ 1",
                    severity="warning",
                )
            )

        # Strictly increasing theta_grid.
        if not np.all(np.diff(self.theta_grid) > 0):
            issues.append(
                CDValidityIssue(
                    code="non-monotone-grid",
                    message="theta_grid must be strictly increasing",
                    severity="error",
                )
            )

        # Signed-confidence monotonicity (the smoothness-pathology flag).
        if self.signed_confidence is not None and not self.is_monotone_inversion():
            issues.append(
                CDValidityIssue(
                    code="non-monotone-signed-confidence",
                    message=(
                        "inversion-based C(θ) is non-monotone; the "
                        "underlying p-value is multimodal (e.g. Dyn-WALDO "
                        "under conflict). Distance metrics still work on "
                        "the density-derived cdf, but this flags the "
                        "diagnostic for downstream consumers."
                    ),
                    severity="warning",
                )
            )

        return issues
