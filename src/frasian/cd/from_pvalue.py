"""Universal CD constructor: Schweder–Hjort density from a tilted p-value.

Given any `(tilting, statistic)` cell that has a `tilting.pvalue(...)`
method, this constructor builds the corresponding confidence
distribution by:

  1. Evaluating `p(θ) = tilting.pvalue(θ_grid, [D], model, prior, statistic)`
     on a fine θ-grid.
  2. Computing the unnormalised density `c̃(θ) = ½ |dp/dθ|` via the
     averaged-absolute-one-sided-differences scheme (kink-robust at
     θ = D for Wald and θ = μ_n for WALDO; central differences cancel
     across those kinks and produce a false zero at the peak of |dp/dθ|).
     `½` is the Schweder-Hjort factor that makes c̃ integrate to ≈ 1
     for unimodal p (p ranges 0 → 1 → 0, so `∫|dp/dθ| dθ = 2`).
  3. Normalising: `Z = ∫ c̃(θ) dθ`, `pdf = c̃ / Z`. For unimodal p,
     Z ≈ 1 exactly. For multimodal p (Dyn-WALDO under conflict, where
     |dp/dθ| has a higher total variation than 2), Z > 1 and the pdf
     is renormalised to a proper probability density. **The renormalised
     pdf is what the framework treats as the CD density** — the
     distance metrics operate on the resulting probability distribution.
  4. Constructing the inversion-based `signed_confidence` from the
     two-sided p-value `p(θ) = 2·min(C(θ), 1−C(θ))`:
        C(θ) = p(θ)/2     on the lower tail (θ ≤ mode_p; C ∈ [0, ½])
        C(θ) = 1 − p(θ)/2 on the upper tail (θ ≥ mode_p; C ∈ [½, 1])
     where `mode_p` is the θ at which p attains its global maximum.
     For unimodal p, this is exactly the conventional confidence
     CDF; for multimodal p, C(θ) is non-monotone and is preserved as
     a *diagnostic* quantity (the smoothness pathology surfacing
     directly in the CD's signed_confidence). It is NOT used by the
     derived `cdf_values` (which are always monotone, derived from
     pdf via cumulative integration).

This is the only constructor used in the framework's experiments.
`cd/from_closed_form.py` provides analytic Gaussian CDs as test
fixtures only.

References
----------
Schweder, T. & Hjort, N. L. (2002). "Confidence and Likelihood."
*Scand. J. Stat.* 29: 309–332. — defines `c(θ) = ½|dp/dθ|`.

Singh, K., Xie, M. & Strawderman, W. E. (2007). "Confidence
distribution (CD) — distribution estimator of a parameter." *IMS
Lecture Notes — Monograph Series* 54: 132–150.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from ..models.base import Model, Prior
from ..statistics.base import TestStatistic
from ..tilting.base import TiltingScheme
from .grid import GridConfidenceDistribution

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _default_theta_grid(
    D: float, sigma: float, *, n: int = 1001, half_width_sigma: float = 8.0
) -> NDArray[np.float64]:
    """θ-grid centred on D, half-width `half_width_sigma · sigma`."""
    half = half_width_sigma * sigma
    return np.linspace(D - half, D + half, n)


def _signed_confidence_curve(
    theta_grid: NDArray[np.float64], pvalues: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Inversion-based confidence curve C(θ) from a two-sided p-value.

    For a two-sided p-value `p(θ) = 2·min(C(θ), 1−C(θ))`, the inversion is

        C(θ) = p(θ)/2          on the lower tail (θ ≤ mode_p; C ∈ [0, ½])
        C(θ) = 1 − p(θ)/2      on the upper tail (θ ≥ mode_p; C ∈ [½, 1])

    where `mode_p` is the θ at which p attains its global maximum
    (= 1 for an exact statistic). For unimodal p, C is monotone
    non-decreasing; for multimodal p, the splits at internal valleys
    induce non-monotone segments — preserved verbatim as the
    diagnostic surface (the smoothness pathology in distributional form).
    """
    mode_idx = int(np.argmax(pvalues))
    out = np.empty_like(pvalues)
    out[:mode_idx] = pvalues[:mode_idx] / 2.0
    out[mode_idx:] = 1.0 - pvalues[mode_idx:] / 2.0
    return out


def build_cd_from_pvalue(
    tilting: TiltingScheme,
    statistic: TestStatistic,
    D: float,
    model: Model,
    prior: Prior,
    *,
    theta_grid: NDArray[np.float64] | None = None,
    sigma_for_grid: float | None = None,
    n_grid: int = 1001,
    half_width_sigma: float = 8.0,
    name: str | None = None,
) -> GridConfidenceDistribution:
    """Build a `GridConfidenceDistribution` from `tilting.pvalue`.

    **n=1 implicit convention** (audit P1 I.4): `D` is a scalar
    observation, passed downstream as `data=np.asarray([float(D)])`.
    The Schweder-Hjort density therefore reflects the p-value at a
    single observation — using `σ` rather than `σ/sqrt(n)` for the NN
    closed-form path (the framework's sandbox is n=1), and a single
    likelihood draw for the generic path. Callers wanting CDs from
    multi-observation data should call the underlying `tilting.pvalue`
    on a θ-grid + the multi-observation data array directly and
    construct a `GridConfidenceDistribution` from the resulting
    Schweder-Hjort density.

    Parameters
    ----------
    tilting, statistic, D, model, prior
        The cell + observed datum to build a CD for. `D` is a scalar
        single observation; see the n=1 convention note above.
    theta_grid : ndarray, optional
        Explicit θ-grid. If None, defaults to a 1001-point grid centred
        on D with half-width `half_width_sigma · σ` (σ from `model.sigma`
        when available, else `sigma_for_grid`).
    sigma_for_grid : float, optional
        Fallback σ for the default grid construction when `model.sigma`
        is not exposed.
    n_grid, half_width_sigma : int, float
        Defaults for the auto-grid.
    name : str, optional
        Display name for the resulting CD; defaults to
        `(tilting.name, statistic.name)@D=...`.

    Returns
    -------
    GridConfidenceDistribution
        Always with non-negative, Z-normalised `pdf_values` and an
        inversion-based `signed_confidence` (possibly non-monotone).
    """
    if theta_grid is None:
        sigma = float(getattr(model, "sigma", sigma_for_grid or 1.0))
        theta_grid = _default_theta_grid(
            float(D),
            sigma,
            n=n_grid,
            half_width_sigma=half_width_sigma,
        )
    theta_grid = np.asarray(theta_grid, dtype=np.float64)

    # 1. Evaluate p(θ) on the grid via the tilting's selector-aware method.
    pvals = np.asarray(
        tilting.pvalue(theta_grid, np.asarray([float(D)]), model, prior, statistic),
        dtype=np.float64,
    )
    pvals = np.clip(pvals, 0.0, 1.0)  # numerical guard

    # 2. Schweder-Hjort density c̃ = ½ |dp/dθ|.
    # At kink points (e.g. θ = D for Wald, where |D − θ| is non-smooth)
    # central differences cancel and produce a false zero; we instead
    # average the absolute *one-sided* differences, which agrees with
    # central diffs on smooth regions but recovers the correct |dp/dθ|
    # at kinks.
    pvals_j = jnp.asarray(pvals, dtype=jnp.float64)
    theta_j = jnp.asarray(theta_grid, dtype=jnp.float64)
    forward_inner = jnp.abs(jnp.diff(pvals_j)) / jnp.diff(theta_j)
    # Replicate the legacy boundary handling exactly: forward[-1] = forward[-2],
    # backward[0] = backward[1] (which equals forward[0]).
    forward = jnp.concatenate([forward_inner, forward_inner[-1:]])
    backward = jnp.concatenate([forward_inner[:1], forward_inner])
    abs_dp_dtheta = 0.5 * (forward + backward)
    c_unnorm = 0.5 * abs_dp_dtheta

    # 3. Z-normalise so pdf integrates to 1 exactly.
    Z = float(jnp.trapezoid(c_unnorm, theta_j))
    if Z <= 0.0:
        raise ValueError(
            f"density normalisation Z = {Z:.3e} is non-positive; "
            f"likely the p-value is constant on the grid (no support). "
            f"Try a wider θ-grid or check the tilting/statistic cell."
        )
    pdf_values = np.asarray(c_unnorm / Z, dtype=np.float64)

    # 4. Inversion-based C(θ) — preserved verbatim as auxiliary diagnostic.
    signed = _signed_confidence_curve(theta_grid, pvals)

    cell_name = (
        name
        if name is not None
        else (
            f"({getattr(tilting, 'cell_name', tilting.name)}, "
            f"{statistic.name})@D={float(D):+.3f}"
        )
    )

    return GridConfidenceDistribution(
        name=cell_name,
        theta_grid=theta_grid,
        pdf_values=pdf_values,
        signed_confidence=signed,
        metadata={
            "tilting": getattr(tilting, "cell_name", tilting.name),
            "statistic": statistic.name,
            "D": float(D),
            "n_grid": int(theta_grid.size),
            "Z_normalisation": Z,
        },
    )
