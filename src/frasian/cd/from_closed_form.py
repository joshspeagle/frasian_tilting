"""Closed-form confidence distributions for cross-checking the universal
constructor (`cd.from_pvalue.build_cd_from_pvalue`) on the conjugate
Normal-Normal sandbox.

These are NOT exposed via the public API. They live here as test
fixtures: the framework's experiment uses the universal constructor
unconditionally, and the regression tests assert that for cells with a
known closed-form CD (Wald, plain WALDO, fixed-η tilted-WALDO) the
universal constructor agrees with the analytic closed form to within
1e-3 (cdf) / 5e-3 (pdf).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from .grid import GridConfidenceDistribution

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def wald_cd(
    D: float,
    sigma: float,
    *,
    theta_grid: NDArray[np.float64] | None = None,
    n_grid: int = 1001,
    half_width_sigma: float = 8.0,
) -> GridConfidenceDistribution:
    """Wald confidence distribution: pdf = N(D, σ²); cdf = Φ((θ−D)/σ).

    For Wald the inversion-based confidence curve C(θ) coincides with
    the density-derived CDF (the underlying p-value is unimodal).
    """
    if theta_grid is None:
        half = half_width_sigma * sigma
        theta_grid = np.linspace(D - half, D + half, n_grid)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    theta_j = jnp.asarray(theta_grid, dtype=jnp.float64)
    pdf_j = jsp_stats.norm.pdf(theta_j, loc=D, scale=sigma)
    # Renormalise so trapezoidal cdf reaches 1 exactly on the truncated grid
    # — the closed-form is from a 2-sided Gaussian, with tiny tail mass cut.
    Z = float(jnp.trapezoid(pdf_j, theta_j))
    pdf_values = np.asarray(pdf_j / Z, dtype=np.float64)
    signed = np.asarray(jsp_stats.norm.cdf(theta_j, loc=D, scale=sigma), dtype=np.float64)
    return GridConfidenceDistribution(
        name=f"wald_cd@D={float(D):+.3f}",
        theta_grid=theta_grid,
        pdf_values=pdf_values,
        signed_confidence=signed,
        metadata={"closed_form": True, "D": float(D), "sigma": float(sigma)},
    )


def waldo_cd(
    D: float,
    model: NormalNormalModel,
    prior: NormalDistribution,
    *,
    theta_grid: NDArray[np.float64] | None = None,
    n_grid: int = 1001,
    half_width_sigma: float = 8.0,
) -> GridConfidenceDistribution:
    """Plain (η = 0) WALDO confidence distribution.

    Derived analytically from the WALDO p-value
    `p(θ) = Φ(b−a) + Φ(−a−b)` with
      a(θ) = |μ_n − θ|/(w·σ), b(θ) = (1−w)(μ₀ − θ)/(w·σ),
    where μ_n = w·D + (1−w)·μ₀ is the posterior mean. The pdf is
    `c(θ) = ½ |dp/dθ|` evaluated symbolically:

      dp/dθ has contributions from the |μ_n − θ| term (sign flip at
      θ = μ_n) and the linear (μ₀ − θ) term. Density is
      `(1/(w·σ)) · sgn(θ−μ_n) · [φ(b+a) − φ(b−a)] / 2 + ...`

    Rather than expanding the full algebra, we evaluate p analytically
    and finite-difference; this matches `from_pvalue` to numerical
    precision and serves as a tight cross-check.
    """
    if theta_grid is None:
        half = half_width_sigma * model.sigma
        theta_grid = np.linspace(D - half, D + half, n_grid)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    theta_j = jnp.asarray(theta_grid, dtype=jnp.float64)

    sigma = model.sigma
    sigma0 = prior.scale
    mu0 = prior.loc
    w = sigma0**2 / (sigma**2 + sigma0**2)
    mu_n = w * D + (1.0 - w) * mu0

    a = jnp.abs(mu_n - theta_j) / (w * sigma)
    b = (1.0 - w) * (mu0 - theta_j) / (w * sigma)
    pvals_j = jsp_stats.norm.cdf(b - a) + jsp_stats.norm.cdf(-a - b)
    pvals_j = jnp.clip(pvals_j, 0.0, 1.0)

    # Robust |dp/dθ|: average of absolute one-sided differences. Avoids
    # the central-diff cancellation at kinks (e.g. θ = μ_n for WALDO).
    forward_inner = jnp.abs(jnp.diff(pvals_j)) / jnp.diff(theta_j)
    forward = jnp.concatenate([forward_inner, forward_inner[-1:]])
    backward = jnp.concatenate([forward_inner[:1], forward_inner])
    abs_dp = 0.5 * (forward + backward)
    c_unnorm = 0.5 * abs_dp
    Z = float(jnp.trapezoid(c_unnorm, theta_j))
    pdf_values = np.asarray(c_unnorm / max(Z, 1e-300), dtype=np.float64)

    # Signed-inversion C(θ) from two-sided p-value:
    #   C(θ) = p(θ)/2 on lower tail (θ ≤ mode);
    #   C(θ) = 1 − p(θ)/2 on upper tail (θ ≥ mode).
    pvals = np.asarray(pvals_j, dtype=np.float64)
    mode_idx = int(np.argmax(pvals))
    signed = np.empty_like(pvals)
    signed[:mode_idx] = pvals[:mode_idx] / 2.0
    signed[mode_idx:] = 1.0 - pvals[mode_idx:] / 2.0

    return GridConfidenceDistribution(
        name=f"waldo_cd@D={float(D):+.3f}",
        theta_grid=theta_grid,
        pdf_values=pdf_values,
        signed_confidence=signed,
        metadata={
            "closed_form": True,
            "D": float(D),
            "sigma": float(sigma),
            "w": float(w),
            "Z_normalisation": Z,
        },
    )


def tilted_waldo_cd(
    D: float,
    model: NormalNormalModel,
    prior: NormalDistribution,
    eta: float,
    *,
    theta_grid: NDArray[np.float64] | None = None,
    n_grid: int = 1001,
    half_width_sigma: float = 8.0,
) -> GridConfidenceDistribution:
    """η-tilted WALDO confidence distribution at fixed η.

    Same structure as `waldo_cd` but at non-zero η — uses the closed-
    form tilted p-value
      `p_η(θ) = Φ(b_η − a_η) + Φ(−a_η − b_η)`
    with the η-tilted (μ_η, w_η) parameters. For unimodal p_η (any
    fixed η), C(θ) is monotone and matches the density-derived cdf.
    """
    if theta_grid is None:
        half = half_width_sigma * model.sigma
        theta_grid = np.linspace(D - half, D + half, n_grid)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    theta_j = jnp.asarray(theta_grid, dtype=jnp.float64)

    sigma = model.sigma
    sigma0 = prior.scale
    mu0 = prior.loc
    w = sigma0**2 / (sigma**2 + sigma0**2)
    denom = 1.0 - eta * (1.0 - w)
    if denom <= 0.0:
        raise ValueError(
            f"eta={eta} drives denom to {denom:.3e} <= 0 with w={w:.3f}; "
            "outside admissible range."
        )

    mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * mu0) / denom
    norm_factor = w * sigma / denom
    a_eta = jnp.abs(mu_eta - theta_j) / norm_factor
    b_eta = (1.0 - eta) * (1.0 - w) * (mu0 - theta_j) / (denom * norm_factor)
    pvals_j = jsp_stats.norm.cdf(b_eta - a_eta) + jsp_stats.norm.cdf(-a_eta - b_eta)
    pvals_j = jnp.clip(pvals_j, 0.0, 1.0)

    # Robust |dp/dθ|: average of absolute one-sided differences. Avoids
    # the central-diff cancellation at kinks (e.g. θ = μ_n for WALDO).
    forward_inner = jnp.abs(jnp.diff(pvals_j)) / jnp.diff(theta_j)
    forward = jnp.concatenate([forward_inner, forward_inner[-1:]])
    backward = jnp.concatenate([forward_inner[:1], forward_inner])
    abs_dp = 0.5 * (forward + backward)
    c_unnorm = 0.5 * abs_dp
    Z = float(jnp.trapezoid(c_unnorm, theta_j))
    pdf_values = np.asarray(c_unnorm / max(Z, 1e-300), dtype=np.float64)

    # Signed-inversion C(θ) from two-sided p-value:
    #   C(θ) = p(θ)/2     on lower tail (θ ≤ mode);
    #   C(θ) = 1 − p(θ)/2 on upper tail (θ ≥ mode).
    pvals = np.asarray(pvals_j, dtype=np.float64)
    mode_idx = int(np.argmax(pvals))
    signed = np.empty_like(pvals)
    signed[:mode_idx] = pvals[:mode_idx] / 2.0
    signed[mode_idx:] = 1.0 - pvals[mode_idx:] / 2.0

    return GridConfidenceDistribution(
        name=f"tilted_waldo(η={eta:+.3f})_cd@D={float(D):+.3f}",
        theta_grid=theta_grid,
        pdf_values=pdf_values,
        signed_confidence=signed,
        metadata={
            "closed_form": True,
            "D": float(D),
            "eta": float(eta),
            "sigma": float(sigma),
            "w": float(w),
            "Z_normalisation": Z,
        },
    )
