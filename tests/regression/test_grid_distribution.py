"""Test GridDistribution (numerical 1D Distribution on a theta-grid).

Validates against `NormalDistribution` since the construction-from-
log-density path can build any normalisable density and the Normal
case has a closed-form reference for every protocol method.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.tilting._grid_distribution import (
    GridDistribution,
    grid_distribution_from_log_density,
)


@pytest.mark.L0
def test_grid_distribution_normal_reference():
    """Build a GridDistribution from N(0.5, 1.0)'s log-pdf; protocol
    methods agree with the closed-form Normal to atol 1e-3 (linear
    interp on 1024 points)."""
    ref = NormalDistribution(loc=0.5, scale=1.0)
    theta = np.linspace(-6.0, 7.0, 1024)
    log_d = np.asarray(ref.logpdf(theta))
    gd = grid_distribution_from_log_density(theta, log_d)

    assert abs(gd.mean() - ref.mean()) < 1e-9
    assert abs(gd.var() - ref.var()) < 1e-6
    assert abs(float(gd.pdf(0.5)) - float(ref.pdf(0.5))) < 1e-4
    assert abs(float(gd.cdf(1.5)) - float(ref.cdf(1.5))) < 1e-4
    assert abs(float(gd.quantile(0.7)) - float(ref.quantile(0.7))) < 1e-3


@pytest.mark.L0
def test_grid_distribution_sample_matches_target_moments():
    """Inverse-CDF sampling reproduces the target distribution's mean
    and std to ~1/sqrt(n) MC tolerance."""
    ref = NormalDistribution(loc=0.5, scale=1.0)
    theta = np.linspace(-6.0, 7.0, 1024)
    log_d = np.asarray(ref.logpdf(theta))
    gd = grid_distribution_from_log_density(theta, log_d)
    rng = np.random.default_rng(0)
    n = 5000
    samples = gd.sample(rng, n)
    assert abs(samples.mean() - 0.5) < 3.0 / np.sqrt(n)  # ~3-sigma
    assert abs(samples.std() - 1.0) < 5.0 / np.sqrt(n)


@pytest.mark.L0
def test_grid_distribution_rejects_non_finite_normaliser():
    """If the log-density has NaNs or the trapezoidal Z is 0, raise.

    Silent NaN propagation through a GridDistribution downstream is
    the failure mode this guard prevents.
    """
    theta = np.linspace(0.0, 1.0, 16)
    bad_log_d = np.full_like(theta, np.nan)
    with pytest.raises(ValueError, match="non-finite|non-positive"):
        grid_distribution_from_log_density(theta, bad_log_d)


@pytest.mark.L0
def test_grid_distribution_logpdf_outside_support_clipped():
    """Outside the grid, pdf=0 → logpdf clipped to log(1e-300), not -inf
    (so downstream arithmetic doesn't propagate NaN under jit)."""
    ref = NormalDistribution(loc=0.0, scale=1.0)
    theta = np.linspace(-3.0, 3.0, 256)
    gd = grid_distribution_from_log_density(theta, np.asarray(ref.logpdf(theta)))
    far = float(gd.logpdf(100.0))
    assert np.isfinite(far)
    assert far < -600.0  # -log(1e-300) ≈ 690


@pytest.mark.L0
def test_grid_distribution_shape_validation():
    """size mismatch and 1-point grid raise."""
    with pytest.raises(ValueError, match="equal length"):
        GridDistribution(
            theta_grid=np.linspace(0, 1, 5), pdf_values=np.ones(4)
        )
    with pytest.raises(ValueError, match="at least 2 points"):
        GridDistribution(
            theta_grid=np.array([0.5]), pdf_values=np.array([1.0])
        )
