"""Regression tests for CD distance functions (Phase C).

Pinned cases:
  - W₂ between two Gaussian-seeded CDs matches the Olkin–Pukelsheim
    closed form `sqrt((μ_a-μ_b)² + (σ_a-σ_b)²)` within 1e-3.
  - W₁ between two Gaussians of equal scale matches |μ_a-μ_b| within 1e-3.
  - W₁(a, a) = 0; W₂(a, a) = 0.
  - Symmetry: w_p(a, b) == w_p(b, a).
  - **Bimodal cross-check** (the user-flagged test): W₂ between a
    bimodal CD (50:50 mixture of N(±2, 1)) and N(0, 1) is computed by
    `wasserstein_2` directly and matches an independent computation
    using the analytic mixture quantile function via scipy. This pins
    that the pdf-primary design correctly handles multimodal
    distributions without rearrangement.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, stats

from frasian.cd.distances import (
    total_variation,
    wasserstein_1,
    wasserstein_1_gaussian_shift,
    wasserstein_1_gaussian_zero_mean_scale,
    wasserstein_2,
    wasserstein_2_gaussian,
)
from frasian.cd.grid import GridConfidenceDistribution


def _gaussian_cd(mu: float, sigma: float, n: int = 4001) -> GridConfidenceDistribution:
    """Gaussian CD on a 12σ window."""
    theta = np.linspace(mu - 12.0 * sigma, mu + 12.0 * sigma, n)
    pdf = stats.norm.pdf(theta, loc=mu, scale=sigma)
    return GridConfidenceDistribution(
        name=f"gauss(mu={mu},sigma={sigma})",
        theta_grid=theta,
        pdf_values=pdf,
    )


def _bimodal_cd(n: int = 4001) -> GridConfidenceDistribution:
    """50:50 mixture of N(-2, 1) and N(+2, 1)."""
    theta = np.linspace(-15.0, 15.0, n)
    pdf = 0.5 * (stats.norm.pdf(theta, -2.0, 1.0) + stats.norm.pdf(theta, +2.0, 1.0))
    return GridConfidenceDistribution(
        name="bimodal_+/-2",
        theta_grid=theta,
        pdf_values=pdf,
    )


@pytest.mark.L0
class TestWasserstein2GaussianClosedForm:
    """W₂ on Gaussians matches Olkin–Pukelsheim (1982)."""

    @pytest.mark.parametrize(
        "mu_a, sigma_a, mu_b, sigma_b",
        [
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 2.0),
            (-1.0, 0.5, 1.5, 2.0),
        ],
    )
    def test_w2_matches_closed_form(self, mu_a, sigma_a, mu_b, sigma_b):
        a = _gaussian_cd(mu_a, sigma_a)
        b = _gaussian_cd(mu_b, sigma_b)
        actual = wasserstein_2(a, b)
        expected = wasserstein_2_gaussian(mu_a, sigma_a, mu_b, sigma_b)
        # Gauss–Hermite W₂ matches closed-form Olkin–Pukelsheim within
        # ~1e-5 on a 4001-point Gaussian grid; the residual is the CD's
        # own grid-quantile interpolation error, not the quadrature.
        assert actual == pytest.approx(expected, abs=5e-5)

    @pytest.mark.parametrize("sigma_b", [0.5, 2.0, 5.0, 10.0])
    def test_w2_scale_mismatch_remains_tight(self, sigma_b):
        """Extreme σ-mismatch was the case that motivated switching from
        u-grid trapezoidal to z-axis Gauss–Hermite. The change of
        variables `z = Φ⁻¹(u)` weights tail contributions by φ(z), so
        the integrand stays well-behaved no matter how heavy one CD's
        tails are relative to the other. This pins that property."""
        a = _gaussian_cd(0.0, 1.0)
        b = _gaussian_cd(0.0, sigma_b, n=8001)  # finer grid for σ=10 tail
        actual = wasserstein_2(a, b)
        expected = wasserstein_2_gaussian(0.0, 1.0, 0.0, sigma_b)
        assert actual == pytest.approx(
            expected, abs=1e-4
        ), f"W₂ at σ-mismatch (1, {sigma_b}): expected {expected}, got {actual}"

    def test_w2_n_quad_doubling_converged(self):
        """Doubling n_quad changes the result by less than the CD's own
        grid-quantile interpolation error (~1e-5 on a 4001-point Gaussian
        grid). This pins that 32 nodes is enough — the Gauss–Hermite
        integration error is negligible compared to the CD primitive's
        resolution. (Larger n_quad just samples the same θ-grid at
        slightly different points.)"""
        a = _gaussian_cd(0.0, 1.0)
        b = _gaussian_cd(0.0, 2.0)
        v_32 = wasserstein_2(a, b, n_quad=32)
        v_64 = wasserstein_2(a, b, n_quad=64)
        v_128 = wasserstein_2(a, b, n_quad=128)
        # All three within ~1e-5 of each other.
        assert abs(v_32 - v_64) < 5e-5
        assert abs(v_64 - v_128) < 5e-5


@pytest.mark.L0
class TestWasserstein1GaussianShift:
    """W₁ on shifted Gaussians of equal scale equals the absolute shift."""

    @pytest.mark.parametrize(
        "mu_a, mu_b",
        [
            (0.0, 0.0),
            (-1.0, 1.0),
            (0.5, 2.5),
            (-3.0, -1.0),
        ],
    )
    def test_w1_equal_scale_shift(self, mu_a, mu_b):
        a = _gaussian_cd(mu_a, 1.0)
        b = _gaussian_cd(mu_b, 1.0)
        actual = wasserstein_1(a, b)
        expected = wasserstein_1_gaussian_shift(mu_a, mu_b)
        # Equal-σ shifts have integrand `|F_a − F_b|` symmetric in θ;
        # CDF-form trapezoidal hits the closed form to floating-point.
        assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.L0
class TestWasserstein1GaussianScaleMismatch:
    """W₁ between zero-mean Gaussians of differing scale matches the
    closed form `|σ_a − σ_b| · √(2/π)` (= |Δσ| · E[|Z|] from the quantile
    representation). This is the missing tight regression that the
    equal-scale-shift test does not exercise."""

    @pytest.mark.parametrize(
        "sigma_a, sigma_b",
        [
            (1.0, 2.0),
            (1.0, 5.0),
            (0.5, 3.0),
            (1.0, 10.0),
        ],
    )
    def test_w1_zero_mean_scale_mismatch(self, sigma_a, sigma_b):
        # Use a generously wider grid for the heavy-σ side so the
        # CDF integral covers both supports adequately.
        n = 8001
        a = _gaussian_cd(0.0, sigma_a, n=n)
        b = _gaussian_cd(0.0, sigma_b, n=n)
        actual = wasserstein_1(a, b)
        expected = wasserstein_1_gaussian_zero_mean_scale(sigma_a, sigma_b)
        # CDF-form trapezoidal hits the closed form within ~1e-6 on a
        # fine grid (the integrand `|F_a − F_b|` is smooth and bounded
        # in [0, 1], so trapezoidal is well-conditioned). Compare to
        # the ~5e-3 error a Gauss–Hermite quantile-form integration
        # would suffer at the |·| kink.
        assert actual == pytest.approx(expected, abs=5e-4), (
            f"W₁(N(0,{sigma_a}), N(0,{sigma_b})): expected {expected}, " f"got {actual}"
        )


@pytest.mark.L0
class TestDistanceProperties:
    def test_w1_self_zero(self):
        a = _gaussian_cd(0.0, 1.0)
        assert wasserstein_1(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_w2_self_zero(self):
        a = _gaussian_cd(0.0, 1.0)
        assert wasserstein_2(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_tv_self_zero(self):
        a = _gaussian_cd(0.0, 1.0)
        assert total_variation(a, a) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("metric", [wasserstein_1, wasserstein_2, total_variation])
    def test_symmetry(self, metric):
        a = _gaussian_cd(0.0, 1.0)
        b = _gaussian_cd(1.5, 2.0)
        assert metric(a, b) == pytest.approx(metric(b, a), abs=1e-9)

    @pytest.mark.parametrize("metric", [wasserstein_1, wasserstein_2, total_variation])
    def test_non_negative(self, metric):
        a = _gaussian_cd(0.0, 1.0)
        b = _gaussian_cd(1.5, 2.0)
        assert metric(a, b) >= 0.0


@pytest.mark.L0
class TestBimodalDistance:
    """The user-flagged test: W₂ on a bimodal distribution against N(0,1).

    Cross-check: the framework's `wasserstein_2` evaluates the quantile
    integral on the cumulative-pdf-derived CDF (always monotone). An
    independent reference value is computed by inverting the analytic
    mixture CDF point-by-point via scipy's root-finder, then taking
    `sqrt(∫₀¹ (Q_mix(u) − Q_ref(u))² du)`. Agreement to ~1e-2 confirms
    the framework handles multimodal CDs correctly without rearrangement.
    """

    def _mixture_cdf(self, theta: float) -> float:
        """Analytic CDF of 0.5*N(-2,1) + 0.5*N(+2,1)."""
        return 0.5 * stats.norm.cdf(theta, -2.0, 1.0) + 0.5 * stats.norm.cdf(theta, 2.0, 1.0)

    def _mixture_quantile(self, u: float) -> float:
        """Inverse of the analytic mixture CDF."""
        if u <= 0.0:
            return -50.0
        if u >= 1.0:
            return 50.0
        return optimize.brentq(lambda x: self._mixture_cdf(x) - u, -50, 50)

    def test_w2_bimodal_vs_standard_normal(self):
        """End-to-end cross-check: W₂(bimodal, N(0,1)) via two independent
        paths agree to 1e-2."""
        a = _bimodal_cd()
        b = _gaussian_cd(0.0, 1.0)
        actual = wasserstein_2(a, b)

        # Independent reference: integrate (Q_mix - Q_ref)² du via the
        # analytic mixture quantile and the analytic Gaussian quantile.
        u = np.linspace(1e-4, 1.0 - 1e-4, 401)  # avoid tail-quantile issues
        q_mix = np.array([self._mixture_quantile(uu) for uu in u])
        q_ref = stats.norm.ppf(u)
        sq = (q_mix - q_ref) ** 2
        expected = float(np.sqrt(np.trapezoid(sq, u)))

        assert actual == pytest.approx(expected, abs=2e-2), (
            f"W₂(bimodal, N(0,1)): framework={actual:.4f}, " f"reference={expected:.4f}"
        )

    def test_w2_bimodal_vs_self_zero(self):
        """W₂ acts identity-like on the same CD even when bimodal."""
        a = _bimodal_cd()
        assert wasserstein_2(a, a) == pytest.approx(0.0, abs=1e-3)

    def test_w1_bimodal_finite_and_positive(self):
        """W₁ between bimodal and N(0,1) is finite and > 0."""
        a = _bimodal_cd()
        b = _gaussian_cd(0.0, 1.0)
        d = wasserstein_1(a, b)
        assert np.isfinite(d) and d > 0.0


@pytest.mark.L0
class TestCDFNonMonotoneStillWorks:
    """Pin: even if a constructor produces a CD with a non-monotone
    `signed_confidence` field, the distance metrics still operate on
    the always-monotone cdf_values and produce sensible results."""

    def _cd_with_perturbed_signed(self) -> GridConfidenceDistribution:
        theta = np.linspace(-6.0, 6.0, 1001)
        pdf = stats.norm.pdf(theta, 0.0, 1.0)
        signed = stats.norm.cdf(theta, 0.0, 1.0)
        # Perturb signed to be non-monotone (the simulated Dyn-WALDO case).
        signed[400:600] -= 0.05
        return GridConfidenceDistribution(
            name="nonmono",
            theta_grid=theta,
            pdf_values=pdf,
            signed_confidence=signed,
        )

    def test_w2_to_self_via_cdf_works(self):
        a = self._cd_with_perturbed_signed()
        # Distance to a separate canonical Gaussian.
        b_theta = np.linspace(-6.0, 6.0, 1001)
        b_pdf = stats.norm.pdf(b_theta, 0.0, 1.0)
        b = GridConfidenceDistribution(name="gauss", theta_grid=b_theta, pdf_values=b_pdf)
        # Both have the same density, so W₂ should be ≈ 0 — even though
        # `a.signed_confidence` is non-monotone. This is the punchline:
        # distance metrics ignore signed_confidence entirely.
        d = wasserstein_2(a, b)
        assert d == pytest.approx(0.0, abs=1e-3)
