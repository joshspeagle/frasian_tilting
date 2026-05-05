"""Regression tests for `GridConfidenceDistribution` (Phase B).

The pdf-primary design means cdf is always derived as the cumulative
trapezoidal integral of `pdf_values`, hence always monotone non-decreasing
even for multimodal pdfs (the bimodal case the user explicitly flagged
as needing a test).

Tests cover:
  - Closed-form match against scipy.stats.norm for unimodal Gaussian pdf.
  - Asymmetric (skew-normal) pdf: mean ≠ median ≠ mode.
  - Bimodal pdf (50:50 mixture of N(-2,1) + N(+2,1)): cdf is monotone,
    quantile is well-defined, secondary_modes detects the second peak,
    interval is finite, mean ≈ 0, median ≈ 0, mode ≈ ±2.
  - Non-monotone signed_confidence: validate() flags it but cdf/quantile
    keep working (they read from the always-monotone cdf_values).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from frasian.cd.grid import GridConfidenceDistribution


def _gaussian_cd(
    name: str, mu: float, sigma: float, n: int = 1001, with_signed: bool = True
) -> GridConfidenceDistribution:
    theta = np.linspace(mu - 12.0 * sigma, mu + 12.0 * sigma, n)
    pdf = stats.norm.pdf(theta, loc=mu, scale=sigma)
    signed = stats.norm.cdf(theta, loc=mu, scale=sigma) if with_signed else None
    return GridConfidenceDistribution(
        name=name,
        theta_grid=theta,
        pdf_values=pdf,
        signed_confidence=signed,
    )


def _bimodal_cd(n: int = 2001) -> GridConfidenceDistribution:
    """50:50 mixture of N(-2, 1) and N(+2, 1)."""
    theta = np.linspace(-12.0, 12.0, n)
    pdf = 0.5 * (
        stats.norm.pdf(theta, loc=-2.0, scale=1.0) + stats.norm.pdf(theta, loc=+2.0, scale=1.0)
    )
    return GridConfidenceDistribution(
        name="bimodal",
        theta_grid=theta,
        pdf_values=pdf,
    )


@pytest.mark.L0
class TestGaussianCDClosedForm:
    """Seeded with a Gaussian pdf, summaries match scipy reference."""

    @pytest.mark.parametrize("mu, sigma", [(0.0, 1.0), (1.5, 0.5), (-2.0, 2.0)])
    def test_mean_median_mode(self, mu, sigma):
        cd = _gaussian_cd("gauss", mu, sigma)
        assert cd.mean() == pytest.approx(mu, abs=1e-3)
        assert cd.median() == pytest.approx(mu, abs=1e-3)
        assert cd.mode() == pytest.approx(mu, abs=2e-2)  # grid-spacing tol

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.32])
    def test_interval(self, alpha):
        mu, sigma = 0.5, 1.0
        cd = _gaussian_cd("gauss", mu, sigma)
        lo, hi = cd.interval(alpha)
        z = stats.norm.ppf(1 - alpha / 2)
        assert lo == pytest.approx(mu - z * sigma, abs=2e-3)
        assert hi == pytest.approx(mu + z * sigma, abs=2e-3)

    def test_quantile_matches_scipy(self):
        mu, sigma = 0.0, 1.0
        cd = _gaussian_cd("gauss", mu, sigma)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert cd.quantile(q) == pytest.approx(stats.norm.ppf(q), abs=2e-3)

    def test_pdf_matches_scipy(self):
        mu, sigma = 0.0, 1.0
        cd = _gaussian_cd("gauss", mu, sigma, n=4001)
        thetas = np.linspace(-3, 3, 13)
        np.testing.assert_allclose(
            cd.pdf(thetas),
            stats.norm.pdf(thetas, mu, sigma),
            atol=1e-3,
        )

    def test_cdf_is_monotone(self):
        cd = _gaussian_cd("gauss", 0.0, 1.0)
        diffs = np.diff(cd.cdf_values)
        assert (diffs >= -1e-12).all()

    def test_cdf_endpoints(self):
        cd = _gaussian_cd("gauss", 0.0, 1.0)
        assert cd.cdf_values[0] == pytest.approx(0.0, abs=1e-6)
        assert cd.cdf_values[-1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.L0
class TestSkewNormalCD:
    """Asymmetric pdf: mean ≠ median ≠ mode."""

    def test_skew_normal_summaries_differ(self):
        a = 4.0  # skew parameter
        theta = np.linspace(-6.0, 8.0, 2001)
        pdf = stats.skewnorm.pdf(theta, a, loc=0.0, scale=1.0)
        cd = GridConfidenceDistribution(
            name="skew",
            theta_grid=theta,
            pdf_values=pdf,
        )

        # All three should be finite and pairwise distinct.
        m, med, mo = cd.mean(), cd.median(), cd.mode()
        for v in (m, med, mo):
            assert np.isfinite(v)
        # Mean > median > mode for right-skewed (positive a).
        assert m > med > mo, f"expected mean > median > mode; got {m, med, mo}"

        # Cross-check against scipy reference values within tolerance.
        ref_mean = stats.skewnorm.mean(a)
        ref_median = stats.skewnorm.median(a)
        assert m == pytest.approx(ref_mean, abs=1e-2)
        assert med == pytest.approx(ref_median, abs=1e-2)


@pytest.mark.L0
class TestBimodalCD:
    """User-flagged: bimodal pdf must work end-to-end without
    rearrangement. cdf is monotone (derived from non-negative pdf),
    quantile is well-defined, secondary_modes finds the second peak."""

    def test_cdf_monotone_for_bimodal(self):
        cd = _bimodal_cd()
        diffs = np.diff(cd.cdf_values)
        assert (diffs >= -1e-12).all(), "cdf must be monotone even for bimodal pdf"

    def test_pdf_integrates_to_one(self):
        cd = _bimodal_cd()
        mass = np.trapezoid(cd.pdf_values, cd.theta_grid)
        assert mass == pytest.approx(1.0, abs=1e-3)

    def test_mean_is_zero_by_symmetry(self):
        cd = _bimodal_cd()
        assert cd.mean() == pytest.approx(0.0, abs=1e-2)

    def test_median_is_zero_by_symmetry(self):
        cd = _bimodal_cd()
        assert cd.median() == pytest.approx(0.0, abs=1e-2)

    def test_mode_is_one_of_the_peaks(self):
        cd = _bimodal_cd()
        mode = cd.mode()
        # Symmetry breaks the tie via grid index; either peak is correct.
        assert abs(abs(mode) - 2.0) < 0.05

    def test_secondary_modes_detects_other_peak(self):
        cd = _bimodal_cd()
        secondary = cd.secondary_modes(prominence_frac=0.1)
        assert (
            len(secondary) >= 1
        ), f"expected at least one secondary mode for bimodal pdf; got {secondary}"
        # Combined with primary mode, both peaks (~ ±2) accounted for.
        all_peaks = sorted([cd.mode()] + secondary)
        assert min(all_peaks) < -1.5
        assert max(all_peaks) > 1.5

    def test_quantile_well_defined_for_bimodal(self):
        cd = _bimodal_cd()
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
            v = cd.quantile(q)
            assert np.isfinite(v)
        # Monotone in q.
        qs = np.linspace(0.01, 0.99, 50)
        vs = cd.quantile(qs)
        assert (np.diff(vs) >= -1e-9).all()

    def test_interval_for_bimodal_is_wide(self):
        """The 95% interval of a bimodal CD with peaks at ±2, scale 1
        each, must straddle both peaks."""
        cd = _bimodal_cd()
        lo, hi = cd.interval(0.05)
        assert lo < -2.0 < 2.0 < hi


@pytest.mark.L0
class TestNonMonotoneSignedConfidence:
    """`validate()` flags non-monotonicity in `signed_confidence`, but
    pdf/cdf/quantile/interval keep working because they read from the
    always-monotone density-derived `cdf_values`."""

    def _build_with_nonmonotone_signed(self) -> GridConfidenceDistribution:
        """Build a CD whose pdf is well-behaved (Gaussian) but whose
        signed_confidence has been intentionally perturbed to be
        non-monotone — simulates the Dyn-WALDO multimodal-p case."""
        theta = np.linspace(-6.0, 6.0, 1001)
        pdf = stats.norm.pdf(theta, loc=0.0, scale=1.0)
        signed = stats.norm.cdf(theta, loc=0.0, scale=1.0)
        # Perturb a small interior region downward so signed becomes non-monotone.
        signed[400:600] -= 0.05
        return GridConfidenceDistribution(
            name="nonmono",
            theta_grid=theta,
            pdf_values=pdf,
            signed_confidence=signed,
        )

    def test_validate_flags_non_monotone(self):
        cd = self._build_with_nonmonotone_signed()
        issues = cd.validate()
        codes = [i.code for i in issues]
        assert "non-monotone-signed-confidence" in codes

    def test_is_monotone_inversion_returns_false(self):
        cd = self._build_with_nonmonotone_signed()
        assert cd.is_monotone_inversion() is False

    def test_cdf_still_monotone(self):
        cd = self._build_with_nonmonotone_signed()
        diffs = np.diff(cd.cdf_values)
        assert (diffs >= -1e-12).all()

    def test_quantile_still_works(self):
        cd = self._build_with_nonmonotone_signed()
        for q in [0.1, 0.5, 0.9]:
            v = cd.quantile(q)
            assert np.isfinite(v)

    def test_interval_still_works(self):
        cd = self._build_with_nonmonotone_signed()
        lo, hi = cd.interval(0.05)
        assert lo < hi


@pytest.mark.L0
class TestValidateBoundaryCases:
    def test_clean_gaussian_no_issues(self):
        cd = _gaussian_cd("gauss", 0.0, 1.0)
        issues = cd.validate()
        # signed_confidence is the proper Gaussian cdf — monotone, no issues.
        assert all(i.severity == "warning" for i in issues), issues

    def test_negative_pdf_is_flagged(self):
        theta = np.linspace(-3, 3, 11)
        pdf = stats.norm.pdf(theta, 0.0, 1.0)
        pdf[5] = -0.01  # perturb negative
        cd = GridConfidenceDistribution(
            name="bad",
            theta_grid=theta,
            pdf_values=pdf,
        )
        issues = cd.validate()
        codes = [i.code for i in issues]
        assert "negative-pdf" in codes
