"""Property tests for the `ConfidenceDistribution` protocol.

Concretely tests `GridConfidenceDistribution` (the only concrete impl)
against the protocol invariants from `cd/base.py`. The invariants are
restated and verified across hypothesis-generated parameter values.

Higher-level statistical-validity tests (KS-uniformity of cdf(θ_true)
under H0) live in `test_post_selection_coverage.py` and the upcoming
`test_cd_validity.py` (Phase D); those run on actual `(tilting,
statistic)` cells, not on synthetic CDs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from frasian.cd.grid import GridConfidenceDistribution

_MU = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=2.0, allow_nan=False)
_ALPHA = st.floats(min_value=0.01, max_value=0.5, allow_nan=False)


def _gaussian_cd(mu: float, sigma: float, n: int = 1001) -> GridConfidenceDistribution:
    theta = np.linspace(mu - 12.0 * sigma, mu + 12.0 * sigma, n)
    pdf = stats.norm.pdf(theta, loc=mu, scale=sigma)
    return GridConfidenceDistribution(
        name="gauss",
        theta_grid=theta,
        pdf_values=pdf,
    )


@pytest.mark.L1
@pytest.mark.properties
class TestCDProtocolInvariants:
    """Properties any `ConfidenceDistribution` implementation must satisfy."""

    @given(mu=_MU, sigma=_SIGMA)
    @settings(max_examples=40, deadline=None)
    def test_pdf_non_negative(self, mu, sigma):
        cd = _gaussian_cd(mu, sigma)
        assert (cd.pdf_values >= 0).all()

    @given(mu=_MU, sigma=_SIGMA)
    @settings(max_examples=40, deadline=None)
    def test_cdf_monotone(self, mu, sigma):
        cd = _gaussian_cd(mu, sigma)
        diffs = np.diff(cd.cdf_values)
        # Allow tiny numerical noise from the cumulative integral.
        assert (diffs >= -1e-12).all()

    @given(mu=_MU, sigma=_SIGMA)
    @settings(max_examples=40, deadline=None)
    def test_cdf_endpoints(self, mu, sigma):
        cd = _gaussian_cd(mu, sigma)
        # On a 12σ window, cdf(−12σ) ≈ 0 and cdf(+12σ) ≈ 1.
        assert cd.cdf_values[0] == pytest.approx(0.0, abs=1e-6)
        assert cd.cdf_values[-1] == pytest.approx(1.0, abs=1e-3)

    @given(mu=_MU, sigma=_SIGMA, alpha=_ALPHA)
    @settings(max_examples=30, deadline=None)
    def test_interval_lower_le_upper(self, mu, sigma, alpha):
        cd = _gaussian_cd(mu, sigma)
        lo, hi = cd.interval(alpha)
        assert lo <= hi

    @given(mu=_MU, sigma=_SIGMA, alpha=_ALPHA)
    @settings(max_examples=30, deadline=None)
    def test_interval_matches_quantile_pair(self, mu, sigma, alpha):
        cd = _gaussian_cd(mu, sigma)
        lo, hi = cd.interval(alpha)
        assert lo == pytest.approx(float(cd.quantile(alpha / 2)), abs=1e-9)
        assert hi == pytest.approx(float(cd.quantile(1 - alpha / 2)), abs=1e-9)

    @given(mu=_MU, sigma=_SIGMA)
    @settings(max_examples=30, deadline=None)
    def test_quantile_monotone_in_q(self, mu, sigma):
        cd = _gaussian_cd(mu, sigma)
        qs = np.linspace(0.01, 0.99, 50)
        vs = cd.quantile(qs)
        assert (np.diff(vs) >= -1e-9).all()

    @given(mu=_MU, sigma=_SIGMA)
    @settings(max_examples=30, deadline=None)
    def test_pdf_integrates_to_one(self, mu, sigma):
        cd = _gaussian_cd(mu, sigma)
        mass = float(np.trapezoid(cd.pdf_values, cd.theta_grid))
        assert mass == pytest.approx(1.0, abs=2e-3)

    @given(mu=_MU, sigma=_SIGMA)
    @settings(max_examples=20, deadline=None)
    def test_mean_matches_truth_for_gaussian(self, mu, sigma):
        cd = _gaussian_cd(mu, sigma)
        assert cd.mean() == pytest.approx(mu, abs=2e-3)

    @given(mu=_MU, sigma=_SIGMA)
    @settings(max_examples=20, deadline=None)
    def test_median_matches_truth_for_gaussian(self, mu, sigma):
        cd = _gaussian_cd(mu, sigma)
        assert cd.median() == pytest.approx(mu, abs=2e-3)


@pytest.mark.L1
@pytest.mark.properties
class TestCDValidateAgreesWithBuiltins:
    """`validate()` should be silent for clean Gaussian CDs and noisy for
    obviously-broken CDs."""

    def test_clean_gaussian_no_errors(self):
        cd = _gaussian_cd(0.0, 1.0)
        issues = cd.validate()
        # Some warnings are tolerable (mass-not-unit at coarser grids); no errors.
        for issue in issues:
            assert issue.severity != "error", issue

    def test_negative_pdf_is_error(self):
        theta = np.linspace(-3, 3, 11)
        pdf = stats.norm.pdf(theta, 0.0, 1.0)
        pdf[5] = -0.01
        cd = GridConfidenceDistribution(
            name="bad",
            theta_grid=theta,
            pdf_values=pdf,
        )
        issues = cd.validate()
        assert any(i.code == "negative-pdf" and i.severity == "error" for i in issues)

    def test_shape_mismatch_is_error(self):
        theta = np.linspace(-3, 3, 11)
        pdf = np.zeros(10)  # wrong length
        cd = GridConfidenceDistribution(
            name="mismatched",
            theta_grid=theta,
            pdf_values=pdf,
        )
        issues = cd.validate()
        assert any(i.code == "shape-mismatch" and i.severity == "error" for i in issues)
