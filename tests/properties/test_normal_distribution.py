"""Property tests for NormalDistribution conformance to the Distribution protocol."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frasian.models.distributions import NormalDistribution

# Conservative bounds avoid float overflow and pathological ppf inversions.
_LOC = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_SCALE = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)


@pytest.mark.L1
@pytest.mark.properties
class TestNormalDistributionInvariants:
    @given(loc=_LOC, scale=_SCALE)
    @settings(max_examples=100, deadline=None)
    def test_pdf_nonnegative(self, loc, scale):
        d = NormalDistribution(loc=loc, scale=scale)
        xs = np.linspace(loc - 6 * scale, loc + 6 * scale, 21)
        assert np.all(d.pdf(xs) >= 0.0)

    @given(loc=_LOC, scale=_SCALE)
    @settings(max_examples=100, deadline=None)
    def test_pdf_integrates_to_one(self, loc, scale):
        d = NormalDistribution(loc=loc, scale=scale)
        xs = np.linspace(loc - 12 * scale, loc + 12 * scale, 1001)
        integral = np.trapezoid(d.pdf(xs), xs)
        np.testing.assert_allclose(integral, 1.0, atol=5e-4)

    @given(loc=_LOC, scale=_SCALE)
    @settings(max_examples=100, deadline=None)
    def test_cdf_monotone(self, loc, scale):
        d = NormalDistribution(loc=loc, scale=scale)
        xs = np.linspace(loc - 6 * scale, loc + 6 * scale, 51)
        cdfs = d.cdf(xs)
        assert np.all(np.diff(cdfs) >= -1e-12)

    @given(loc=_LOC, scale=_SCALE, q=st.floats(min_value=1e-3, max_value=1.0 - 1e-3))
    @settings(max_examples=100, deadline=None)
    def test_quantile_round_trip(self, loc, scale, q):
        d = NormalDistribution(loc=loc, scale=scale)
        x = d.quantile(q)
        np.testing.assert_allclose(d.cdf(x), q, atol=1e-9)

    @given(loc=_LOC, scale=_SCALE)
    @settings(max_examples=100, deadline=None)
    def test_mean_var_match_params(self, loc, scale):
        d = NormalDistribution(loc=loc, scale=scale)
        np.testing.assert_allclose(d.mean(), loc, atol=1e-12)
        np.testing.assert_allclose(d.var(), scale**2, atol=1e-12)

    def test_invalid_scale_rejected(self):
        with pytest.raises(ValueError):
            NormalDistribution(loc=0.0, scale=0.0)
        with pytest.raises(ValueError):
            NormalDistribution(loc=0.0, scale=-1.0)
