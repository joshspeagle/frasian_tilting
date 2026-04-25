"""Property tests for the WaldoStatistic.

Invariants:
  - p ∈ [0, 1] for all inputs.
  - p-value at theta = mu_n (the WALDO mode) equals 1.
  - p-value is continuous in theta (no infinite jumps); the *Lipschitz
    constant* is the load-bearing object of study and is quantified by
    the Step-5 smoothness diagnostic, not pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel, posterior_params
from frasian.statistics.waldo import WaldoStatistic

_THETA = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_D = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_SIGMA0 = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestWaldoInvariants:
    @given(theta=_THETA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0)
    @settings(max_examples=100, deadline=None)
    def test_pvalue_in_unit_interval(self, theta, D, sigma, sigma0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        p = WaldoStatistic().pvalue(theta, np.asarray([D]), model, prior)
        assert 0.0 <= float(p) <= 1.0

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0)
    @settings(max_examples=80, deadline=None)
    def test_pvalue_at_mode_equals_one(self, D, sigma, sigma0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        mu_n, _, _ = posterior_params(D, 0.0, sigma, sigma0)
        p = WaldoStatistic().pvalue(float(mu_n), np.asarray([D]), model, prior)
        np.testing.assert_allclose(p, 1.0, atol=1e-10)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0)
    @settings(max_examples=40, deadline=None)
    def test_pvalue_no_jumps_in_theta(self, D, sigma, sigma0):
        """Continuity: no infinite-magnitude jumps on a fine grid.

        The *Lipschitz constant* of the WALDO p-value scales as
        1/(w*sigma), which is itself the central object of study —
        sharp transitions in the derivative are exactly what the Step-5
        smoothness diagnostic quantifies. Here we verify only that the
        function is continuous (no impossible jumps).
        """
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        thetas = np.linspace(-6, 6, 4001)
        ps = WaldoStatistic().pvalue(thetas, np.asarray([D]), model, prior)
        assert np.max(np.abs(np.diff(ps))) < 0.5
