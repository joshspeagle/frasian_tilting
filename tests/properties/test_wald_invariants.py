"""Property tests for the WaldStatistic.

Invariants:
  - p ∈ [0, 1] for all inputs.
  - Under H0, p-values are Uniform[0, 1] (statistical L3).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic

_THETA = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_D = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestWaldInvariants:
    @given(theta=_THETA, D=_D, sigma=_SIGMA)
    @settings(max_examples=100, deadline=None)
    def test_pvalue_in_unit_interval(self, theta, D, sigma):
        model = NormalNormalModel(sigma=sigma)
        p = WaldStatistic().pvalue(theta, np.asarray([D]), model)
        assert 0.0 <= float(p) <= 1.0


@pytest.mark.L3
class TestWaldUniformPvalueUnderH0:
    """Statistical-tier: Wald p-values are Uniform[0,1] under H0."""

    def test_ks_uniform(self):
        rng = np.random.default_rng(42)
        sigma = 1.0
        theta_true = 0.7
        n = 5000
        Ds = rng.normal(loc=theta_true, scale=sigma, size=n)
        model = NormalNormalModel(sigma=sigma)
        ps = np.array(
            [float(WaldStatistic().pvalue(theta_true, np.asarray([D]), model)) for D in Ds]
        )
        ks_stat, ks_p = stats.kstest(ps, "uniform")
        assert ks_p > 0.01, f"KS p-value too low: {ks_p}, ks_stat={ks_stat}"
