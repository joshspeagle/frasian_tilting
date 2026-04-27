"""Property tests for the dynamic-η CI machinery.

Pinned invariants from `docs/methods/dynamic_ci_experiment.md` and
`src/frasian/tilting/power_law.py:dynamic_tilted_confidence_interval`:

  - mean_width > 0
  - mean_regions >= 1
  - For Wald cell: width = 2 z sigma exactly, regions = 1.
  - Regions are strictly increasing intervals (no overlap, no inversions).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.eta_selectors import NumericalEtaSelector
from frasian.tilting.power_law import PowerLawTilting


_D = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False)
_SIGMA0 = st.floats(min_value=0.5, max_value=2.0, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestDynamicCIInvariants:
    @given(D=_D, sigma0=_SIGMA0)
    @settings(max_examples=8, deadline=None)
    def test_regions_are_increasing_intervals(self, D, sigma0):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        scheme = PowerLawTilting()
        selector = NumericalEtaSelector(sigma=1.0, mu0=0.0)
        regions, _, _ = scheme.dynamic_tilted_confidence_interval(
            0.05, D, model, prior, "waldo", selector,
            n_grid=151, coarse_n=11,
        )
        prev_hi = -np.inf
        for lo, hi in regions:
            assert lo < hi, f"inverted interval at D={D}, sigma0={sigma0}"
            assert lo >= prev_hi, "overlapping regions"
            prev_hi = hi

    @given(D=_D)
    @settings(max_examples=6, deadline=None)
    def test_wald_width_independent_of_D(self, D):
        """Wald CI width is constant in D — verify under dynamic dispatch."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        selector = NumericalEtaSelector(sigma=1.0, mu0=0.0)
        _, total, n_reg = scheme.dynamic_tilted_confidence_interval(
            0.05, D, model, prior, "wald", selector,
            n_grid=151, coarse_n=11,
        )
        z = stats.norm.ppf(0.975)
        assert n_reg == 1
        np.testing.assert_allclose(total, 2 * z, atol=0.05)
