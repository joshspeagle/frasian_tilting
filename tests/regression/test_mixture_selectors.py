"""Integration tests: MixtureTilting with each selector class.

Covers Fixed, Numerical (static-with-context), and DynamicNumerical.
LearnedDynamic is exercised in Stage C (after training fixtures).
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import (
    DynamicNumericalEtaSelector,
    FixedEtaSelector,
    NumericalEtaSelector,
)
from frasian.tilting.mixture import MixtureTilting


@pytest.fixture
def fixtures():
    # w = sigma_0^2 / (sigma^2 + sigma_0^2) = 0.5 (well-conditioned for selectors).
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    D = np.asarray([0.5])
    return model, prior, D


@pytest.mark.L4
class TestMixtureWithSelectors:
    def test_fixed_eta_zero(self, fixtures):
        model, prior, D = fixtures
        til = MixtureTilting(selector=FixedEtaSelector(eta=0.0))
        ci = til.confidence_interval(0.05, D, model, prior, WaldoStatistic())
        assert ci[0] < ci[1] and np.all(np.isfinite(ci))

    def test_fixed_eta_half(self, fixtures):
        model, prior, D = fixtures
        til = MixtureTilting(selector=FixedEtaSelector(eta=0.5))
        ci = til.confidence_interval(0.05, D, model, prior, WaldoStatistic())
        assert ci[0] < ci[1] and np.all(np.isfinite(ci))

    def test_numerical_static(self, fixtures):
        model, prior, D = fixtures
        til = MixtureTilting(selector=NumericalEtaSelector())
        ci = til.confidence_interval(0.05, D, model, prior, WaldoStatistic())
        assert ci[0] < ci[1] and np.all(np.isfinite(ci))

    def test_dynamic_numerical(self, fixtures):
        model, prior, D = fixtures
        til = MixtureTilting(
            selector=DynamicNumericalEtaSelector(n_grid=201, coarse_n=15)
        )
        regions = til.confidence_regions(0.05, D, model, prior, WaldoStatistic())
        assert len(regions) >= 1
        for lo, hi in regions:
            assert lo < hi and np.isfinite(lo) and np.isfinite(hi)

    def test_dynamic_numerical_strong_conflict(self):
        """In the bimodal regime |Delta| > 2*sqrt(w), multi-region CIs are allowed."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=-3.0, scale=1.0)
        D = np.asarray([3.0])
        til = MixtureTilting(
            selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)
        )
        regions = til.confidence_regions(0.05, D, model, prior, WaldoStatistic())
        assert len(regions) >= 1
