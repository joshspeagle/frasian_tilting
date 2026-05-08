"""Property tests for the EtaSelector contract.

The EtaSelector protocol declares three flags every implementation must
expose: `name`, `is_dynamic`, and `is_post_selection`. The
`is_post_selection` flag distinguishes selectors whose `select(...)`
reads `D = data.mean()` (post-selection inference, undercovers) from
selectors whose `select(...)` is independent of D (calibrated). These
tests pin the flags against the documented behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.base import EtaSelector
from frasian.tilting.eta_selectors import (
    DynamicNumericalEtaSelector,
    FixedEtaSelector,
    NumericalEtaSelector,
)
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L1
@pytest.mark.properties
class TestEtaSelectorContract:
    """Pin the documented `is_post_selection` flag per selector."""

    def test_fixed_is_not_post_selection(self):
        sel = FixedEtaSelector(eta=0.3)
        assert sel.is_post_selection is False
        assert sel.is_dynamic is False
        assert isinstance(sel, EtaSelector)

    def test_numerical_is_post_selection(self):
        """NumericalEtaSelector reads D = data.mean() in select() — flag must be True."""
        sel = NumericalEtaSelector()
        assert sel.is_post_selection is True
        assert sel.is_dynamic is False
        assert isinstance(sel, EtaSelector)

    def test_dynamic_numerical_is_not_post_selection(self):
        """DynamicNumericalEtaSelector via select_grid is θ-only (calibrated)."""
        sel = DynamicNumericalEtaSelector()
        assert sel.is_post_selection is False
        assert sel.is_dynamic is True
        assert isinstance(sel, EtaSelector)


@pytest.mark.L1
@pytest.mark.properties
class TestFixedEtaSelectorBehavior:
    """FixedEtaSelector returns the same η regardless of inputs."""

    def test_constant_across_data(self):
        sel = FixedEtaSelector(eta=0.42)
        scheme = PowerLawTilting()
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldoStatistic()
        for D in [-3.0, 0.0, 1.5, 4.0]:
            eta = sel.select(
                scheme,
                data=np.asarray([D]),
                model=model,
                prior=prior,
                alpha=0.05,
                statistic=stat,
            )
            assert eta == pytest.approx(0.42)

    def test_constant_across_alpha(self):
        sel = FixedEtaSelector(eta=0.0)
        scheme = PowerLawTilting()
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldoStatistic()
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            eta = sel.select(
                scheme,
                data=np.asarray([1.0]),
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=stat,
            )
            assert eta == 0.0


@pytest.mark.L1
@pytest.mark.properties
class TestDynamicSelectorThetaOnly:
    """`select_grid(theta_grid, ...)` must be independent of `data` (no D-leak)."""

    def test_select_grid_data_independent(self):
        """The same theta_grid + (model, prior, alpha) yields the same η-curve
        regardless of which D-supplying call invoked it (through select_grid,
        which doesn't take data at all).
        """
        sel = DynamicNumericalEtaSelector()
        scheme = PowerLawTilting()
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldoStatistic()
        theta_grid = np.linspace(-2.0, 2.0, 9)
        eta_a = sel.select_grid(
            theta_grid, scheme, statistic=stat, model=model, prior=prior, alpha=0.05
        )
        eta_b = sel.select_grid(
            theta_grid, scheme, statistic=stat, model=model, prior=prior, alpha=0.05
        )
        np.testing.assert_array_equal(eta_a, eta_b)
