"""Confirm the n=1 contract is enforced at the OT/power-law CI surface.

Earlier code silently used ``data.mean()`` so n>1 inputs produced
silently mis-scaled results (effective σ should have been σ/√n, not σ).
The new behaviour raises NotImplementedError with a clear message.

Tier 1.5-O8 in the audit.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L2
@pytest.mark.parametrize("scheme", [PowerLawTilting(), OTTilting()])
def test_confidence_regions_rejects_n_gt_1(scheme) -> None:
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    statistic = WaldoStatistic()
    multi_obs = np.array([0.1, 0.2, 0.3])

    with pytest.raises(NotImplementedError, match="single-observation"):
        scheme.confidence_regions(0.05, multi_obs, model, prior, statistic)


@pytest.mark.L2
@pytest.mark.parametrize("scheme", [PowerLawTilting(), OTTilting()])
def test_pvalue_rejects_n_gt_1(scheme) -> None:
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    statistic = WaldoStatistic()
    multi_obs = np.array([0.1, 0.2])

    with pytest.raises(NotImplementedError, match="single-observation"):
        scheme.pvalue(np.array([0.0]), multi_obs, model, prior, statistic)


@pytest.mark.L2
@pytest.mark.parametrize("scheme", [PowerLawTilting(), OTTilting()])
def test_n_eq_1_still_works(scheme) -> None:
    """n=1 is the supported contract; verify it still functions."""
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    statistic = WaldoStatistic()
    single_obs = np.array([0.5])

    regions = scheme.confidence_regions(0.05, single_obs, model, prior, statistic)
    assert len(regions) >= 1
    for lo, hi in regions:
        assert lo < hi
