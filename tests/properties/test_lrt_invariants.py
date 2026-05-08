"""Property tests for LRTStatistic (stub).

Audit P0-18: also pin that the stub *raises* NotImplementedError on
every protocol method, rather than silently returning a wrong value.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.lrt import LRTStatistic

_STUB_REASON = "stub - see docs/methods/lrt.md"


@pytest.mark.L1
@pytest.mark.properties
class TestLRTInvariants:
    @pytest.mark.skip(reason=_STUB_REASON)
    def test_pvalue_in_unit_interval(self):
        """p-value in [0, 1] for all inputs."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_pvalue_at_mle_equals_one(self):
        """pvalue(theta=MLE) = 1 (mode property)."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_matches_wald_on_normal_location(self):
        """On the Normal-location sandbox, lrt.pvalue == wald.pvalue."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_uniform_under_h0(self):
        """Under H0, p-values are Uniform[0, 1] (KS test)."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_ci_matches_wald_on_normal_location(self):
        """On Normal-location, lrt.confidence_interval == wald.CI."""


@pytest.mark.L1
@pytest.mark.properties
class TestLRTStubActuallyRaises:
    """Audit P0-18: pin that each stub method raises NotImplementedError."""

    @pytest.fixture
    def fixtures(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        return model, prior

    def test_evaluate_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"lrt"):
            LRTStatistic().evaluate(0.5, np.asarray([1.0]), model, prior)

    def test_pvalue_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"lrt"):
            LRTStatistic().pvalue(0.5, np.asarray([1.0]), model, prior)

    def test_acceptance_region_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"lrt"):
            LRTStatistic().acceptance_region(0.05, 0.5, model, prior)

    def test_confidence_interval_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"lrt"):
            LRTStatistic().confidence_interval(0.05, np.asarray([1.0]), model, prior)
