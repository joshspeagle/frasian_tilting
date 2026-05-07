"""Property tests for BartlettCorrectedLRT (stub).

Audit P0-18: pin that each stub method raises NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.bartlett import BartlettCorrectedLRT

_STUB_REASON = "stub - see docs/methods/bartlett.md"


@pytest.mark.L1
@pytest.mark.properties
class TestBartlettInvariants:
    @pytest.mark.skip(reason=_STUB_REASON)
    def test_pvalue_in_unit_interval(self):
        """p-value in [0, 1] for all inputs."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_matches_lrt_on_normal_location(self):
        """E[LRT] = 1 on Normal-location, so the correction is trivial."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_decorator_pattern_commutes_with_registry(self):
        """BartlettCorrected(LRT()) doesn't require base re-registration."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_uniform_under_h0(self):
        """Under H0, p-values are Uniform[0, 1] (asymptotic)."""


@pytest.mark.L1
@pytest.mark.properties
class TestBartlettStubActuallyRaises:
    """Audit P0-18: pin that each stub method raises NotImplementedError."""

    @pytest.fixture
    def fixtures(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        return model, prior

    def test_evaluate_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"bartlett"):
            BartlettCorrectedLRT().evaluate(0.5, np.asarray([1.0]), model, prior)

    def test_pvalue_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"bartlett"):
            BartlettCorrectedLRT().pvalue(0.5, np.asarray([1.0]), model, prior)

    def test_confidence_interval_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"bartlett"):
            BartlettCorrectedLRT().confidence_interval(
                0.05, np.asarray([1.0]), model, prior
            )
