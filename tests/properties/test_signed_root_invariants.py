"""Property tests for SignedRootStatistic (stub).

Audit P0-18: pin that each stub method raises NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.signed_root import SignedRootStatistic

_STUB_REASON = "stub - see docs/methods/signed_root.md"


@pytest.mark.L1
@pytest.mark.properties
class TestSignedRootInvariants:
    @pytest.mark.skip(reason=_STUB_REASON)
    def test_pvalue_in_unit_interval(self):
        """p-value in [0, 1] for all inputs."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_r_zero_at_mle(self):
        """r(theta=MLE) = 0 by definition of the signed root."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_matches_wald_on_normal_location(self):
        """On Normal-location, signed_root.pvalue == wald.pvalue."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_uniform_under_h0(self):
        """Under H0, p-values are Uniform[0, 1]."""


@pytest.mark.L1
@pytest.mark.properties
class TestSignedRootStubActuallyRaises:
    """Audit P0-18: pin that each stub method raises NotImplementedError."""

    @pytest.fixture
    def fixtures(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        return model, prior

    def test_evaluate_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"signed_root"):
            SignedRootStatistic().evaluate(0.5, np.asarray([1.0]), model, prior)

    def test_pvalue_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"signed_root"):
            SignedRootStatistic().pvalue(0.5, np.asarray([1.0]), model, prior)

    def test_confidence_interval_raises(self, fixtures):
        model, prior = fixtures
        with pytest.raises(NotImplementedError, match=r"signed_root"):
            SignedRootStatistic().confidence_interval(0.05, np.asarray([1.0]), model, prior)
