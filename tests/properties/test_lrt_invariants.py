"""Skipped property tests for LRTStatistic (stub)."""

from __future__ import annotations

import pytest

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
