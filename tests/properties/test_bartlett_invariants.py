"""Skipped property tests for BartlettCorrectedLRT (stub)."""

from __future__ import annotations

import pytest

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
