"""Skipped property tests for SignedRootStatistic (stub)."""

from __future__ import annotations

import pytest

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
