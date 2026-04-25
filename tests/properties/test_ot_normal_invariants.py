"""Skipped property tests for OTNormalTilting (stub).

Each test corresponds to an invariant in `docs/methods/ot_normal.md`.
Flipping `skip` → passing is the unit of progress when the stub is
implemented via `/propose-method ot_normal`.
"""

from __future__ import annotations

import pytest

_STUB_REASON = "stub - see docs/methods/ot_normal.md"


@pytest.mark.L1
@pytest.mark.properties
class TestOTNormalInvariants:
    @pytest.mark.skip(reason=_STUB_REASON)
    def test_identity_at_eta_identity(self):
        """tilt(eta=eta_identity) returns the chosen reference Gaussian."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_continuous_in_eta(self):
        """No clamps, no NaNs in the admissible range."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_output_sigma_positive(self):
        """W2-geodesic must keep sigma_t > 0 along the path."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_smoothness_lipschitz_below_threshold(self):
        """Step-5 smoothness diagnostic: lipschitz_eta < 1.0 (claim)."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_admissible_range_is_unit_interval(self):
        """t in [0, 1] by construction."""
