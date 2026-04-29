"""Skipped property tests for GeodesicNormalTilting (stub).

Mirrors invariants in `docs/methods/geodesic_normal.md`. Flipping
`skip` → passing is the unit of progress.
"""

from __future__ import annotations

import pytest

_STUB_REASON = "stub - see docs/methods/geodesic_normal.md"


@pytest.mark.L1
@pytest.mark.properties
class TestGeodesicNormalInvariants:
    @pytest.mark.skip(reason=_STUB_REASON)
    def test_identity_at_eta_identity(self):
        """tilt(eta=eta_identity) returns the chosen reference Gaussian."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_output_sigma_positive(self):
        """Fisher-Rao path stays in the open half-plane sigma > 0."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_reduces_to_vertical_line_when_means_match(self):
        """Equal-mean special case: path is sigma-only."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_smoothness_lipschitz_below_threshold(self):
        """Step-5 diagnostic: lipschitz_eta < 1.0 (claim)."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_differs_from_ot_normal_when_variances_differ(self):
        """Fisher-Rao != W2 unless sigma_a = sigma_b."""
