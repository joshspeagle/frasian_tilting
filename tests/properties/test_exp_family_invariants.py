"""Skipped property tests for ExpFamilyTilting (stub)."""

from __future__ import annotations

import pytest

_STUB_REASON = "stub - see docs/methods/exp_family.md"


@pytest.mark.L1
@pytest.mark.properties
class TestExpFamilyInvariants:
    @pytest.mark.skip(reason=_STUB_REASON)
    def test_identity_at_eta_identity(self):
        """tilt(eta=eta_identity) returns the chosen reference Gaussian."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_natural_parameter_recovery(self):
        """Inverse map (mu_t, sigma_t) ↔ natural-param interpolation closes."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_avoid_sigma_zero_singularity(self):
        """Path must not pass through the natural-parameter singularity."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_smoothness_similar_to_geodesic_normal(self):
        """On Gaussians, expected to be very close to Fisher-Rao."""
