"""Skipped property tests for MixtureTilting (stub)."""

from __future__ import annotations

import pytest

_STUB_REASON = "stub - see docs/methods/mixture.md"


@pytest.mark.L1
@pytest.mark.properties
class TestMixtureInvariants:
    @pytest.mark.skip(reason=_STUB_REASON)
    def test_eta_zero_returns_prior(self):
        """tilt(eta=0) must return the prior exactly."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_eta_one_returns_posterior(self):
        """tilt(eta=1) must return the posterior exactly."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_density_integrates_to_one(self):
        """Convex combination of two normalised densities is normalised."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_continuous_in_eta(self):
        """Linear in eta by construction."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_waldo_ci_record_nan_when_bimodal(self):
        """Cell evaluator records NaN when the mixture is heavily bimodal."""
