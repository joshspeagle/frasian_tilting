"""Round-trip invariants for delta/eta transforms in `frasian.learned.transforms`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frasian.learned.transforms import (delta_inverse, delta_transform,
                                          eta_inverse, eta_inverse_powerlaw,
                                          eta_min_powerlaw,
                                          eta_transform,
                                          eta_transform_powerlaw)


@pytest.mark.L1
@pytest.mark.properties
class TestDeltaTransform:
    @given(delta=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=80, deadline=None)
    def test_round_trip(self, delta):
        """Δ → Δ' → Δ identity to atol 1e-10."""
        delta_prime = delta_transform(delta)
        recovered = delta_inverse(delta_prime)
        np.testing.assert_allclose(recovered, delta, atol=1e-10)

    @given(delta=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=80, deadline=None)
    def test_image_in_unit_interval(self, delta):
        """Δ' = Δ/(1+Δ) ∈ [0, 1)."""
        delta_prime = delta_transform(delta)
        assert 0.0 <= delta_prime < 1.0


@pytest.mark.L1
@pytest.mark.properties
class TestEtaTransformPowerLaw:
    @given(w=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
           eta=st.floats(min_value=-0.99, max_value=0.99, allow_nan=False))
    @settings(max_examples=80, deadline=None)
    def test_round_trip(self, w, eta):
        """η → η' → η identity for power_law (within admissible range).

        Skip cases below η_min(w) where the forward map produces η' < 0.
        """
        if eta <= eta_min_powerlaw(w) + 1e-3:
            return
        eta_prime = eta_transform_powerlaw(eta, w)
        recovered = eta_inverse_powerlaw(eta_prime, w)
        np.testing.assert_allclose(recovered, eta, atol=1e-10)

    @given(w=st.floats(min_value=0.05, max_value=0.95, allow_nan=False))
    @settings(max_examples=40, deadline=None)
    def test_endpoints(self, w):
        """η_min(w) ↦ 0; η = 1 ↦ 1."""
        eta_lo = eta_min_powerlaw(w)
        np.testing.assert_allclose(eta_transform_powerlaw(eta_lo, w),
                                     0.0, atol=1e-10)
        np.testing.assert_allclose(eta_transform_powerlaw(1.0, w),
                                     1.0, atol=1e-10)


@pytest.mark.L1
@pytest.mark.properties
class TestTransformBoundarySafety:
    """Skeptic-Phase-B item 3: transforms must not return NaN/Inf at
    boundary inputs the framework's grids actually reach."""

    def test_delta_transform_handles_inf(self):
        """delta_transform(inf) returns the clamped maximum, not NaN."""
        out = delta_transform(np.inf)
        assert np.isfinite(out)
        assert out < 1.0

    def test_delta_transform_handles_large_finite(self):
        """delta_transform(1e9) gives a finite Δ' ≈ 1."""
        out = delta_transform(1e9)
        assert np.isfinite(out)
        assert out < 1.0
        assert out > 0.999

    def test_delta_inverse_clamps_close_to_one(self):
        """delta_inverse(1.0) does not produce inf."""
        out = delta_inverse(1.0)
        assert np.isfinite(out)

    def test_delta_inverse_array(self):
        """Vector inputs preserve the clamp."""
        out = delta_inverse(np.array([0.0, 0.5, 0.9, 1.0, 1.5]))
        assert np.all(np.isfinite(out))

    def test_eta_min_powerlaw_at_w_close_to_one(self):
        """eta_min_powerlaw(0.999) is finite (clamps w)."""
        out = eta_min_powerlaw(0.999)
        assert np.isfinite(out)

    def test_eta_inverse_powerlaw_at_w_close_to_one(self):
        """eta_inverse_powerlaw at w ≈ 1 is finite."""
        out = eta_inverse_powerlaw(0.5, 0.999)
        assert np.isfinite(out)

    def test_round_trip_at_boundary_w(self):
        """delta_transform then delta_inverse round-trips for any Δ ∈ [0, 1e6]."""
        for delta in [0.0, 1e-3, 1.0, 100.0, 1e6]:
            recovered = delta_inverse(delta_transform(delta))
            np.testing.assert_allclose(recovered, delta, rtol=1e-6,
                                         err_msg=f"failed at delta={delta}")


@pytest.mark.L1
@pytest.mark.properties
class TestSchemeDispatch:
    def test_powerlaw_dispatch(self):
        """`eta_transform("power_law", ...)` matches direct call."""
        eta, w = 0.3, 0.5
        np.testing.assert_allclose(
            eta_transform("power_law", eta, w),
            eta_transform_powerlaw(eta, w),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            eta_inverse("power_law", 0.65, w),
            eta_inverse_powerlaw(0.65, w),
            atol=1e-12,
        )

    def test_ot_dispatch_is_identity(self):
        """For `ot`, η-transforms are the identity (admissible range is [0, 1])."""
        eta, w = 0.3, 0.5
        np.testing.assert_allclose(
            eta_transform("ot", eta, w), eta, atol=1e-12
        )
        np.testing.assert_allclose(
            eta_inverse("ot", eta, w), eta, atol=1e-12
        )

    def test_unknown_scheme_raises(self):
        """`eta_transform("unknown", ...)` raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="unknown"):
            eta_transform("unknown", 0.3, 0.5)
