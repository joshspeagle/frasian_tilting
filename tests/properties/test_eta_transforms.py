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
