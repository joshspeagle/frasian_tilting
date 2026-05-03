"""Regression: general `eta_transform_general` agrees with per-scheme shims.

The new `eta_transform_general(eta, eta_low, eta_high)` is the universal
affine map from the admissible η range to [0, 1]. The legacy
per-scheme shims (`eta_transform_powerlaw`, `eta_transform_ot`, etc.)
must produce identical numerical output after the refactor.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.learned.transforms import (
    admissible_range_for_scheme, admissible_range_ot,
    admissible_range_powerlaw, eta_inverse_general, eta_inverse_ot,
    eta_inverse_powerlaw, eta_transform_general, eta_transform_ot,
    eta_transform_powerlaw,
)


@pytest.mark.L0
@pytest.mark.parametrize("w_val", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_general_matches_per_scheme_powerlaw(w_val):
    """eta_transform_general(η, η_low, η_high) ≡ eta_transform_powerlaw(η, w)."""
    eta_grid = np.linspace(-w_val / (1.0 - w_val), 1.0, 11)
    eta_low, eta_high = admissible_range_powerlaw(w_val)
    out_general = eta_transform_general(eta_grid, eta_low, eta_high)
    out_legacy = eta_transform_powerlaw(eta_grid, w_val)
    np.testing.assert_allclose(out_general, out_legacy, atol=1e-12)


@pytest.mark.L0
@pytest.mark.parametrize("w_val", [0.1, 0.5, 0.9])
def test_general_matches_per_scheme_ot(w_val):
    """eta_transform_general matches eta_transform_ot."""
    eta_grid = np.linspace(0.0, 1.0, 11)
    eta_low, eta_high = admissible_range_ot(w_val)
    out_general = eta_transform_general(eta_grid, eta_low, eta_high)
    out_legacy = eta_transform_ot(eta_grid, w_val)
    np.testing.assert_allclose(out_general, out_legacy, atol=1e-12)


@pytest.mark.L0
def test_general_round_trip():
    """η → η' → η round-trips for arbitrary η_low/η_high."""
    eta = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    eta_low = -0.5
    eta_high = 1.5
    eta_prime = eta_transform_general(eta, eta_low, eta_high)
    eta_recovered = eta_inverse_general(eta_prime, eta_low, eta_high)
    np.testing.assert_allclose(eta_recovered, eta, atol=1e-12)


@pytest.mark.L0
def test_admissible_range_for_scheme_dispatch():
    """`admissible_range_for_scheme` dispatches correctly per scheme name."""
    pl_low, pl_high = admissible_range_for_scheme("power_law", 0.5)
    np.testing.assert_allclose(pl_low, -1.0)
    np.testing.assert_allclose(pl_high, 1.0)

    ot_low, ot_high = admissible_range_for_scheme("ot", 0.5)
    np.testing.assert_allclose(ot_low, 0.0)
    np.testing.assert_allclose(ot_high, 1.0)


@pytest.mark.L0
def test_admissible_range_for_scheme_unregistered_raises():
    with pytest.raises(NotImplementedError, match="fisher_rao"):
        admissible_range_for_scheme("fisher_rao", 0.5)
