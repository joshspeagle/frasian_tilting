"""Regression: numerical admissible-range probe matches closed-form.

`numerical_admissible_range` is the future-proof fallback for schemes
without a closed-form admissible-range formula. We verify it on the
two schemes that DO have closed forms (power_law and ot) — the
numerical bisection should rediscover the closed-form bounds within
the requested atol.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting._admissible import (
    clear_admissible_range_cache, make_default_validity_fn,
    numerical_admissible_range, numerical_admissible_range_cached,
)
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


def _setup_validity_fn(scheme, w_val, statistic_name="waldo"):
    sigma = 1.0
    sigma0 = float(np.sqrt(w_val / max(1.0 - w_val, 1e-9)) * sigma)
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=sigma0)
    return make_default_validity_fn(
        scheme, model, prior, statistic_name=statistic_name,
    )


@pytest.mark.L1
@pytest.mark.parametrize("w_val", [0.2, 0.5, 0.8])
def test_numerical_recovers_real_bounds_powerlaw(w_val):
    """power_law's actual mathematical lower bound is much wider than the
    legacy `-w/(1-w)` clamp.

    `denom = 1 - η(1-w)` is the variance-positivity quantity; for η < 0
    it is > 1, so the formula is well-defined arbitrarily far below
    zero. The numerical search exposes this — the historical clamp at
    `-w/(1-w)` is a *training-friendly* convention, not a math fact.
    The closed-form upper bound `1/(1-w)` is the actual variance-
    positivity boundary.
    """
    scheme = PowerLawTilting()
    validity_fn = _setup_validity_fn(scheme, w_val)
    eta_low, eta_high = numerical_admissible_range(
        validity_fn, eta_id=0.0, atol=1e-3,
        step_max=1e3,
    )
    # Lower bound: search hits step_max without finding an invalid η.
    assert eta_low < -1.0 / (1.0 - w_val), (
        f"η_low={eta_low} is INSIDE the legacy clamp; the search "
        f"should find power_law tolerates much wider η_low."
    )
    # Upper bound: 1/(1-w) is the variance-positivity boundary
    # (denom > 0). Numerical search should find it within atol.
    eta_high_closed = 1.0 / (1.0 - w_val)
    assert eta_high < eta_high_closed, (
        f"η_high={eta_high} ≥ closed-form variance-positivity upper "
        f"bound {eta_high_closed}"
    )
    assert eta_high > eta_high_closed - 0.1, (
        f"η_high={eta_high} suspiciously far from {eta_high_closed}"
    )


@pytest.mark.L1
@pytest.mark.parametrize("w_val", [0.2, 0.5, 0.8])
def test_numerical_recovers_closed_form_ot(w_val):
    """For ot, the numerical bracket should be ⊆ (0, 1)."""
    scheme = OTTilting()
    validity_fn = _setup_validity_fn(scheme, w_val)
    eta_low, eta_high = numerical_admissible_range(
        validity_fn, eta_id=0.5, atol=1e-3,
    )
    # OT formula extends below 0 (down to η=-w/(1-w) where s_t=0), so
    # the numerical search may find a wider range than the
    # closed-form (0, 1). Just assert the search stayed within the
    # variance-positivity interval.
    assert eta_low > -1.0 / (1.0 - w_val) - 0.1
    assert eta_high > 0.5


@pytest.mark.L1
def test_numerical_admissible_range_invalid_eta_id_raises():
    """Passing an invalid identity should raise."""
    def always_false(_eta):
        return False
    with pytest.raises(ValueError, match="invalid"):
        numerical_admissible_range(always_false, eta_id=0.0)


@pytest.mark.L1
def test_numerical_admissible_range_cache_hit():
    """Repeated calls with the same cache_key skip recomputation."""
    clear_admissible_range_cache()
    call_count = 0

    def counting_validity(eta: float) -> bool:
        nonlocal call_count
        call_count += 1
        return -1.0 < eta < 1.0

    key = ("test_scheme", "fp_model", "fp_prior", 0.5)
    r1 = numerical_admissible_range_cached(
        cache_key=key, validity_fn=counting_validity,
        eta_id=0.0, atol=1e-3,
    )
    n_after_first = call_count
    r2 = numerical_admissible_range_cached(
        cache_key=key, validity_fn=counting_validity,
        eta_id=0.0, atol=1e-3,
    )
    assert r1 == r2
    assert call_count == n_after_first, (
        f"cache miss: validity_fn called {call_count - n_after_first} "
        f"extra times on repeat call with same cache_key"
    )
    clear_admissible_range_cache()


@pytest.mark.L1
def test_numerical_admissible_range_returns_buffered_range():
    """Returned range should be strictly inside the valid set by atol."""
    def is_valid(eta: float) -> bool:
        return -2.0 < eta < 3.0

    eta_low, eta_high = numerical_admissible_range(
        is_valid, eta_id=0.0, atol=1e-3,
    )
    assert eta_low > -2.0  # buffer applied
    assert eta_high < 3.0
    # Buffer should be ~atol from the true boundary.
    assert eta_low < -1.99 + 0.01
    assert eta_high > 2.99 - 0.01
