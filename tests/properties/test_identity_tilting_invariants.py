"""Property tests for the IdentityTilting scheme.

Invariants checked:
  - `tilt(...)` returns the input posterior object verbatim.
  - `is_identity(η)` is True for every η.
  - `confidence_interval` matches the bare statistic's CI exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.identity import IdentityTilting

_D = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_SIGMA0 = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_ETA = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestIdentityTiltingInvariants:
    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0, eta=_ETA)
    @settings(max_examples=80, deadline=None)
    def test_tilt_returns_posterior_verbatim(self, D, sigma, sigma0, eta):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=D, sigma=sigma)
        post = model.posterior(np.asarray([D]), prior)
        tilted = IdentityTilting().tilt(post, prior, lik, eta)
        assert tilted is post

    @given(eta=_ETA)
    @settings(max_examples=20, deadline=None)
    def test_is_identity_always_true(self, eta):
        assert IdentityTilting().is_identity(eta)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0)
    @settings(max_examples=50, deadline=None)
    def test_ci_matches_wald_statistic(self, D, sigma, sigma0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        ident = IdentityTilting()
        wald = WaldStatistic()
        ci_via_tilting = ident.confidence_interval(
            0.05,
            np.asarray([D]),
            model,
            prior,
            wald,
        )
        ci_direct = wald.confidence_interval(
            0.05,
            np.asarray([D]),
            model,
            prior,
        )
        np.testing.assert_allclose(ci_via_tilting, ci_direct, atol=1e-12)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0)
    @settings(max_examples=30, deadline=None)
    def test_ci_matches_waldo_statistic(self, D, sigma, sigma0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        ident = IdentityTilting()
        waldo = WaldoStatistic()
        ci_via_tilting = ident.confidence_interval(
            0.05,
            np.asarray([D]),
            model,
            prior,
            waldo,
        )
        ci_direct = waldo.confidence_interval(
            0.05,
            np.asarray([D]),
            model,
            prior,
        )
        np.testing.assert_allclose(ci_via_tilting, ci_direct, atol=1e-9)

    def test_admissible_range_is_unbounded(self):
        from frasian.tilting.base import TiltingContext

        ctx = TiltingContext(w=0.5, alpha=0.05)
        lo, hi = IdentityTilting().admissible_range(ctx)
        assert np.isinf(lo) and lo < 0
        assert np.isinf(hi) and hi > 0
