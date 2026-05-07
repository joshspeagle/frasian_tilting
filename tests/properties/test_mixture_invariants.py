"""Property tests for MixtureTilting (stub).

Audit P0-18: previously every method here had a `@pytest.mark.skip`-
decorated empty body. That meant the test suite had ZERO checks on the
stub — if a future change accidentally returned a wrong-but-non-raising
value, no test would catch it. We now keep the framework-claim tests as
skip-placeholders (they require the real implementation), but add an
active sanity test that pins each stub method actually raises
NotImplementedError. If `MixtureTilting.tilt(...)` silently starts
returning something, this test fires.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.mixture import MixtureTilting

_STUB_REASON = "stub - see docs/methods/mixture.md"


@pytest.mark.L1
@pytest.mark.properties
class TestMixtureInvariants:
    @pytest.mark.skip(reason=_STUB_REASON)
    def test_eta_zero_returns_posterior(self):
        """tilt(eta=0) must return the posterior exactly (framework convention)."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_eta_one_returns_likelihood_gaussian(self):
        """tilt(eta=1) must return the likelihood-induced Gaussian."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_density_integrates_to_one(self):
        """Convex combination of two normalised densities is normalised."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_continuous_in_eta(self):
        """Linear in eta by construction."""

    @pytest.mark.skip(reason=_STUB_REASON)
    def test_waldo_ci_record_nan_when_bimodal(self):
        """Cell evaluator records NaN when the mixture is heavily bimodal."""


@pytest.mark.L1
@pytest.mark.properties
class TestMixtureStubActuallyRaises:
    """Active checks (audit P0-18) that the stub actually raises rather
    than silently returning a wrong value. If MixtureTilting picks up an
    accidental implementation, these fire.
    """

    @pytest.fixture
    def fixtures(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        lik = GaussianLikelihood(D=1.0, sigma=1.0)
        post = model.posterior(np.asarray([1.0]), prior)
        return model, prior, lik, post

    def test_tilt_raises_notimplemented(self, fixtures):
        _, prior, lik, post = fixtures
        with pytest.raises(NotImplementedError, match=r"mixture"):
            MixtureTilting().tilt(post, prior, lik, 0.5)

    def test_path_raises_notimplemented(self, fixtures):
        _, prior, lik, post = fixtures
        with pytest.raises(NotImplementedError, match=r"mixture"):
            list(MixtureTilting().path(post, prior, lik, np.linspace(0, 1, 5)))

    def test_confidence_interval_raises_notimplemented(self, fixtures):
        model, prior, _, _ = fixtures
        with pytest.raises(NotImplementedError, match=r"mixture"):
            MixtureTilting().confidence_interval(
                0.05, np.asarray([1.0]), model, prior, WaldoStatistic()
            )

    def test_confidence_regions_raises_notimplemented(self, fixtures):
        model, prior, _, _ = fixtures
        with pytest.raises(NotImplementedError, match=r"mixture"):
            MixtureTilting().confidence_regions(
                0.05, np.asarray([1.0]), model, prior, WaldoStatistic()
            )

    def test_pvalue_raises_notimplemented(self, fixtures):
        model, prior, _, _ = fixtures
        with pytest.raises(NotImplementedError, match=r"mixture"):
            MixtureTilting().pvalue(
                np.asarray([0.5]), np.asarray([1.0]), model, prior, WaldoStatistic()
            )

    def test_is_identity_returns_correct_endpoint(self):
        """is_identity should not raise — it returns whether eta is the
        identity element. After Cluster A, identity is at eta=0."""
        m = MixtureTilting()
        assert m.is_identity(0.0) is True
        assert m.is_identity(0.5) is False
        assert m.is_identity(1.0) is False
