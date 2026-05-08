"""Property tests for FisherRaoTilting (stub).

Mirrors invariants in `docs/methods/fisher_rao.md`. Flipping
`skip` -> passing is the unit of progress.

Audit P0-18: also pin that the stub *raises* NotImplementedError
rather than silently returning a wrong value.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.fisher_rao import FisherRaoTilting

_STUB_REASON = "stub - see docs/methods/fisher_rao.md"


@pytest.mark.L1
@pytest.mark.properties
class TestFisherRaoInvariants:
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
    def test_differs_from_ot_when_variances_differ(self):
        """Fisher-Rao != W2 unless sigma_a = sigma_b."""


@pytest.mark.L1
@pytest.mark.properties
class TestFisherRaoStubActuallyRaises:
    """Audit P0-18: pin that each stub method raises NotImplementedError."""

    @pytest.fixture
    def fixtures(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        lik = GaussianLikelihood(D=1.0, sigma=1.0)
        post = model.posterior(np.asarray([1.0]), prior)
        return model, prior, lik, post

    def test_tilt_raises(self, fixtures):
        _, prior, lik, post = fixtures
        with pytest.raises(NotImplementedError, match=r"fisher_rao"):
            FisherRaoTilting().tilt(post, prior, lik, 0.5)

    def test_path_raises(self, fixtures):
        _, prior, lik, post = fixtures
        with pytest.raises(NotImplementedError, match=r"fisher_rao"):
            list(FisherRaoTilting().path(post, prior, lik, np.linspace(0, 1, 5)))

    def test_confidence_interval_raises(self, fixtures):
        model, prior, _, _ = fixtures
        with pytest.raises(NotImplementedError, match=r"fisher_rao"):
            FisherRaoTilting().confidence_interval(
                0.05, np.asarray([1.0]), model, prior, WaldoStatistic()
            )

    def test_confidence_regions_raises(self, fixtures):
        model, prior, _, _ = fixtures
        with pytest.raises(NotImplementedError, match=r"fisher_rao"):
            FisherRaoTilting().confidence_regions(
                0.05, np.asarray([1.0]), model, prior, WaldoStatistic()
            )

    def test_pvalue_raises(self, fixtures):
        model, prior, _, _ = fixtures
        with pytest.raises(NotImplementedError, match=r"fisher_rao"):
            FisherRaoTilting().pvalue(
                np.asarray([0.5]), np.asarray([1.0]), model, prior, WaldoStatistic()
            )

    def test_is_identity_does_not_raise(self):
        """is_identity is a pure equality check; must not raise on the stub."""
        f = FisherRaoTilting()
        assert f.is_identity(0.0) is True
        assert f.is_identity(0.5) is False
