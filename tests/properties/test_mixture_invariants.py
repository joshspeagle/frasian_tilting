"""Property tests for MixtureTilting.

Stage A invariants: tilt(eta=0)=posterior, tilt(eta=1)=likelihood,
density integrates to 1, continuous in eta, refuses inadmissible eta.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.mixture import MixtureTilting


@pytest.mark.L1
@pytest.mark.properties
class TestMixtureInvariants:
    def test_eta_zero_returns_posterior(self):
        """tilt(eta=0) recovers the posterior exactly (framework convention)."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=2.0)
        D = np.asarray([0.5])
        post = model.posterior(D, prior)
        lik = model.likelihood(D)

        tilted = MixtureTilting().tilt(post, prior, lik, 0.0)
        x = np.linspace(-3.0, 4.0, 50)
        np.testing.assert_allclose(
            np.asarray(tilted.pdf(x)),
            np.asarray(post.pdf(x)),
            atol=1e-12,
        )

    def test_eta_one_returns_likelihood_gaussian(self):
        """tilt(eta=1) recovers the likelihood-as-Gaussian exactly."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=2.0)
        D = np.asarray([0.5])
        post = model.posterior(D, prior)
        lik = model.likelihood(D)

        tilted = MixtureTilting().tilt(post, prior, lik, 1.0)
        x = np.linspace(-3.0, 4.0, 50)
        from scipy import stats as sp_stats
        expected = sp_stats.norm.pdf(x, loc=0.5, scale=1.0)
        np.testing.assert_allclose(
            np.asarray(tilted.pdf(x)), expected, atol=1e-12
        )

    def test_density_integrates_to_one(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=2.0)
        D = np.asarray([0.5])
        post = model.posterior(D, prior)
        lik = model.likelihood(D)

        for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tilted = MixtureTilting().tilt(post, prior, lik, eta)
            x = np.linspace(-15.0, 15.0, 6001)
            pdf = np.asarray(tilted.pdf(x), dtype=np.float64)
            Z = float(np.trapezoid(pdf, x))
            assert Z == pytest.approx(1.0, abs=1e-4), f"eta={eta}: Z={Z}"

    def test_continuous_in_eta(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=2.0)
        D = np.asarray([2.0])
        post = model.posterior(D, prior)
        lik = model.likelihood(D)

        eta_grid = np.linspace(0.0, 1.0, 51)
        x_eval = np.asarray([0.5])
        pdfs = np.array(
            [float(MixtureTilting().tilt(post, prior, lik, e).pdf(x_eval)[0])
             for e in eta_grid]
        )
        # Linear interp in eta -> consecutive differences should be small.
        max_diff = float(np.max(np.abs(np.diff(pdfs))))
        assert max_diff < 0.05, f"non-continuous: max consecutive d_pdf = {max_diff}"

    def test_inadmissible_eta_negative_raises(self):
        """eta < 0 on Normal-Normal is never admissible (prior tails go to 0)."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        D = np.asarray([0.0])
        post = model.posterior(D, prior)
        lik = model.likelihood(D)

        with pytest.raises(ValueError, match=r"admissibl|negative|inadmiss"):
            MixtureTilting().tilt(post, prior, lik, -0.5)
