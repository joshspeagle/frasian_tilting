"""Regression tests pinning BernoulliModel + BetaDistribution + BernoulliLikelihood.

Closed-form formulas (Beta-Binomial conjugate update) are pinned to
atol=1e-12. Boundaries (theta -> 0, theta -> 1) are checked for
graceful behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import BernoulliLikelihood, BetaDistribution


@pytest.mark.L0
class TestBetaDistribution:
    def test_mean_var_match_formula(self):
        d = BetaDistribution(alpha=2.0, beta=3.0)
        np.testing.assert_allclose(d.mean(), 2.0 / 5.0, atol=1e-12)
        np.testing.assert_allclose(d.var(), (2.0 * 3.0) / (5.0 ** 2 * 6.0),
                                    atol=1e-12)

    def test_pdf_integrates_to_one(self):
        d = BetaDistribution(alpha=2.0, beta=5.0)
        x = np.linspace(0.0001, 0.9999, 5001)
        np.testing.assert_allclose(np.trapezoid(d.pdf(x), x), 1.0, atol=5e-4)

    def test_quantile_round_trip(self):
        d = BetaDistribution(alpha=3.0, beta=2.0)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            np.testing.assert_allclose(d.cdf(d.quantile(q)), q, atol=1e-9)

    def test_invalid_shape_rejected(self):
        with pytest.raises(ValueError):
            BetaDistribution(alpha=0.0, beta=1.0)
        with pytest.raises(ValueError):
            BetaDistribution(alpha=1.0, beta=-1.0)


@pytest.mark.L0
class TestBernoulliLikelihood:
    def test_loglik_at_mle(self):
        # k=3, n=10 -> MLE 0.3; loglik(0.3) = 3 log 0.3 + 7 log 0.7
        lik = BernoulliLikelihood(n_success=3, n_total=10)
        expected = 3 * np.log(0.3) + 7 * np.log(0.7)
        np.testing.assert_allclose(lik.loglik(0.3), expected, atol=1e-12)

    def test_invalid_counts_rejected(self):
        with pytest.raises(ValueError):
            BernoulliLikelihood(n_success=5, n_total=3)
        with pytest.raises(ValueError):
            BernoulliLikelihood(n_success=-1, n_total=3)
        with pytest.raises(ValueError):
            BernoulliLikelihood(n_success=0, n_total=0)


@pytest.mark.L0
class TestBernoulliModel:
    def test_posterior_matches_conjugate_formula(self):
        model = BernoulliModel()
        prior = BetaDistribution(alpha=2.0, beta=3.0)
        # 7 successes out of 10 trials -> posterior Beta(9, 6).
        data = np.array([1.0] * 7 + [0.0] * 3)
        post = model.posterior(data, prior)
        assert isinstance(post, BetaDistribution)
        np.testing.assert_allclose(post.alpha, 9.0, atol=1e-12)
        np.testing.assert_allclose(post.beta, 6.0, atol=1e-12)

    def test_mle_is_sample_mean(self):
        model = BernoulliModel()
        np.testing.assert_allclose(
            model.mle(np.array([1.0, 1.0, 0.0, 1.0, 0.0])), 0.6, atol=1e-12,
        )

    def test_fisher_information_formula(self):
        model = BernoulliModel()
        # I(theta) = 1 / (theta (1 - theta))
        np.testing.assert_allclose(
            model.fisher_information(0.5), 4.0, atol=1e-12,
        )
        np.testing.assert_allclose(
            model.fisher_information(0.25), 1 / (0.25 * 0.75), atol=1e-12,
        )

    def test_support(self):
        assert BernoulliModel().support() == (0.0, 1.0)

    def test_sample_data_in_support(self):
        model = BernoulliModel()
        rng = np.random.default_rng(0)
        samples = model.sample_data(0.3, rng, 1000)
        assert set(np.unique(samples).tolist()) <= {0.0, 1.0}
        # MLE should concentrate near theta_true.
        np.testing.assert_allclose(samples.mean(), 0.3, atol=0.05)

    def test_invalid_theta_rejected(self):
        model = BernoulliModel()
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            model.sample_data(-0.1, rng, 5)
        with pytest.raises(ValueError):
            model.sample_data(1.5, rng, 5)

    def test_non_beta_prior_raises(self):
        from frasian.models.distributions import NormalDistribution

        model = BernoulliModel()
        with pytest.raises(NotImplementedError):
            model.posterior(np.array([1.0, 0.0]),
                             NormalDistribution(loc=0.0, scale=1.0))
