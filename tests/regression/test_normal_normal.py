"""Regression tests pinning NormalNormalModel math to the legacy formulas.

These are the load-bearing checks that the port did not silently change a
constant. Tolerances are very tight (atol=1e-12) — any drift is a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import (
    NormalNormalModel,
    noncentrality,
    posterior_params,
    prior_residual,
    scaled_conflict,
    weight,
)


@pytest.mark.L0
class TestPosteriorParams:
    """Verbatim formulas from legacy/src/frasian/core.py."""

    def test_balanced_prior_data(self):
        # mu0=0, sigma=sigma0=1: w=0.5, mu_n = 0.5*D, sigma_n = sqrt(0.5)
        mu_n, sigma_n, w = posterior_params(D=2.0, mu0=0.0, sigma=1.0, sigma0=1.0)
        np.testing.assert_allclose(w, 0.5, atol=1e-12)
        np.testing.assert_allclose(mu_n, 1.0, atol=1e-12)
        np.testing.assert_allclose(sigma_n, np.sqrt(0.5), atol=1e-12)

    def test_strong_prior(self):
        # sigma0 << sigma: w small, posterior near prior
        mu_n, sigma_n, w = posterior_params(D=10.0, mu0=0.0, sigma=10.0, sigma0=1.0)
        # w = 1 / (100 + 1) ≈ 0.0099
        np.testing.assert_allclose(w, 1.0 / 101.0, atol=1e-12)
        np.testing.assert_allclose(mu_n, 10.0 / 101.0, atol=1e-12)

    def test_weak_prior(self):
        # sigma0 >> sigma: w near 1, posterior near MLE
        mu_n, sigma_n, w = posterior_params(D=3.0, mu0=0.0, sigma=1.0, sigma0=100.0)
        np.testing.assert_allclose(w, 10000.0 / 10001.0, atol=1e-12)
        np.testing.assert_allclose(mu_n, 3.0 * w, atol=1e-12)

    def test_array_input(self):
        D_arr = np.array([-1.0, 0.0, 1.0, 5.0])
        mu_n, sigma_n, w = posterior_params(D=D_arr, mu0=0.0, sigma=1.0, sigma0=1.0)
        np.testing.assert_allclose(mu_n, 0.5 * D_arr, atol=1e-12)
        np.testing.assert_allclose(sigma_n, np.sqrt(0.5), atol=1e-12)
        np.testing.assert_allclose(w, 0.5, atol=1e-12)


@pytest.mark.L0
class TestWeight:
    def test_formula(self):
        np.testing.assert_allclose(weight(1.0, 1.0), 0.5, atol=1e-12)
        np.testing.assert_allclose(weight(2.0, 1.0), 1.0 / 5.0, atol=1e-12)
        np.testing.assert_allclose(weight(1.0, 2.0), 4.0 / 5.0, atol=1e-12)


@pytest.mark.L0
class TestScaledConflict:
    def test_zero_when_D_equals_mu0(self):
        np.testing.assert_allclose(
            scaled_conflict(D=0.0, mu0=0.0, w=0.5, sigma=1.0), 0.0, atol=1e-12
        )

    def test_sign_convention(self):
        # Positive Delta when D < mu0 (legacy convention)
        delta = scaled_conflict(D=-2.0, mu0=0.0, w=0.5, sigma=1.0)
        np.testing.assert_allclose(delta, 1.0, atol=1e-12)


@pytest.mark.L0
class TestPriorResidual:
    def test_formula(self):
        np.testing.assert_allclose(
            prior_residual(theta=2.0, mu0=0.0, sigma0=1.0), 2.0, atol=1e-12
        )
        np.testing.assert_allclose(
            prior_residual(theta=4.0, mu0=2.0, sigma0=2.0), 1.0, atol=1e-12
        )


@pytest.mark.L0
class TestNoncentrality:
    def test_zero_at_mu0(self):
        nc = noncentrality(theta=0.0, mu0=0.0, w=0.5, sigma=1.0)
        np.testing.assert_allclose(nc, 0.0, atol=1e-12)

    def test_legacy_formula(self):
        # lambda = (1-w)^2 (mu0 - theta)^2 / (w^2 * sigma^2)
        # at theta=2, mu0=0, w=0.5, sigma=1 -> 0.25 * 4 / (0.25 * 1) = 4
        nc = noncentrality(theta=2.0, mu0=0.0, w=0.5, sigma=1.0)
        np.testing.assert_allclose(nc, 4.0, atol=1e-12)


@pytest.mark.L0
class TestNormalNormalModel:
    def test_posterior_uses_correct_params(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        post = model.posterior(np.asarray([2.0]), prior)
        np.testing.assert_allclose(post.loc, 1.0, atol=1e-12)
        np.testing.assert_allclose(post.scale, np.sqrt(0.5), atol=1e-12)

    def test_mle_is_data_mean(self):
        model = NormalNormalModel(sigma=1.0)
        np.testing.assert_allclose(model.mle(np.asarray([1.0, 2.0, 3.0])), 2.0,
                                    atol=1e-12)

    def test_fisher_information_is_inverse_variance(self):
        model = NormalNormalModel(sigma=2.0)
        I = model.fisher_information(0.0)
        np.testing.assert_allclose(I, 0.25, atol=1e-12)

    def test_sigma_must_be_positive(self):
        with pytest.raises(ValueError):
            NormalNormalModel(sigma=0.0)
        with pytest.raises(ValueError):
            NormalNormalModel(sigma=-1.0)

    def test_likelihood_returns_gaussian(self):
        from frasian.models.distributions import GaussianLikelihood

        model = NormalNormalModel(sigma=1.5)
        lik = model.likelihood(np.asarray([3.0]))
        assert isinstance(lik, GaussianLikelihood)
        np.testing.assert_allclose(lik.D, 3.0, atol=1e-12)
        np.testing.assert_allclose(lik.sigma, 1.5, atol=1e-12)
