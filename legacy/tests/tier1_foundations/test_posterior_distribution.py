"""
Tests for Theorem 1: Distribution of (mu_n - theta) under true theta.

Theorem 1: Under the true parameter theta, the posterior mean satisfies:
    (mu_n - theta) | theta ~ N(b(theta), v)
where:
    b(theta) = (1-w)(mu0 - theta) is the bias
    v = w^2 * sigma^2 is the variance
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import (
    posterior_params,
    bias,
    variance,
    posterior_mean_distribution_params,
    weight,
)

from conftest import (
    TestConfig,
    ModelParams,
    simulate_data,
    compute_posterior_means,
    get_model_from_w,
)


@pytest.mark.tier1
class TestPosteriorMeanDistribution:
    """Tests for the distribution of (mu_n - theta) under true theta."""

    def test_mean_of_posterior_mean_error(self, balanced_model, config, rng):
        """Test that E[mu_n - theta | theta] = b(theta) = (1-w)(mu0 - theta)."""
        theta_true = 2.0  # Test at theta != mu0
        model = balanced_model

        # Simulate data and compute posterior means
        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)

        # Compute errors
        errors = mu_n - theta_true

        # Theoretical mean (bias)
        expected_mean = bias(theta_true, model.mu0, model.w)

        # Check empirical mean matches theoretical
        empirical_mean = np.mean(errors)
        assert np.abs(empirical_mean - expected_mean) < 0.05, (
            f"Mean mismatch: empirical={empirical_mean:.4f}, expected={expected_mean:.4f}"
        )

    def test_variance_of_posterior_mean_error(self, balanced_model, config, rng):
        """Test that Var[mu_n - theta | theta] = v = w^2 * sigma^2."""
        theta_true = 2.0
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        errors = mu_n - theta_true

        expected_var = variance(model.w, model.sigma)
        empirical_var = np.var(errors)

        rel_error = np.abs(empirical_var - expected_var) / expected_var
        assert rel_error < config.moment_rtol, (
            f"Variance mismatch: empirical={empirical_var:.4f}, expected={expected_var:.4f}, "
            f"rel_error={rel_error:.4f}"
        )

    def test_normality_of_posterior_mean_error(self, balanced_model, config, rng):
        """Test that (mu_n - theta) follows a Normal distribution."""
        theta_true = 2.0
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        errors = mu_n - theta_true

        # Standardize errors
        expected_mean = bias(theta_true, model.mu0, model.w)
        expected_std = np.sqrt(variance(model.w, model.sigma))
        standardized = (errors - expected_mean) / expected_std

        # KS test against standard normal
        ks_stat, ks_pval = stats.kstest(standardized, 'norm')

        assert ks_pval > config.ks_threshold, (
            f"KS test failed: stat={ks_stat:.4f}, p-value={ks_pval:.4f}"
        )

    @pytest.mark.parametrize("theta_true", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_theorem1_at_various_theta(self, balanced_model, config, rng, theta_true):
        """Test Theorem 1 at various true theta values."""
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        errors = mu_n - theta_true

        expected_mean, expected_var = posterior_mean_distribution_params(
            theta_true, model.mu0, model.sigma, model.sigma0
        )
        expected_std = np.sqrt(expected_var)

        # Check mean
        empirical_mean = np.mean(errors)
        assert np.abs(empirical_mean - expected_mean) < 3 * expected_std / np.sqrt(config.n_samples), (
            f"Mean mismatch at theta={theta_true}: "
            f"empirical={empirical_mean:.4f}, expected={expected_mean:.4f}"
        )

        # Check variance
        empirical_var = np.var(errors)
        rel_error = np.abs(empirical_var - expected_var) / expected_var
        assert rel_error < config.moment_rtol, (
            f"Variance mismatch at theta={theta_true}: rel_error={rel_error:.4f}"
        )

    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    def test_theorem1_at_various_w(self, config, rng, w):
        """Test Theorem 1 at various weight values."""
        theta_true = 2.0
        model = get_model_from_w(w)

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        errors = mu_n - theta_true

        expected_mean = bias(theta_true, model.mu0, model.w)
        expected_var = variance(model.w, model.sigma)

        # Standardize and test normality
        standardized = (errors - expected_mean) / np.sqrt(expected_var)
        ks_stat, ks_pval = stats.kstest(standardized, 'norm')

        assert ks_pval > config.ks_threshold, (
            f"KS test failed at w={w}: stat={ks_stat:.4f}, p-value={ks_pval:.4f}"
        )

    def test_bias_zero_when_theta_equals_mu0(self, balanced_model, config, rng):
        """Test that bias is zero when theta = mu0."""
        theta_true = balanced_model.mu0
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        errors = mu_n - theta_true

        expected_mean = bias(theta_true, model.mu0, model.w)
        assert expected_mean == 0.0, "Theoretical bias should be zero at theta=mu0"

        empirical_mean = np.mean(errors)
        std_error = np.sqrt(variance(model.w, model.sigma) / config.n_samples)
        assert np.abs(empirical_mean) < 3 * std_error, (
            f"Empirical mean {empirical_mean:.4f} not close to zero"
        )

    def test_bias_sign(self, balanced_model, config, rng):
        """Test that bias has correct sign: b(theta) > 0 when theta < mu0."""
        model = balanced_model

        # theta < mu0: expect positive bias (posterior pulled toward mu0)
        theta_below = model.mu0 - 2.0
        D = simulate_data(theta_below, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        errors_below = mu_n - theta_below
        assert np.mean(errors_below) > 0, "Bias should be positive when theta < mu0"

        # theta > mu0: expect negative bias (posterior pulled toward mu0)
        theta_above = model.mu0 + 2.0
        D = simulate_data(theta_above, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        errors_above = mu_n - theta_above
        assert np.mean(errors_above) < 0, "Bias should be negative when theta > mu0"


@pytest.mark.tier1
class TestPosteriorParameters:
    """Tests for posterior parameter formulas."""

    def test_posterior_mean_formula(self):
        """Test mu_n = w*D + (1-w)*mu0."""
        D, mu0, sigma, sigma0 = 5.0, 0.0, 1.0, 1.0
        mu_n, _, w = posterior_params(D, mu0, sigma, sigma0)

        expected = w * D + (1 - w) * mu0
        assert np.isclose(mu_n, expected), f"mu_n={mu_n}, expected={expected}"

    def test_posterior_variance_formula(self):
        """Test sigma_n^2 = w * sigma^2."""
        D, mu0, sigma, sigma0 = 5.0, 0.0, 1.0, 1.0
        mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

        expected_sigma_n = np.sqrt(w) * sigma
        assert np.isclose(sigma_n, expected_sigma_n), (
            f"sigma_n={sigma_n}, expected={expected_sigma_n}"
        )

    def test_weight_formula(self):
        """Test w = sigma0^2 / (sigma^2 + sigma0^2)."""
        sigma, sigma0 = 1.0, 1.0
        w = weight(sigma, sigma0)

        expected = sigma0**2 / (sigma**2 + sigma0**2)
        assert np.isclose(w, expected), f"w={w}, expected={expected}"

    @pytest.mark.parametrize("sigma0,expected_w", [
        (0.5, 0.2),  # Strong prior: w ≈ 0.2
        (1.0, 0.5),  # Balanced: w = 0.5
        (2.0, 0.8),  # Weak prior: w ≈ 0.8
    ])
    def test_weight_varies_with_prior_strength(self, sigma0, expected_w):
        """Test that w varies correctly with prior strength."""
        sigma = 1.0
        w = weight(sigma, sigma0)

        assert np.isclose(w, expected_w, atol=0.01), (
            f"w={w:.3f}, expected≈{expected_w:.1f} for sigma0={sigma0}"
        )

    def test_posterior_mean_limiting_cases(self):
        """Test limiting cases of posterior mean."""
        D, mu0, sigma = 5.0, 0.0, 1.0

        # Very strong prior (sigma0 -> 0): mu_n -> mu0
        mu_n_strong, _, w_strong = posterior_params(D, mu0, sigma, sigma0=0.1)
        assert w_strong < 0.1, f"w should be small for strong prior, got {w_strong}"
        assert mu_n_strong < D * 0.2, "mu_n should be close to mu0 for strong prior"

        # Very weak prior (sigma0 -> inf): mu_n -> D
        mu_n_weak, _, w_weak = posterior_params(D, mu0, sigma, sigma0=10.0)
        assert w_weak > 0.9, f"w should be large for weak prior, got {w_weak}"
        assert np.abs(mu_n_weak - D) < 0.5, "mu_n should be close to D for weak prior"

    def test_vectorized_posterior_params(self):
        """Test that posterior_params handles array inputs."""
        D = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu0, sigma, sigma0 = 0.0, 1.0, 1.0

        mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

        assert len(mu_n) == len(D), "mu_n should have same length as D"
        expected_mu_n = w * D + (1 - w) * mu0
        np.testing.assert_allclose(mu_n, expected_mu_n)
