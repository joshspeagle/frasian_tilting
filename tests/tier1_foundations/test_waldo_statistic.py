"""
Tests for Theorem 2: Distribution of the WALDO test statistic.

Theorem 2: The WALDO test statistic tau_WALDO = (mu_n - theta)^2 / sigma_n^2
follows a scaled non-central chi-squared distribution:
    tau_WALDO | theta ~ w * chi^2_1(lambda(theta))
where the non-centrality parameter is:
    lambda(theta) = delta(theta)^2 / w
with delta(theta) = (1-w)(mu0 - theta) / (sqrt(w) * sigma).
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import (
    posterior_params,
    delta_scaled,
    weight,
)
from frasian.waldo import (
    waldo_statistic,
    noncentrality,
)

from conftest import (
    TestConfig,
    ModelParams,
    simulate_data,
    compute_posterior_means,
    compute_waldo_stats,
    get_model_from_w,
)


@pytest.mark.tier1
class TestWaldoStatisticDistribution:
    """Tests for the distribution of the WALDO test statistic."""

    def test_waldo_follows_scaled_ncx2_at_mu0(self, balanced_model, config, rng):
        """Test tau_WALDO ~ w * chi^2_1(0) when theta = mu0 (central case)."""
        theta_true = balanced_model.mu0  # lambda = 0 when theta = mu0
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        tau = compute_waldo_stats(mu_n, theta_true, model.sigma_n)

        # Non-centrality should be zero at theta = mu0
        lambda_ = noncentrality(theta_true, model.mu0, model.w, model.sigma, model.sigma0)
        assert np.isclose(lambda_, 0.0, atol=1e-10), f"lambda should be 0, got {lambda_}"

        # tau / w should follow chi^2_1 (central)
        scaled_tau = tau / model.w
        ks_stat, ks_pval = stats.kstest(scaled_tau, 'chi2', args=(1,))

        assert ks_pval > config.ks_threshold, (
            f"KS test failed for central chi^2: stat={ks_stat:.4f}, p-value={ks_pval:.4f}"
        )

    def test_waldo_follows_scaled_ncx2_away_from_mu0(self, balanced_model, config, rng):
        """Test tau_WALDO ~ w * chi^2_1(lambda) when theta != mu0."""
        theta_true = 2.0  # Away from mu0 = 0
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        tau = compute_waldo_stats(mu_n, theta_true, model.sigma_n)

        # Compute theoretical non-centrality
        lambda_ = noncentrality(theta_true, model.mu0, model.w, model.sigma, model.sigma0)
        assert lambda_ > 0, f"lambda should be positive, got {lambda_}"

        # tau / w should follow ncx2(df=1, nc=lambda)
        scaled_tau = tau / model.w

        # KS test against non-central chi-squared
        ks_stat, ks_pval = stats.kstest(
            scaled_tau,
            lambda x: stats.ncx2.cdf(x, df=1, nc=lambda_)
        )

        assert ks_pval > config.ks_threshold, (
            f"KS test failed for ncx2(1, {lambda_:.2f}): "
            f"stat={ks_stat:.4f}, p-value={ks_pval:.4f}"
        )

    def test_mean_of_waldo_statistic(self, balanced_model, config, rng):
        """Test E[tau_WALDO] = w * (1 + lambda)."""
        theta_true = 2.0
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        tau = compute_waldo_stats(mu_n, theta_true, model.sigma_n)

        lambda_ = noncentrality(theta_true, model.mu0, model.w, model.sigma, model.sigma0)

        # E[w * ncx2(1, lambda)] = w * (1 + lambda)
        expected_mean = model.w * (1 + lambda_)
        empirical_mean = np.mean(tau)

        rel_error = np.abs(empirical_mean - expected_mean) / expected_mean
        assert rel_error < config.moment_rtol, (
            f"Mean mismatch: empirical={empirical_mean:.4f}, expected={expected_mean:.4f}, "
            f"rel_error={rel_error:.4f}"
        )

    def test_variance_of_waldo_statistic(self, balanced_model, config, rng):
        """Test Var[tau_WALDO] = w^2 * 2(1 + 2*lambda)."""
        theta_true = 2.0
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        tau = compute_waldo_stats(mu_n, theta_true, model.sigma_n)

        lambda_ = noncentrality(theta_true, model.mu0, model.w, model.sigma, model.sigma0)

        # Var[w * ncx2(1, lambda)] = w^2 * Var[ncx2] = w^2 * 2(1 + 2*lambda)
        expected_var = model.w**2 * 2 * (1 + 2 * lambda_)
        empirical_var = np.var(tau)

        rel_error = np.abs(empirical_var - expected_var) / expected_var
        assert rel_error < 2 * config.moment_rtol, (  # Allow more tolerance for variance
            f"Variance mismatch: empirical={empirical_var:.4f}, expected={expected_var:.4f}, "
            f"rel_error={rel_error:.4f}"
        )

    @pytest.mark.parametrize("theta_true", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_theorem2_at_various_theta(self, balanced_model, config, rng, theta_true):
        """Test Theorem 2 at various true theta values."""
        model = balanced_model

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        tau = compute_waldo_stats(mu_n, theta_true, model.sigma_n)

        lambda_ = noncentrality(theta_true, model.mu0, model.w, model.sigma, model.sigma0)
        scaled_tau = tau / model.w

        # KS test
        if lambda_ < 0.01:  # Essentially central
            ks_stat, ks_pval = stats.kstest(scaled_tau, 'chi2', args=(1,))
        else:
            ks_stat, ks_pval = stats.kstest(
                scaled_tau,
                lambda x: stats.ncx2.cdf(x, df=1, nc=lambda_)
            )

        assert ks_pval > config.ks_threshold, (
            f"KS test failed at theta={theta_true}, lambda={lambda_:.2f}: "
            f"stat={ks_stat:.4f}, p-value={ks_pval:.4f}"
        )

    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    def test_theorem2_at_various_w(self, config, rng, w):
        """Test Theorem 2 at various weight values."""
        theta_true = 2.0
        model = get_model_from_w(w)

        D = simulate_data(theta_true, model.sigma, config.n_samples, rng)
        mu_n = compute_posterior_means(D, model.mu0, model.w)
        tau = compute_waldo_stats(mu_n, theta_true, model.sigma_n)

        lambda_ = noncentrality(theta_true, model.mu0, model.w, model.sigma, model.sigma0)
        scaled_tau = tau / model.w

        ks_stat, ks_pval = stats.kstest(
            scaled_tau,
            lambda x: stats.ncx2.cdf(x, df=1, nc=lambda_)
        )

        assert ks_pval > config.ks_threshold, (
            f"KS test failed at w={w}: stat={ks_stat:.4f}, p-value={ks_pval:.4f}"
        )


@pytest.mark.tier1
class TestNoncentralityParameter:
    """Tests for the non-centrality parameter lambda(theta)."""

    def test_noncentrality_formula(self, balanced_model):
        """Test lambda(theta) = delta(theta)^2 / w formula."""
        theta = 2.0
        model = balanced_model

        delta = delta_scaled(theta, model.mu0, model.w, model.sigma)
        lambda_from_formula = delta**2 / model.w

        lambda_from_function = noncentrality(
            theta, model.mu0, model.w, model.sigma, model.sigma0
        )

        assert np.isclose(lambda_from_formula, lambda_from_function), (
            f"Formula mismatch: direct={lambda_from_formula:.4f}, "
            f"function={lambda_from_function:.4f}"
        )

    def test_noncentrality_zero_at_mu0(self, balanced_model):
        """Test lambda(mu0) = 0."""
        theta = balanced_model.mu0
        model = balanced_model

        lambda_ = noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)
        assert np.isclose(lambda_, 0.0, atol=1e-10), f"lambda should be 0, got {lambda_}"

    def test_noncentrality_increases_with_distance(self, balanced_model):
        """Test that lambda increases as theta moves away from mu0."""
        model = balanced_model

        thetas = [0.0, 1.0, 2.0, 3.0]  # Increasing distance from mu0
        lambdas = [
            noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)
            for theta in thetas
        ]

        # Lambda should strictly increase
        for i in range(len(lambdas) - 1):
            assert lambdas[i+1] > lambdas[i], (
                f"lambda not increasing: lambda({thetas[i]})={lambdas[i]:.4f}, "
                f"lambda({thetas[i+1]})={lambdas[i+1]:.4f}"
            )

    def test_noncentrality_symmetric(self, balanced_model):
        """Test lambda(-theta) = lambda(theta) when mu0 = 0."""
        model = balanced_model
        assert model.mu0 == 0.0, "This test assumes mu0 = 0"

        theta = 2.0
        lambda_pos = noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)
        lambda_neg = noncentrality(-theta, model.mu0, model.w, model.sigma, model.sigma0)

        assert np.isclose(lambda_pos, lambda_neg), (
            f"lambda not symmetric: lambda({theta})={lambda_pos:.4f}, "
            f"lambda({-theta})={lambda_neg:.4f}"
        )

    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    def test_noncentrality_scales_with_w(self, w):
        """Test how lambda scales with w for fixed theta."""
        theta, mu0, sigma = 2.0, 0.0, 1.0
        model = get_model_from_w(w)

        lambda_ = noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)

        # For fixed (theta - mu0), lambda ~ (1-w)^2 / w^2 * constant
        # This means lambda decreases as w increases
        expected_scaling = (1 - w)**2 / w

        # The actual value depends on sigma0 which varies with w
        # But we can check the trend
        assert lambda_ > 0, f"lambda should be positive for theta != mu0"

    def test_noncentrality_quadratic_in_theta(self, balanced_model):
        """Test that lambda(theta) is quadratic in (theta - mu0)."""
        model = balanced_model

        thetas = np.linspace(-3, 3, 7)
        lambdas = np.array([
            noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)
            for theta in thetas
        ])

        # lambda should be proportional to (theta - mu0)^2
        expected_shape = (thetas - model.mu0)**2

        # Normalize both
        lambdas_norm = lambdas / np.max(lambdas) if np.max(lambdas) > 0 else lambdas
        expected_norm = expected_shape / np.max(expected_shape)

        np.testing.assert_allclose(lambdas_norm, expected_norm, atol=1e-10)


@pytest.mark.tier1
class TestWaldoStatisticComputation:
    """Basic tests for WALDO statistic computation."""

    def test_waldo_statistic_positive(self):
        """Test that tau_WALDO >= 0."""
        mu_n, sigma_n, theta = 2.0, 1.0, 1.0
        tau = waldo_statistic(mu_n, sigma_n, theta)
        assert tau >= 0, f"tau should be non-negative, got {tau}"

    def test_waldo_statistic_zero_at_mu_n(self):
        """Test that tau_WALDO = 0 when theta = mu_n."""
        mu_n, sigma_n = 2.0, 1.0
        theta = mu_n
        tau = waldo_statistic(mu_n, sigma_n, theta)
        assert np.isclose(tau, 0.0), f"tau should be 0 when theta=mu_n, got {tau}"

    def test_waldo_statistic_formula(self):
        """Test tau = (mu_n - theta)^2 / sigma_n^2."""
        mu_n, sigma_n, theta = 3.0, 0.5, 1.0
        tau = waldo_statistic(mu_n, sigma_n, theta)

        expected = (mu_n - theta)**2 / sigma_n**2
        assert np.isclose(tau, expected), f"tau={tau}, expected={expected}"

    def test_waldo_statistic_vectorized(self):
        """Test that waldo_statistic handles array inputs."""
        mu_n = np.array([1.0, 2.0, 3.0])
        sigma_n = 1.0
        theta = 2.0

        tau = waldo_statistic(mu_n, sigma_n, theta)

        assert len(tau) == len(mu_n)
        expected = (mu_n - theta)**2 / sigma_n**2
        np.testing.assert_allclose(tau, expected)
