"""
Tests for Theorem 3: The WALDO p-value function.

Theorem 3: The WALDO p-value function is:
    p(theta) = Phi(b - a) + Phi(-a - b)
where:
    a(theta) = |mu_n - theta| / (w * sigma) >= 0
    b(theta) = (1 - w) * (mu0 - theta) / (w * sigma)

Verification properties:
- p(mu_n) = 1 (p-value at mode equals 1)
- p(theta) -> 0 as theta -> +/- infinity
- When b = 0: p(theta) = 2*Phi(-a) (symmetric Wald p-value)
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import posterior_params, scaled_conflict
from frasian.waldo import pvalue, pvalue_components, pvalue_from_data

from conftest import (
    TestConfig,
    ModelParams,
    get_model_from_w,
    simulate_data,
)


@pytest.mark.tier2
class TestPvalueFormula:
    """Tests for the p-value formula p(theta) = Phi(b-a) + Phi(-a-b)."""

    def test_pvalue_formula_matches_components(self, balanced_model):
        """Test that p(theta) = Phi(b-a) + Phi(-a-b) using components."""
        D = 2.0
        theta = 1.5
        model = balanced_model

        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        # Get components
        a, b = pvalue_components(theta, mu_n, model.mu0, w, model.sigma)

        # Compute p-value from formula
        p_from_formula = stats.norm.cdf(b - a) + stats.norm.cdf(-a - b)

        # Compute using function
        p_from_function = pvalue(theta, mu_n, model.mu0, w, model.sigma)

        assert np.isclose(p_from_formula, p_from_function, atol=1e-10), (
            f"Formula mismatch: manual={p_from_formula:.6f}, function={p_from_function:.6f}"
        )

    def test_a_is_nonnegative(self, balanced_model):
        """Test that a(theta) >= 0 for all theta."""
        D = 2.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        thetas = np.linspace(-10, 10, 100)
        for theta in thetas:
            a, _ = pvalue_components(theta, mu_n, model.mu0, w, model.sigma)
            assert a >= 0, f"a should be non-negative, got {a} at theta={theta}"

    def test_pvalue_in_unit_interval(self, balanced_model):
        """Test that 0 <= p(theta) <= 1 for all theta."""
        D = 2.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        thetas = np.linspace(-10, 10, 100)
        for theta in thetas:
            p = pvalue(theta, mu_n, model.mu0, w, model.sigma)
            assert 0 <= p <= 1, f"p-value out of bounds: {p} at theta={theta}"


@pytest.mark.tier2
class TestPvalueAtMode:
    """Tests verifying p(mu_n) = 1."""

    def test_pvalue_equals_one_at_mode(self, balanced_model):
        """Test p(mu_n) = 1."""
        D = 2.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        p_at_mode = pvalue(mu_n, mu_n, model.mu0, w, model.sigma)

        assert np.isclose(p_at_mode, 1.0, atol=1e-10), (
            f"p(mu_n) should be 1.0, got {p_at_mode}"
        )

    def test_a_equals_zero_at_mode(self, balanced_model):
        """Test that a(mu_n) = 0."""
        D = 2.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        a, b = pvalue_components(mu_n, mu_n, model.mu0, w, model.sigma)

        assert np.isclose(a, 0.0, atol=1e-10), f"a(mu_n) should be 0, got {a}"

    def test_pvalue_sum_at_mode(self, balanced_model):
        """Test that Phi(b) + Phi(-b) = 1 when a = 0."""
        D = 2.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        _, b = pvalue_components(mu_n, mu_n, model.mu0, w, model.sigma)

        sum_phi = stats.norm.cdf(b) + stats.norm.cdf(-b)
        assert np.isclose(sum_phi, 1.0, atol=1e-10), (
            f"Phi(b) + Phi(-b) should be 1, got {sum_phi}"
        )

    @pytest.mark.parametrize("D", [-5.0, -2.0, 0.0, 2.0, 5.0])
    def test_pvalue_at_mode_for_various_D(self, balanced_model, D):
        """Test p(mu_n) = 1 for various data values."""
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        p_at_mode = pvalue(mu_n, mu_n, model.mu0, w, model.sigma)

        assert np.isclose(p_at_mode, 1.0, atol=1e-10), (
            f"p(mu_n) should be 1.0 for D={D}, got {p_at_mode}"
        )


@pytest.mark.tier2
class TestPvalueLimits:
    """Tests verifying p(theta) -> 0 as theta -> +/- infinity."""

    def test_pvalue_approaches_zero_large_positive_theta(self, balanced_model):
        """Test p(theta) -> 0 as theta -> +infinity."""
        D = 0.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        large_thetas = [10, 50, 100, 500]
        for theta in large_thetas:
            p = pvalue(theta, mu_n, model.mu0, w, model.sigma)
            assert p < 0.01, f"p({theta}) should be near 0, got {p}"

    def test_pvalue_approaches_zero_large_negative_theta(self, balanced_model):
        """Test p(theta) -> 0 as theta -> -infinity."""
        D = 0.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        large_thetas = [-10, -50, -100, -500]
        for theta in large_thetas:
            p = pvalue(theta, mu_n, model.mu0, w, model.sigma)
            assert p < 0.01, f"p({theta}) should be near 0, got {p}"

    def test_pvalue_monotonic_away_from_mode(self, balanced_model):
        """Test that p(theta) decreases as |theta - mu_n| increases (approximately)."""
        D = 2.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        # Check on one side
        thetas = [mu_n, mu_n + 1, mu_n + 2, mu_n + 3, mu_n + 4]
        pvals = [pvalue(theta, mu_n, model.mu0, w, model.sigma) for theta in thetas]

        # Should generally decrease (not strictly due to asymmetry)
        assert pvals[0] > pvals[-1], "p-value should decrease away from mode"


@pytest.mark.tier2
class TestPvalueSymmetry:
    """Tests for the symmetry property: when b = 0, p(theta) = 2*Phi(-a)."""

    def test_wald_pvalue_when_theta_gives_b_zero(self, balanced_model):
        """Test that p(theta) = 2*Phi(-a) when b(theta) = 0.

        b(theta) = 0 when theta = mu0 (prior mean).
        """
        D = 2.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        # Test at theta = mu0, where b = 0
        theta = model.mu0
        a, b = pvalue_components(theta, mu_n, model.mu0, w, model.sigma)

        assert np.isclose(b, 0.0, atol=1e-10), f"b should be 0 at theta=mu0, got {b}"

        # Check formula reduces to 2*Phi(-a)
        p_formula = pvalue(theta, mu_n, model.mu0, w, model.sigma)
        p_wald = 2 * stats.norm.cdf(-a)

        assert np.isclose(p_formula, p_wald, atol=1e-10), (
            f"When b=0: p={p_formula:.6f} should equal 2*Phi(-a)={p_wald:.6f}"
        )

    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    def test_symmetry_at_mu0_for_various_w(self, w):
        """Test p(mu0) = 2*Phi(-a) for various weights."""
        D = 2.0
        model = get_model_from_w(w)
        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        theta = model.mu0
        a, b = pvalue_components(theta, mu_n, model.mu0, w, model.sigma)

        p_formula = pvalue(theta, mu_n, model.mu0, w, model.sigma)
        p_wald = 2 * stats.norm.cdf(-a)

        assert np.isclose(p_formula, p_wald, atol=1e-10), (
            f"Symmetry failed at w={w}: p={p_formula:.6f}, 2*Phi(-a)={p_wald:.6f}"
        )


@pytest.mark.tier2
class TestPvalueStandardizedCoordinates:
    """Tests for p-value in standardized coordinates u = (theta - mu_n) / (w*sigma)."""

    def test_pvalue_formula_in_u_coordinates(self, balanced_model):
        """Test the standardized coordinate formula for p(u).

        For u >= 0: p(u) = Phi(Delta - (2-w)*u) + Phi(-w*u - Delta)
        For u < 0: p(u) = Phi(Delta + w*u) + Phi((2-w)*u - Delta)
        """
        D = 2.0
        model = balanced_model
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        Delta = scaled_conflict(D, model.mu0, w, model.sigma)

        # Test several u values
        us = [-2.0, -1.0, 0.0, 1.0, 2.0]
        for u in us:
            # Convert u to theta
            theta = mu_n + w * model.sigma * u

            # Compute p-value using main formula
            p_main = pvalue(theta, mu_n, model.mu0, w, model.sigma)

            # Compute using u-coordinate formula
            if u >= 0:
                p_u = (stats.norm.cdf(Delta - (2 - w) * u) +
                       stats.norm.cdf(-w * u - Delta))
            else:
                p_u = (stats.norm.cdf(Delta + w * u) +
                       stats.norm.cdf((2 - w) * u - Delta))

            assert np.isclose(p_main, p_u, atol=1e-8), (
                f"Formula mismatch at u={u}: main={p_main:.6f}, u_coord={p_u:.6f}"
            )


@pytest.mark.tier2
class TestPvalueFromData:
    """Tests for the convenience function pvalue_from_data."""

    def test_pvalue_from_data_matches_manual(self, balanced_model):
        """Test that pvalue_from_data gives same result as manual computation."""
        D = 2.0
        theta = 1.5
        model = balanced_model

        # Manual computation
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        p_manual = pvalue(theta, mu_n, model.mu0, w, model.sigma)

        # Using convenience function
        p_auto = pvalue_from_data(theta, D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(p_manual, p_auto, atol=1e-10), (
            f"Mismatch: manual={p_manual}, auto={p_auto}"
        )


@pytest.mark.tier2
class TestPvalueFrequentistProperty:
    """Test that under true theta, p(theta) ~ Uniform(0,1)."""

    @pytest.mark.slow
    def test_pvalue_uniform_under_true_theta(self, balanced_model, config, rng):
        """Test that p(theta_true) is uniformly distributed (confidence distribution property)."""
        theta_true = 2.0
        model = balanced_model

        # Simulate many data sets and compute p-values at true theta
        D_samples = simulate_data(theta_true, model.sigma, config.n_samples, rng)

        pvals = []
        for D in D_samples:
            mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)
            p = pvalue(theta_true, mu_n, model.mu0, w, model.sigma)
            pvals.append(p)

        pvals = np.array(pvals)

        # KS test against Uniform(0,1)
        ks_stat, ks_pval = stats.kstest(pvals, 'uniform')

        assert ks_pval > config.ks_threshold, (
            f"p(theta_true) should be uniform: KS stat={ks_stat:.4f}, p-value={ks_pval:.4f}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("theta_true", [-2.0, 0.0, 2.0])
    def test_pvalue_uniform_at_various_theta(self, balanced_model, config, rng, theta_true):
        """Test uniformity at various true theta values."""
        model = balanced_model

        D_samples = simulate_data(theta_true, model.sigma, config.n_samples, rng)

        pvals = []
        for D in D_samples:
            mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)
            p = pvalue(theta_true, mu_n, model.mu0, w, model.sigma)
            pvals.append(p)

        pvals = np.array(pvals)

        ks_stat, ks_pval = stats.kstest(pvals, 'uniform')

        assert ks_pval > config.ks_threshold, (
            f"p(theta_true={theta_true}) should be uniform: "
            f"KS stat={ks_stat:.4f}, p-value={ks_pval:.4f}"
        )
