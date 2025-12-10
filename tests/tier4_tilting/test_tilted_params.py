"""
Tests for Theorems 6-8: Tilted posterior parameters and p-values.

Theorem 6: Tilted posterior parameters
    mu_eta = [w*D + (1-eta)*(1-w)*mu0] / [1 - eta*(1-w)]
    sigma_eta^2 = w*sigma^2 / [1 - eta*(1-w)]

Corollary:
    - eta = 0: Recovers WALDO (mu_eta = mu_n, sigma_eta = sigma_n)
    - eta = 1: Recovers Wald (mu_eta = D, sigma_eta = sigma)

Theorem 7: Non-centrality reduction
    lambda_eta = (1-eta)^2 * lambda_0

Theorem 8: Tilted p-value formula
    p_eta(theta) = Phi(b_eta - a_eta) + Phi(-a_eta - b_eta)
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import posterior_params, weight
from frasian.waldo import pvalue, noncentrality, wald_ci, confidence_interval
from frasian.tilting import (
    tilted_params,
    tilted_noncentrality,
    tilted_pvalue,
    tilted_ci,
    tilted_mode,
)

from conftest import (
    TestConfig,
    ModelParams,
    get_model_from_w,
    data_for_conflict,
    simulate_data,
)


@pytest.mark.tier4
class TestTiltedParameters:
    """Tests for Theorem 6: Tilted posterior parameters."""

    def test_tilted_at_eta_zero_equals_waldo(self, balanced_model):
        """Test that eta=0 recovers WALDO parameters."""
        D = 2.0
        model = balanced_model
        eta = 0.0

        # WALDO parameters
        mu_n, sigma_n, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        # Tilted parameters at eta=0
        mu_eta, sigma_eta, _ = tilted_params(D, model.mu0, model.sigma, model.sigma0, eta)

        assert np.isclose(mu_eta, mu_n, atol=1e-10), (
            f"mu_eta at eta=0 should equal mu_n: {mu_eta} vs {mu_n}"
        )
        assert np.isclose(sigma_eta, sigma_n, atol=1e-10), (
            f"sigma_eta at eta=0 should equal sigma_n: {sigma_eta} vs {sigma_n}"
        )

    def test_tilted_at_eta_one_equals_wald(self, balanced_model):
        """Test that eta=1 recovers Wald parameters."""
        D = 2.0
        model = balanced_model
        eta = 1.0

        # Tilted parameters at eta=1
        mu_eta, sigma_eta, _ = tilted_params(D, model.mu0, model.sigma, model.sigma0, eta)

        # Should equal D and sigma (Wald/MLE parameters)
        assert np.isclose(mu_eta, D, atol=1e-10), (
            f"mu_eta at eta=1 should equal D: {mu_eta} vs {D}"
        )
        assert np.isclose(sigma_eta, model.sigma, atol=1e-10), (
            f"sigma_eta at eta=1 should equal sigma: {sigma_eta} vs {model.sigma}"
        )

    def test_tilted_mean_moves_toward_mle(self, balanced_model):
        """Test that mu_eta moves from mu_n toward D as eta increases."""
        D = 3.0
        model = balanced_model

        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        means = []
        for eta in etas:
            mu_eta, _, _ = tilted_params(D, model.mu0, model.sigma, model.sigma0, eta)
            means.append(mu_eta)

        # Mean should increase monotonically (since D > mu0 = 0)
        for i in range(len(means) - 1):
            assert means[i+1] >= means[i] - 1e-10, (
                f"mu_eta should increase with eta: {means}"
            )

        # First should equal mu_n, last should equal D
        assert np.isclose(means[0], mu_n)
        assert np.isclose(means[-1], D)

    def test_tilted_variance_increases_with_eta(self, balanced_model):
        """Test that sigma_eta increases from sigma_n toward sigma as eta increases."""
        D = 2.0
        model = balanced_model

        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        sigma_n = model.sigma_n

        variances = []
        for eta in etas:
            _, sigma_eta, _ = tilted_params(D, model.mu0, model.sigma, model.sigma0, eta)
            variances.append(sigma_eta)

        # Variance should increase monotonically
        for i in range(len(variances) - 1):
            assert variances[i+1] >= variances[i] - 1e-10, (
                f"sigma_eta should increase with eta: {variances}"
            )

        assert np.isclose(variances[0], sigma_n)
        assert np.isclose(variances[-1], model.sigma)

    @pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_tilted_formula_direct(self, balanced_model, eta):
        """Test the tilted parameter formulas directly."""
        D = 2.0
        model = balanced_model
        w = model.w

        mu_eta, sigma_eta, _ = tilted_params(D, model.mu0, model.sigma, model.sigma0, eta)

        # Check formula: mu_eta = [w*D + (1-eta)*(1-w)*mu0] / [1 - eta*(1-w)]
        denom = 1 - eta * (1 - w)
        expected_mu = (w * D + (1 - eta) * (1 - w) * model.mu0) / denom

        # Check formula: sigma_eta^2 = w*sigma^2 / [1 - eta*(1-w)]
        expected_sigma = np.sqrt(w * model.sigma**2 / denom)

        assert np.isclose(mu_eta, expected_mu, atol=1e-10), (
            f"mu_eta formula mismatch at eta={eta}"
        )
        assert np.isclose(sigma_eta, expected_sigma, atol=1e-10), (
            f"sigma_eta formula mismatch at eta={eta}"
        )


@pytest.mark.tier4
class TestTiltedNoncentrality:
    """Tests for Theorem 7: Non-centrality reduction."""

    def test_noncentrality_reduction_formula(self, balanced_model):
        """Test lambda_eta = (1-eta)^2 * lambda_0."""
        theta = 2.0
        model = balanced_model

        # Base non-centrality
        lambda0 = noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)

        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for eta in etas:
            lambda_eta = tilted_noncentrality(lambda0, eta)
            expected = (1 - eta)**2 * lambda0

            assert np.isclose(lambda_eta, expected, atol=1e-10), (
                f"lambda_eta formula mismatch at eta={eta}: "
                f"got {lambda_eta}, expected {expected}"
            )

    def test_noncentrality_zero_at_eta_one(self, balanced_model):
        """Test that lambda_eta = 0 when eta = 1 (Wald case)."""
        theta = 2.0
        model = balanced_model

        lambda0 = noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)
        lambda_1 = tilted_noncentrality(lambda0, eta=1.0)

        assert np.isclose(lambda_1, 0.0, atol=1e-10), (
            f"lambda at eta=1 should be 0, got {lambda_1}"
        )

    def test_noncentrality_unchanged_at_eta_zero(self, balanced_model):
        """Test that lambda_eta = lambda_0 when eta = 0 (WALDO case)."""
        theta = 2.0
        model = balanced_model

        lambda0 = noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)
        lambda_0 = tilted_noncentrality(lambda0, eta=0.0)

        assert np.isclose(lambda_0, lambda0, atol=1e-10), (
            f"lambda at eta=0 should equal lambda0"
        )

    def test_noncentrality_decreases_with_eta(self, balanced_model):
        """Test that lambda_eta decreases monotonically with eta."""
        theta = 2.0
        model = balanced_model

        lambda0 = noncentrality(theta, model.mu0, model.w, model.sigma, model.sigma0)

        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        lambdas = [tilted_noncentrality(lambda0, eta) for eta in etas]

        for i in range(len(lambdas) - 1):
            assert lambdas[i+1] <= lambdas[i] + 1e-10, (
                f"lambda should decrease with eta: {lambdas}"
            )


@pytest.mark.tier4
class TestTiltedPvalue:
    """Tests for Theorem 8: Tilted p-value formula."""

    def test_tilted_pvalue_at_eta_zero_equals_waldo(self, balanced_model):
        """Test that tilted p-value at eta=0 equals WALDO p-value."""
        D = 2.0
        theta = 1.5
        model = balanced_model

        # WALDO p-value
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        p_waldo = pvalue(theta, mu_n, model.mu0, w, model.sigma)

        # Tilted p-value at eta=0
        p_tilted = tilted_pvalue(theta, D, model.mu0, model.sigma, model.sigma0, eta=0.0)

        assert np.isclose(p_waldo, p_tilted, atol=1e-10), (
            f"Tilted p-value at eta=0 should equal WALDO: {p_tilted} vs {p_waldo}"
        )

    def test_tilted_pvalue_at_eta_one_is_wald(self, balanced_model):
        """Test that tilted p-value at eta=1 is symmetric (Wald-like)."""
        D = 2.0
        theta = 1.5
        model = balanced_model

        p_tilted = tilted_pvalue(theta, D, model.mu0, model.sigma, model.sigma0, eta=1.0)

        # At eta=1, the p-value should be Wald: 2*Phi(-|D - theta|/sigma)
        p_wald = 2 * stats.norm.cdf(-np.abs(D - theta) / model.sigma)

        assert np.isclose(p_tilted, p_wald, atol=1e-10), (
            f"Tilted p-value at eta=1 should equal Wald: {p_tilted} vs {p_wald}"
        )

    def test_tilted_pvalue_at_mode_equals_one(self, balanced_model):
        """Test that p(mu_eta) = 1 for all eta."""
        D = 2.0
        model = balanced_model

        for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mu_eta, _, _ = tilted_params(D, model.mu0, model.sigma, model.sigma0, eta)
            p_at_mode = tilted_pvalue(mu_eta, D, model.mu0, model.sigma, model.sigma0, eta)

            assert np.isclose(p_at_mode, 1.0, atol=1e-10), (
                f"p(mu_eta) should be 1 at eta={eta}, got {p_at_mode}"
            )

    @pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_tilted_pvalue_in_unit_interval(self, balanced_model, eta):
        """Test that 0 <= p_eta <= 1 for various theta."""
        D = 2.0
        model = balanced_model

        thetas = np.linspace(-5, 10, 50)
        for theta in thetas:
            p = tilted_pvalue(theta, D, model.mu0, model.sigma, model.sigma0, eta)
            assert 0 <= p <= 1 + 1e-10, f"p-value out of bounds at theta={theta}, eta={eta}"


@pytest.mark.tier4
class TestTiltedCI:
    """Tests for tilted confidence intervals."""

    def test_tilted_ci_at_eta_zero_equals_waldo(self, balanced_model):
        """Test that tilted CI at eta=0 equals WALDO CI."""
        D = 2.0
        model = balanced_model

        # WALDO CI
        waldo_lower, waldo_upper = confidence_interval(
            D, model.mu0, model.sigma, model.sigma0
        )

        # Tilted CI at eta=0
        tilt_lower, tilt_upper = tilted_ci(
            D, model.mu0, model.sigma, model.sigma0, eta=0.0
        )

        assert np.isclose(waldo_lower, tilt_lower, atol=0.01), (
            f"Lower bounds mismatch: WALDO={waldo_lower}, tilted={tilt_lower}"
        )
        assert np.isclose(waldo_upper, tilt_upper, atol=0.01), (
            f"Upper bounds mismatch: WALDO={waldo_upper}, tilted={tilt_upper}"
        )

    def test_tilted_ci_at_eta_one_equals_wald(self, balanced_model):
        """Test that tilted CI at eta=1 equals Wald CI."""
        D = 2.0
        model = balanced_model

        # Wald CI
        wald_lower, wald_upper = wald_ci(D, model.sigma)

        # Tilted CI at eta=1
        tilt_lower, tilt_upper = tilted_ci(
            D, model.mu0, model.sigma, model.sigma0, eta=1.0
        )

        assert np.isclose(wald_lower, tilt_lower, atol=0.01), (
            f"Lower bounds mismatch: Wald={wald_lower}, tilted={tilt_lower}"
        )
        assert np.isclose(wald_upper, tilt_upper, atol=0.01), (
            f"Upper bounds mismatch: Wald={wald_upper}, tilted={tilt_upper}"
        )

    def test_tilted_ci_contains_mode(self, balanced_model):
        """Test that tilted CI contains its mode mu_eta."""
        D = 2.0
        model = balanced_model

        for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mu_eta = tilted_mode(D, model.mu0, model.sigma, model.sigma0, eta)
            lower, upper = tilted_ci(D, model.mu0, model.sigma, model.sigma0, eta)

            assert lower < mu_eta < upper, (
                f"CI should contain mode at eta={eta}: "
                f"CI=({lower:.2f}, {upper:.2f}), mode={mu_eta:.2f}"
            )


@pytest.mark.tier4
class TestTiltedCoverage:
    """Tests for tilted CI coverage properties."""

    @pytest.mark.slow
    @pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
    def test_tilted_coverage_at_various_eta(self, balanced_model, config, rng, eta):
        """Test that tilted CI maintains ~95% coverage at various eta."""
        theta_true = 2.0
        model = balanced_model

        D_samples = simulate_data(theta_true, model.sigma, config.n_coverage, rng)
        covered = 0

        for D in D_samples:
            lower, upper = tilted_ci(D, model.mu0, model.sigma, model.sigma0, eta)
            if lower <= theta_true <= upper:
                covered += 1

        coverage = covered / config.n_coverage

        assert abs(coverage - 0.95) < config.coverage_tol, (
            f"Tilted coverage at eta={eta}: {coverage:.3f}, expected ~0.95"
        )
