"""
Tests for Theorems 4-5: Mode and Mean of the WALDO confidence distribution.

Theorem 4: The mode of the WALDO CD equals mu_n (posterior mean).
           This is where p(theta) = 1.

Key properties (from Schweder-Hjort methodology):
- WALDO CD is a 50-50 Gaussian mixture: 0.5 * N(D, σ²) + 0.5 * N(μ*, σ*²)
- Mode = μ_n (posterior mean)
- Mean = (μ_n + (1-w)D) / (2-w)
- The mean lies between the mode (mu_n) and the MLE (D).
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import posterior_params, scaled_conflict
from frasian.confidence import (
    pvalue_mode,
    pvalue_at_mode,
    verify_mode_is_max,
    waldo_cd_params,
    waldo_cd_density,
    waldo_cd_mean,
    waldo_cd_mode,
    wald_cd_mean,
    wald_cd_mode,
)

from conftest import (
    TestConfig,
    ModelParams,
    get_model_from_w,
    data_for_conflict,
)


@pytest.mark.tier2
class TestModeEqualsPosteriorMean:
    """Tests for Theorem 4: Mode of WALDO CD equals mu_n."""

    def test_mode_equals_mu_n(self, balanced_model):
        """Test that waldo_cd_mode returns mu_n."""
        D = 2.0
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mode, mu_n, atol=1e-10), (
            f"Mode should equal mu_n: mode={mode}, mu_n={mu_n}"
        )

    def test_pvalue_mode_equals_mu_n(self, balanced_model):
        """Test that pvalue_mode (legacy function) returns mu_n."""
        D = 2.0
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mode, mu_n, atol=1e-10), (
            f"Mode should equal mu_n: mode={mode}, mu_n={mu_n}"
        )

    def test_pvalue_at_mode_is_one(self, balanced_model):
        """Test that p(mode) = 1."""
        D = 2.0
        model = balanced_model

        p_at_mode = pvalue_at_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(p_at_mode, 1.0, atol=1e-10), (
            f"p(mode) should be 1.0, got {p_at_mode}"
        )

    def test_mode_is_maximum(self, balanced_model):
        """Test that the mode is indeed the maximum of p(theta)."""
        D = 2.0
        model = balanced_model

        is_max, p_at_mode, max_found = verify_mode_is_max(
            D, model.mu0, model.sigma, model.sigma0
        )

        # p_at_mode should be 1.0 (the theoretical max)
        assert p_at_mode >= max_found - 1e-6, (
            f"Mode is not maximum: p(mode)={p_at_mode}, max found={max_found}"
        )
        assert np.isclose(p_at_mode, 1.0, atol=1e-10), (
            f"p(mode) should be 1.0, got {p_at_mode}"
        )

    @pytest.mark.parametrize("D", [-5.0, -2.0, 0.0, 2.0, 5.0])
    def test_mode_equals_mu_n_various_D(self, balanced_model, D):
        """Test mode = mu_n for various data values."""
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mode, mu_n), f"Mode mismatch at D={D}: mode={mode}, mu_n={mu_n}"

    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    def test_mode_equals_mu_n_various_w(self, w):
        """Test mode = mu_n for various weight values."""
        D = 2.0
        model = get_model_from_w(w)

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mode, mu_n), f"Mode mismatch at w={w}: mode={mode}, mu_n={mu_n}"


@pytest.mark.tier2
class TestWaldoCDMeanFormula:
    """Tests for the WALDO CD mean: E[θ] = (μ_n + (1-w)D) / (2-w)."""

    def test_mean_formula(self, balanced_model):
        """Test the closed-form mean formula."""
        D = 2.0
        model = balanced_model

        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        # Expected mean from formula
        expected_mean = (mu_n + (1 - w) * D) / (2 - w)

        # Computed mean
        computed_mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(computed_mean, expected_mean, atol=1e-10), (
            f"Mean formula mismatch: computed={computed_mean:.4f}, expected={expected_mean:.4f}"
        )

    def test_mean_equals_mode_when_D_equals_mu0(self, balanced_model):
        """Test that mean ≈ mode when D = mu0 (no conflict)."""
        D = balanced_model.mu0  # Delta = 0
        model = balanced_model

        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)
        mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        # When D = mu0, both mu_n = mu0, so mean = (mu0 + (1-w)*mu0)/(2-w) = mu0
        assert np.isclose(mean, mode, atol=1e-6), (
            f"When D=mu0: mean should equal mode. mean={mean:.4f}, mode={mode:.4f}"
        )

    @pytest.mark.parametrize("D", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_mean_formula_various_D(self, balanced_model, D):
        """Test mean formula for various data values."""
        model = balanced_model

        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        expected = (mu_n + (1 - w) * D) / (2 - w)
        computed = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(computed, expected, atol=1e-10), (
            f"Mean mismatch at D={D}: computed={computed:.4f}, expected={expected:.4f}"
        )


@pytest.mark.tier2
class TestMeanBetweenModeAndMLE:
    """Tests for the property: mu_n < E[theta] < D (when D > mu_n)."""

    def test_mean_between_mode_and_mle_D_above_mu0(self, balanced_model):
        """Test mu_n < E[theta] < D when D > mu0."""
        D = 3.0  # D > mu0 = 0
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)
        mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        assert D > model.mu0, "Precondition: D > mu0"
        assert mode < D, f"Mode should be less than D: mode={mode}, D={D}"
        assert mode < mean < D, (
            f"Mean should be between mode and D: mode={mode:.4f}, mean={mean:.4f}, D={D:.4f}"
        )

    def test_mean_between_mode_and_mle_D_below_mu0(self, balanced_model):
        """Test D < E[theta] < mu_n when D < mu0."""
        D = -3.0  # D < mu0 = 0
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)
        mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        assert D < model.mu0, "Precondition: D < mu0"
        assert mode > D, f"Mode should be greater than D: mode={mode}, D={D}"
        assert D < mean < mode, (
            f"Mean should be between D and mode: D={D:.4f}, mean={mean:.4f}, mode={mode:.4f}"
        )

    @pytest.mark.parametrize("D", [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0])
    def test_mean_ordering_various_D(self, balanced_model, D):
        """Test mean ordering for various data values."""
        model = balanced_model

        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)
        mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        # Mean should be between mode and D
        if D > model.mu0:
            assert mode <= mean <= D, (
                f"Mean ordering violated at D={D}: mode={mode:.4f}, mean={mean:.4f}, D={D:.4f}"
            )
        else:
            assert D <= mean <= mode, (
                f"Mean ordering violated at D={D}: D={D:.4f}, mean={mean:.4f}, mode={mode:.4f}"
            )

    def test_mean_pulled_toward_mle(self, balanced_model):
        """Test that mean is closer to MLE than mode is."""
        D = 3.0
        model = balanced_model

        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)
        mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        dist_mode_to_mle = np.abs(D - mode)
        dist_mean_to_mle = np.abs(D - mean)

        assert dist_mean_to_mle < dist_mode_to_mle, (
            f"Mean should be closer to MLE than mode is: "
            f"dist(mean,mle)={dist_mean_to_mle:.4f}, dist(mode,mle)={dist_mode_to_mle:.4f}"
        )


@pytest.mark.tier2
class TestWaldoCDMixtureProperties:
    """Tests for the WALDO CD as a Gaussian mixture."""

    def test_mixture_parameters(self, balanced_model):
        """Test that mixture parameters are computed correctly."""
        D = 2.0
        model = balanced_model

        params = waldo_cd_params(D, model.mu0, model.sigma, model.sigma0)
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)

        # Component 1: N(D, sigma^2)
        assert np.isclose(params['component1_mean'], D)
        assert np.isclose(params['component1_std'], model.sigma)

        # Component 2: N(mu*, sigma*^2)
        # mu* = (wD + 2(1-w)mu0) / (2-w)
        expected_mu_star = (w * D + 2 * (1 - w) * model.mu0) / (2 - w)
        # sigma* = w*sigma / (2-w)
        expected_sigma_star = w * model.sigma / (2 - w)

        assert np.isclose(params['mu_star'], expected_mu_star, atol=1e-10), (
            f"mu* mismatch: got {params['mu_star']}, expected {expected_mu_star}"
        )
        assert np.isclose(params['sigma_star'], expected_sigma_star, atol=1e-10), (
            f"sigma* mismatch: got {params['sigma_star']}, expected {expected_sigma_star}"
        )

    def test_density_integrates_to_one(self, balanced_model):
        """Test that the WALDO CD density integrates to 1."""
        D = 2.0
        model = balanced_model

        # Create fine grid for integration
        theta_grid = np.linspace(-10, 15, 10000)
        density = waldo_cd_density(theta_grid, D, model.mu0, model.sigma, model.sigma0)

        integral = np.trapezoid(density, theta_grid)

        assert np.isclose(integral, 1.0, atol=0.01), (
            f"WALDO CD should integrate to 1, got {integral:.4f}"
        )

    def test_mean_matches_numerical_integration(self, balanced_model):
        """Test that closed-form mean matches numerical integration."""
        D = 2.0
        model = balanced_model

        # Closed form mean
        closed_mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        # Numerical mean
        theta_grid = np.linspace(-10, 15, 10000)
        density = waldo_cd_density(theta_grid, D, model.mu0, model.sigma, model.sigma0)
        numerical_mean = np.trapezoid(theta_grid * density, theta_grid)

        assert np.isclose(closed_mean, numerical_mean, atol=0.01), (
            f"Mean mismatch: closed={closed_mean:.4f}, numerical={numerical_mean:.4f}"
        )


@pytest.mark.tier2
class TestWaldCDComparison:
    """Tests comparing Wald CD to WALDO CD."""

    def test_wald_mode_equals_D(self, balanced_model):
        """Test that Wald CD mode equals D (MLE)."""
        D = 2.0

        mode = wald_cd_mode(D)
        assert np.isclose(mode, D), f"Wald mode should be D={D}, got {mode}"

    def test_wald_mean_equals_D(self, balanced_model):
        """Test that Wald CD mean equals D (MLE)."""
        D = 2.0

        mean = wald_cd_mean(D)
        assert np.isclose(mean, D), f"Wald mean should be D={D}, got {mean}"

    def test_waldo_mode_different_from_wald(self, balanced_model):
        """Test that WALDO mode differs from Wald mode when there's conflict."""
        D = 3.0  # D != mu0, so conflict exists
        model = balanced_model

        wald_mode = wald_cd_mode(D)
        waldo_mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)

        assert not np.isclose(waldo_mode, wald_mode), (
            f"WALDO mode should differ from Wald mode when D != mu0"
        )

    def test_waldo_mean_different_from_wald(self, balanced_model):
        """Test that WALDO mean differs from Wald mean when there's conflict."""
        D = 3.0
        model = balanced_model

        wald_mean = wald_cd_mean(D)
        waldo_mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        assert not np.isclose(waldo_mean, wald_mean), (
            f"WALDO mean should differ from Wald mean when D != mu0"
        )


@pytest.mark.tier2
class TestEstimatorConsistency:
    """Tests for consistency between different estimator computations."""

    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    @pytest.mark.parametrize("D", [-2.0, 0.0, 2.0])
    def test_mode_and_mean_consistency(self, w, D):
        """Test that mode and mean are computed consistently across parameters."""
        model = get_model_from_w(w)

        mode = waldo_cd_mode(D, model.mu0, model.sigma, model.sigma0)
        mean = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)
        mle = D

        # Mode should always equal mu_n
        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        assert np.isclose(mode, mu_n), f"Mode != mu_n at w={w}, D={D}"

        # Mean should be finite and between mode and MLE
        assert np.isfinite(mean), f"Mean not finite at w={w}, D={D}"

        # Check mean is between mode and MLE (inclusive for edge cases)
        min_val, max_val = min(mode, mle), max(mode, mle)
        assert min_val - 1e-6 <= mean <= max_val + 1e-6, (
            f"Mean not between mode and MLE at w={w}, D={D}: "
            f"mode={mode:.4f}, mean={mean:.4f}, mle={mle:.4f}"
        )
