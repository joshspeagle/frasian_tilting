"""
Tests for Theorems 4-5: Mode and Mean of the WALDO confidence distribution.

Theorem 4: The unique maximizer of WALDO's p-value function is theta* = mu_n.
Theorem 5: The mean of the normalized p-value function is given by a closed-form
expression involving Phi and phi evaluated at Delta.

Key property: The mean lies between the mode (mu_n) and the MLE (D).
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import posterior_params, scaled_conflict
from frasian.confidence import (
    pvalue_mode,
    pvalue_mean,
    pvalue_mean_numerical,
    pvalue_mean_closed_form,
    numerical_mode,
    verify_mode_is_max,
    mean_between_mode_and_mle,
    pvalue_at_mode,
    sample_confidence_dist,
)

from conftest import (
    TestConfig,
    ModelParams,
    get_model_from_w,
    data_for_conflict,
)


@pytest.mark.tier2
class TestModeEqualsPosteriorMean:
    """Tests for Theorem 4: Mode of p-value function equals mu_n."""

    def test_mode_equals_mu_n(self, balanced_model):
        """Test that pvalue_mode returns mu_n."""
        D = 2.0
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mode, mu_n, atol=1e-10), (
            f"Mode should equal mu_n: mode={mode}, mu_n={mu_n}"
        )

    def test_numerical_mode_matches_analytical(self, balanced_model):
        """Test that numerical optimization finds the same mode."""
        D = 2.0
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        numerical = numerical_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(numerical, mu_n, atol=0.001), (
            f"Numerical mode should match mu_n: numerical={numerical}, mu_n={mu_n}"
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

        # p_at_mode should be 1.0 (the theoretical max), and max_found should be <= 1.0
        # They may not be exactly equal due to discretization in the search
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
        mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mode, mu_n), f"Mode mismatch at D={D}: mode={mode}, mu_n={mu_n}"

    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    def test_mode_equals_mu_n_various_w(self, w):
        """Test mode = mu_n for various weight values."""
        D = 2.0
        model = get_model_from_w(w)

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mode, mu_n), f"Mode mismatch at w={w}: mode={mode}, mu_n={mu_n}"


@pytest.mark.tier2
class TestMeanFormula:
    """Tests for Theorem 5: Closed-form expression for the mean."""

    def test_closed_form_matches_numerical(self, balanced_model):
        """Test that closed-form mean matches numerical integration."""
        D = 2.0
        model = balanced_model

        mean_closed = pvalue_mean_closed_form(D, model.mu0, model.sigma, model.sigma0)
        mean_numerical = pvalue_mean_numerical(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mean_closed, mean_numerical, rtol=0.01), (
            f"Mean mismatch: closed={mean_closed:.4f}, numerical={mean_numerical:.4f}"
        )

    @pytest.mark.parametrize("D", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_closed_form_matches_numerical_various_D(self, balanced_model, D):
        """Test closed-form vs numerical for various data values."""
        model = balanced_model

        mean_closed = pvalue_mean_closed_form(D, model.mu0, model.sigma, model.sigma0)
        mean_numerical = pvalue_mean_numerical(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mean_closed, mean_numerical, rtol=0.02), (
            f"Mean mismatch at D={D}: closed={mean_closed:.4f}, numerical={mean_numerical:.4f}"
        )

    def test_mean_equals_mode_when_delta_zero(self, balanced_model):
        """Test that mean = mode when Delta = 0 (D = mu0)."""
        D = balanced_model.mu0  # Delta = 0
        model = balanced_model

        mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)
        mean = pvalue_mean(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mean, mode, atol=0.01 * model.sigma), (
            f"When Delta=0: mean should equal mode. mean={mean:.4f}, mode={mode:.4f}"
        )


@pytest.mark.tier2
class TestMeanBetweenModeAndMLE:
    """Tests for Section 5.2.1: Mean lies between mode and MLE."""

    def test_mean_between_mode_and_mle_D_above_mu0(self, balanced_model):
        """Test mu_n < E[theta] < D when D > mu0."""
        D = 3.0  # D > mu0 = 0
        model = balanced_model

        is_between, mode, mean, mle = mean_between_mode_and_mle(
            D, model.mu0, model.sigma, model.sigma0
        )

        assert D > model.mu0, "Precondition: D > mu0"
        assert mode < D, f"Mode should be less than D: mode={mode}, D={D}"
        assert is_between, (
            f"Mean should be between mode and MLE: mode={mode:.4f}, mean={mean:.4f}, mle={mle:.4f}"
        )

    def test_mean_between_mode_and_mle_D_below_mu0(self, balanced_model):
        """Test D < E[theta] < mu_n when D < mu0."""
        D = -3.0  # D < mu0 = 0
        model = balanced_model

        is_between, mode, mean, mle = mean_between_mode_and_mle(
            D, model.mu0, model.sigma, model.sigma0
        )

        assert D < model.mu0, "Precondition: D < mu0"
        assert mode > D, f"Mode should be greater than D: mode={mode}, D={D}"
        assert is_between, (
            f"Mean should be between MLE and mode: mle={mle:.4f}, mean={mean:.4f}, mode={mode:.4f}"
        )

    @pytest.mark.parametrize("D", [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0])
    def test_mean_ordering_various_D(self, balanced_model, D):
        """Test mean ordering for various data values."""
        model = balanced_model

        is_between, mode, mean, mle = mean_between_mode_and_mle(
            D, model.mu0, model.sigma, model.sigma0
        )

        assert is_between, (
            f"Mean ordering violated at D={D}: mode={mode:.4f}, mean={mean:.4f}, mle={mle:.4f}"
        )

    def test_mean_pulled_toward_mle(self, balanced_model):
        """Test that mean is closer to MLE than mode is."""
        D = 3.0
        model = balanced_model

        _, mode, mean, mle = mean_between_mode_and_mle(
            D, model.mu0, model.sigma, model.sigma0
        )

        dist_mode_to_mle = np.abs(mle - mode)
        dist_mean_to_mle = np.abs(mle - mean)

        assert dist_mean_to_mle < dist_mode_to_mle, (
            f"Mean should be closer to MLE than mode is: "
            f"dist(mean,mle)={dist_mean_to_mle:.4f}, dist(mode,mle)={dist_mode_to_mle:.4f}"
        )


@pytest.mark.tier2
class TestMeanSpecialCases:
    """Tests for special cases of the mean formula."""

    def test_mean_approaches_mle_as_w_approaches_one(self):
        """Test that mean -> D as w -> 1 (uninformative prior)."""
        D = 3.0

        # w close to 1
        model = get_model_from_w(0.95)
        mean_high_w = pvalue_mean(D, model.mu0, model.sigma, model.sigma0)

        # Mean should be close to D
        assert np.abs(mean_high_w - D) < 0.5, (
            f"With weak prior (w~1), mean should be close to D={D}, got {mean_high_w}"
        )

    def test_mean_between_mode_and_mle_with_strong_prior(self):
        """Test that mean is between mode and MLE even with strong prior.

        Note: With strong prior (w~0), both mode and mean are pulled toward mu0,
        but mean is still between mode and MLE (not necessarily close to mode).
        """
        D = 3.0

        # w close to 0
        model = get_model_from_w(0.1)
        mean_low_w = pvalue_mean(D, model.mu0, model.sigma, model.sigma0)
        mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)
        mle = D

        # Mode should be close to mu0
        assert np.abs(mode - model.mu0) < np.abs(D - model.mu0), (
            f"Mode should be between mu0 and D: mode={mode:.4f}"
        )

        # Mean should be between mode and MLE (or very close to boundaries)
        is_between, _, _, _ = mean_between_mode_and_mle(
            D, model.mu0, model.sigma, model.sigma0
        )
        assert is_between, (
            f"Mean should be between mode and MLE: "
            f"mode={mode:.4f}, mean={mean_low_w:.4f}, mle={mle:.4f}"
        )

    @pytest.mark.parametrize("Delta", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_mean_at_various_conflict_levels(self, Delta):
        """Test mean formula at various prior-data conflict levels."""
        w = 0.5
        model = get_model_from_w(w)

        # Compute D to get desired Delta
        D = data_for_conflict(Delta, model.mu0, model.w, model.sigma)

        mean_closed = pvalue_mean_closed_form(D, model.mu0, model.sigma, model.sigma0)
        mean_numerical = pvalue_mean_numerical(D, model.mu0, model.sigma, model.sigma0)

        assert np.isclose(mean_closed, mean_numerical, rtol=0.03), (
            f"Mean mismatch at Delta={Delta}: closed={mean_closed:.4f}, numerical={mean_numerical:.4f}"
        )


@pytest.mark.tier2
class TestSamplingFromConfidenceDistribution:
    """Tests for sampling from the confidence distribution."""

    @pytest.mark.slow
    def test_sample_mean_matches_analytical(self, balanced_model, rng):
        """Test that sample mean approximates analytical mean."""
        D = 2.0
        model = balanced_model
        n_samples = 5000

        samples = sample_confidence_dist(
            D, model.mu0, model.sigma, model.sigma0,
            n_samples=n_samples, rng=rng
        )

        analytical_mean = pvalue_mean(D, model.mu0, model.sigma, model.sigma0)
        sample_mean = np.mean(samples)

        # Allow for sampling error
        assert np.abs(sample_mean - analytical_mean) < 0.1, (
            f"Sample mean should match analytical: "
            f"sample={sample_mean:.4f}, analytical={analytical_mean:.4f}"
        )

    @pytest.mark.slow
    def test_sample_mode_near_analytical(self, balanced_model, rng):
        """Test that sample mode (peak of histogram) is near analytical mode."""
        D = 2.0
        model = balanced_model
        n_samples = 5000

        samples = sample_confidence_dist(
            D, model.mu0, model.sigma, model.sigma0,
            n_samples=n_samples, rng=rng
        )

        analytical_mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)

        # Use histogram to estimate mode
        hist, bin_edges = np.histogram(samples, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        sample_mode = bin_centers[np.argmax(hist)]

        # Allow for discretization and sampling error
        assert np.abs(sample_mode - analytical_mode) < 0.3, (
            f"Sample mode should be near analytical: "
            f"sample={sample_mode:.4f}, analytical={analytical_mode:.4f}"
        )


@pytest.mark.tier2
class TestEstimatorConsistency:
    """Tests for consistency between different estimator computations."""

    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    @pytest.mark.parametrize("D", [-2.0, 0.0, 2.0])
    def test_mode_and_mean_consistency(self, w, D):
        """Test that mode and mean are computed consistently across parameters."""
        model = get_model_from_w(w)

        mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)
        mean = pvalue_mean(D, model.mu0, model.sigma, model.sigma0)
        mle = D

        # Mode should always equal mu_n
        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        assert np.isclose(mode, mu_n), f"Mode != mu_n at w={w}, D={D}"

        # Mean should be finite and reasonable
        assert np.isfinite(mean), f"Mean not finite at w={w}, D={D}"
        assert np.abs(mean - mu_n) < np.abs(mle - mu_n) + 0.1, (
            f"Mean too far from mu_n at w={w}, D={D}"
        )
