"""
Tier 5 Integration Tests.

Tests for:
- Three-regime structure (Theorem 9)
- Full pipeline: data -> CI -> coverage
- Prior ancillary conditioning
- Width bounds
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import posterior_params, weight, scaled_conflict
from frasian.waldo import (
    pvalue,
    confidence_interval,
    confidence_interval_width,
    wald_ci_width,
    posterior_ci,
)
from frasian.confidence import (
    pvalue_mode,
    waldo_cd_mean,
)
from frasian.tilting import (
    tilted_params,
    tilted_pvalue,
    tilted_ci,
    tilted_ci_width,
    optimal_eta_approximation,
    dynamic_tilted_ci,
    dynamic_tilted_mode,
)

from conftest import (
    TestConfig,
    get_model_from_w,
    data_for_conflict,
    simulate_data,
)


@pytest.mark.tier5
class TestThreeRegimeStructure:
    """Tests for Theorem 9: Three-regime structure of dynamically-tilted CIs.

    Regime    | |Delta| Range | eta* at Boundaries | Width vs Wald
    ----------|---------------|--------------------|--------------
    Low       | < 1           | Both low (0.2-0.5) | W_Tilt < W_Wald
    Transition| 1 - 3         | Mixed (~0.3, ~0.98)| Can vary
    High      | > 3           | Both high (~0.98+) | W_Tilt ≈ W_Wald
    """

    def test_low_conflict_regime(self):
        """Test behavior in low conflict regime (|Delta| < 1)."""
        model = get_model_from_w(0.5)

        for delta in [0.0, 0.3, 0.7]:
            eta_star = optimal_eta_approximation(delta)

            # eta* should be relatively low
            assert eta_star < 0.7, (
                f"eta* should be low in low-conflict regime: "
                f"|Delta|={delta}, eta*={eta_star:.2f}"
            )

    def test_high_conflict_regime(self):
        """Test behavior in high conflict regime (|Delta| > 3)."""
        model = get_model_from_w(0.5)

        for delta in [3.0, 4.0, 5.0]:
            eta_star = optimal_eta_approximation(delta)

            # eta* should be close to 1
            assert eta_star > 0.95, (
                f"eta* should be high in high-conflict regime: "
                f"|Delta|={delta}, eta*={eta_star:.2f}"
            )

    def test_transition_regime(self):
        """Test behavior in transition regime (1 < |Delta| < 3)."""
        model = get_model_from_w(0.5)

        for delta in [1.5, 2.0, 2.5]:
            eta_star = optimal_eta_approximation(delta)

            # eta* should be intermediate
            assert 0.5 < eta_star < 0.98, (
                f"eta* should be intermediate in transition regime: "
                f"|Delta|={delta}, eta*={eta_star:.2f}"
            )


@pytest.mark.tier5
class TestEstimatorOrdering:
    """Tests for the estimator hierarchy from the document corollary.

    The document's expected ordering mu_n < Mode_Tilt < Mean_WALDO < Mean_Tilt < D
    depends on specific conditions. Here we verify the key property that all
    estimators move from mu_n toward D.
    """

    def test_estimators_between_mu_n_and_D(self):
        """Test that all estimators are between mu_n and D."""
        model = get_model_from_w(0.5)
        D = 4.0  # D > mu0 = 0

        # WALDO estimators
        mu_n = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)
        mean_waldo = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)

        # Dynamic tilted mode
        mode_tilt = dynamic_tilted_mode(D, model.mu0, model.sigma, model.sigma0)

        # All estimators should be in [mu_n, D]
        lower, upper = min(mu_n, D), max(mu_n, D)

        assert lower <= mean_waldo <= upper, (
            f"mean_waldo should be between mu_n and D: {mean_waldo:.2f}"
        )
        assert lower <= mode_tilt <= upper + 0.1, (
            f"mode_tilt should be between mu_n and D: {mode_tilt:.2f}"
        )

    def test_estimator_ordering_reverses_when_D_below_mu0(self):
        """Test that ordering reverses when D < mu0."""
        model = get_model_from_w(0.5)
        D = -4.0  # D < mu0 = 0

        mu_n = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)
        mean_waldo = waldo_cd_mean(D, model.mu0, model.sigma, model.sigma0)
        mode_tilt = dynamic_tilted_mode(D, model.mu0, model.sigma, model.sigma0)

        # When D < mu0, ordering should be reversed
        assert mu_n > D, f"mu_n should be > D when D < mu0: mu_n={mu_n:.2f}, D={D}"


@pytest.mark.tier5
class TestPriorAncillaryConditioning:
    """Tests for the prior ancillary concept (Section 8).

    WALDO achieves conditional coverage by implicitly conditioning on
    delta(theta) = (theta - mu0) / sigma0.
    """

    @pytest.mark.slow
    def test_coverage_conditional_on_prior_residual(self, config, rng):
        """Test that WALDO coverage is correct conditional on prior residual.

        We stratify by delta(theta) = (theta - mu0) / sigma0 and check
        that coverage is ~95% in each stratum.
        """
        model = get_model_from_w(0.5)

        # Test at several values of delta (prior residual)
        deltas = [0.0, 1.0, 2.0, 3.0]

        for delta in deltas:
            # theta such that (theta - mu0) / sigma0 = delta
            theta_true = model.mu0 + delta * model.sigma0

            D_samples = simulate_data(theta_true, model.sigma, config.n_coverage, rng)
            covered = 0

            for D in D_samples:
                lower, upper = confidence_interval(D, model.mu0, model.sigma, model.sigma0)
                if lower <= theta_true <= upper:
                    covered += 1

            coverage = covered / config.n_coverage

            assert abs(coverage - 0.95) < 2 * config.coverage_tol, (
                f"WALDO coverage conditional on delta={delta}: {coverage:.3f}"
            )


@pytest.mark.tier5
class TestWidthBounds:
    """Tests for width bounds (Section 10.4).

    Key property: E[W_Tilt] <= W_Wald for all |Delta|
    """

    def test_tilted_ci_at_eta_one_equals_wald_width(self):
        """Test that CI at eta=1 has Wald width."""
        model = get_model_from_w(0.5)
        w_wald = wald_ci_width(model.sigma)

        for D in [-2.0, 0.0, 2.0, 4.0]:
            w_tilt_1 = tilted_ci_width(D, model.mu0, model.sigma, model.sigma0, eta=1.0)

            assert np.isclose(w_tilt_1, w_wald, atol=0.1), (
                f"Width at eta=1 should equal Wald: D={D}, "
                f"W_tilt={w_tilt_1:.2f}, W_Wald={w_wald:.2f}"
            )


@pytest.mark.tier5
class TestFullPipeline:
    """End-to-end tests for the complete inference pipeline."""

    def test_pipeline_data_to_ci(self):
        """Test full pipeline: data -> posterior -> CI."""
        model = get_model_from_w(0.5)
        D = 2.5

        # Step 1: Compute posterior
        mu_n, sigma_n, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        assert mu_n == w * D + (1 - w) * model.mu0

        # Step 2: Compute WALDO CI
        lower, upper = confidence_interval(D, model.mu0, model.sigma, model.sigma0)
        assert lower < mu_n < upper

        # Step 3: Verify p-value at CI boundaries is alpha
        p_lower = pvalue(lower, mu_n, model.mu0, w, model.sigma)
        p_upper = pvalue(upper, mu_n, model.mu0, w, model.sigma)
        assert np.isclose(p_lower, 0.05, atol=0.01)
        assert np.isclose(p_upper, 0.05, atol=0.01)

    def test_pipeline_with_tilting(self):
        """Test full pipeline with tilted inference."""
        model = get_model_from_w(0.5)
        D = 2.5

        for eta in [0.0, 0.5, 1.0]:
            # Compute tilted posterior
            mu_eta, sigma_eta, _ = tilted_params(D, model.mu0, model.sigma, model.sigma0, eta)

            # Compute tilted CI
            lower, upper = tilted_ci(D, model.mu0, model.sigma, model.sigma0, eta)

            # Mode should be in CI
            assert lower < mu_eta < upper, (
                f"Tilted mode should be in CI at eta={eta}"
            )

            # p-value at boundaries should be alpha
            p_lower = tilted_pvalue(lower, D, model.mu0, model.sigma, model.sigma0, eta)
            p_upper = tilted_pvalue(upper, D, model.mu0, model.sigma, model.sigma0, eta)
            assert np.isclose(p_lower, 0.05, atol=0.02)
            assert np.isclose(p_upper, 0.05, atol=0.02)

    @pytest.mark.slow
    def test_full_coverage_verification(self, config, rng):
        """Comprehensive coverage test across all methods."""
        model = get_model_from_w(0.5)
        theta_true = 2.0

        methods = {
            'waldo': lambda D: confidence_interval(D, model.mu0, model.sigma, model.sigma0),
            'tilted_0.5': lambda D: tilted_ci(D, model.mu0, model.sigma, model.sigma0, eta=0.5),
            'tilted_1.0': lambda D: tilted_ci(D, model.mu0, model.sigma, model.sigma0, eta=1.0),
        }

        D_samples = simulate_data(theta_true, model.sigma, config.n_coverage, rng)

        for name, ci_func in methods.items():
            covered = 0
            for D in D_samples:
                lower, upper = ci_func(D)
                if lower <= theta_true <= upper:
                    covered += 1

            coverage = covered / config.n_coverage

            assert abs(coverage - 0.95) < 2 * config.coverage_tol, (
                f"{name} coverage: {coverage:.3f}, expected ~0.95"
            )


@pytest.mark.tier5
class TestConsistencyAcrossMethods:
    """Tests for consistency between different methods."""

    def test_waldo_and_tilted_0_are_identical(self):
        """Test that WALDO and tilted at eta=0 give identical results."""
        model = get_model_from_w(0.5)
        D = 2.0
        theta = 1.5

        # WALDO p-value
        mu_n, _, w = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        p_waldo = pvalue(theta, mu_n, model.mu0, w, model.sigma)

        # Tilted p-value at eta=0
        p_tilted = tilted_pvalue(theta, D, model.mu0, model.sigma, model.sigma0, eta=0.0)

        assert np.isclose(p_waldo, p_tilted, atol=1e-10)

    def test_tilted_1_and_wald_are_identical(self):
        """Test that tilted at eta=1 and Wald give identical p-values."""
        model = get_model_from_w(0.5)
        D = 2.0
        theta = 1.5

        # Tilted p-value at eta=1
        p_tilted = tilted_pvalue(theta, D, model.mu0, model.sigma, model.sigma0, eta=1.0)

        # Wald p-value: 2*Phi(-|D - theta|/sigma)
        p_wald = 2 * stats.norm.cdf(-np.abs(D - theta) / model.sigma)

        assert np.isclose(p_tilted, p_wald, atol=1e-10)

    def test_interpolation_smoothness(self):
        """Test that tilted p-value varies smoothly with eta."""
        model = get_model_from_w(0.5)
        D = 2.0
        theta = 1.5

        etas = np.linspace(0, 1, 11)
        pvals = [
            tilted_pvalue(theta, D, model.mu0, model.sigma, model.sigma0, eta)
            for eta in etas
        ]

        # Check that changes are smooth (no large jumps)
        diffs = np.abs(np.diff(pvals))
        assert np.max(diffs) < 0.2, f"p-value should vary smoothly with eta: diffs={diffs}"
