"""
Tests for optimal tilting (Section 10).

Key results from the document:

Optimal tilting table (w = 0.5, alpha = 0.05):
| |Delta| | eta*(theta) | E[W]/W_Wald |
|---------|-------------|-------------|
| 0.0     | 0.23        | 0.85        |
| 0.5     | 0.52        | 0.88        |
| 1.0     | 0.77        | 0.93        |
| 2.0     | 0.94        | 0.97        |
| 3.0     | 0.97        | 0.99        |
| 5.0     | 0.99        | ~1.00       |

Power-law scaling (Section 10.3):
    1 - eta*(|Delta|) ≈ 0.18 / |Delta|^{1.7}
"""

import pytest
import numpy as np
from scipy import stats

from frasian.tilting import (
    optimal_eta_approximation,
    optimal_eta_numerical,
    tilted_ci_width,
    dynamic_tilted_pvalue,
    dynamic_tilted_ci,
    dynamic_tilted_mode,
    tilted_mode,
    tilted_params,
)
from frasian.waldo import wald_ci_width, confidence_interval_width
from frasian.core import posterior_params
from frasian.confidence import pvalue_mode

from conftest import (
    TestConfig,
    ModelParams,
    get_model_from_w,
    data_for_conflict,
)


@pytest.mark.tier4
class TestOptimalTiltingApproximation:
    """Tests for the power-law approximation of optimal eta."""

    def test_approximation_at_zero_conflict(self):
        """Test eta* ≈ 0.23 at |Delta| = 0."""
        eta_star = optimal_eta_approximation(0.0)

        # Document says eta* ≈ 0.23 at Delta=0
        assert np.isclose(eta_star, 0.23, atol=0.05), (
            f"eta* at |Delta|=0 should be ~0.23, got {eta_star}"
        )

    @pytest.mark.parametrize("abs_Delta,expected_eta,tolerance", [
        (0.5, 0.52, 0.15),
        (1.0, 0.77, 0.15),
        (2.0, 0.94, 0.10),
        (3.0, 0.97, 0.05),
        (5.0, 0.99, 0.02),
    ])
    def test_approximation_at_various_conflicts(self, abs_Delta, expected_eta, tolerance):
        """Test eta* approximation at various |Delta| values."""
        eta_star = optimal_eta_approximation(abs_Delta)

        assert np.abs(eta_star - expected_eta) < tolerance, (
            f"eta* at |Delta|={abs_Delta}: got {eta_star:.2f}, expected ~{expected_eta:.2f}"
        )

    def test_approximation_increases_with_conflict(self):
        """Test that eta* increases with |Delta|."""
        deltas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        etas = [optimal_eta_approximation(d) for d in deltas]

        for i in range(len(etas) - 1):
            assert etas[i+1] >= etas[i] - 0.01, (
                f"eta* should increase with |Delta|: {etas}"
            )

    def test_approximation_approaches_one(self):
        """Test that eta* -> 1 as |Delta| -> infinity."""
        large_deltas = [5.0, 10.0, 20.0, 50.0]

        for delta in large_deltas:
            eta_star = optimal_eta_approximation(delta)
            assert eta_star > 0.95, f"eta* should approach 1 at large |Delta|={delta}"

    def test_power_law_formula(self):
        """Test the power-law formula: 1 - eta* ≈ 0.18 / |Delta|^1.7."""
        deltas = [1.0, 2.0, 3.0, 5.0]

        for delta in deltas:
            eta_star = optimal_eta_approximation(delta)
            one_minus_eta = 1 - eta_star

            # Expected from formula
            expected_one_minus = 0.18 / delta**1.7

            # Allow some tolerance since we clamp to [0,1]
            assert np.abs(one_minus_eta - expected_one_minus) < 0.1, (
                f"Power-law mismatch at |Delta|={delta}: "
                f"1-eta*={one_minus_eta:.3f}, expected={expected_one_minus:.3f}"
            )


@pytest.mark.tier4
class TestOptimalTiltingWidth:
    """Tests for CI width properties under optimal tilting.

    Note: The optimal tilting is designed to minimize *expected* CI width
    over the sampling distribution of D, not the width for a single D value.
    Individual CI widths may vary.
    """

    def test_tilted_width_varies_with_eta(self):
        """Test that CI width changes with tilting parameter."""
        model = get_model_from_w(0.5)
        D = 2.0

        widths = []
        for eta in [0.0, 0.5, 1.0]:
            w = tilted_ci_width(D, model.mu0, model.sigma, model.sigma0, eta)
            widths.append(w)

        # Widths should be different for different eta
        assert not np.allclose(widths, widths[0]), "Widths should vary with eta"

    def test_tilted_width_at_eta_one_equals_wald(self):
        """Test that width at eta=1 equals Wald width."""
        model = get_model_from_w(0.5)
        D = 2.0

        w_wald = wald_ci_width(model.sigma)
        w_tilt_1 = tilted_ci_width(D, model.mu0, model.sigma, model.sigma0, eta=1.0)

        assert np.isclose(w_tilt_1, w_wald, atol=0.05), (
            f"Width at eta=1 should equal Wald: W_tilt={w_tilt_1:.2f}, W_Wald={w_wald:.2f}"
        )


@pytest.mark.tier4
class TestDynamicTilting:
    """Tests for dynamic tilting functions."""

    def test_dynamic_pvalue_at_various_theta(self, balanced_model):
        """Test that dynamic p-value is well-behaved."""
        D = 2.0
        model = balanced_model

        # Test at several theta values
        thetas = [-1.0, 0.0, 1.0, 2.0, 3.0]
        pvals = []
        for theta in thetas:
            p = dynamic_tilted_pvalue(theta, D, model.mu0, model.sigma, model.sigma0)
            pvals.append(p)
            assert 0 <= p <= 1, f"p-value out of bounds at theta={theta}"

        # p-values should be highest near the data D and decrease away from it
        assert max(pvals) > 0.5, "At least one p-value should be reasonably high"

    def test_dynamic_pvalue_in_unit_interval(self, balanced_model):
        """Test that dynamic p-value is in [0, 1]."""
        D = 2.0
        model = balanced_model

        thetas = np.linspace(-5, 10, 50)
        for theta in thetas:
            p = dynamic_tilted_pvalue(theta, D, model.mu0, model.sigma, model.sigma0)
            assert 0 <= p <= 1 + 1e-10, f"p-value out of bounds at theta={theta}"

    def test_dynamic_ci_contains_posterior_mean(self, balanced_model):
        """Test that dynamic CI contains the posterior mean."""
        D = 2.0
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        lower, upper = dynamic_tilted_ci(D, model.mu0, model.sigma, model.sigma0)

        assert lower < mu_n < upper, (
            f"Dynamic CI should contain mu_n: CI=({lower:.2f}, {upper:.2f}), mu_n={mu_n:.2f}"
        )


@pytest.mark.tier4
class TestDynamicMode:
    """Tests for the dynamically-tilted mode (Theorem 10)."""

    def test_dynamic_mode_fixed_point(self, balanced_model):
        """Test that dynamic mode satisfies theta* = mu_{eta*(theta*)}."""
        D = 2.0
        model = balanced_model

        mode = dynamic_tilted_mode(D, model.mu0, model.sigma, model.sigma0)

        # Compute |Delta| at the mode
        w = model.w
        Delta_mode = np.abs((1 - w) * (model.mu0 - mode) / model.sigma)

        # Get eta* for this |Delta|
        eta_star = optimal_eta_approximation(Delta_mode)

        # Compute mu_eta at this eta*
        mu_eta, _, _ = tilted_params(D, model.mu0, model.sigma, model.sigma0, eta_star)

        # mode should equal mu_eta (fixed-point condition)
        assert np.isclose(mode, mu_eta, atol=0.01), (
            f"Fixed-point violated: mode={mode:.4f}, mu_eta={mu_eta:.4f}"
        )

    def test_dynamic_mode_between_mu_n_and_mle(self, balanced_model):
        """Test that dynamic mode is between mu_n and D."""
        D = 3.0
        model = balanced_model

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        mode = dynamic_tilted_mode(D, model.mu0, model.sigma, model.sigma0)

        # Mode should be between mu_n and D
        lower, upper = min(mu_n, D), max(mu_n, D)

        assert lower - 0.1 <= mode <= upper + 0.1, (
            f"Mode should be between mu_n and D: "
            f"mode={mode:.4f}, mu_n={mu_n:.4f}, D={D:.4f}"
        )

    def test_dynamic_mode_converges(self, balanced_model):
        """Test that fixed-point iteration converges."""
        D = 2.0
        model = balanced_model

        # This should not raise an exception
        mode = dynamic_tilted_mode(D, model.mu0, model.sigma, model.sigma0)

        assert np.isfinite(mode), "Mode should converge to a finite value"


@pytest.mark.tier4
class TestEstimatorHierarchy:
    """Tests for the estimator ordering from the document corollary.

    Expected ordering: mu_n < Mode_Tilt < Mean_WALDO < Mean_Tilt < D
    (when D > mu0)
    """

    def test_mode_ordering(self, balanced_model):
        """Test that WALDO mode < dynamic mode when D > mu0."""
        D = 3.0  # D > mu0 = 0
        model = balanced_model

        waldo_mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)
        dynamic_mode = dynamic_tilted_mode(D, model.mu0, model.sigma, model.sigma0)

        # Dynamic mode should be >= WALDO mode (both moving toward D)
        assert dynamic_mode >= waldo_mode - 0.01, (
            f"Dynamic mode should be >= WALDO mode: "
            f"dynamic={dynamic_mode:.4f}, waldo={waldo_mode:.4f}"
        )

    def test_mode_less_than_mle(self, balanced_model):
        """Test that both modes are less than MLE when D > mu0."""
        D = 3.0
        model = balanced_model

        waldo_mode = pvalue_mode(D, model.mu0, model.sigma, model.sigma0)
        dynamic_mode = dynamic_tilted_mode(D, model.mu0, model.sigma, model.sigma0)

        assert waldo_mode < D, f"WALDO mode should be < D: {waldo_mode} vs {D}"
        assert dynamic_mode < D + 0.01, f"Dynamic mode should be <= D: {dynamic_mode} vs {D}"


@pytest.mark.tier4
@pytest.mark.slow
class TestOptimalTiltingNumerical:
    """Tests for numerical computation of optimal eta."""

    def test_numerical_eta_at_zero_conflict(self, rng):
        """Test numerical eta* at |Delta| = 0."""
        eta_star = optimal_eta_numerical(
            abs_Delta=0.0, w=0.5, n_sims=500, rng=rng
        )

        # Should be roughly in the range from the document
        assert 0.1 < eta_star < 0.5, (
            f"Numerical eta* at |Delta|=0: {eta_star}, expected ~0.23"
        )

    def test_numerical_eta_increases_with_conflict(self, rng):
        """Test that numerical eta* increases with |Delta|."""
        deltas = [0.0, 1.0, 2.0]
        etas = []

        for delta in deltas:
            eta = optimal_eta_numerical(
                abs_Delta=delta, w=0.5, n_sims=300, rng=rng
            )
            etas.append(eta)

        # Should generally increase (allow some noise)
        assert etas[-1] > etas[0] - 0.1, (
            f"eta* should increase with |Delta|: {etas}"
        )
