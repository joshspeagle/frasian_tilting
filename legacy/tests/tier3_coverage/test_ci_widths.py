"""
Tests for confidence interval widths (Section 6).

Key findings from the document:
- At Delta = 0 (no conflict): W_Post < W_WALDO < W_Wald
- At large |Delta| (strong conflict): W_WALDO > W_Wald
- WALDO CI is asymmetric when Delta != 0

CI Width Table (w = 0.5, alpha = 0.05):
| Delta | W_Wald | W_Post | W_WALDO | Asym  |
|-------|--------|--------|---------|-------|
| 0     | 3.92   | 2.77   | 3.29    | 0.00  |
| -1    | 3.92   | 2.77   | 3.63    | 1.66  |
| -1.5  | 3.92   | 2.77   | 4.21    | 2.08  |
| -2.5  | 3.92   | 2.77   | 5.53    | 2.76  |
| -5    | 3.92   | 2.77   | 8.86    | 4.43  |
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import posterior_params, scaled_conflict
from frasian.waldo import (
    confidence_interval,
    confidence_interval_width,
    wald_ci_width,
    posterior_ci_width,
    ci_asymmetry,
)

from conftest import (
    TestConfig,
    ModelParams,
    get_model_from_w,
    data_for_conflict,
)


@pytest.mark.tier3
class TestCIWidthOrdering:
    """Tests for CI width ordering at different conflict levels."""

    def test_width_ordering_at_zero_conflict(self):
        """Test W_Post < W_WALDO < W_Wald when Delta = 0."""
        w = 0.5
        sigma = 1.0
        model = get_model_from_w(w, sigma=sigma)

        # D = mu0 gives Delta = 0
        D = model.mu0

        w_wald = wald_ci_width(model.sigma)
        w_post = posterior_ci_width(model.sigma, model.sigma0)
        w_waldo = confidence_interval_width(D, model.mu0, model.sigma, model.sigma0)

        assert w_post < w_waldo < w_wald, (
            f"Ordering violated at Delta=0: W_Post={w_post:.2f}, "
            f"W_WALDO={w_waldo:.2f}, W_Wald={w_wald:.2f}"
        )

    def test_waldo_wider_than_wald_at_high_conflict(self):
        """Test W_WALDO > W_Wald when prior-data conflict is large."""
        w = 0.5
        sigma = 1.0
        model = get_model_from_w(w, sigma=sigma)

        # High conflict: Delta = -2.5
        D = data_for_conflict(-2.5, model.mu0, model.w, model.sigma)

        w_wald = wald_ci_width(model.sigma)
        w_waldo = confidence_interval_width(D, model.mu0, model.sigma, model.sigma0)

        assert w_waldo > w_wald, (
            f"WALDO should be wider than Wald at high conflict: "
            f"W_WALDO={w_waldo:.2f}, W_Wald={w_wald:.2f}"
        )

    def test_posterior_width_constant(self):
        """Test that posterior CI width doesn't depend on data."""
        model = get_model_from_w(0.5)

        D_values = [-5.0, -2.0, 0.0, 2.0, 5.0]
        w_post = posterior_ci_width(model.sigma, model.sigma0)

        # Posterior CI width is constant (depends only on sigma, sigma0)
        for D in D_values:
            mu_n, sigma_n, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
            z = stats.norm.ppf(0.975)
            computed_width = 2 * z * sigma_n

            assert np.isclose(computed_width, w_post), (
                f"Posterior width should be constant: {computed_width:.4f} != {w_post:.4f}"
            )

    def test_wald_width_constant(self):
        """Test that Wald CI width doesn't depend on data."""
        sigma = 1.0
        w_wald = wald_ci_width(sigma)

        # Wald width = 2 * z_{0.975} * sigma
        expected = 2 * stats.norm.ppf(0.975) * sigma

        assert np.isclose(w_wald, expected), (
            f"Wald width: {w_wald:.4f}, expected {expected:.4f}"
        )


@pytest.mark.tier3
class TestCIWidthVsConflict:
    """Tests for how CI width varies with prior-data conflict."""

    def test_waldo_width_increases_with_conflict(self):
        """Test that WALDO CI width increases with |Delta|."""
        model = get_model_from_w(0.5)

        # Test at increasing conflict levels
        deltas = [0.0, -1.0, -2.0, -3.0]
        widths = []

        for delta in deltas:
            D = data_for_conflict(delta, model.mu0, model.w, model.sigma)
            width = confidence_interval_width(D, model.mu0, model.sigma, model.sigma0)
            widths.append(width)

        # Width should increase (not strictly, but generally)
        assert widths[-1] > widths[0], (
            f"WALDO width should increase with |Delta|: widths={widths}"
        )

    @pytest.mark.parametrize("delta,expected_min,expected_max", [
        (0.0, 3.0, 3.5),      # At zero conflict
        (-1.0, 3.5, 4.0),     # Mild conflict
        (-2.5, 5.0, 6.0),     # High conflict
    ])
    def test_waldo_width_approximate_values(self, delta, expected_min, expected_max):
        """Test WALDO width is in expected range at various conflict levels."""
        model = get_model_from_w(0.5)
        D = data_for_conflict(delta, model.mu0, model.w, model.sigma)

        width = confidence_interval_width(D, model.mu0, model.sigma, model.sigma0)

        assert expected_min < width < expected_max, (
            f"WALDO width at Delta={delta}: {width:.2f}, "
            f"expected in ({expected_min}, {expected_max})"
        )


@pytest.mark.tier3
class TestCIAsymmetry:
    """Tests for CI asymmetry properties."""

    def test_symmetric_at_zero_conflict(self):
        """Test CI is symmetric when Delta = 0."""
        model = get_model_from_w(0.5)
        D = model.mu0  # Delta = 0

        asymmetry = ci_asymmetry(D, model.mu0, model.sigma, model.sigma0)

        assert np.abs(asymmetry) < 0.1, (
            f"CI should be symmetric at Delta=0: asymmetry={asymmetry:.4f}"
        )

    def test_asymmetric_with_conflict(self):
        """Test CI is asymmetric when Delta != 0."""
        model = get_model_from_w(0.5)
        D = data_for_conflict(-2.0, model.mu0, model.w, model.sigma)

        asymmetry = ci_asymmetry(D, model.mu0, model.sigma, model.sigma0)

        # With D > mu0 (negative Delta means D > mu0 for our convention),
        # CI should extend further toward D (positive asymmetry)
        assert np.abs(asymmetry) > 0.5, (
            f"CI should be asymmetric at Delta=-2: asymmetry={asymmetry:.4f}"
        )

    def test_asymmetry_direction(self):
        """Test that CI extends further toward the MLE (D)."""
        model = get_model_from_w(0.5)

        # D > mu0: CI should extend further upward (toward D)
        D_high = 3.0
        mu_n_high, _, _ = posterior_params(D_high, model.mu0, model.sigma, model.sigma0)
        lower_high, upper_high = confidence_interval(D_high, model.mu0, model.sigma, model.sigma0)

        upper_dist = upper_high - mu_n_high
        lower_dist = mu_n_high - lower_high

        assert upper_dist > lower_dist, (
            f"CI should extend further toward D={D_high}: "
            f"upper_dist={upper_dist:.2f}, lower_dist={lower_dist:.2f}"
        )

    def test_asymmetry_increases_with_conflict(self):
        """Test that asymmetry increases with |Delta|."""
        model = get_model_from_w(0.5)

        deltas = [-0.5, -1.0, -2.0, -3.0]
        asymmetries = []

        for delta in deltas:
            D = data_for_conflict(delta, model.mu0, model.w, model.sigma)
            asym = np.abs(ci_asymmetry(D, model.mu0, model.sigma, model.sigma0))
            asymmetries.append(asym)

        # Asymmetry should generally increase
        assert asymmetries[-1] > asymmetries[0], (
            f"Asymmetry should increase with |Delta|: {asymmetries}"
        )


@pytest.mark.tier3
class TestCIWidthTable:
    """Reproduce CI width table from Section 6."""

    def test_reproduce_width_table(self):
        """Reproduce the CI width table from the document.

        Expected (w = 0.5, alpha = 0.05):
        | Delta | W_Wald | W_Post | W_WALDO | Asym |
        |-------|--------|--------|---------|------|
        | 0     | 3.92   | 2.77   | 3.29    | 0.00 |
        | -1    | 3.92   | 2.77   | 3.63    | 1.66 |
        | -1.5  | 3.92   | 2.77   | 4.21    | 2.08 |
        | -2.5  | 3.92   | 2.77   | 5.53    | 2.76 |
        | -5    | 3.92   | 2.77   | 8.86    | 4.43 |
        """
        model = get_model_from_w(0.5)

        w_wald = wald_ci_width(model.sigma)
        w_post = posterior_ci_width(model.sigma, model.sigma0)

        # Check constant widths
        assert np.isclose(w_wald, 3.92, atol=0.02), f"W_Wald={w_wald:.2f}, expected 3.92"
        assert np.isclose(w_post, 2.77, atol=0.02), f"W_Post={w_post:.2f}, expected 2.77"

        # Check WALDO widths at various Delta values
        test_cases = [
            (0.0, 3.29, 0.2),
            (-1.0, 3.63, 0.3),
            (-1.5, 4.21, 0.3),
            (-2.5, 5.53, 0.4),
            (-5.0, 8.86, 0.5),
        ]

        for delta, expected_width, tolerance in test_cases:
            D = data_for_conflict(delta, model.mu0, model.w, model.sigma)
            width = confidence_interval_width(D, model.mu0, model.sigma, model.sigma0)

            assert np.abs(width - expected_width) < tolerance, (
                f"W_WALDO at Delta={delta}: {width:.2f}, expected {expected_width:.2f}"
            )

    def test_intermediate_value_ordering(self):
        """Test W_Post < W_WALDO(0) < W_Wald."""
        model = get_model_from_w(0.5)
        D = model.mu0  # Delta = 0

        w_wald = wald_ci_width(model.sigma)
        w_post = posterior_ci_width(model.sigma, model.sigma0)
        w_waldo = confidence_interval_width(D, model.mu0, model.sigma, model.sigma0)

        assert w_post < w_waldo < w_wald, (
            f"Ordering at Delta=0: W_Post={w_post:.2f} < "
            f"W_WALDO={w_waldo:.2f} < W_Wald={w_wald:.2f}"
        )


@pytest.mark.tier3
class TestCIBounds:
    """Tests for CI bound properties."""

    def test_ci_contains_mode(self):
        """Test that WALDO CI contains the mode (posterior mean)."""
        model = get_model_from_w(0.5)
        D = 2.0

        mu_n, _, _ = posterior_params(D, model.mu0, model.sigma, model.sigma0)
        lower, upper = confidence_interval(D, model.mu0, model.sigma, model.sigma0)

        assert lower < mu_n < upper, (
            f"CI should contain mode: CI=({lower:.2f}, {upper:.2f}), mode={mu_n:.2f}"
        )

    def test_ci_bounds_move_with_data(self):
        """Test that CI bounds shift as data changes."""
        model = get_model_from_w(0.5)

        D_values = [-2.0, 0.0, 2.0, 4.0]
        lowers = []
        uppers = []

        for D in D_values:
            lower, upper = confidence_interval(D, model.mu0, model.sigma, model.sigma0)
            lowers.append(lower)
            uppers.append(upper)

        # Both bounds should generally increase with D
        assert lowers[-1] > lowers[0], "Lower bound should increase with D"
        assert uppers[-1] > uppers[0], "Upper bound should increase with D"

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_ci_width_varies_with_alpha(self, alpha):
        """Test that CI width changes appropriately with alpha."""
        model = get_model_from_w(0.5)
        D = 2.0

        width = confidence_interval_width(D, model.mu0, model.sigma, model.sigma0, alpha)

        # Smaller alpha should give wider CI
        if alpha == 0.01:
            assert width > 3.5, f"99% CI should be wide: {width:.2f}"
        elif alpha == 0.10:
            assert width < 3.5, f"90% CI should be narrower: {width:.2f}"
