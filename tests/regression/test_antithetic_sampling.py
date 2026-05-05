"""Regression: antithetic ``2θ − D`` pairing reduces MC variance.

Closes audit finding 1.2-NN1.

For Normal-Normal symmetry (likelihood ``N(θ, σ²)``), if ``D = θ + δ``
then ``D' = 2θ − D = θ − δ`` is exactly anti-correlated with ``D``.
Averaging the loss over the (D, D') pair halves the variance on
even loss components (e.g., the integrated p-value, which is even
under ``D ↔ 2θ − D``).

We pin two things:

1. The arithmetic identity ``antithetic_pair(θ, D) == 2θ − D`` element-
   wise. Pure numpy, no torch needed.

2. Variance reduction: the empirical variance of a Normal-Normal
   width estimator under (D, D') pairing is at most half the variance
   of independent Monte Carlo at the same effective sample size. This
   doesn't require torch — we use a closed-form Wald p-value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from frasian.learned.training._validity_data import (
    antithetic_pair,
    sample_data_per_theta,
)


def _wald_pvalue_grid(theta_grid: np.ndarray, D: float, sigma: float) -> np.ndarray:
    z = np.abs(D - theta_grid) / sigma
    return 2.0 * (1.0 - 0.5 * (1.0 + np.array([math.erf(zi / math.sqrt(2.0)) for zi in z])))


def _integrated_p(D: float, theta_grid: np.ndarray, sigma: float) -> float:
    p = _wald_pvalue_grid(theta_grid, D, sigma)
    return float(np.trapezoid(p, theta_grid))


@pytest.mark.L1
def test_antithetic_pair_arithmetic_identity():
    """``antithetic_pair(θ, D)`` returns ``2θ − D`` exactly."""
    rng = np.random.default_rng(seed=42)
    theta = rng.standard_normal(50)
    D = theta + 0.7 * rng.standard_normal(50)
    paired = antithetic_pair(theta, D)
    np.testing.assert_array_equal(paired, 2.0 * theta - D)


@pytest.mark.L1
def test_antithetic_pair_shape_check_raises_on_mismatch():
    theta = np.array([0.0, 1.0])
    D = np.array([0.5])
    with pytest.raises(ValueError, match="matching shapes"):
        antithetic_pair(theta, D)


@pytest.mark.L1
def test_sample_data_per_theta_antithetic_returns_2N():
    """``antithetic=True`` returns a (2N,) array with the second half
    equal to ``2θ − D[:N]``."""

    class _StubModel:
        sigma: float = 1.0

        def sample_data(self, theta_scalar, rng, n=1):
            return np.array([theta_scalar + rng.standard_normal()])

    theta = np.array([-1.0, 0.0, 1.0])
    rng = np.random.default_rng(seed=7)
    out = sample_data_per_theta(_StubModel(), theta, rng, antithetic=True)
    assert out.shape == (6,)
    primary, partner = out[:3], out[3:]
    np.testing.assert_array_equal(partner, 2.0 * theta - primary)


def _shifted_p(D: float, theta_grid: np.ndarray, sigma: float) -> float:
    """Asymmetric loss: shifted/skewed integrand so f has odd Taylor part.

    ``f(D) = ∫ p_wald(θ; D, σ) · weight(θ) dθ`` with
    ``weight(θ) = exp(0.4·θ)``. The weight breaks the reflection
    symmetry of the unweighted integral, so ``f`` has both even and
    odd Taylor components in D about θ_true. Antithetic pairing
    (which exactly cancels odd terms) then reduces variance.
    """
    p = _wald_pvalue_grid(theta_grid, D, sigma)
    weight = np.exp(0.4 * theta_grid)
    return float(np.trapezoid(p * weight, theta_grid))


@pytest.mark.L2
def test_antithetic_variance_reduction_on_skewed_width():
    """Antithetic pairing reduces variance on the skew-weighted Wald
    integrated p-value: at least 2× cheaper than IID at the same
    total sample budget.

    Why a skewed integrand? An *unweighted* Wald p-value integrated
    over a grid centred at θ_true is reflection-symmetric in D about
    θ_true (a function of |D − θ_true| only), making the antithetic
    estimator degenerate (same variance as one IID draw). Real
    training-loss components have asymmetric Taylor structure
    (φ-prior asymmetry, statistic non-linearity); we mimic that here
    with an exponential weight ``exp(0.4·θ)``.

    Variance threshold: at n_pairs=2000 the antithetic estimator
    must beat the IID estimator at the same 2·n_pairs total samples
    by at least 2×.
    """
    sigma = 1.0
    theta_true = 0.0
    theta_grid = np.linspace(-5.0, 5.0, 401)
    n_pairs = 2000
    rng_iid = np.random.default_rng(seed=2026)
    rng_anti = np.random.default_rng(seed=4242)  # different seed → independent runs

    D_iid = rng_iid.standard_normal(2 * n_pairs) * sigma + theta_true
    iid_vals = np.array([_shifted_p(d, theta_grid, sigma) for d in D_iid])
    var_iid = float(np.var(iid_vals, ddof=1)) / (2 * n_pairs)  # variance of mean

    D_primary = rng_anti.standard_normal(n_pairs) * sigma + theta_true
    D_partner = 2.0 * theta_true - D_primary
    primary_vals = np.array([_shifted_p(d, theta_grid, sigma) for d in D_primary])
    partner_vals = np.array([_shifted_p(d, theta_grid, sigma) for d in D_partner])
    paired_means = 0.5 * (primary_vals + partner_vals)
    var_anti = float(np.var(paired_means, ddof=1)) / n_pairs

    assert var_anti <= 0.5 * var_iid, (
        f"antithetic variance {var_anti:.4e} not ≤ 0.5 × IID {var_iid:.4e}; "
        f"ratio={var_anti / var_iid:.3f}."
    )
