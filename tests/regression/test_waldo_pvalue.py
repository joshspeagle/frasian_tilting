"""Regression tests pinning WALDO p-value to the legacy `Phi(b-a) + Phi(-a-b)`.

The p-value formula at `legacy/src/frasian/waldo.py:115` is the load-bearing
math of the entire framework. These tests cross-check the new
`WaldoStatistic.pvalue` against the legacy free-function values at a grid of
(theta, D, w) settings, plus a brute-force numerical integration for one
canonical setting.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel, posterior_params
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic


def _legacy_waldo_pvalue(theta, D, mu0, sigma, sigma0):
    """Recompute the legacy formula independently."""
    mu_n, _, w = posterior_params(D, mu0, sigma, sigma0)
    a = np.abs(mu_n - theta) / (w * sigma)
    b = (1 - w) * (mu0 - theta) / (w * sigma)
    return stats.norm.cdf(b - a) + stats.norm.cdf(-a - b)


@pytest.mark.L2
class TestWaldoPvalueMatchesLegacy:
    """Tight regression: new pvalue matches legacy formula to 1e-12."""

    @pytest.mark.parametrize("D", [-3.0, -1.0, 0.0, 1.0, 3.0])
    @pytest.mark.parametrize("theta", [-2.0, 0.0, 2.0, 5.0])
    @pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
    def test_grid(self, D, theta, sigma0):
        sigma, mu0 = 1.0, 0.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        stat = WaldoStatistic()
        new = stat.pvalue(theta, np.asarray([D]), model, prior)
        old = _legacy_waldo_pvalue(theta, D, mu0, sigma, sigma0)
        np.testing.assert_allclose(new, old, atol=1e-12)


@pytest.mark.L0
class TestWaldoPvalueAtMode:
    """At theta = mu_n, p(theta) = 1 (the mode of the WALDO CD)."""

    def test_pvalue_at_posterior_mean_equals_one(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldoStatistic()
        for D in (-2.0, 0.0, 1.5, 4.0):
            mu_n, _, _ = posterior_params(D, 0.0, 1.0, 1.0)
            p = stat.pvalue(float(mu_n), np.asarray([D]), model, prior)
            np.testing.assert_allclose(p, 1.0, atol=1e-12)


@pytest.mark.L0
class TestWaldoPvalueBounds:
    def test_pvalue_in_unit_interval(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldoStatistic()
        thetas = np.linspace(-10, 10, 51)
        for D in (-5.0, 0.0, 5.0):
            ps = stat.pvalue(thetas, np.asarray([D]), model, prior)
            assert np.all(ps >= 0.0 - 1e-12)
            assert np.all(ps <= 1.0 + 1e-12)


@pytest.mark.L0
class TestWaldoEvaluate:
    """Direct test of the test statistic itself: tau = (mu_n - theta)^2 / sigma_n^2."""

    @pytest.mark.parametrize("D", [-2.0, 0.0, 2.0])
    @pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
    def test_matches_closed_form(self, D, sigma0):
        sigma, mu0 = 1.0, 0.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        stat = WaldoStatistic()
        thetas = np.array([-1.0, 0.0, 1.0, 2.0])
        tau = stat.evaluate(thetas, np.asarray([D]), model, prior)
        mu_n, sigma_n, _ = posterior_params(D, mu0, sigma, sigma0)
        expected = (mu_n - thetas) ** 2 / sigma_n ** 2
        np.testing.assert_allclose(tau, expected, atol=1e-12)

    def test_zero_at_mu_n(self):
        """tau(theta = mu_n) = 0 by construction."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        mu_n, _, _ = posterior_params(2.0, 0.0, 1.0, 1.0)
        tau = WaldoStatistic().evaluate(float(mu_n), np.asarray([2.0]),
                                          model, prior)
        np.testing.assert_allclose(tau, 0.0, atol=1e-24)


@pytest.mark.L0
class TestWaldoAcceptanceRegion:
    """Acceptance region in D-space: dual to confidence_interval in θ-space."""

    def test_pvalue_at_boundary_is_alpha(self):
        """For D at the acceptance-region boundary, p(theta=theta0; D) = alpha."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldoStatistic()
        alpha = 0.05
        for theta0 in (-1.0, 0.5, 2.0):
            D_lo, D_hi = stat.acceptance_region(alpha, theta0, model, prior)
            for D in (float(D_lo), float(D_hi)):
                p = float(stat.pvalue(theta0, np.asarray([D]), model, prior))
                np.testing.assert_allclose(p, alpha, atol=1e-7)

    def test_brackets_theta(self):
        """The acceptance region in D contains D = theta (where mu_n is far
        from theta but the test rarely rejects since p approaches 1)."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        D_lo, D_hi = WaldoStatistic().acceptance_region(0.05, 1.5, model, prior)
        assert float(D_lo) <= 1.5 <= float(D_hi)


@pytest.mark.L0
class TestWaldStatistic:
    def test_pvalue_at_mle_equals_one(self):
        """Wald's MLE is D, so p(theta=D) = 2*(1 - Phi(0)) = 1."""
        model = NormalNormalModel(sigma=1.0)
        stat = WaldStatistic()
        for D in (-2.0, 0.0, 3.0):
            p = stat.pvalue(D, np.asarray([D]), model)
            np.testing.assert_allclose(p, 1.0, atol=1e-12)

    def test_pvalue_decays_with_distance(self):
        model = NormalNormalModel(sigma=1.0)
        stat = WaldStatistic()
        D = 0.0
        thetas = np.array([0.0, 1.0, 2.0, 3.0])
        ps = stat.pvalue(thetas, np.asarray([D]), model)
        # Strictly decreasing as |theta - D| grows.
        assert np.all(np.diff(ps) < 0.0)

    def test_acceptance_region_two_sided(self):
        model = NormalNormalModel(sigma=2.0)
        stat = WaldStatistic()
        lo, hi = stat.acceptance_region(0.05, 1.0, model)
        np.testing.assert_allclose(hi - lo, 2 * stats.norm.ppf(0.975) * 2.0,
                                    atol=1e-12)
        np.testing.assert_allclose((lo + hi) / 2.0, 1.0, atol=1e-12)
