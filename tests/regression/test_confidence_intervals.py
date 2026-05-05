"""Regression tests for `confidence_interval` on Wald and WALDO statistics.

These pin the new CI inversion against (a) the legacy closed-form Wald
formula and (b) an independent re-derivation of the WALDO CI by direct
numerical solve of `pvalue(theta) = alpha`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, stats

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel, posterior_params
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic


@pytest.mark.L0
class TestWaldCI:
    @pytest.mark.parametrize("D", [-2.0, 0.0, 1.5, 5.0])
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_closed_form(self, D, sigma):
        model = NormalNormalModel(sigma=sigma)
        stat = WaldStatistic()
        alpha = 0.05
        lo, hi = stat.confidence_interval(alpha, np.asarray([D]), model)
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        np.testing.assert_allclose(lo, D - z * sigma, atol=1e-12)
        np.testing.assert_allclose(hi, D + z * sigma, atol=1e-12)

    def test_width_independent_of_D(self):
        model = NormalNormalModel(sigma=1.0)
        stat = WaldStatistic()
        widths = []
        for D in (-3.0, 0.0, 5.0):
            lo, hi = stat.confidence_interval(0.05, np.asarray([D]), model)
            widths.append(hi - lo)
        np.testing.assert_allclose(widths, [widths[0]] * 3, atol=1e-12)


def _legacy_waldo_ci(D, mu0, sigma, sigma0, alpha):
    """Independent re-derivation: numerical solve of pvalue == alpha."""
    mu_n, _, w = posterior_params(D, mu0, sigma, sigma0)

    def f(theta):
        a = np.abs(mu_n - theta) / (w * sigma)
        b = (1 - w) * (mu0 - theta) / (w * sigma)
        return stats.norm.cdf(b - a) + stats.norm.cdf(-a - b) - alpha

    half = 4.0 * sigma
    while True:
        try:
            lo = optimize.brentq(f, mu_n - half, mu_n)
            break
        except ValueError:
            half *= 2
    half = 4.0 * sigma
    while True:
        try:
            hi = optimize.brentq(f, mu_n, mu_n + half)
            break
        except ValueError:
            half *= 2
    return lo, hi


@pytest.mark.L2
class TestWaldoCIMatchesLegacy:
    @pytest.mark.parametrize("D", [-2.0, 0.0, 2.0])
    @pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
    def test_matches(self, D, sigma0):
        sigma, mu0, alpha = 1.0, 0.0, 0.05
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        stat = WaldoStatistic()
        new_lo, new_hi = stat.confidence_interval(alpha, np.asarray([D]), model, prior)
        old_lo, old_hi = _legacy_waldo_ci(D, mu0, sigma, sigma0, alpha)
        np.testing.assert_allclose(new_lo, old_lo, atol=1e-7)
        np.testing.assert_allclose(new_hi, old_hi, atol=1e-7)


@pytest.mark.L0
class TestWaldoCIBracketsMode:
    """Mode (mu_n) is always inside the CI; coverage corollary."""

    @pytest.mark.parametrize("D", [-2.0, 0.0, 1.5])
    def test_mu_n_inside(self, D):
        sigma, sigma0, mu0, alpha = 1.0, 1.0, 0.0, 0.05
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        stat = WaldoStatistic()
        lo, hi = stat.confidence_interval(alpha, np.asarray([D]), model, prior)
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        assert lo <= float(mu_n) <= hi
