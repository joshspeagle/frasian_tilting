"""Regression tests pinning PowerLawTilting.tilted_pvalue + tilted_CI.

Theorem 8 (legacy `tilting.py:104-209`): the tilted WALDO p-value has the
form Phi(b_eta - a_eta) + Phi(-a_eta - b_eta), with a_eta and b_eta
defined by the closed forms in the legacy code.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel, posterior_params
from frasian.tilting.power_law import PowerLawTilting


def _legacy_tilted_pvalue_waldo(theta, D, mu0, sigma, sigma0, eta):
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
    denom = 1 - eta * (1 - w)
    mu_eta = (w * D + (1 - eta) * (1 - w) * mu0) / denom
    norm_factor = w * sigma / denom
    a = np.abs(mu_eta - theta) / norm_factor
    b = (1 - eta) * (1 - w) * (mu0 - theta) / (denom * norm_factor)
    return stats.norm.cdf(b - a) + stats.norm.cdf(-a - b)


@pytest.mark.L2
class TestTiltedPvalueMatchesLegacy:
    @pytest.mark.parametrize("D", [-2.0, 0.0, 2.0])
    @pytest.mark.parametrize("eta", [-0.4, 0.0, 0.3, 0.7])
    @pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("theta", [-1.0, 0.5, 2.5])
    def test_waldo(self, D, eta, sigma0, theta):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        scheme = PowerLawTilting()
        new = scheme.tilted_pvalue(theta, D, model, prior, eta, "waldo")
        old = _legacy_tilted_pvalue_waldo(theta, D, 0.0, 1.0, sigma0, eta)
        np.testing.assert_allclose(new, old, atol=1e-12)


@pytest.mark.L0
class TestTiltedPvalueIdentity:
    def test_waldo_eta_zero_recovers_untilted(self):
        """At eta=0, tilted_pvalue(waldo) must match the untilted WALDO p-value."""
        from frasian.statistics.waldo import WaldoStatistic

        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        stat = WaldoStatistic()
        for D in (-1.5, 0.0, 2.0):
            for theta in (-2.0, 0.5, 1.5):
                tilted = scheme.tilted_pvalue(theta, D, model, prior, 0.0,
                                                "waldo")
                untilted = stat.pvalue(theta, np.asarray([D]), model, prior)
                np.testing.assert_allclose(tilted, untilted, atol=1e-12)

    def test_wald_pvalue_eta_independent(self):
        """Wald p-value is eta-independent (statistic ignores prior)."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        D, theta = 1.5, 0.7
        for eta in (-0.4, 0.0, 0.5, 0.9):
            p = scheme.tilted_pvalue(theta, D, model, prior, eta, "wald")
            expected = 2.0 * stats.norm.sf(np.abs(D - theta))
            np.testing.assert_allclose(p, expected, atol=1e-12)


@pytest.mark.L0
class TestTiltedConfidenceInterval:
    @pytest.mark.parametrize("eta", [-0.3, 0.0, 0.5, 0.9])
    def test_pvalue_at_endpoints_equals_alpha(self, eta):
        """CI endpoints solve p(theta) = alpha by construction."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        D = 1.5
        alpha = 0.05
        lo, hi = scheme.tilted_confidence_interval(
            alpha, D, model, prior, eta, "waldo",
        )
        p_lo = scheme.tilted_pvalue(lo, D, model, prior, eta, "waldo")
        p_hi = scheme.tilted_pvalue(hi, D, model, prior, eta, "waldo")
        np.testing.assert_allclose(p_lo, alpha, atol=1e-7)
        np.testing.assert_allclose(p_hi, alpha, atol=1e-7)

    def test_eta_zero_matches_waldo_ci(self):
        """At eta=0, tilted CI matches the untilted WALDO CI."""
        from frasian.statistics.waldo import WaldoStatistic

        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        stat = WaldoStatistic()
        D, alpha = 1.5, 0.05
        tilted = scheme.tilted_confidence_interval(alpha, D, model, prior,
                                                     0.0, "waldo")
        untilted = stat.confidence_interval(alpha, np.asarray([D]), model,
                                              prior)
        np.testing.assert_allclose(tilted[0], untilted[0], atol=1e-7)
        np.testing.assert_allclose(tilted[1], untilted[1], atol=1e-7)

    def test_eta_one_matches_wald(self):
        """At eta=1, tilted Waldo CI should approach the Wald CI."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        D, alpha = 1.5, 0.05
        # Use eta near 1; exactly 1 would require denom = w (well-defined).
        eta = 1.0 - 1e-3
        lo, hi = scheme.tilted_confidence_interval(alpha, D, model, prior,
                                                     eta, "waldo")
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(lo, D - z, atol=0.05)
        np.testing.assert_allclose(hi, D + z, atol=0.05)

    def test_unsupported_statistic_raises(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        with pytest.raises(NotImplementedError):
            scheme.tilted_pvalue(0.0, 1.0, model, prior, 0.0, "lrt")
