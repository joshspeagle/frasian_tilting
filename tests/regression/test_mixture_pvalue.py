"""Regression tests for MixtureTilting closed-form WALDO/Wald p-values.

Endpoint reductions:
  eta=0 -> bare WALDO formula (atol 1e-12)
  eta=1 -> bare two-sided Wald (atol 1e-12)
Interior cross-check: closed-form vs direct Monte-Carlo at eta=0.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import FixedEtaSelector
from frasian.tilting.mixture import MixtureTilting


def _bare_waldo_pvalue(theta, D, mu0, sigma, sigma0):
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
    mu_n = w * D + (1.0 - w) * mu0
    a = abs(mu_n - theta) / (w * sigma)
    b = (1.0 - w) * (mu0 - theta) / (w * sigma)
    from scipy.stats import norm
    return float(norm.cdf(b - a) + norm.cdf(-a - b))


def _bare_wald_pvalue(theta, D, sigma):
    z = abs(D - theta) / sigma
    from scipy.stats import norm
    return float(2.0 * norm.sf(z))


@pytest.mark.L0
class TestMixtureWaldoEndpoints:
    @pytest.fixture
    def fixtures(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=2.0)
        D_arr = np.asarray([0.7])
        return model, prior, D_arr

    def test_eta_zero_reduces_to_bare_waldo(self, fixtures):
        model, prior, D = fixtures
        til = MixtureTilting()
        for theta in [-1.0, -0.3, 0.0, 0.5, 1.5]:
            p_mix = float(
                til.pvalue(np.asarray([theta]), D, model, prior, WaldoStatistic())[0]
            )
            p_ref = _bare_waldo_pvalue(theta, D=0.7, mu0=0.0, sigma=1.0, sigma0=2.0)
            assert p_mix == pytest.approx(p_ref, abs=1e-12), (
                f"mixture pvalue at theta={theta} (eta=0) = {p_mix}; "
                f"bare WALDO = {p_ref}"
            )

    def test_eta_zero_explicit_via_fixed_selector(self, fixtures):
        model, prior, D = fixtures
        til = MixtureTilting(selector=FixedEtaSelector(eta=0.0))
        for theta in [-1.0, 0.0, 1.0]:
            p_mix = float(
                til.pvalue(np.asarray([theta]), D, model, prior, WaldoStatistic())[0]
            )
            p_ref = _bare_waldo_pvalue(theta, D=0.7, mu0=0.0, sigma=1.0, sigma0=2.0)
            assert p_mix == pytest.approx(p_ref, abs=1e-12)

    def test_eta_one_reduces_to_bare_wald(self, fixtures):
        model, prior, D = fixtures
        til = MixtureTilting(selector=FixedEtaSelector(eta=1.0))
        for theta in [-1.0, 0.0, 0.7, 1.5]:
            p_mix = float(
                til.pvalue(np.asarray([theta]), D, model, prior, WaldoStatistic())[0]
            )
            p_ref = _bare_wald_pvalue(theta, D=0.7, sigma=1.0)
            assert p_mix == pytest.approx(p_ref, abs=1e-12), (
                f"mixture pvalue at theta={theta} (eta=1) = {p_mix}; "
                f"bare Wald = {p_ref}"
            )

    def test_wald_statistic_only_at_eta_zero(self, fixtures):
        """Wald accepts only identity tilting; mixture at eta!=0 must refuse."""
        model, prior, D = fixtures
        til = MixtureTilting(selector=FixedEtaSelector(eta=0.5))
        # Wald + non-identity mixture should be flagged incompatible.
        assert not WaldStatistic().accepts_tilting(til), (
            "Wald should refuse non-identity MixtureTilting"
        )


@pytest.mark.L3
class TestMixtureWaldoInteriorMC:
    """Closed-form vs direct MC at interior eta=0.5, mild and strong conflict."""

    @pytest.fixture
    def fixtures(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        return model, prior

    @pytest.mark.parametrize(
        "D_val,theta",
        [
            (0.5, 0.2),   # mild conflict, theta near mu_n
            (0.5, 1.5),   # mild conflict, theta in tail
            (3.0, 1.5),   # strong conflict (|Delta| ~= 1.5), bimodal regime
        ],
    )
    def test_closed_form_matches_mc_at_eta_half(self, fixtures, D_val, theta):
        model, prior = fixtures
        D = np.asarray([D_val])
        til = MixtureTilting(selector=FixedEtaSelector(eta=0.5))
        p_closed = float(
            til.pvalue(np.asarray([theta]), D, model, prior, WaldoStatistic())[0]
        )
        # Direct MC: sample X ~ N(theta, 1), compute t(theta; X), count
        # exceedances over t_obs(theta).
        n_mc = 100_000
        rng = np.random.default_rng(0xCAFE)
        X = rng.normal(loc=theta, scale=1.0, size=n_mc)
        # Tilted moments at X.
        w = 1.0 / 2.0  # sigma=1, sigma_0=1 -> w=0.5
        mu0 = 0.0
        sigma = 1.0
        sigma_n_sq = w * sigma ** 2
        eta = 0.5
        alpha_lin = w + eta * (1.0 - w)
        mu_til_X = alpha_lin * X + (1.0 - alpha_lin) * mu0
        var_til_X = (
            (1.0 - eta) * sigma_n_sq + eta * sigma ** 2
            + (1.0 - eta) * eta * (1.0 - w) ** 2 * (mu0 - X) ** 2
        )
        t_X = (mu_til_X - theta) ** 2 / var_til_X
        # t_obs at observed D
        mu_til_D = alpha_lin * D_val + (1.0 - alpha_lin) * mu0
        var_til_D = (
            (1.0 - eta) * sigma_n_sq + eta * sigma ** 2
            + (1.0 - eta) * eta * (1.0 - w) ** 2 * (mu0 - D_val) ** 2
        )
        t_obs = (mu_til_D - theta) ** 2 / var_til_D
        p_mc = float(np.mean(t_X >= t_obs))
        # 4-sigma MC bound: SE = sqrt(p(1-p)/n) <= 0.5/sqrt(n).
        se = float(np.sqrt(max(p_mc * (1 - p_mc), 1e-6) / n_mc))
        assert abs(p_closed - p_mc) < 4.0 * se, (
            f"closed={p_closed:.6f}, MC={p_mc:.6f} (4sig={4*se:.6f}); "
            f"D={D_val}, theta={theta}, eta=0.5"
        )


@pytest.mark.L1
class TestMixtureCI:
    @pytest.fixture
    def fixtures(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=2.0)
        D = np.asarray([0.5])
        return model, prior, D

    def test_eta_zero_ci_matches_bare_waldo(self, fixtures):
        from frasian.tilting.identity import IdentityTilting
        model, prior, D = fixtures
        til = MixtureTilting(selector=FixedEtaSelector(eta=0.0))
        til_ref = IdentityTilting()
        ci_mix = til.confidence_interval(0.05, D, model, prior, WaldoStatistic())
        ci_ref = til_ref.confidence_interval(0.05, D, model, prior, WaldoStatistic())
        np.testing.assert_allclose(ci_mix, ci_ref, atol=1e-6)

    def test_ci_returned_ordered(self, fixtures):
        model, prior, D = fixtures
        til = MixtureTilting(selector=FixedEtaSelector(eta=0.5))
        lo, hi = til.confidence_interval(0.05, D, model, prior, WaldoStatistic())
        assert lo < hi

    def test_eta_one_ci_matches_bare_wald(self, fixtures):
        from frasian.tilting.identity import IdentityTilting
        model, prior, D = fixtures
        til = MixtureTilting(selector=FixedEtaSelector(eta=1.0))
        til_ref = IdentityTilting()
        ci_mix = til.confidence_interval(0.05, D, model, prior, WaldStatistic())
        ci_ref = til_ref.confidence_interval(0.05, D, model, prior, WaldStatistic())
        np.testing.assert_allclose(ci_mix, ci_ref, atol=1e-6)
