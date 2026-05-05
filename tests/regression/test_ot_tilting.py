"""Regression tests pinning OTTilting against the W2-on-Gaussians closed form.

W2 geodesic between N(mu_a, sigma_a^2) and N(mu_b, sigma_b^2):
  mu_t    = (1 - t) * mu_a + t * mu_b
  sigma_t = (1 - t) * sigma_a + t * sigma_b      (linear in sigma, NOT sigma^2)

Plus the boundary contracts:
  - eta = 0 reproduces the input posterior (W2 identity element).
  - eta = 1 reproduces the likelihood-induced Gaussian N(D, sigma^2).
  - eta outside [0, 1] raises TiltingDomainError.

Plus the OT-tilted WALDO p-value closed form:
  s_t      = (w + eta * (1 - w)) * sigma
  mu_t     = (1 - eta) * mu_n + eta * D
  a(theta) = |mu_t - theta| / s_t
  b(theta) = (1 - eta) * (1 - w) * (mu0 - theta) / s_t
  p(theta) = Phi(b - a) + Phi(-a - b).

Plus the general 1D quantile-mixture identity for non-Gaussian endpoints
(Beta-Beta) — exercises the QuantileMixturePath code path.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from frasian import TiltingDomainError
from frasian.models.distributions import BetaDistribution, GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.ot import OTTilting
from frasian.tilting.quantile_mixture import QuantileMixturePath


def _setup(D=2.0, mu0=0.0, sigma=1.0, sigma0=1.0):
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)
    posterior = model.posterior(np.asarray([D]), prior)
    return model, prior, likelihood, posterior


@pytest.mark.L0
class TestOTClosedFormGaussianPath:
    @pytest.mark.parametrize("D", [-2.0, 0.0, 1.5])
    @pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
    def test_w2_path_linear_in_mu_and_sigma(self, D, eta, sigma0):
        """W2 geodesic on Gaussians: linear in (mu, sigma)."""
        sigma, mu0 = 1.0, 0.0
        _, prior, likelihood, posterior = _setup(D=D, mu0=mu0, sigma=sigma, sigma0=sigma0)
        scheme = OTTilting()
        tilted = scheme.tilt(posterior, prior, likelihood, eta)

        mu_a, sigma_a = posterior.loc, posterior.scale
        mu_b, sigma_b = D, sigma
        mu_expected = (1.0 - eta) * mu_a + eta * mu_b
        sigma_expected = (1.0 - eta) * sigma_a + eta * sigma_b
        np.testing.assert_allclose(tilted.loc, mu_expected, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, sigma_expected, atol=1e-12)


@pytest.mark.L0
class TestOTIdentityElement:
    def test_eta_zero_recovers_posterior(self):
        _, prior, likelihood, posterior = _setup(D=1.5)
        tilted = OTTilting().tilt(posterior, prior, likelihood, 0.0)
        np.testing.assert_allclose(tilted.loc, posterior.loc, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, posterior.scale, atol=1e-12)

    @pytest.mark.parametrize("D", [-1.0, 0.5, 2.5])
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_eta_one_recovers_likelihood_gaussian(self, D, sigma):
        _, prior, likelihood, posterior = _setup(D=D, sigma=sigma, sigma0=1.0)
        tilted = OTTilting().tilt(posterior, prior, likelihood, 1.0)
        np.testing.assert_allclose(tilted.loc, D, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, sigma, atol=1e-12)


@pytest.mark.L0
class TestOTAdmissibleRange:
    @pytest.mark.parametrize("eta", [-0.001, -0.5, 1.001, 1.5, 100.0])
    def test_out_of_range_raises(self, eta):
        _, prior, likelihood, posterior = _setup()
        with pytest.raises(TiltingDomainError):
            OTTilting().tilt(posterior, prior, likelihood, eta)


@pytest.mark.L0
class TestOTDynamicTiltedPvalueGuard:
    """Pin the pre-validation guard in `OTTilting.dynamic_tilted_pvalue`.

    The guard surfaces the offending index/value cleanly rather than
    populating `out` partially before raising mid-loop. A future refactor
    that drops the pre-validation back into the loop would silently
    succeed-then-fail; these tests catch that.
    """

    def _setup_dyn(self):
        sigma, mu0, sigma0 = 1.0, 0.0, 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        D = 2.0
        theta_arr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        return model, prior, D, theta_arr

    def test_nan_eta_raises_with_index(self):
        model, prior, D, theta_arr = self._setup_dyn()
        eta_at_theta = np.array([0.5, np.nan, 0.7], dtype=np.float64)
        with pytest.raises(TiltingDomainError, match=r"index 1"):
            OTTilting().dynamic_tilted_pvalue(
                theta_arr,
                D,
                model,
                prior,
                "waldo",
                eta_at_theta,
            )

    def test_eta_above_one_raises_with_index(self):
        model, prior, D, theta_arr = self._setup_dyn()
        eta_at_theta = np.array([0.5, 1.5, 0.7], dtype=np.float64)
        with pytest.raises(TiltingDomainError, match=r"index 1"):
            OTTilting().dynamic_tilted_pvalue(
                theta_arr,
                D,
                model,
                prior,
                "waldo",
                eta_at_theta,
            )

    def test_eta_below_zero_raises_with_index(self):
        model, prior, D, theta_arr = self._setup_dyn()
        eta_at_theta = np.array([-0.1, 0.5, 0.7], dtype=np.float64)
        with pytest.raises(TiltingDomainError, match=r"index 0"):
            OTTilting().dynamic_tilted_pvalue(
                theta_arr,
                D,
                model,
                prior,
                "waldo",
                eta_at_theta,
            )

    def test_valid_eta_returns_finite_pvalues(self):
        """Positive-path: `eta_at_theta = [0.0, 0.5, 1.0]` does NOT raise
        and returns finite p-values in [0, 1]. Without this, the guard's
        valid-path is never exercised."""
        model, prior, D, theta_arr = self._setup_dyn()
        eta_at_theta = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        p = OTTilting().dynamic_tilted_pvalue(
            theta_arr,
            D,
            model,
            prior,
            "waldo",
            eta_at_theta,
        )
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
        assert p_arr.shape == theta_arr.shape
        assert np.all(np.isfinite(p_arr))
        assert np.all(p_arr >= 0.0) and np.all(p_arr <= 1.0)


@pytest.mark.L0
class TestOTTiltedWaldoPvalue:
    @pytest.mark.parametrize("eta", [0.0, 0.3, 0.6, 0.9])
    @pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("D", [-1.0, 0.5, 2.0])
    def test_closed_form_against_explicit_recompute(self, eta, sigma0, D):
        sigma, mu0 = 1.0, 0.0
        model, prior, _, _ = _setup(D=D, mu0=mu0, sigma=sigma, sigma0=sigma0)
        scheme = OTTilting()
        thetas = np.linspace(D - 4 * sigma, D + 4 * sigma, 17)
        # Reference: recompute by the documented closed form.
        w = sigma0**2 / (sigma**2 + sigma0**2)
        mu_n = w * D + (1.0 - w) * mu0
        mu_t = (1.0 - eta) * mu_n + eta * D
        s_t = (w + eta * (1.0 - w)) * sigma
        a = np.abs(mu_t - thetas) / s_t
        b = (1.0 - eta) * (1.0 - w) * (mu0 - thetas) / s_t
        p_expected = stats.norm.cdf(b - a) + stats.norm.cdf(-a - b)
        p_actual = scheme.tilted_pvalue(thetas, D, model, prior, eta, "waldo")
        np.testing.assert_allclose(p_actual, p_expected, atol=1e-12)

    @pytest.mark.parametrize("D", [-1.0, 0.0, 1.5, 3.0])
    @pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
    def test_eta_zero_matches_bare_waldo(self, D, sigma0):
        from frasian.statistics.waldo import WaldoStatistic

        sigma, mu0 = 1.0, 0.0
        model, prior, _, _ = _setup(D=D, mu0=mu0, sigma=sigma, sigma0=sigma0)
        thetas = np.linspace(D - 3 * sigma, D + 3 * sigma, 13)
        ot_p = OTTilting().tilted_pvalue(thetas, D, model, prior, 0.0, "waldo")
        bare_p = WaldoStatistic().pvalue(thetas, np.asarray([D]), model, prior)
        np.testing.assert_allclose(ot_p, bare_p, atol=1e-12)

    @pytest.mark.parametrize("D", [-1.0, 0.5, 2.0])
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_eta_one_matches_bare_wald(self, D, sigma):
        sigma0, mu0 = 1.0, 0.0
        model, prior, _, _ = _setup(D=D, mu0=mu0, sigma=sigma, sigma0=sigma0)
        thetas = np.linspace(D - 3 * sigma, D + 3 * sigma, 13)
        ot_p = OTTilting().tilted_pvalue(thetas, D, model, prior, 1.0, "waldo")
        wald_p = 2.0 * stats.norm.sf(np.abs(D - thetas) / sigma)
        np.testing.assert_allclose(ot_p, wald_p, atol=1e-12)

    def test_wald_statistic_eta_independent(self):
        """Wald ignores the prior, so OT-tilted Wald == bare Wald at any eta."""
        model, prior, _, _ = _setup(D=1.5)
        thetas = np.linspace(-2.0, 4.0, 11)
        scheme = OTTilting()
        for eta in (0.0, 0.3, 0.7, 1.0):
            ot_p = scheme.tilted_pvalue(thetas, 1.5, model, prior, eta, "wald")
            wald_p = 2.0 * stats.norm.sf(np.abs(1.5 - thetas) / 1.0)
            np.testing.assert_allclose(ot_p, wald_p, atol=1e-12)


@pytest.mark.L0
class TestQuantileMixturePathBetaEndpoints:
    """General 1D quantile-mixture: exercises non-Gaussian endpoints."""

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("u", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_quantile_is_linear_in_endpoints(self, t, u):
        p = BetaDistribution(alpha=2.0, beta=5.0)
        q = BetaDistribution(alpha=5.0, beta=2.0)
        path = QuantileMixturePath(p=p, q=q, t=t)
        actual = float(path.quantile(np.asarray(u)))
        expected = float((1.0 - t) * p.quantile(u) + t * q.quantile(u))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_endpoint_recovery_on_beta(self):
        p = BetaDistribution(alpha=2.0, beta=5.0)
        q = BetaDistribution(alpha=5.0, beta=2.0)
        u_grid = np.linspace(0.01, 0.99, 25)

        path0 = QuantileMixturePath(p=p, q=q, t=0.0)
        np.testing.assert_allclose(path0.quantile(u_grid), p.quantile(u_grid), atol=1e-12)

        path1 = QuantileMixturePath(p=p, q=q, t=1.0)
        np.testing.assert_allclose(path1.quantile(u_grid), q.quantile(u_grid), atol=1e-12)

    def test_mean_is_linear_in_endpoints(self):
        p = BetaDistribution(alpha=2.0, beta=5.0)
        q = BetaDistribution(alpha=5.0, beta=2.0)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            path = QuantileMixturePath(p=p, q=q, t=t)
            expected = (1.0 - t) * p.mean() + t * q.mean()
            np.testing.assert_allclose(path.mean(), expected, atol=1e-12)

    def test_cdf_quantile_round_trip_on_beta(self):
        p = BetaDistribution(alpha=2.0, beta=5.0)
        q = BetaDistribution(alpha=5.0, beta=2.0)
        path = QuantileMixturePath(p=p, q=q, t=0.4)
        # Round-trip: x -> u -> x must be the identity (atol depends on tail).
        u_in = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        x = path.quantile(u_in)
        u_out = path.cdf(x)
        np.testing.assert_allclose(u_out, u_in, atol=1e-8)


@pytest.mark.L2
class TestQuantileMixtureBoundary:
    """Pin the cdf/pdf outside-support behaviour.

    The cdf returns exact 0.0 below support and exact 1.0 above support.
    The pdf returns exact 0.0 outside support, computed symmetrically
    with cdf's boundary detector (against the endpoint quantiles), so
    a future refactor of cdf's exact-boundary return value can not
    silently re-introduce ~3.6e-9 chain-rule garbage in the pdf.
    """

    def test_cdf_below_support_returns_zero(self):
        # Beta(2,5) and Beta(5,2) both supported on [0,1]. The path's
        # support is [0, 1]; -1.0 is strictly below.
        p = BetaDistribution(alpha=2.0, beta=5.0)
        q = BetaDistribution(alpha=5.0, beta=2.0)
        path = QuantileMixturePath(p=p, q=q, t=0.5)
        x_below = np.asarray(-1.0)
        u = path.cdf(x_below)
        np.testing.assert_array_equal(np.asarray(u), 0.0)

    def test_cdf_above_support_returns_one(self):
        p = BetaDistribution(alpha=2.0, beta=5.0)
        q = BetaDistribution(alpha=5.0, beta=2.0)
        path = QuantileMixturePath(p=p, q=q, t=0.5)
        x_above = np.asarray(2.0)
        u = path.cdf(x_above)
        np.testing.assert_array_equal(np.asarray(u), 1.0)

    def test_pdf_outside_support_is_exact_zero(self):
        """pdf at x outside support must be exactly 0.0 (not ~3.6e-9).

        Pins 1.5-O3's fix: the chain-rule expression underflows to a
        tiny positive number near the support boundary, but the
        outside-support mask zeroes it out.
        """
        p = BetaDistribution(alpha=2.0, beta=5.0)
        q = BetaDistribution(alpha=5.0, beta=2.0)
        path = QuantileMixturePath(p=p, q=q, t=0.5)
        # Symmetric outside-support detection: anything below the lower
        # endpoint quantile or above the upper.
        x_below = np.asarray(-0.5)
        x_above = np.asarray(1.5)
        np.testing.assert_array_equal(np.asarray(path.pdf(x_below)), 0.0)
        np.testing.assert_array_equal(np.asarray(path.pdf(x_above)), 0.0)

    def test_pdf_inside_support_is_finite_positive(self):
        p = BetaDistribution(alpha=2.0, beta=5.0)
        q = BetaDistribution(alpha=5.0, beta=2.0)
        path = QuantileMixturePath(p=p, q=q, t=0.5)
        # 0.5 is interior to the support of every Beta(α, β) with α, β > 0.
        x_in = np.asarray(0.5)
        f = float(np.asarray(path.pdf(x_in)))
        assert np.isfinite(f) and f > 0.0


@pytest.mark.L2
class TestOTPathAgreementGaussianEndpoints:
    """Fast-path (closed-form) vs general path (QuantileMixturePath) on Gaussians.

    On Gaussian endpoints both paths must agree — the fast path is the
    closed form of the general 1D W2 geodesic, so any drift between them
    is silent garbage. Pins quantile and pdf agreement.
    """

    @pytest.mark.parametrize(
        "mu_p,sigma_p,mu_q,sigma_q",
        [
            (0.0, 1.0, 0.0, 2.0),  # balanced: same mean, different scale
            (3.0, 1.0, 0.0, 1.0),  # conflict: same scale, different mean
            (-1.0, 0.5, 2.0, 1.5),  # mixed: both differ
        ],
    )
    def test_fast_path_matches_general_path(
        self,
        mu_p,
        sigma_p,
        mu_q,
        sigma_q,
    ):
        t = 0.5
        p_normal = NormalDistribution(loc=mu_p, scale=sigma_p)
        q_normal = NormalDistribution(loc=mu_q, scale=sigma_q)

        # Fast path: linear interpolation in (mu, sigma).
        mu_fast = (1.0 - t) * mu_p + t * mu_q
        sigma_fast = (1.0 - t) * sigma_p + t * sigma_q
        fast = NormalDistribution(loc=mu_fast, scale=sigma_fast)

        # General path: QuantileMixturePath wrapping the same Gaussian endpoints.
        general = QuantileMixturePath(p=p_normal, q=q_normal, t=t)

        # Quantile agreement at 50 q points (atol 1e-12).
        q_grid = np.linspace(0.01, 0.99, 50)
        qf = np.asarray([float(fast.quantile(np.asarray(u))) for u in q_grid])
        qg = np.asarray([float(general.quantile(np.asarray(u))) for u in q_grid])
        np.testing.assert_allclose(qg, qf, atol=1e-12)

        # PDF agreement at 50 theta points (atol 1e-9). Choose a theta range
        # well inside the support so the chain-rule pdf is numerically stable.
        lo = mu_fast - 3.0 * sigma_fast
        hi = mu_fast + 3.0 * sigma_fast
        theta_grid = np.linspace(lo, hi, 50)
        pdf_fast = np.asarray([float(fast.pdf(np.asarray(x))) for x in theta_grid])
        pdf_general = np.asarray([float(general.pdf(np.asarray(x))) for x in theta_grid])
        np.testing.assert_allclose(pdf_general, pdf_fast, atol=1e-9)
