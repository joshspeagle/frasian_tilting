"""Regression tests pinning OTTilting against the W2-on-Gaussians closed form.

W2 geodesic between N(mu_a, sigma_a^2) and N(mu_b, sigma_b^2):
  mu_t    = (1 - t) * mu_a + t * mu_b
  sigma_t = (1 - t) * sigma_a + t * sigma_b      (linear in sigma, NOT sigma^2)

Plus the boundary contracts:
  - eta = 0 reproduces the input posterior (W2 identity element).
  - eta = 1 reproduces the likelihood-induced Gaussian N(D, sigma^2).
  - eta is admissible whenever the tilted distribution is well-defined
    along the W2 displacement line (audit P0-4): Gaussian fast path
    requires sigma_t = (1-eta)*sigma_post + eta*sigma > 0; closed-form
    pvalue requires s_t = (w + eta*(1-w))*sigma > 0 (i.e. eta > -w/(1-w)).
    The geodesic *segment* is [0, 1], but extrapolation along the same
    straight line is admissible while the math holds.

Plus the OT-tilted WALDO p-value closed form:
  s_t      = (w + eta * (1 - w)) * sigma
  mu_t     = (1 - eta) * mu_n + eta * D
  a(theta) = |mu_t - theta| / s_t
  b(theta) = (1 - eta) * (1 - w) * (mu0 - theta) / s_t
  p(theta) = Phi(b - a) + Phi(-a - b).

"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from frasian import TiltingDomainError
from frasian.models.distributions import GaussianLikelihood, NormalDistribution
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
    """Phase A audit: OT now extrapolates along the W2 displacement line.

    The geodesic *segment* is [0, 1]; eta outside that traces the same
    straight line in Wasserstein space. The Gaussian fast path requires
    sigma_t > 0; the closed-form pvalue requires s_t > 0 ⇔ eta > -w/(1-w).
    With the default _setup() (sigma=1, sigma0=1 → w=0.5, sigma_post≈0.707,
    sigma_lik=1.0), sigma_t crosses zero at eta ≈ -2.41.
    """

    @pytest.mark.parametrize("eta", [-0.5, -2.0, 1.5, 100.0])
    def test_extrapolation_admissible_when_math_holds(self, eta):
        """eta outside [0, 1] is now admissible whenever sigma_t > 0."""
        _, prior, likelihood, posterior = _setup()
        tilted = OTTilting().tilt(posterior, prior, likelihood, eta)
        assert tilted.scale > 0.0

    @pytest.mark.parametrize("eta", [-3.0, -10.0, -1e6, float("inf"), float("nan")])
    def test_out_of_range_raises(self, eta):
        """Only non-finite eta or eta that drives sigma_t<=0 raises."""
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

    def test_eta_below_admissibility_raises_with_index(self):
        """The closed-form WALDO pvalue requires s_t > 0 ⇔ eta > -w/(1-w).
        With w=0.5 (sigma=sigma0=1), the bound is eta > -1.0.
        """
        model, prior, D, theta_arr = self._setup_dyn()
        # -1.5 is below the eta_lower = -1.0 bound → must raise at index 0.
        eta_at_theta = np.array([-1.5, 0.5, 0.7], dtype=np.float64)
        with pytest.raises(TiltingDomainError, match=r"index 0"):
            OTTilting().dynamic_tilted_pvalue(
                theta_arr,
                D,
                model,
                prior,
                "waldo",
                eta_at_theta,
            )

    def test_extrapolated_eta_above_one_does_not_raise(self):
        """eta > 1 is now admissible (W2 displacement line beyond the segment)."""
        model, prior, D, theta_arr = self._setup_dyn()
        eta_at_theta = np.array([0.5, 1.5, 0.7], dtype=np.float64)
        p = OTTilting().dynamic_tilted_pvalue(
            theta_arr,
            D,
            model,
            prior,
            "waldo",
            eta_at_theta,
        )
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
        assert np.all(np.isfinite(p_arr))
        assert np.all(p_arr >= 0.0) and np.all(p_arr <= 1.0)

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
