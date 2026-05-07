"""Property tests for the OTTilting scheme.

Invariants checked (one per `docs/methods/ot.md` invariant):

  - Identity element: tilt(eta=0) returns the input posterior (W2-style).
  - Endpoint at eta=1: tilt returns the likelihood-induced Gaussian.
  - Continuity in eta: small change in eta -> small change in (mu_t, sigma_t).
  - Output sigma stays positive on the admissible range.
  - Admissible range: any finite eta with sigma_t > 0 (Gaussian path) /
    s_t > 0 (closed-form pvalue, eta > -w/(1-w)). The geodesic *segment*
    is [0, 1] but the W2 displacement line extends in both directions
    along the same straight line in Wasserstein space.
  - Quantile-mixture identity: F_t^{-1}(u) = (1-t) F_p^{-1}(u) + t F_q^{-1}(u)
    on the general 1D path (with NormalDistribution endpoints, since
    GaussianLikelihood is the only conversion currently implemented).
  - tilted-WALDO endpoint recovery: at eta=0 reduces to bare WALDO,
    at eta=1 reduces to bare two-sided Wald.
  - Out-of-domain eta raises TiltingDomainError, never returns NaN.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frasian import TiltingDomainError
from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.ot import OTTilting
from frasian.tilting.quantile_mixture import QuantileMixturePath

_D = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_SIGMA0 = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_ETA = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestOTInvariants:
    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0)
    @settings(max_examples=60, deadline=None)
    def test_identity_at_eta_zero(self, D, sigma, sigma0):
        """tilt(eta=0) returns the input posterior (W2 identity element)."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=D, sigma=sigma)
        post = model.posterior(np.asarray([D]), prior)
        tilted = OTTilting().tilt(post, prior, lik, 0.0)
        np.testing.assert_allclose(tilted.loc, post.loc, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, post.scale, atol=1e-12)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0)
    @settings(max_examples=60, deadline=None)
    def test_endpoint_at_eta_one(self, D, sigma, sigma0):
        """tilt(eta=1) returns the likelihood-induced Gaussian N(D, sigma)."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=D, sigma=sigma)
        post = model.posterior(np.asarray([D]), prior)
        tilted = OTTilting().tilt(post, prior, lik, 1.0)
        np.testing.assert_allclose(tilted.loc, D, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, sigma, atol=1e-12)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0, eta=_ETA)
    @settings(max_examples=80, deadline=None)
    def test_output_sigma_positive(self, D, sigma, sigma0, eta):
        """W2-geodesic must keep sigma_t > 0 for eta in [0, 1]."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=D, sigma=sigma)
        post = model.posterior(np.asarray([D]), prior)
        tilted = OTTilting().tilt(post, prior, lik, eta)
        assert tilted.scale > 0.0

    @given(
        D=_D,
        sigma=_SIGMA,
        sigma0=_SIGMA0,
        eta=st.floats(min_value=0.0, max_value=0.99, allow_nan=False),
    )
    @settings(max_examples=60, deadline=None)
    def test_continuous_in_eta(self, D, sigma, sigma0, eta):
        """A small change in eta produces a small change in (mu_t, sigma_t).

        The W2-on-Gaussians path is **linear** in (mu, sigma), so the
        Lipschitz constant is bounded by max(|mu_n - D|, |sigma_n - sigma|).
        We check a Lipschitz bound of 10 (a wide upper bound for the
        Hypothesis test ranges).
        """
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=D, sigma=sigma)
        post = model.posterior(np.asarray([D]), prior)
        scheme = OTTilting()
        h = 1e-4
        eta2 = eta + h
        a = scheme.tilt(post, prior, lik, eta)
        b = scheme.tilt(post, prior, lik, eta2)
        bound = 10.0 * h + 1e-9
        assert abs(a.loc - b.loc) <= bound
        assert abs(a.scale - b.scale) <= bound

    def test_out_of_domain_raises(self):
        """Non-finite eta or extrapolation that drives sigma_t<=0 must raise.

        The W2 displacement line accepts any finite eta in principle; the
        Gaussian fast path requires sigma_t = (1-eta)*sigma_a + eta*sigma_b
        > 0. With sigma_post < sigma_lik, large positive eta is fine but
        large negative eta drives sigma_t through zero; vice versa.
        """
        sigma, sigma0 = 1.0, 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=2.0, sigma=sigma)
        post = model.posterior(np.asarray([2.0]), prior)
        # sigma_post = sqrt(w)*sigma = sqrt(0.5) ≈ 0.707, sigma_lik = 1.0.
        # sigma_t = (1-eta)*0.707 + eta*1.0 — positive for all eta >= 0;
        # crosses zero at eta = -0.707/(1.0-0.707) ≈ -2.41.
        with pytest.raises(TiltingDomainError):
            OTTilting().tilt(post, prior, lik, -10.0)  # sigma_t < 0
        with pytest.raises(TiltingDomainError):
            OTTilting().tilt(post, prior, lik, float("inf"))
        with pytest.raises(TiltingDomainError):
            OTTilting().tilt(post, prior, lik, float("nan"))

    def test_extrapolation_beyond_segment_admissible(self):
        """W2 displacement line: eta>1 and small eta<0 produce valid Gaussians."""
        sigma, sigma0 = 1.0, 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=2.0, sigma=sigma)
        post = model.posterior(np.asarray([2.0]), prior)
        # eta=2 (extrapolating past likelihood): sigma_t = -1*0.707 + 2*1
        # = 1.293 > 0; mu_t extrapolates linearly past D.
        tilted_above = OTTilting().tilt(post, prior, lik, 2.0)
        assert tilted_above.scale > 0.0
        # eta=-0.5 (extrapolating past posterior, but not enough to flip sigma):
        # sigma_t = 1.5*0.707 - 0.5*1.0 = 0.561 > 0.
        tilted_below = OTTilting().tilt(post, prior, lik, -0.5)
        assert tilted_below.scale > 0.0

    @given(
        t=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        u=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        mu_a=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
        sigma_a=st.floats(min_value=0.3, max_value=2.0, allow_nan=False),
        mu_b=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
        sigma_b=st.floats(min_value=0.3, max_value=2.0, allow_nan=False),
    )
    @settings(max_examples=80, deadline=None)
    def test_quantile_mixture_identity(self, t, u, mu_a, sigma_a, mu_b, sigma_b):
        """General 1D path: F_t^{-1}(u) = (1-t) F_p^{-1}(u) + t F_q^{-1}(u)."""
        p = NormalDistribution(loc=mu_a, scale=sigma_a)
        q = NormalDistribution(loc=mu_b, scale=sigma_b)
        path = QuantileMixturePath(p=p, q=q, t=t)
        lhs = float(path.quantile(np.asarray(u)))
        rhs = float((1.0 - t) * p.quantile(u) + t * q.quantile(u))
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    @given(
        D=_D,
        sigma=_SIGMA,
        sigma0=_SIGMA0,
        theta=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=60, deadline=None)
    def test_tilted_pvalue_reduces_to_bare_waldo_at_eta_zero(self, D, sigma, sigma0, theta):
        """At eta=0, tilted_pvalue (waldo) must equal bare WALDO p-value."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        scheme = OTTilting()
        bare = float(WaldoStatistic().pvalue(theta, np.asarray([D]), model, prior))
        tilted = float(scheme.tilted_pvalue(theta, D, model, prior, 0.0, "waldo"))
        np.testing.assert_allclose(tilted, bare, atol=1e-12)

    @given(
        D=_D,
        sigma=_SIGMA,
        sigma0=_SIGMA0,
        theta=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=60, deadline=None)
    def test_tilted_pvalue_reduces_to_bare_wald_at_eta_one(self, D, sigma, sigma0, theta):
        """At eta=1, tilted_pvalue (waldo) must equal bare two-sided Wald."""
        from scipy import stats

        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        scheme = OTTilting()
        tilted = float(scheme.tilted_pvalue(theta, D, model, prior, 1.0, "waldo"))
        wald = float(2.0 * stats.norm.sf(abs(D - theta) / sigma))
        np.testing.assert_allclose(tilted, wald, atol=1e-12)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0, eta=_ETA)
    @settings(max_examples=60, deadline=None)
    def test_tilted_pvalue_at_mu_t_equals_one(self, D, sigma, sigma0, eta):
        """p(theta = mu_t(eta)) = 1 exactly (mode invariant; a=0 -> Phi(b)+Phi(-b)=1).

        At theta = mu_t the residual a vanishes, leaving Phi(b)+Phi(-b)=1
        by symmetry. This is the WALDO-mode invariant for any tilting that
        preserves the two-Phi structure of the formula.
        """
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        w = sigma0**2 / (sigma**2 + sigma0**2)
        mu_n = w * D + (1.0 - w) * 0.0
        mu_t = (1.0 - eta) * mu_n + eta * D
        scheme = OTTilting()
        p = float(scheme.tilted_pvalue(mu_t, D, model, prior, eta, "waldo"))
        np.testing.assert_allclose(p, 1.0, atol=1e-12)

    @given(
        sigma=_SIGMA, sigma0=_SIGMA0, eta=st.floats(min_value=0.0, max_value=0.99, allow_nan=False)
    )
    @settings(max_examples=40, deadline=None)
    def test_s_t_strictly_increasing_in_eta(self, sigma, sigma0, eta):
        """s_t(eta) = (w + eta*(1-w))*sigma is strictly increasing in eta;
        s_t(0) = w*sigma; s_t(1) = sigma."""
        w = sigma0**2 / (sigma**2 + sigma0**2)
        h = 1e-3
        s_t_lo = (w + eta * (1.0 - w)) * sigma
        s_t_hi = (w + (eta + h) * (1.0 - w)) * sigma
        assert s_t_hi > s_t_lo
        # endpoint values
        np.testing.assert_allclose((w + 0.0 * (1.0 - w)) * sigma, w * sigma)
        np.testing.assert_allclose((w + 1.0 * (1.0 - w)) * sigma, sigma)
