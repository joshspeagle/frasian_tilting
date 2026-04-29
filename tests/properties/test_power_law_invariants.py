"""Property tests for the PowerLawTilting scheme.

Invariants checked:
  - Identity element: tilt(eta=0) returns the input posterior.
  - Tilted posterior pdf integrates to 1 (normalized).
  - Continuity in eta (Lipschitz bound on (mu_eta, sigma_eta)).
  - Out-of-domain eta raises TiltingDomainError, never NaN.
  - Theorem 7 connection: noncentrality scales as (1-eta)^2.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frasian import TiltingDomainError
from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel, posterior_params
from frasian.tilting.power_law import PowerLawTilting

_D = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_SIGMA0 = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_ETA = st.floats(min_value=-0.6, max_value=0.95, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestPowerLawInvariants:
    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0)
    @settings(max_examples=80, deadline=None)
    def test_identity_at_eta_zero(self, D, sigma, sigma0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=D, sigma=sigma)
        post = model.posterior(np.asarray([D]), prior)
        tilted = PowerLawTilting().tilt(post, prior, lik, 0.0)
        np.testing.assert_allclose(tilted.loc, post.loc, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, post.scale, atol=1e-12)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0, eta=_ETA)
    @settings(max_examples=80, deadline=None)
    def test_pdf_integrates_to_one(self, D, sigma, sigma0, eta):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=D, sigma=sigma)
        post = model.posterior(np.asarray([D]), prior)
        tilted = PowerLawTilting().tilt(post, prior, lik, eta)
        # Integrate on a 12-sigma window around the tilted mean.
        xs = np.linspace(tilted.loc - 12 * tilted.scale,
                         tilted.loc + 12 * tilted.scale, 2001)
        integral = np.trapezoid(tilted.pdf(xs), xs)
        np.testing.assert_allclose(integral, 1.0, atol=5e-4)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0, eta=_ETA)
    @settings(max_examples=60, deadline=None)
    def test_continuous_in_eta(self, D, sigma, sigma0, eta):
        """A small change in eta produces a small change in (mu_eta, sigma_eta)."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=D, sigma=sigma)
        post = model.posterior(np.asarray([D]), prior)
        scheme = PowerLawTilting()
        h = 1e-4
        eta2 = min(0.95, eta + h)
        a = scheme.tilt(post, prior, lik, eta)
        b = scheme.tilt(post, prior, lik, eta2)
        # Crude Lipschitz bound; actual values depend on (D, sigma, sigma0).
        bound = 1e3 * (eta2 - eta) + 1e-6
        assert abs(a.loc - b.loc) <= bound
        assert abs(a.scale - b.scale) <= bound

    def test_out_of_domain_raises_not_nan(self):
        sigma, sigma0 = 1.0, 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        lik = GaussianLikelihood(D=2.0, sigma=sigma)
        post = model.posterior(np.asarray([2.0]), prior)
        # w = 0.5; eta = 2 -> denom = 0; eta > 2 -> denom < 0
        with pytest.raises(TiltingDomainError):
            PowerLawTilting().tilt(post, prior, lik, 3.0)

    @given(eta=st.floats(min_value=-0.4, max_value=0.9, allow_nan=False),
           sigma0=st.floats(min_value=0.5, max_value=2.0, allow_nan=False))
    @settings(max_examples=40, deadline=None)
    def test_noncentrality_quadratic_decay(self, eta, sigma0):
        """Theorem 7: lambda_eta(theta) = (1 - eta)^2 * lambda_0(theta).

        Verifies the scaling by computing lambda_eta via the actual
        Theorem-6 closed forms. Under H0 (D ~ N(theta, sigma)):

            mu_eta - theta = bias_eta + (w/denom)(D - theta),
            bias_eta       = (1-eta)(1-w)(mu0 - theta) / denom,
            sigma_eta^2    = w * sigma^2 / denom,
            denom          = 1 - eta*(1-w).

        Var_D[(mu_eta - theta)/sigma_eta] = w/denom, so the tilted
        WALDO statistic distributes as (w/denom) * chi^2_1(lambda_eta) with

            lambda_eta = bias_eta^2 * denom^2 / (w^2 * sigma^2)
                       = (1 - eta)^2 * lambda_0,

        where lambda_0 is `models.normal_normal.noncentrality` at eta=0.
        Both sides are computed via the actual implementations — no
        tautology — and the closed-form bias_eta is cross-checked
        against `PowerLawTilting.tilt(...).loc` at D=theta.
        """
        from frasian.models.normal_normal import noncentrality as _nc, weight

        sigma, mu0 = 1.0, 0.0
        theta = mu0 + 0.5 * sigma0  # representative test point

        w = weight(sigma, sigma0)
        denom = 1.0 - eta * (1.0 - w)
        bias_eta = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / denom

        # Cross-check the closed-form bias against the actual tilt() output:
        # tilting with D = theta makes mu_eta = E_H0[mu_eta], so the tilted
        # location must equal theta + bias_eta.
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        post_h0 = model.posterior(np.asarray([theta]), prior)
        lik_h0 = GaussianLikelihood(D=theta, sigma=sigma)
        tilted_h0 = PowerLawTilting().tilt(post_h0, prior, lik_h0, eta)
        np.testing.assert_allclose(
            tilted_h0.loc - theta, bias_eta, atol=1e-12,
            err_msg="closed-form bias_eta disagrees with tilt() output",
        )

        # LHS: lambda_eta from the tilted-statistic distribution.
        lambda_eta_LHS = bias_eta ** 2 * denom ** 2 / (w ** 2 * sigma ** 2)

        # RHS: (1 - eta)^2 * lambda_0.
        lambda_0 = float(_nc(theta, mu0, w, sigma))
        lambda_eta_RHS = (1.0 - eta) ** 2 * lambda_0

        np.testing.assert_allclose(lambda_eta_LHS, lambda_eta_RHS, atol=1e-12)
