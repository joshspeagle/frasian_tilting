"""Regression tests pinning PowerLawTilting against legacy `tilted_params`.

Theorem 6 in the legacy derivations:
  denom    = 1 - eta * (1 - w)
  mu_eta   = (w*D + (1-eta)*(1-w)*mu0) / denom
  sigma_eta^2 = w * sigma^2 / denom

Plus the boundary contracts:
  - eta = 0 reproduces the WALDO posterior.
  - eta = 1 reproduces the Wald limit (mu_eta = D, sigma_eta = sigma).
  - eta outside the admissible range raises TiltingDomainError.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian import TiltingDomainError
from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel, posterior_params
from frasian.tilting.eta_selectors import NumericalEtaSelector
from frasian.tilting.power_law import PowerLawTilting


def _setup(D=2.0, mu0=0.0, sigma=1.0, sigma0=1.0):
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)
    posterior = model.posterior(np.asarray([D]), prior)
    return model, prior, likelihood, posterior


@pytest.mark.L0
class TestPowerLawClosedForm:
    @pytest.mark.parametrize("D", [-2.0, 0.0, 1.5])
    @pytest.mark.parametrize("eta", [-0.4, -0.1, 0.0, 0.3, 0.7, 0.95])
    @pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
    def test_matches_theorem_6(self, D, eta, sigma0):
        sigma, mu0 = 1.0, 0.0
        _, prior, likelihood, posterior = _setup(D=D, mu0=mu0, sigma=sigma, sigma0=sigma0)
        scheme = PowerLawTilting()
        tilted = scheme.tilt(posterior, prior, likelihood, eta)

        w = sigma0**2 / (sigma**2 + sigma0**2)
        denom = 1 - eta * (1 - w)
        mu_eta_expected = (w * D + (1 - eta) * (1 - w) * mu0) / denom
        sigma_eta_expected = np.sqrt(w * sigma**2 / denom)
        np.testing.assert_allclose(tilted.loc, mu_eta_expected, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, sigma_eta_expected, atol=1e-12)


@pytest.mark.L0
class TestPowerLawIdentityElement:
    def test_eta_zero_recovers_waldo_posterior(self):
        model, prior, likelihood, posterior = _setup(D=1.5)
        scheme = PowerLawTilting()
        tilted = scheme.tilt(posterior, prior, likelihood, 0.0)
        # eta = 0 should reproduce the input posterior (the WALDO posterior).
        mu_n, sigma_n, _ = posterior_params(1.5, 0.0, 1.0, 1.0)
        np.testing.assert_allclose(tilted.loc, mu_n, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, sigma_n, atol=1e-12)

    def test_is_identity_method(self):
        scheme = PowerLawTilting()
        assert scheme.is_identity(0.0) is True
        assert scheme.is_identity(0.5) is False


@pytest.mark.L0
class TestPowerLawWaldLimit:
    def test_eta_one_recovers_wald(self):
        """At eta=1, mu_eta = D and sigma_eta = sigma."""
        model, prior, likelihood, posterior = _setup(D=2.0, sigma=1.0, sigma0=1.0)
        scheme = PowerLawTilting()
        tilted = scheme.tilt(posterior, prior, likelihood, 1.0)
        np.testing.assert_allclose(tilted.loc, 2.0, atol=1e-12)
        np.testing.assert_allclose(tilted.scale, 1.0, atol=1e-12)


@pytest.mark.L0
class TestPowerLawDomain:
    def test_eta_below_admissible_raises(self):
        # w = 0.5 -> denom = 1 - eta*0.5; denom <= 0 when eta >= 2.
        # The other tail: eta < -1 makes mu_eta blow up (denom > 1 but variance fine).
        # Strictly out-of-domain when denom <= 0: try eta = 2.5.
        _, prior, likelihood, posterior = _setup()
        scheme = PowerLawTilting()
        with pytest.raises(TiltingDomainError):
            scheme.tilt(posterior, prior, likelihood, 2.5)

    def test_numerical_selector_eta_bounds_brackets_finite_window(self):
        """Phase 3a-1.5: η-bounds moved out of the public protocol into
        `NumericalEtaSelector._eta_bounds(model, prior, scheme=...)`
        (internal). This replaces the old `scheme.admissible_range(context)`
        test.

        Updated 2026-05-11: bracket is now scheme-neutral (-50, 50) per
        the deriver agents' formal admissibility derivations
        (docs/superpowers/specs/2026-05-11-{pl,ot,fr}-admissibility-derivation.md);
        PL upper-only, OT lower-only, FR geodesically complete — no shared
        NN-formula captures all three.
        """
        sigma, mu0, w = 1.0, 0.0, 0.5
        sigma0 = float(np.sqrt(w / (1.0 - w)) * sigma)
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        sel = NumericalEtaSelector()
        scheme = PowerLawTilting()
        lo, hi = sel._eta_bounds(model, prior, scheme=scheme)
        assert lo < scheme.param_space.eta_identity < hi

    def test_numerical_selector_eta_bounds_finite_at_extreme_w(self):
        """η-bracket remains finite + ordered regardless of (model, prior)
        scale parameters.

        Updated 2026-05-11: previously asserted the closed-form NN
        `_normal_normal_w` raised for non-NN inputs and produced a
        w-dependent bracket. Per deriver A.3 the spurious lower bound
        from PL's NN formula is removed; the new bracket is the
        scheme-neutral wide default (-50, 50), self-correcting via
        +inf-on-failure inside the optimizer.
        """
        sigma, mu0 = 1.0, 0.0
        # Extreme prior scale used to push the legacy `_normal_normal_w`
        # path; the new bracket is independent of (sigma, sigma0).
        sigma0 = 1e6
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        sel = NumericalEtaSelector()
        lo, hi = sel._eta_bounds(model, prior, scheme=PowerLawTilting())
        # Bounds should still be finite + ordered.
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < hi


@pytest.mark.L0
class TestPowerLawPath:
    def test_path_yields_distribution_per_eta(self):
        _, prior, likelihood, posterior = _setup()
        scheme = PowerLawTilting()
        ts = np.linspace(-0.4, 0.9, 7)
        out = list(scheme.path(posterior, prior, likelihood, ts))
        assert len(out) == 7
        # First element is at eta=-0.4
        first = scheme.tilt(posterior, prior, likelihood, ts[0])
        np.testing.assert_allclose(out[0].loc, first.loc, atol=1e-12)
        np.testing.assert_allclose(out[0].scale, first.scale, atol=1e-12)
