"""Selector-aware `tilting.pvalue(...)` semantics.

Pinning the protocol-level guarantees:
  - `IdentityTilting.pvalue` ≡ `statistic.pvalue` (no tilt).
  - `PowerLawTilting(FixedEta(0.0)).pvalue` ≡ `IdentityTilting.pvalue`
    when paired with WALDO (η=0 is the WALDO identity).
  - `PowerLawTilting(FixedEta(η)).pvalue` ≡ `tilted_pvalue(η)` for
    arbitrary fixed η (static dispatch).
  - `PowerLawTilting(DynamicNumerical()).pvalue` matches the dynamic
    p-value computed with the selector-derived `eta_at_theta` lookup.

The CD constructor (Phase D) consumes `tilting.pvalue` to evaluate
p(θ) on a fine θ-grid; these tests pin the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import (
    DynamicNumericalEtaSelector,
    FixedEtaSelector,
    _NamedStatistic,
)
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L0
class TestTiltingPvalueDispatch:
    def setup_method(self):
        self.model = NormalNormalModel(sigma=1.0)
        self.prior = NormalDistribution(loc=0.0, scale=1.0)
        self.D = np.asarray([1.5])
        self.thetas = np.linspace(-3.0, 4.0, 11)

    def test_identity_wald_delegates(self):
        scheme_p = IdentityTilting().pvalue(
            self.thetas, self.D, self.model, self.prior, WaldStatistic(),
        )
        direct = WaldStatistic().pvalue(
            self.thetas, self.D, self.model, self.prior,
        )
        np.testing.assert_allclose(scheme_p, direct, atol=1e-12)

    def test_identity_waldo_delegates(self):
        scheme_p = IdentityTilting().pvalue(
            self.thetas, self.D, self.model, self.prior, WaldoStatistic(),
        )
        direct = WaldoStatistic().pvalue(
            self.thetas, self.D, self.model, self.prior,
        )
        np.testing.assert_allclose(scheme_p, direct, atol=1e-12)

    def test_power_law_fixed_eta_zero_matches_identity(self):
        """η=0 is WALDO's identity element, so power_law[fixed=0] ≡ identity
        when paired with WALDO."""
        scheme = PowerLawTilting(selector=FixedEtaSelector(eta=0.0))
        scheme_p = scheme.pvalue(
            self.thetas, self.D, self.model, self.prior, WaldoStatistic(),
        )
        ident_p = IdentityTilting().pvalue(
            self.thetas, self.D, self.model, self.prior, WaldoStatistic(),
        )
        np.testing.assert_allclose(scheme_p, ident_p, atol=1e-12)

    @pytest.mark.parametrize("eta", [-0.4, 0.0, 0.3, 0.7])
    def test_power_law_fixed_eta_matches_tilted_pvalue(self, eta):
        """Static dispatch: scheme.pvalue should match tilted_pvalue(eta)."""
        scheme = PowerLawTilting(selector=FixedEtaSelector(eta=eta))
        scheme_p = scheme.pvalue(
            self.thetas, self.D, self.model, self.prior, WaldoStatistic(),
        )
        direct = scheme.tilted_pvalue(
            self.thetas, float(self.D[0]), self.model, self.prior,
            eta, "waldo",
        )
        np.testing.assert_allclose(scheme_p, direct, atol=1e-12)

    def test_power_law_dynamic_matches_dynamic_tilted_pvalue(self):
        """Dynamic dispatch: scheme.pvalue should match
        dynamic_tilted_pvalue when given the same eta_at_theta lookup
        the selector would produce."""
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0,
                                            n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)

        scheme_p = scheme.pvalue(
            self.thetas, self.D, self.model, self.prior, WaldoStatistic(),
        )

        # Reconstruct the eta_at_theta lookup the dispatch would use.
        sigma = self.model.sigma
        mu0 = self.prior.loc
        sigma0 = self.prior.scale
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        abs_delta_theta = np.abs((1.0 - w) * (mu0 - self.thetas) / sigma)
        coarse_n = 25
        ad_max = float(abs_delta_theta.max()) + 1e-6
        coarse_grid = np.linspace(0.0, ad_max, coarse_n)
        coarse_eta = sel.select_grid(
            coarse_grid, scheme,
            statistic=_NamedStatistic("waldo"), w=w, alpha=0.05,
        )
        eta_at_theta = np.interp(abs_delta_theta, coarse_grid, coarse_eta)

        direct = scheme.dynamic_tilted_pvalue(
            self.thetas, float(self.D[0]), self.model, self.prior,
            "waldo", eta_at_theta,
        )

        np.testing.assert_allclose(scheme_p, direct, atol=1e-9)

    def test_pvalue_returns_array_of_correct_shape(self):
        scheme = PowerLawTilting(selector=FixedEtaSelector(eta=0.0))
        result = scheme.pvalue(
            self.thetas, self.D, self.model, self.prior, WaldoStatistic(),
        )
        assert result.shape == self.thetas.shape
        assert np.all((0.0 <= result) & (result <= 1.0))
