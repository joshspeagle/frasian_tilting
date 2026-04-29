"""Regression tests for `PowerLawTilting.dynamic_tilted_*`.

Pin the dynamic p-value and CI inversion against (a) the static
tilted-pvalue formula at η=eta_at_theta (consistency check) and
(b) the Wald baseline (eta-independence of Wald means dynamic = static).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.eta_selectors import NumericalEtaSelector
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L0
class TestDynamicTiltedPvalue:
    def test_matches_static_when_eta_constant(self):
        """If eta_at_theta is constant, dynamic_tilted_pvalue must match
        the static tilted_pvalue at that eta for every θ."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        thetas = np.linspace(-3, 5, 17)
        for eta in (-0.4, 0.0, 0.5, 0.9):
            eta_arr = np.full_like(thetas, eta)
            dyn = scheme.dynamic_tilted_pvalue(
                thetas, 1.5, model, prior, "waldo", eta_arr,
            )
            for i, th in enumerate(thetas):
                static = scheme.tilted_pvalue(float(th), 1.5, model, prior,
                                                eta, "waldo")
                np.testing.assert_allclose(dyn[i], static, atol=1e-12)

    def test_wald_dynamic_equals_static(self):
        """Wald p-value is η-independent, so dynamic must match static
        regardless of the (silly) η values supplied."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        thetas = np.linspace(-2, 4, 11)
        # Random eta values to confirm independence.
        rng = np.random.default_rng(0)
        eta_arr = rng.uniform(-0.4, 0.9, size=thetas.size)
        dyn = scheme.dynamic_tilted_pvalue(
            thetas, 1.0, model, prior, "wald", eta_arr,
        )
        for i, th in enumerate(thetas):
            expected = 2.0 * stats.norm.sf(abs(1.0 - float(th)))
            np.testing.assert_allclose(dyn[i], expected, atol=1e-12)

    def test_shape_mismatch_raises(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        with pytest.raises(ValueError):
            scheme.dynamic_tilted_pvalue(
                np.array([0.0, 1.0]), 1.0, model, prior, "waldo",
                np.array([0.0]),  # wrong shape
            )


@pytest.mark.L0
class TestDynamicTiltedConfidenceInterval:
    def test_wald_dynamic_ci_equals_static(self):
        """For Wald, dynamic CI must equal static D ± z*sigma."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        selector = NumericalEtaSelector(sigma=1.0, mu0=0.0)
        regions, total, n_reg = scheme.dynamic_tilted_confidence_interval(
            0.05, 1.5, model, prior, "wald", selector,
            n_grid=201, coarse_n=11,
        )
        assert n_reg == 1
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(regions[0][0], 1.5 - z, atol=0.02)
        np.testing.assert_allclose(regions[0][1], 1.5 + z, atol=0.02)
        np.testing.assert_allclose(total, 2 * z, atol=0.04)

    def test_pvalue_at_endpoints_close_to_alpha(self):
        """The dynamic CI inversion solves p(θ) = α at each crossing."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        selector = NumericalEtaSelector(sigma=1.0, mu0=0.0)
        alpha = 0.05
        regions, _, _ = scheme.dynamic_tilted_confidence_interval(
            alpha, 1.5, model, prior, "waldo", selector,
            n_grid=201, coarse_n=11,
        )
        assert len(regions) >= 1
        # Recompute p at each endpoint via the same dynamic procedure.
        for lo, hi in regions:
            for theta in (lo, hi):
                # Direct dynamic-pvalue evaluation
                ad = abs((1 - 0.5) * (0.0 - theta) / 1.0)
                # Use the same coarse grid as the inversion did
                from frasian.tilting.eta_selectors import _NamedStatistic
                coarse = np.linspace(0.0, 8.0, 11)
                eta_grid = selector.select_grid(
                    coarse, scheme,
                    statistic=_NamedStatistic("waldo"), w=0.5, alpha=alpha,
                )
                eta = float(np.interp(ad, coarse, eta_grid))
                p = float(scheme.tilted_pvalue(theta, 1.5, model, prior,
                                                eta, "waldo"))
                assert abs(p - alpha) < 0.02

    def test_returns_at_least_one_region(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        selector = NumericalEtaSelector(sigma=1.0, mu0=0.0)
        for D in (-3.0, 0.0, 2.0, 6.0):
            regions, _, n_reg = scheme.dynamic_tilted_confidence_interval(
                0.05, D, model, prior, "waldo", selector,
                n_grid=151, coarse_n=11,
            )
            assert n_reg >= 1, f"empty CI at D={D}"
            for lo, hi in regions:
                assert lo < hi
