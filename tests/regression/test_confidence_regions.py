"""Multi-region CI semantics: `tilting.confidence_regions` invariants.

Pinning the protocol-level guarantees:
  - Identity tilting always returns a single-element list.
  - Static-η `power_law` always returns a single-element list (the
    p-value is unimodal at fixed η).
  - Dynamic-η `power_law` may return >1 regions at conflict-band D
    where the dynamic p-value is multimodal.
  - Within a region list: regions are θ-sorted and disjoint.
  - Union width ≤ convex-hull width always; equal for single-region.

These invariants underpin the union semantics of `coverage` and `width`.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.statistics.wald import WaldStatistic
from frasian.tilting.eta_selectors import (DynamicNumericalEtaSelector,
                                              FixedEtaSelector,
                                              NumericalEtaSelector)
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


def _model_prior(sigma=1.0, sigma0=1.0):
    return NormalNormalModel(sigma=sigma), NormalDistribution(loc=0.0, scale=sigma0)


@pytest.mark.L0
class TestConfidenceRegionsBasics:
    def test_identity_wald_is_single_region(self):
        m, p = _model_prior()
        regions = IdentityTilting().confidence_regions(
            0.05, np.asarray([1.5]), m, p, WaldStatistic(),
        )
        assert len(regions) == 1
        lo, hi = regions[0]
        assert lo < hi

    def test_identity_waldo_is_single_region(self):
        m, p = _model_prior()
        regions = IdentityTilting().confidence_regions(
            0.05, np.asarray([1.5]), m, p, WaldoStatistic(),
        )
        assert len(regions) == 1

    def test_power_law_static_is_single_region(self):
        m, p = _model_prior()
        plain = PowerLawTilting()  # FixedEta(0.0)
        regions = plain.confidence_regions(
            0.05, np.asarray([1.5]), m, p, WaldoStatistic(),
        )
        assert len(regions) == 1
        # And matches identity-WALDO exactly (η=0 redundancy).
        ident_regions = IdentityTilting().confidence_regions(
            0.05, np.asarray([1.5]), m, p, WaldoStatistic(),
        )
        np.testing.assert_allclose(regions[0], ident_regions[0], atol=1e-9)

    def test_power_law_numerical_static_is_single_region(self):
        m, p = _model_prior()
        sel_static = NumericalEtaSelector(sigma=1.0, mu0=0.0)
        scheme = PowerLawTilting(selector=sel_static)
        for D in (-2.0, 0.0, 2.0, 5.0):
            regions = scheme.confidence_regions(
                0.05, np.asarray([D]), m, p, WaldoStatistic(),
            )
            assert len(regions) == 1, f"static η* should be single-region at D={D}"

    def test_dynamic_pvalue_is_bimodal_in_shape(self):
        """The dynamic p-value's *shape* is bimodal at conflict-band D
        (two local maxima as a function of θ). Whether this materialises
        as multi-region CIs at a given α depends on whether the dip
        between the maxima drops below α — which on the realistic
        sandbox (σ=1, σ₀=1, α=0.05) it does not, because the secondary
        peak's height stays well above 0.05.

        This test pins the structural bimodality property, which is the
        smoothness pathology the framework cares about even when CIs
        themselves remain single-region. The CD experiment (Phase E)
        will detect this via the `signed_confidence` non-monotonicity.
        """
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0,
                                            n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)

        # Direct dynamic-p evaluation; count local maxima.
        from frasian.tilting.eta_selectors import _NamedStatistic
        sigma = 1.0
        mu0 = 0.0
        w = 0.5
        D = 2.0
        thetas = np.linspace(D - 8, D + 8, 1001)
        abs_d = np.abs((1.0 - w) * (mu0 - thetas) / sigma)
        coarse_grid = np.linspace(0.0, abs_d.max() + 1e-6, 25)
        coarse_eta = sel.select_grid(
            coarse_grid, scheme,
            statistic=_NamedStatistic("waldo"), w=w, alpha=0.05,
        )
        eta_at_theta = np.interp(abs_d, coarse_grid, coarse_eta)
        p_dyn = scheme.dynamic_tilted_pvalue(
            thetas, D, m, p, "waldo", eta_at_theta,
        )

        diff = np.diff(p_dyn)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        local_maxima = [i for i in sign_changes if diff[i] > 0]
        # We expect ≥ 2 local maxima at conflict-band D.
        assert len(local_maxima) >= 2, (
            f"dynamic-p at D={D} should be bimodal (≥2 local maxima); "
            f"got {len(local_maxima)}"
        )

    def test_multi_region_api_plumbs_through(self):
        """If the dynamic engine *did* produce multi-region CIs (e.g. with
        a synthetic multi-modal selector), `confidence_regions` would
        return them. We pin only the API contract here: the return type
        is a list, and length ≥ 1. The empirical multi-region case is
        rare on the realistic sandbox (see test above) but the plumbing
        must work regardless."""
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0,
                                            n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        for D in np.linspace(-3.0, 3.0, 11):
            regions = scheme.confidence_regions(
                0.05, np.asarray([float(D)]), m, p, WaldoStatistic(),
            )
            assert isinstance(regions, list)
            assert len(regions) >= 1
            for r in regions:
                assert isinstance(r, tuple) and len(r) == 2


@pytest.mark.L1
class TestConfidenceRegionsStructure:
    def test_regions_are_sorted_and_disjoint(self):
        """For any cell that returns regions, lo_i < hi_i and hi_i ≤ lo_{i+1}."""
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0,
                                            n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        for D in np.linspace(-3.0, 3.0, 11):
            regions = scheme.confidence_regions(
                0.05, np.asarray([float(D)]), m, p, WaldoStatistic(),
            )
            for r in regions:
                lo, hi = r
                assert lo < hi, f"empty region {r} at D={D}"
            for k in range(len(regions) - 1):
                assert regions[k][1] <= regions[k + 1][0], (
                    f"overlapping/unsorted regions at D={D}: {regions}"
                )

    def test_union_width_le_convex_hull(self):
        """For multi-region cells, union ≤ convex hull always; equal when
        single-region. This is the relationship that motivated the uplift."""
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0,
                                            n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        for D in np.linspace(-3.0, 3.0, 11):
            regions = scheme.confidence_regions(
                0.05, np.asarray([float(D)]), m, p, WaldoStatistic(),
            )
            union = sum(hi - lo for lo, hi in regions)
            hull = max(r[1] for r in regions) - min(r[0] for r in regions)
            assert union <= hull + 1e-12, f"union>hull at D={D}: {regions}"
            if len(regions) == 1:
                assert abs(union - hull) < 1e-12

    def test_confidence_interval_matches_convex_hull_of_regions(self):
        """`confidence_interval` should equal `(min lo, max hi)` of regions."""
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0,
                                            n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        for D in (-2.0, 0.0, 1.5, 3.0):
            regions = scheme.confidence_regions(
                0.05, np.asarray([float(D)]), m, p, WaldoStatistic(),
            )
            ci_lo, ci_hi = scheme.confidence_interval(
                0.05, np.asarray([float(D)]), m, p, WaldoStatistic(),
            )
            assert ci_lo == pytest.approx(min(r[0] for r in regions))
            assert ci_hi == pytest.approx(max(r[1] for r in regions))
