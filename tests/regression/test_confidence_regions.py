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
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import (
    DynamicNumericalEtaSelector,
    FixedEtaSelector,
    NumericalEtaSelector,
)
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


def _model_prior(sigma=1.0, sigma0=1.0):
    return NormalNormalModel(sigma=sigma), NormalDistribution(loc=0.0, scale=sigma0)


@pytest.mark.L0
class TestConfidenceRegionsBasics:
    def test_identity_wald_is_single_region(self):
        m, p = _model_prior()
        regions = IdentityTilting().confidence_regions(
            0.05,
            np.asarray([1.5]),
            m,
            p,
            WaldStatistic(),
        )
        assert len(regions) == 1
        lo, hi = regions[0]
        assert lo < hi

    def test_identity_waldo_is_single_region(self):
        m, p = _model_prior()
        regions = IdentityTilting().confidence_regions(
            0.05,
            np.asarray([1.5]),
            m,
            p,
            WaldoStatistic(),
        )
        assert len(regions) == 1

    def test_power_law_static_is_single_region(self):
        m, p = _model_prior()
        plain = PowerLawTilting()  # FixedEta(0.0)
        regions = plain.confidence_regions(
            0.05,
            np.asarray([1.5]),
            m,
            p,
            WaldoStatistic(),
        )
        assert len(regions) == 1
        # And matches identity-WALDO exactly (η=0 redundancy).
        ident_regions = IdentityTilting().confidence_regions(
            0.05,
            np.asarray([1.5]),
            m,
            p,
            WaldoStatistic(),
        )
        np.testing.assert_allclose(regions[0], ident_regions[0], atol=1e-9)

    def test_power_law_numerical_static_is_single_region(self):
        m, p = _model_prior()
        sel_static = NumericalEtaSelector(sigma=1.0, mu0=0.0)
        scheme = PowerLawTilting(selector=sel_static)
        for D in (-2.0, 0.0, 2.0, 5.0):
            regions = scheme.confidence_regions(
                0.05,
                np.asarray([D]),
                m,
                p,
                WaldoStatistic(),
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
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)

        # Direct dynamic-p evaluation; count local maxima.
        # Phase 3a-1.5: selector signature is θ-keyed throughout.
        from frasian.tilting.eta_selectors import _NamedStatistic

        sigma = 1.0
        mu0 = 0.0
        D = 2.0
        thetas = np.linspace(D - 8, D + 8, 1001)
        coarse_theta = np.linspace(D - 8, D + 8, 25)
        coarse_eta = sel.select_grid(
            coarse_theta,
            scheme,
            statistic=_NamedStatistic("waldo"),
            model=m,
            prior=p,
            alpha=0.05,
        )
        eta_at_theta = np.interp(thetas, coarse_theta, coarse_eta)
        p_dyn = scheme.dynamic_tilted_pvalue(
            thetas,
            D,
            m,
            p,
            "waldo",
            eta_at_theta,
        )

        diff = np.diff(p_dyn)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        local_maxima = [i for i in sign_changes if diff[i] > 0]
        # We expect ≥ 2 local maxima at conflict-band D.
        assert len(local_maxima) >= 2, (
            f"dynamic-p at D={D} should be bimodal (≥2 local maxima); " f"got {len(local_maxima)}"
        )

    def test_multi_region_api_plumbs_through(self):
        """If the dynamic engine *did* produce multi-region CIs (e.g. with
        a synthetic multi-modal selector), `confidence_regions` would
        return them. We pin only the API contract here: the return type
        is a list, and length ≥ 1. The empirical multi-region case is
        rare on the realistic sandbox (see test above) but the plumbing
        must work regardless."""
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        for D in np.linspace(-3.0, 3.0, 11):
            regions = scheme.confidence_regions(
                0.05,
                np.asarray([float(D)]),
                m,
                p,
                WaldoStatistic(),
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
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        for D in np.linspace(-3.0, 3.0, 11):
            regions = scheme.confidence_regions(
                0.05,
                np.asarray([float(D)]),
                m,
                p,
                WaldoStatistic(),
            )
            for r in regions:
                lo, hi = r
                assert lo < hi, f"empty region {r} at D={D}"
            for k in range(len(regions) - 1):
                assert (
                    regions[k][1] <= regions[k + 1][0]
                ), f"overlapping/unsorted regions at D={D}: {regions}"

    def test_union_width_le_convex_hull(self):
        """For multi-region cells, union ≤ convex hull always; equal when
        single-region. This is the relationship that motivated the uplift."""
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        for D in np.linspace(-3.0, 3.0, 11):
            regions = scheme.confidence_regions(
                0.05,
                np.asarray([float(D)]),
                m,
                p,
                WaldoStatistic(),
            )
            union = sum(hi - lo for lo, hi in regions)
            hull = max(r[1] for r in regions) - min(r[0] for r in regions)
            assert union <= hull + 1e-12, f"union>hull at D={D}: {regions}"
            if len(regions) == 1:
                assert abs(union - hull) < 1e-12

    def test_confidence_interval_matches_convex_hull_of_regions(self):
        """`confidence_interval` should equal `(min lo, max hi)` of regions."""
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        for D in (-2.0, 0.0, 1.5, 3.0):
            regions = scheme.confidence_regions(
                0.05,
                np.asarray([float(D)]),
                m,
                p,
                WaldoStatistic(),
            )
            ci_lo, ci_hi = scheme.confidence_interval(
                0.05,
                np.asarray([float(D)]),
                m,
                p,
                WaldoStatistic(),
            )
            assert ci_lo == pytest.approx(min(r[0] for r in regions))
            assert ci_hi == pytest.approx(max(r[1] for r in regions))


@pytest.mark.L0
class TestMultiRegionEmpiricallyExercised:
    """Pins that the multi-region branch is actually reachable.

    At standard α=0.05 the dynamic p-value is bimodal in *shape* but its
    inter-peak valley sits well above 0.05, so CIs collapse to a single
    region — no multi-region path exercised at production α. To verify
    the plumbing actually emits >1 regions under the conditions where it
    should, we use a *high* α=0.86 at (D=3, w=0.5): the dynamic p-value's
    twin peaks (≈0.91 and ≈1.00) and inter-peak valley (≈0.83) straddle
    this α level, producing two disjoint regions.

    This is per the user's request to "adjust the test alpha level to
    get a bimodal region" — the test confirms multi-region wiring works.
    """

    def test_dyn_waldo_two_regions_at_alpha_0p86(self):
        m, p = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        regions = scheme.confidence_regions(
            0.86,
            np.asarray([3.0]),
            m,
            p,
            WaldoStatistic(),
        )
        assert len(regions) == 2, f"expected 2 regions at α=0.86 / D=3 / w=0.5; got {regions}"
        # Disjoint and sorted (re-asserts class-level invariant on this case).
        for r in regions:
            assert r[0] < r[1]
        assert regions[0][1] < regions[1][0]
        # Convex hull is strictly greater than union — the gap between
        # peaks accounts for the difference.
        union = sum(hi - lo for lo, hi in regions)
        hull = regions[1][1] - regions[0][0]
        assert hull > union + 1e-6, (
            f"expected hull strictly greater than union; got hull={hull}, " f"union={union}"
        )
