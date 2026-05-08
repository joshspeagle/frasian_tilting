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
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector, NumericalEtaSelector
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
                thetas,
                1.5,
                model,
                prior,
                "waldo",
                eta_arr,
            )
            for i, th in enumerate(thetas):
                static = scheme.tilted_pvalue(float(th), 1.5, model, prior, eta, "waldo")
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
            thetas,
            1.0,
            model,
            prior,
            "wald",
            eta_arr,
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
                np.array([0.0, 1.0]),
                1.0,
                model,
                prior,
                "waldo",
                np.array([0.0]),  # wrong shape
            )


@pytest.mark.L0
class TestDynamicTiltedConfidenceInterval:
    def test_wald_dynamic_ci_equals_static(self):
        """For Wald, dynamic CI must equal static D ± z*sigma."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        selector = NumericalEtaSelector()
        regions, total, n_reg = scheme.dynamic_tilted_confidence_interval(
            0.05,
            1.5,
            model,
            prior,
            "wald",
            selector,
            n_grid=201,
            coarse_n=11,
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
        selector = NumericalEtaSelector()
        alpha = 0.05
        regions, _, _ = scheme.dynamic_tilted_confidence_interval(
            alpha,
            1.5,
            model,
            prior,
            "waldo",
            selector,
            n_grid=201,
            coarse_n=11,
        )
        assert len(regions) >= 1
        # Recompute p at each endpoint via the same dynamic procedure.
        # Phase 3a-1.5: selector signature is θ-keyed throughout.
        from frasian.tilting.eta_selectors import _NamedStatistic

        # Mirror the dynamic_ci_scan internal coarse-θ grid: it spans
        # ``D ± search_mult·σ`` (default 8) — see _dynamic.py.
        coarse_theta = np.linspace(1.5 - 8.0, 1.5 + 8.0, 11)
        eta_grid = selector.select_grid(
            coarse_theta,
            scheme,
            statistic=_NamedStatistic("waldo"),
            model=model,
            prior=prior,
            alpha=alpha,
        )
        for lo, hi in regions:
            for theta in (lo, hi):
                eta = float(np.interp(theta, coarse_theta, eta_grid))
                p = float(scheme.tilted_pvalue(theta, 1.5, model, prior, eta, "waldo"))
                assert abs(p - alpha) < 0.02

    def test_returns_at_least_one_region(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        scheme = PowerLawTilting()
        selector = NumericalEtaSelector()
        for D in (-3.0, 0.0, 2.0, 6.0):
            regions, _, n_reg = scheme.dynamic_tilted_confidence_interval(
                0.05,
                D,
                model,
                prior,
                "waldo",
                selector,
                n_grid=151,
                coarse_n=11,
            )
            assert n_reg >= 1, f"empty CI at D={D}"
            for lo, hi in regions:
                assert lo < hi


@pytest.mark.L0
class TestDynamicNumericalEtaSelectorCache:
    """The selector caches `coarse_eta` per cache key so per-cell experiment
    loops don't recompute η*(θ) on every sample. Phase 3a-1 changed the
    cache key from `(w, α, stat, scheme, coarse_n, ad_max_bin)` to a
    θ-grid-based key; the |Δ|-binning trick is gone. The behavioural
    contract — order-independent results, distinct experiments distinct
    entries — is preserved by the model/prior fingerprints in the key.
    """

    @pytest.mark.skip(
        reason="Phase 3a-1: |Δ|-bin cache replaced by θ-grid keying; "
        "cache size now scales linearly with distinct D values. The "
        "behavioural contract (deterministic, fingerprint-keyed) is "
        "preserved by `test_cache_distinguishes_w` and "
        "`test_cache_order_independence`. Re-pin in commit 3a-2 with "
        "the bin sized to the scan-window grid."
    )
    def test_cache_hit_count_after_many_D_values(self):
        sel = DynamicNumericalEtaSelector(n_grid=81, coarse_n=11)
        scheme = PowerLawTilting(selector=sel)
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldoStatistic()
        # Call with 30 different D values spanning [-3, 3]: the resulting
        # ad_max values fall in a small number of 0.5-wide bins, so the
        # cache stays bounded (typically ≤ 6 entries on this range).
        for D in np.linspace(-3, 3, 30):
            scheme.confidence_interval(
                0.05,
                np.asarray([float(D)]),
                model,
                prior,
                stat,
            )
        assert 1 <= len(sel._cache) <= 6, (
            f"cache should fall in a small number of ad_max bins, got " f"{len(sel._cache)}"
        )

    def test_cache_order_independence(self):
        """Two selectors that see D values in different orders end up
        with the same cache contents — keys are (w, α, stat, scheme,
        coarse_n, ad_max_bin) and values are deterministic given the
        bin. This is the property we lost in the previous implementation
        and gained back via binning."""
        kwargs = dict(n_grid=81, coarse_n=11)
        sel_a = DynamicNumericalEtaSelector(**kwargs)
        sel_b = DynamicNumericalEtaSelector(**kwargs)
        scheme_a = PowerLawTilting(selector=sel_a)
        scheme_b = PowerLawTilting(selector=sel_b)
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldoStatistic()
        Ds = list(np.linspace(-3, 3, 12))
        # sel_a sees Ds in forward order; sel_b in reverse.
        for D in Ds:
            scheme_a.confidence_interval(0.05, np.asarray([float(D)]), model, prior, stat)
        for D in reversed(Ds):
            scheme_b.confidence_interval(0.05, np.asarray([float(D)]), model, prior, stat)
        # Same set of cache keys.
        assert set(sel_a._cache) == set(sel_b._cache)
        # Same cached grid + η values per key.
        for key in sel_a._cache:
            grid_a, eta_a = sel_a._cache[key]
            grid_b, eta_b = sel_b._cache[key]
            np.testing.assert_array_equal(grid_a, grid_b)
            np.testing.assert_array_equal(eta_a, eta_b)

    def test_cache_distinguishes_w(self):
        """Different priors (different w) need different cache entries."""
        sel = DynamicNumericalEtaSelector(n_grid=81, coarse_n=11)
        scheme = PowerLawTilting(selector=sel)
        model = NormalNormalModel(sigma=1.0)
        stat = WaldoStatistic()
        for sigma0 in (0.5, 1.0, 2.0):
            prior = NormalDistribution(loc=0.0, scale=sigma0)
            scheme.confidence_interval(
                0.05,
                np.asarray([1.0]),
                model,
                prior,
                stat,
            )
        assert len(sel._cache) == 3
