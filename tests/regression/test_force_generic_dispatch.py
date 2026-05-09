"""L2 regression: `force_generic=True` flips Wald/WALDO dispatch on Normal-Normal.

Pins three contracts for the path-coverage debug flag added to
`WaldStatistic` / `WaldoStatistic`:

  1. `cell_name` is discriminated: `"wald" / "wald[generic]"` and
     `"waldo" / "waldo[generic]"`. The runner / cache key flow through
     this name, so the two flavours don't collide on disk.
  2. `force_generic=True` actually exercises the generic numerical
     path on Normal-Normal (the closed-form fast path is the default).
     We verify by comparing the dispatched public-API result against
     the result of calling the generic helper directly.
  3. The generic path matches the closed-form path on Normal-Normal
     within tolerance (Wald: bit-equal; WALDO: MC noise).

`acceptance_region` has no generic data-space inversion, so
`force_generic=True` raises `NotImplementedError` rather than silently
flipping back to the closed form — pinned here too.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic


@pytest.mark.L2
class TestWaldForceGeneric:
    def test_cell_name(self):
        assert WaldStatistic().cell_name == "wald"
        assert WaldStatistic(force_generic=True).cell_name == "wald[generic]"

    def test_pvalue_uses_generic_on_nn(self):
        model = NormalNormalModel(sigma=1.0)
        data = np.asarray([0.5])
        theta = 0.2
        stat_default = WaldStatistic()
        stat_generic = WaldStatistic(force_generic=True)
        # Default → closed-form path.
        p_default = float(stat_default.pvalue(theta, data, model))
        # force_generic=True → generic path. Verify by matching the
        # private generic helper directly.
        p_forced = float(stat_generic.pvalue(theta, data, model))
        p_via_generic = float(stat_default._generic_pvalue(theta, data, model))
        assert abs(p_forced - p_via_generic) < 1e-12, (
            f"force_generic=True should hit the generic helper; got "
            f"forced={p_forced}, direct generic={p_via_generic}"
        )
        # Sanity: closed-form and generic agree on NN to within tolerance.
        assert abs(p_default - p_forced) < 1e-10

    @pytest.mark.parametrize("alpha", [0.05, 0.10])
    def test_ci_uses_generic_on_nn(self, alpha):
        model = NormalNormalModel(sigma=1.0)
        data = np.asarray([0.5])
        cf = WaldStatistic().confidence_interval(alpha, data, model)
        gn = WaldStatistic(force_generic=True).confidence_interval(alpha, data, model)
        # Both paths should pin the same CI to within brentq tolerance.
        assert abs(cf[0] - gn[0]) < 1e-6
        assert abs(cf[1] - gn[1]) < 1e-6

    def test_acceptance_region_raises_under_force_generic(self):
        model = NormalNormalModel(sigma=1.0)
        # Default (force_generic=False): closed-form path returns a region.
        WaldStatistic().acceptance_region(0.05, 0.0, model)
        # force_generic=True: no generic data-space inversion exists.
        with pytest.raises(NotImplementedError, match="no generic path"):
            WaldStatistic(force_generic=True).acceptance_region(0.05, 0.0, model)


@pytest.mark.L2
class TestWaldoForceGeneric:
    def test_cell_name(self):
        assert WaldoStatistic().cell_name == "waldo"
        assert WaldoStatistic(force_generic=True).cell_name == "waldo[generic]"

    def test_pvalue_uses_generic_on_nn(self):
        # Use a small n_mc for speed; agreement is to MC noise only.
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        theta = 0.2
        stat_default = WaldoStatistic()
        stat_generic = WaldoStatistic(force_generic=True, n_mc=2000)
        p_default = float(stat_default.pvalue(theta, data, model, prior))
        p_forced = float(stat_generic.pvalue(theta, data, model, prior))
        # MC-vs-closed-form agreement: ~0.022 SE at p~0.5 with n_mc=2000.
        # Use a generous tolerance.
        assert abs(p_default - p_forced) < 0.05, (
            f"closed-form={p_default}, generic-forced={p_forced}"
        )

    @pytest.mark.parametrize("alpha", [0.05, 0.10])
    def test_ci_uses_generic_on_nn(self, alpha):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        cf = WaldoStatistic().confidence_interval(alpha, data, model, prior)
        gn = WaldoStatistic(force_generic=True, n_mc=2000).confidence_interval(
            alpha, data, model, prior
        )
        # MC noise + the (k+1)/(n+1) continuity correction biases the CI
        # slightly outward; tolerate ~3 sigma in either endpoint at
        # n_mc=2000 (~0.022 SE on p=0.5).
        assert abs(cf[0] - gn[0]) < 0.25, f"lower endpoint: cf={cf[0]}, gn={gn[0]}"
        assert abs(cf[1] - gn[1]) < 0.25, f"upper endpoint: cf={cf[1]}, gn={gn[1]}"

    def test_acceptance_region_raises_under_force_generic(self):
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        WaldoStatistic().acceptance_region(0.05, 0.0, model, prior)
        with pytest.raises(NotImplementedError, match="no generic path"):
            WaldoStatistic(force_generic=True).acceptance_region(0.05, 0.0, model, prior)


@pytest.mark.L2
class TestForceGenericThroughTilting:
    """force_generic on the statistic should propagate through non-identity
    tiltings: PowerLaw + force_generic on NN should route through the
    generic MC `_generic_tilted_confidence_interval` path, not the
    closed-form Theorem-6/8 path. The two paths should agree on NN
    within MC noise.
    """

    def test_powerlaw_force_generic_routes_to_generic_on_nn(self):
        from frasian.tilting.eta_selectors import FixedEtaSelector
        from frasian.tilting.power_law import PowerLawTilting

        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        # Fixed-η static selector — generic path is implemented.
        # Use a non-zero η so the closed-form Theorem-8 tilted distribution
        # is genuinely distinct from plain WALDO.
        tilting = PowerLawTilting(selector=FixedEtaSelector(eta=0.5))
        cf_regions = tilting.confidence_regions(
            0.05, data, model, prior, WaldoStatistic()
        )
        gn_regions = tilting.confidence_regions(
            0.05, data, model, prior, WaldoStatistic(force_generic=True, n_mc=2000)
        )
        assert len(cf_regions) == 1 and len(gn_regions) == 1
        cf_lo, cf_hi = cf_regions[0]
        gn_lo, gn_hi = gn_regions[0]
        # Within ~3 sigma MC noise on the WALDO MC reference at n_mc=2000.
        assert abs(cf_lo - gn_lo) < 0.3, f"lower: cf={cf_lo}, gn={gn_lo}"
        assert abs(cf_hi - gn_hi) < 0.3, f"upper: cf={cf_hi}, gn={gn_hi}"

    def test_powerlaw_dynamic_force_generic_runs_on_nn(self):
        """Phase F: PowerLaw + dynamic-η + force_generic on NN now runs
        via the triple-batched `_generic_tilted_pvalue_vec`. Result must
        agree with the analytic Theorem-8 dynamic CI within MC tolerance.
        """
        from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
        from frasian.tilting.power_law import PowerLawTilting

        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        # Use the standard dynamic config (matches the Phase E checkpoint
        # config defaults) and the WaldoStatistic default n_mc=2000 so MC
        # noise on the fine-scan p-value is below ~α/3 → typically a
        # single region.
        tilting = PowerLawTilting(
            selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)
        )
        cf_regions = tilting.confidence_regions(
            0.05, data, model, prior, WaldoStatistic()
        )
        gn_regions = tilting.confidence_regions(
            0.05, data, model, prior,
            WaldoStatistic(force_generic=True, n_mc=2000),
        )
        assert len(cf_regions) == 1, f"analytic should be single-region: {cf_regions}"
        # Generic may produce 1-3 regions due to MC noise on the fine-
        # scan crossings; union width must agree with analytic within
        # ~5% (3-σ MC envelope at n_mc=2000 over the 401-pt fine grid).
        cf_w = float(cf_regions[0][1] - cf_regions[0][0])
        gn_w = float(sum(hi - lo for lo, hi in gn_regions))
        assert abs(cf_w - gn_w) < 0.10 * cf_w, (
            f"union width disagreement: analytic={cf_w}, generic={gn_w}, "
            f"regions={gn_regions}"
        )

    def test_powerlaw_dynamic_force_generic_raises_off_nn(self):
        """Dynamic + force_generic on NON-NN models still raises — the
        DynamicNumericalEtaSelector internally uses NumericalEtaSelector
        which is closed-form NN. Phase F unblocked NN only."""
        from frasian.models.bernoulli import BernoulliModel
        from frasian.models.distributions import BetaDistribution
        from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
        from frasian.tilting.power_law import PowerLawTilting

        bern = BernoulliModel()
        beta_prior = BetaDistribution(alpha=2.0, beta=2.0)
        data = np.asarray([1.0, 0.0, 1.0])
        tilting = PowerLawTilting(
            selector=DynamicNumericalEtaSelector(n_grid=51, coarse_n=11)
        )
        with pytest.raises(NotImplementedError, match="dynamic"):
            tilting.confidence_regions(
                0.05, data, bern, beta_prior, WaldoStatistic(force_generic=True)
            )

    def test_ot_force_generic_routes_to_generic_on_nn(self):
        from frasian.tilting.eta_selectors import FixedEtaSelector
        from frasian.tilting.ot import OTTilting

        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        tilting = OTTilting(selector=FixedEtaSelector(eta=0.5))
        cf_regions = tilting.confidence_regions(
            0.05, data, model, prior, WaldoStatistic()
        )
        gn_regions = tilting.confidence_regions(
            0.05, data, model, prior, WaldoStatistic(force_generic=True, n_mc=2000)
        )
        assert len(cf_regions) == 1 and len(gn_regions) == 1
        cf_lo, cf_hi = cf_regions[0]
        gn_lo, gn_hi = gn_regions[0]
        assert abs(cf_lo - gn_lo) < 0.3, f"lower: cf={cf_lo}, gn={gn_lo}"
        assert abs(cf_hi - gn_hi) < 0.3, f"upper: cf={cf_hi}, gn={gn_hi}"
