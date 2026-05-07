"""Phase 3c regression tests for PowerLawTilting generic tilted-pvalue + CI.

Pin three contracts:

1. **Cross-path agreement on Normal-Normal** (L3, MC tolerance): the
   generic MC path's CI matches the closed-form CI within ~3 * SE.
2. **Bernoulli smoke** at multiple levels:
   - L0: dispatch correctness — non-Normal pair routes through the
     generic path, returns sensible (lo, hi) on [0, 1] (uses
     `n_mc=50` via direct `_generic_tilted_confidence_interval` call
     to keep wall time < 5 s).
   - L3: full public-API end-to-end (`confidence_regions` /
     `confidence_interval`) at default `n_mc=200`. Slower (~30-60 s).
3. **Dynamic selector raises** (L0): the dynamic CI scanner remains
   Normal-Normal-only in Phase 3c.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import BetaDistribution, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import (
    DynamicNumericalEtaSelector,
    FixedEtaSelector,
)
from frasian.tilting.power_law import (
    PowerLawTilting,
    _generic_tilted_confidence_interval,
    _generic_tilted_pvalue,
)


@pytest.mark.L0
def test_dynamic_selector_raises_on_bernoulli():
    """Dynamic-η selector path remains Normal-Normal-only in Phase 3c.

    The dynamic CI scanner builds its θ-window from `D ± search_mult * sigma`,
    which is Normal-Normal-flavoured. Using a dynamic selector with
    Bernoulli must raise a clear error.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    scheme = PowerLawTilting(selector=DynamicNumericalEtaSelector())
    statistic = WaldoStatistic()
    data = np.asarray([1.0, 0.0, 1.0])

    with pytest.raises(NotImplementedError, match="dynamic"):
        scheme.confidence_regions(0.10, data, model, prior, statistic)


@pytest.mark.L0
def test_generic_tilted_pvalue_bernoulli_smoke_low_nmc():
    """`_generic_tilted_pvalue` runs end-to-end on Bernoulli at n_mc=50.

    Direct internal call — keeps wall time low. Asserts pvalue is
    finite + in (0, 1] (strict > 0 thanks to +1 smoothing).
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0])

    p = _generic_tilted_pvalue(
        theta=0.5, data=data, model=model, prior=prior, eta=0.0,
        statistic_name="waldo", n_mc=50,
    )
    assert np.isfinite(p)
    assert 0.0 < p <= 1.0


@pytest.mark.L0
def test_generic_tilted_ci_bernoulli_smoke_low_nmc():
    """`_generic_tilted_confidence_interval` runs end-to-end at n_mc=50."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    lo, hi = _generic_tilted_confidence_interval(
        alpha=0.10, data=data, model=model, prior=prior, eta=0.0,
        statistic_name="waldo", n_mc=50,
    )
    assert np.isfinite(lo) and np.isfinite(hi)
    assert 0.0 <= lo < hi <= 1.0
    # MLE = 4/6 ≈ 0.667; CI should contain it.
    assert lo <= 4.0 / 6.0 <= hi


@pytest.mark.L0
def test_generic_tilted_pvalue_reproducible_within_process():
    """Two calls with identical inputs return bit-identical pvalues
    (CRN-seeded; same root cause as the Phase 2 WALDO blake2b fix)."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0])
    p1 = _generic_tilted_pvalue(
        theta=0.4, data=data, model=model, prior=prior, eta=0.2,
        statistic_name="waldo", n_mc=30,
    )
    p2 = _generic_tilted_pvalue(
        theta=0.4, data=data, model=model, prior=prior, eta=0.2,
        statistic_name="waldo", n_mc=30,
    )
    assert p1 == p2, f"non-deterministic generic tilted pvalue: {p1} != {p2}"


@pytest.mark.L0
def test_generic_tilted_pvalue_wald_delegates_to_generic_wald():
    """For statistic_name='wald', the generic tilted pvalue is
    eta-independent and delegates to WaldStatistic._generic_pvalue."""
    from frasian.statistics.wald import WaldStatistic
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0])

    p_wald_direct = float(np.asarray(WaldStatistic()._generic_pvalue(0.5, data, model)))
    p_tilted_at_eta_05 = _generic_tilted_pvalue(
        theta=0.5, data=data, model=model, prior=prior, eta=0.5,
        statistic_name="wald", n_mc=10,  # n_mc unused for wald
    )
    p_tilted_at_eta_00 = _generic_tilted_pvalue(
        theta=0.5, data=data, model=model, prior=prior, eta=0.0,
        statistic_name="wald", n_mc=10,
    )
    # Wald is eta-independent: tilted pvalue at any eta == bare Wald pvalue.
    assert p_tilted_at_eta_05 == p_wald_direct
    assert p_tilted_at_eta_00 == p_wald_direct


@pytest.mark.L3
@pytest.mark.parametrize("eta", [0.0, 0.3])
@pytest.mark.parametrize("D", [-0.5, 0.0, 1.0])
def test_generic_ci_matches_closed_form_normal_normal(eta, D):
    """Generic vs closed-form CI on Normal-Normal at static η.

    Generic: MC with n_mc=500 + brentq inversion. Tolerance ~3 *
    sigma_post / sqrt(n_mc) plus a coarse-grid floor.
    """
    sigma, sigma0 = 1.0, 1.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=sigma0)
    data = np.asarray([float(D)])
    alpha = 0.10
    scheme = PowerLawTilting(selector=FixedEtaSelector(eta=eta))
    statistic = WaldoStatistic()

    cf_lo, cf_hi = scheme.tilted_confidence_interval(
        alpha, float(D), model, prior, eta, statistic.name
    )
    gn_lo, gn_hi = _generic_tilted_confidence_interval(
        alpha, data, model, prior, eta, statistic.name, n_mc=500,
    )

    sigma_post = float(np.sqrt(model.posterior(data, prior).var()))
    tol = 3.0 * sigma_post / np.sqrt(500) + 0.10
    assert abs(cf_lo - gn_lo) < tol, (
        f"CI lower disagreement: closed={cf_lo:.4f} generic={gn_lo:.4f} (tol={tol:.4f})"
    )
    assert abs(cf_hi - gn_hi) < tol, (
        f"CI upper disagreement: closed={cf_hi:.4f} generic={gn_hi:.4f} (tol={tol:.4f})"
    )


@pytest.mark.L3
def test_confidence_regions_bernoulli_full_smoke():
    """Full public-API end-to-end on (BernoulliModel, BetaDistribution).

    Slower (~30-60 s) — uses default n_mc=200. Pins that the static-
    selector dispatch in `confidence_regions` works without bypass.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    scheme = PowerLawTilting(selector=FixedEtaSelector(eta=0.3))
    statistic = WaldoStatistic()
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])  # 4 / 6

    regions = scheme.confidence_regions(0.10, data, model, prior, statistic)
    assert len(regions) == 1
    lo, hi = regions[0]
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= 4.0 / 6.0 <= hi
