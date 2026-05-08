"""Phase 3d regression tests for OTTilting generic path.

Mirrors `test_power_law_generic_tilt.py` and `test_power_law_generic_pvalue_ci.py`
for OT. Three contracts:

1. **Cross-path agreement on Normal-Normal**: at FixedEtaSelector(eta),
   the generic-path CI on (NormalNormalModel, NormalDistribution)
   matches the closed-form CI within MC tolerance.
2. **Bernoulli end-to-end smoke**: `confidence_regions` on
   (BernoulliModel, BetaDistribution) returns finite (lo, hi) on
   [0, 1] containing the MLE. The new OT generic path uses the
   QuantileMixturePath geodesic between posterior (Beta) and
   likelihood-as-distribution (GridDistribution from log L).
3. **Dynamic-selector + Bernoulli raises** (parity with power_law).
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
    NumericalEtaSelector,
)
from frasian.tilting.ot import (
    OTTilting,
    _generic_tilted_confidence_interval_ot,
    _generic_tilted_pvalue_ot,
    _generic_tilt_ot,
)
from frasian.tilting.quantile_mixture import QuantileMixturePath


@pytest.mark.L0
def test_ot_generic_tilt_runs_on_bernoulli_smoke():
    """`_generic_tilt_ot` returns a QuantileMixturePath on Bernoulli with
    finite moments. The endpoints are the Beta posterior and a
    GridDistribution likelihood-as-distribution; the geodesic at t=eta
    is well-defined.
    """
    from frasian.tilting._generic_pvalue import _resolve_support
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    posterior = model.posterior(data, prior)
    likelihood = model.likelihood(data)
    support = _resolve_support(model, data)

    for eta in [0.0, 0.3, 0.7, 1.0]:
        qmp = _generic_tilt_ot(
            posterior, likelihood, eta,
            model=model, data=data, support=support,
        )
        assert isinstance(qmp, QuantileMixturePath)
        m = qmp.mean()
        v = qmp.var()
        assert 0.0 <= m <= 1.0, f"OT tilted mean {m} outside [0, 1] at eta={eta}"
        assert v > 0.0, f"OT tilted var {v} non-positive at eta={eta}"


@pytest.mark.L0
def test_ot_dynamic_selector_raises_on_bernoulli():
    """Dynamic-η + non-Normal-Normal raises (parity with power_law)."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    scheme = OTTilting(selector=DynamicNumericalEtaSelector())
    data = np.asarray([1.0, 0.0, 1.0])
    with pytest.raises(NotImplementedError, match="dynamic"):
        scheme.confidence_regions(0.10, data, model, prior, WaldoStatistic())


@pytest.mark.L0
def test_ot_numerical_selector_raises_on_bernoulli():
    """NumericalEtaSelector + Bernoulli raises clearly via the
    `_normal_normal_w` isinstance check (skeptic finding #9 confirmed
    not a real bug; pin the behaviour)."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    scheme = OTTilting(selector=NumericalEtaSelector())
    data = np.asarray([1.0, 0.0, 1.0])
    with pytest.raises(NotImplementedError, match="NormalNormalModel"):
        scheme.confidence_regions(0.10, data, model, prior, WaldoStatistic())


@pytest.mark.L0
def test_ot_generic_tilted_pvalue_bernoulli_smoke():
    """`_generic_tilted_pvalue_ot` runs on Bernoulli at low n_mc."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0])

    p = _generic_tilted_pvalue_ot(
        theta=0.5, data=data, model=model, prior=prior, eta=0.3,
        statistic_name="waldo", n_mc=50,
    )
    assert np.isfinite(p)
    assert 0.0 < p <= 1.0


@pytest.mark.L0
def test_ot_generic_tilted_pvalue_reproducible_within_process():
    """Two calls with identical inputs return bit-identical pvalues
    (CRN-seeded; same blake2b stable hash as power_law)."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0])
    p1 = _generic_tilted_pvalue_ot(
        theta=0.4, data=data, model=model, prior=prior, eta=0.2,
        statistic_name="waldo", n_mc=30,
    )
    p2 = _generic_tilted_pvalue_ot(
        theta=0.4, data=data, model=model, prior=prior, eta=0.2,
        statistic_name="waldo", n_mc=30,
    )
    assert p1 == p2


@pytest.mark.L0
def test_ot_generic_tilted_pvalue_wald_delegates():
    """For statistic_name='wald', delegates to WaldStatistic._generic_pvalue
    (eta-independent)."""
    from frasian.statistics.wald import WaldStatistic
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0])

    p_wald = float(np.asarray(WaldStatistic()._generic_pvalue(0.5, data, model)))
    p_ot_at_0 = _generic_tilted_pvalue_ot(
        theta=0.5, data=data, model=model, prior=prior, eta=0.0,
        statistic_name="wald", n_mc=10,
    )
    p_ot_at_5 = _generic_tilted_pvalue_ot(
        theta=0.5, data=data, model=model, prior=prior, eta=0.5,
        statistic_name="wald", n_mc=10,
    )
    assert p_ot_at_0 == p_wald
    assert p_ot_at_5 == p_wald


@pytest.mark.L0
def test_ot_generic_tilted_ci_bernoulli_smoke_low_nmc():
    """`_generic_tilted_confidence_interval_ot` runs end-to-end at n_mc=50."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    lo, hi = _generic_tilted_confidence_interval_ot(
        alpha=0.10, data=data, model=model, prior=prior, eta=0.0,
        statistic_name="waldo", n_mc=50,
    )
    assert np.isfinite(lo) and np.isfinite(hi)
    assert 0.0 <= lo < hi <= 1.0
    assert lo <= 4.0 / 6.0 <= hi


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("D", [-0.5, 1.0])
def test_ot_generic_ci_matches_closed_form_normal_normal(eta, D):
    """Generic vs closed-form CI on Normal-Normal at static η for OT.

    The Gaussian fast path in `tilt()` returns a NormalDistribution;
    the generic path (when forced) uses QuantileMixturePath. Both should
    agree within MC tolerance.

    Trimmed parametrize grid (η ∈ {0, 0.3, 1.0} × D ∈ {-0.5, 1.0}):
    spans geodesic endpoints + an interior point on η, and one conflict
    + one non-conflict draw on D — covers the same regression surface
    as the original 4×3 grid.

    Marked ``@slow`` (~2 min wall): cross-path agreement check is a
    full-tier concern, not per-PR. The L0 ``_generic_tilt_ot`` smoke
    pins the construction at n_mc=50 on every PR.
    """
    sigma, sigma0 = 1.0, 1.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=sigma0)
    data = np.asarray([float(D)])
    alpha = 0.10
    scheme = OTTilting(selector=FixedEtaSelector(eta=eta))
    statistic = WaldoStatistic()

    n_mc = 500
    cf_lo, cf_hi = scheme.tilted_confidence_interval(
        alpha, float(D), model, prior, eta, statistic.name
    )
    gn_lo, gn_hi = _generic_tilted_confidence_interval_ot(
        alpha, data, model, prior, eta, statistic.name, n_mc=n_mc,
    )

    sigma_post = float(np.sqrt(model.posterior(data, prior).var()))
    tol = 3.0 * sigma_post / np.sqrt(n_mc) + 0.10
    assert abs(cf_lo - gn_lo) < tol, (
        f"OT CI lower disagreement at D={D}, eta={eta}: "
        f"closed={cf_lo:.4f} generic={gn_lo:.4f} (tol={tol:.4f})"
    )
    assert abs(cf_hi - gn_hi) < tol, (
        f"OT CI upper disagreement at D={D}, eta={eta}: "
        f"closed={cf_hi:.4f} generic={gn_hi:.4f} (tol={tol:.4f})"
    )
