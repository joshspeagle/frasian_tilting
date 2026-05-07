"""Phase 3e: Bernoulli end-to-end smoke through power_law and ot.

The marquee Phase 3 deliverable: PowerLaw and OT tilting cells
work end-to-end on (BernoulliModel, BetaDistribution) — `tilt()`,
`tilted_pvalue`, `confidence_regions`, `confidence_interval`, and
`pvalue` all run without raising and produce sensible outputs.

This is an **L4** test (end-to-end, slow MC) because it exercises
the full generic-path stack with `n_mc=200` defaults: `tilt()`
construction, generic tilted-pvalue via MC reference, brentq CI
inversion, and explicit boundary detection. Wall time ~30 s per
scheme on dev hardware.

Combined with the L0 smoke tests in
`test_power_law_generic_tilt.py` / `test_power_law_generic_pvalue_ci.py`
/ `test_ot_generic_tilt.py`, this test pins that nothing was missed
at the integration level: a user can construct a `(BernoulliModel,
BetaDistribution)` pair with a power_law or ot scheme + WALDO
statistic + FixedEtaSelector and get a working CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import BetaDistribution
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import FixedEtaSelector
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L4
@pytest.mark.parametrize("scheme_cls", [PowerLawTilting, OTTilting])
@pytest.mark.parametrize("eta", [0.0, 0.3])
def test_full_public_api_bernoulli(scheme_cls, eta):
    """Full public API on (BernoulliModel, BetaDistribution).

    For each (scheme_cls, eta), exercise:
    - `confidence_regions(alpha, data, model, prior, statistic)`
    - `confidence_interval(alpha, data, model, prior, statistic)`
    - `pvalue(theta, data, model, prior, statistic)` at multiple θ

    Pin: all three return finite values on [0, 1]; CI contains the
    MLE; pvalue is in (0, 1].

    Pinned at default n_mc=200 — slow but realistic. The L0/L3 layer
    tests exercise the same surface at lower n_mc for fast-path
    coverage.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    statistic = WaldoStatistic()
    scheme = scheme_cls(selector=FixedEtaSelector(eta=eta))
    # 4 successes / 6 trials; MLE = 0.667.
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    alpha = 0.10

    regions = scheme.confidence_regions(alpha, data, model, prior, statistic)
    assert len(regions) == 1
    lo, hi = regions[0]
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= 4.0 / 6.0 <= hi, (
        f"{scheme_cls.__name__} eta={eta}: MLE 0.667 not in CI ({lo:.3f}, {hi:.3f})"
    )

    ci_lo, ci_hi = scheme.confidence_interval(alpha, data, model, prior, statistic)
    assert (ci_lo, ci_hi) == (lo, hi)  # Single-region case: convex hull == region.

    theta_arr = np.linspace(0.1, 0.9, 5)
    p = scheme.pvalue(theta_arr, data, model, prior, statistic)
    p_arr = np.asarray(p)
    assert p_arr.shape == theta_arr.shape
    assert np.all(p_arr > 0)
    assert np.all(p_arr <= 1.0)


@pytest.mark.L4
@pytest.mark.parametrize("scheme_cls", [PowerLawTilting, OTTilting])
def test_bernoulli_extreme_data_returns_support_boundary(scheme_cls):
    """All-ones Bernoulli data: CI upper bound IS support_hi (=1.0).

    Phase 3c-fix1's explicit boundary detection should return the
    support endpoint cleanly without silently falling through brentq's
    bracket-doubling exhaust + except.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    scheme = scheme_cls(selector=FixedEtaSelector(eta=0.0))
    data = np.ones(10, dtype=np.float64)  # all successes

    regions = scheme.confidence_regions(0.10, data, model, prior, WaldoStatistic())
    lo, hi = regions[0]
    assert hi == 1.0, (
        f"{scheme_cls.__name__}: expected upper bound at support 1.0, got {hi:.6f}"
    )
    assert 0.0 <= lo < 1.0
