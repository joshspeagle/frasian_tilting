"""Phase 3e: Bernoulli end-to-end smoke through power_law and ot.

The marquee Phase 3 deliverable: PowerLaw and OT tilting cells
work end-to-end on (BernoulliModel, BetaDistribution) — `tilt()`,
`tilted_pvalue`, `confidence_regions`, `confidence_interval`, and
`pvalue` all run without raising and produce sensible outputs.

This is an **L4** test (end-to-end, slow MC) because it exercises
the full generic-path stack: `tilt()` construction, generic tilted-
pvalue via MC reference, brentq CI inversion, and explicit boundary
detection. We use ``confidence_regions`` and ``pvalue`` directly
(default n_mc=200 internally — set in the scheme). Wall time ~30 s
per cell on dev hardware.

Combined with the L0 smoke tests in
`test_power_law_generic_tilt.py` / `test_power_law_generic_pvalue_ci.py`
/ `test_ot_generic_tilt.py`, this test pins that nothing was missed
at the integration level: a user can construct a `(BernoulliModel,
BetaDistribution)` pair with a power_law or ot scheme + WALDO
statistic + FixedEtaSelector and get a working CI.

Phase 4 wrap: only the (PowerLawTilting, eta=0.0) cell is exercised
at the full public API surface; the other cells call into the same
``_generic_tilted_*`` helpers and are pinned by their own L0/L2
tests. Trimming the parametrize space cuts L4 wall-time ~4x without
losing integration-level coverage of the public API.
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
@pytest.mark.slow
def test_full_public_api_bernoulli_powerlaw():
    """Full public API on (BernoulliModel, BetaDistribution, power_law,
    waldo) at FixedEtaSelector(eta=0). One canonical cell at the L4
    integration layer; the remaining (scheme × eta) combinations are
    pinned individually by L0/L2 tests in
    ``test_power_law_generic_pvalue_ci.py`` and ``test_ot_generic_tilt.py``.

    For the chosen cell, exercise:
    - `confidence_regions(alpha, data, model, prior, statistic)`
    - `pvalue(theta, data, model, prior, statistic)` at multiple θ

    Pin: both return finite values on [0, 1]; CI contains the MLE;
    pvalue is in (0, 1].

    Marked ``@slow`` (wall ~80 s): the public API for non-Normal pairs
    routes through the generic MC path with default ``n_mc=200`` —
    not overridable from the surface this test exercises. The cheaper
    L0 paths in ``test_power_law_generic_pvalue_ci.py`` already pin
    each generic helper at ``n_mc=50``; this test's marginal value is
    confirming the public-API plumbing on top, which is a full-tier
    concern, not a per-PR concern.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    statistic = WaldoStatistic()
    scheme = PowerLawTilting(selector=FixedEtaSelector(eta=0.0))
    # 4 successes / 6 trials; MLE = 0.667.
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    alpha = 0.10

    regions = scheme.confidence_regions(alpha, data, model, prior, statistic)
    assert len(regions) == 1
    lo, hi = regions[0]
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= 4.0 / 6.0 <= hi, (
        f"power_law eta=0.0: MLE 0.667 not in CI ({lo:.3f}, {hi:.3f})"
    )

    theta_arr = np.linspace(0.1, 0.9, 5)
    p = scheme.pvalue(theta_arr, data, model, prior, statistic)
    p_arr = np.asarray(p)
    assert p_arr.shape == theta_arr.shape
    assert np.all(p_arr > 0)
    assert np.all(p_arr <= 1.0)


@pytest.mark.L4
@pytest.mark.slow
def test_full_public_api_bernoulli_ot_smoke():
    """OT scheme integration smoke: only ``confidence_regions`` to
    cover the OT-specific generic dispatch path; other surfaces are
    pinned by dedicated tests. ``@slow`` for the same reason as the
    power_law sibling: default ``n_mc=200`` is wired by the public
    API, not overridable here, and the cheaper helpers in
    ``test_ot_generic_tilt.py`` already pin the math at ``n_mc=50``."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    scheme = OTTilting(selector=FixedEtaSelector(eta=0.0))
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    regions = scheme.confidence_regions(0.10, data, model, prior, WaldoStatistic())
    assert len(regions) == 1
    lo, hi = regions[0]
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= 4.0 / 6.0 <= hi


@pytest.mark.L4
@pytest.mark.slow
def test_bernoulli_extreme_data_returns_support_boundary():
    """All-ones Bernoulli data: CI upper bound IS support_hi (=1.0).

    Phase 3c-fix1's explicit boundary detection should return the
    support endpoint cleanly without silently falling through brentq's
    bracket-doubling exhaust + except. Tested on power_law only;
    ot follows the same brentq path.

    ``@slow`` (~30 s): boundary-detection bug regression. The cheaper
    L0 dispatch tests cover the non-boundary path; this case forces
    the brentq-exhaustion branch and is full-tier territory.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    scheme = PowerLawTilting(selector=FixedEtaSelector(eta=0.0))
    data = np.ones(10, dtype=np.float64)  # all successes

    regions = scheme.confidence_regions(0.10, data, model, prior, WaldoStatistic())
    lo, hi = regions[0]
    assert hi == 1.0, (
        f"power_law: expected upper bound at support 1.0, got {hi:.6f}"
    )
    assert 0.0 <= lo < 1.0
