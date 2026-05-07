"""Smoke test for ``scripts.regen_headline._compute_table``.

The Phase 5 regression (skeptic vector #1) was that the script's
``identity.confidence_interval(...)`` calls were missing the ``prior``
positional argument, so every Wald / bare-WALDO rep raised
``TypeError`` and was silently swallowed by ``except Exception: pass``.
The headline regen emitted NaN for two of four rows without any error.

This smoke test runs ``_compute_table`` (or, when torch is unavailable,
the equivalent bare-statistic path) at ``n_reps=2`` and asserts that
the Wald / bare WALDO rows are finite. A regression in
``identity.confidence_interval`` signature would crash every rep at
both rows; the per-cell failure-counter also added in this fix would
then ``raise RuntimeError`` rather than silently emit NaN, but the
test belt-and-braces verifies the no-NaN contract on the happy path.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.L4
def test_compute_table_smoke_no_nan() -> None:
    """End-to-end: run the script's inner table builder at n_reps=2 and
    assert no row has NaN. Catches signature regressions in the
    ``identity.confidence_interval(alpha, data, model, prior, statistic)``
    call sites and the four ``except Exception`` swallows that masked
    them.
    """
    from scripts.regen_headline import _compute_table

    rows = _compute_table([0.0, 1.0], n_reps=2)
    for label, vals in rows.items():
        assert len(vals) == 2, f"{label}: expected 2 θ values, got {len(vals)}"
        for v in vals:
            assert np.isfinite(v), f"{label}: NaN at one or more θ values"


@pytest.mark.L4
def test_identity_rows_are_finite_no_torch() -> None:
    """Reproduce the script's identity-tilting Wald + bare WALDO rows
    via the bare-statistic path (no learned-eta checkpoint loaded) —
    exercises exactly the call sites the skeptic flagged (vector #1).
    If ``identity.confidence_interval`` ever drifts back to a 4-arg
    signature mismatch, this test crashes.
    """
    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel
    from frasian.statistics.wald import WaldStatistic
    from frasian.statistics.waldo import WaldoStatistic
    from frasian.tilting.identity import IdentityTilting

    sigma = 1.0
    sigma0 = 1.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=sigma0)
    wald = WaldStatistic()
    waldo = WaldoStatistic()
    identity = IdentityTilting()

    alpha = 0.05
    rng = np.random.default_rng(0)
    n_reps = 2
    for theta_true in (0.0, 1.0):
        D_samples = rng.normal(loc=theta_true, scale=sigma, size=n_reps)
        for D in D_samples:
            data = np.array([D])
            # Both calls must succeed with the 5-arg signature; a
            # signature drift back to 4 args would raise TypeError.
            lo_w, hi_w = identity.confidence_interval(alpha, data, model, prior, wald)
            assert np.isfinite(hi_w - lo_w)
            assert lo_w < hi_w
            lo_o, hi_o = identity.confidence_interval(alpha, data, model, prior, waldo)
            assert np.isfinite(hi_o - lo_o)
            assert lo_o < hi_o
