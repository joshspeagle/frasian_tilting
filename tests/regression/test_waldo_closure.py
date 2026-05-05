"""Regression: WaldoStatistic.acceptance_region closure binding.

Pins the B023 fix in `src/frasian/statistics/waldo.py:94-96`. The
inner `f(D_val, _theta=theta_val)` closure must capture the
*per-iteration* `theta_val` via default-arg binding — NOT the
last-iteration value via late binding. A regression that drops the
default-arg binding back to a bare closure (`def f(D_val)` referencing
free `theta_val`) would have every iteration solve at the *last*
`theta_val`, producing identical `D_lo` / `D_hi` for every i.

Two assertions catch this:
  - `D_lo[i] < theta_arr[i] < D_hi[i]` for each i (centring is per-θ).
  - `D_hi[0]` and `D_hi[2]` differ by more than 5.0 — proving the
    closure binding is per-iteration, not last-iteration sticky.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic


@pytest.mark.L2
class TestWaldoAcceptanceRegionClosure:
    def test_acceptance_region_centres_per_theta(self):
        """Per-θ binding: each (D_lo[i], D_hi[i]) must contain theta_arr[i]."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        alpha = 0.05
        theta_arr = np.array([1.0, 5.0, 10.0], dtype=np.float64)

        D_lo, D_hi = WaldoStatistic().acceptance_region(
            alpha, theta_arr, model, prior,
        )
        D_lo = np.atleast_1d(np.asarray(D_lo, dtype=np.float64))
        D_hi = np.atleast_1d(np.asarray(D_hi, dtype=np.float64))

        # Per-θ centring: D_lo[i] < theta_arr[i] < D_hi[i] for each i.
        # Late-binding regression would put theta_arr[-1]=10.0 inside
        # *every* (D_lo[i], D_hi[i]), and theta_arr[0]=1.0 outside the
        # interval centred on 10.0 — caught here.
        for i, theta_i in enumerate(theta_arr):
            assert D_lo[i] < theta_i < D_hi[i], (
                f"D_lo[{i}]={D_lo[i]} < theta={theta_i} < "
                f"D_hi[{i}]={D_hi[i]} failed"
            )

    def test_acceptance_region_strictly_monotone_in_theta(self):
        """As theta sweeps right, the acceptance region must shift right too."""
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        alpha = 0.05
        theta_arr = np.array([1.0, 5.0, 10.0], dtype=np.float64)

        D_lo, D_hi = WaldoStatistic().acceptance_region(
            alpha, theta_arr, model, prior,
        )
        D_lo = np.atleast_1d(np.asarray(D_lo, dtype=np.float64))
        D_hi = np.atleast_1d(np.asarray(D_hi, dtype=np.float64))

        # Monotone: theta sweeping right => D_lo, D_hi both sweep right.
        assert D_lo[0] < D_lo[1] < D_lo[2], (
            f"D_lo not strictly increasing: {D_lo!r}"
        )
        assert D_hi[0] < D_hi[1] < D_hi[2], (
            f"D_hi not strictly increasing: {D_hi!r}"
        )

    def test_acceptance_region_nontrivial_gap_across_theta(self):
        """The gap between first/last D_hi must be > 5.0.

        With theta_arr=[1.0, 5.0, 10.0] and σ=1, late-binding would
        return |D_hi[0] - D_hi[2]| ≈ 0 (all centred on 10.0). The
        per-iteration binding produces |D_hi[0] - D_hi[2]| ≈ 9 (each
        centred on its own θ).
        """
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        alpha = 0.05
        theta_arr = np.array([1.0, 5.0, 10.0], dtype=np.float64)

        D_lo, D_hi = WaldoStatistic().acceptance_region(
            alpha, theta_arr, model, prior,
        )
        D_lo = np.atleast_1d(np.asarray(D_lo, dtype=np.float64))
        D_hi = np.atleast_1d(np.asarray(D_hi, dtype=np.float64))

        gap_hi = abs(float(D_hi[2]) - float(D_hi[0]))
        gap_lo = abs(float(D_lo[2]) - float(D_lo[0]))
        assert gap_hi > 5.0, (
            f"|D_hi[2] - D_hi[0]| = {gap_hi} <= 5.0; closure may be "
            f"binding last-iteration theta_val (B023 regression)."
        )
        assert gap_lo > 5.0, (
            f"|D_lo[2] - D_lo[0]| = {gap_lo} <= 5.0; closure may be "
            f"binding last-iteration theta_val (B023 regression)."
        )
