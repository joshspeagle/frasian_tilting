"""Pinned regression: `NumericalEtaSelector` undercovers at α=0.05.

This is the post-selection inference failure that motivated making
`DynamicNumericalEtaSelector` the framework's calibrated default. The
static η*-opt selector minimizes CI width per D, yielding CIs that are
strictly narrower than WALDO and that approach Wald at large |Δ| — but
the procedure is post-selection (η chosen as a function of D, then the
narrow CI reported), so empirical coverage falls below the nominal
1−α. Empirical observation at w=0.5 with `n_reps=600`:

    θ_true   |Δ|    WALDO   static η*-opt   dynamic-η
    -2.0     1.0    0.95    0.93            0.95
    -1.0     0.5    0.95    0.93            0.95

This test pins the inequality `coverage(static) < coverage(dynamic)`
on a tiny grid; if a future change to the optimizer or selector
accidentally restores nominal coverage, that's diagnostic of either a
real fix (great — surface it) or a mis-implementation (concerning —
investigate).
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector, NumericalEtaSelector
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L3
@pytest.mark.slow
class TestPostSelectionCoverage:
    """Pin the coverage gap between the calibrated dynamic selector and
    the post-selection static-η*-opt selector."""

    def test_static_optimum_undercovers_dynamic_does_not(self):
        sigma, mu0, w, alpha = 1.0, 0.0, 0.5, 0.05
        sigma0 = float(np.sqrt(w / (1 - w)) * sigma)  # = 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        waldo = WaldoStatistic()

        ident = IdentityTilting()
        static_opt = PowerLawTilting(
            selector=NumericalEtaSelector(),
        )
        dyn = PowerLawTilting(
            selector=DynamicNumericalEtaSelector(
                sigma=sigma,
                mu0=mu0,
                n_grid=201,
                coarse_n=11,
            ),
        )

        # Pick a θ_true with |Δ|=1 — comfortably in the conflict band where
        # the post-selection effect is visible without huge n_reps.
        theta_true = -2.0
        n_reps = 600
        rng = np.random.default_rng(20260501)
        Ds = rng.normal(theta_true, sigma, size=n_reps)

        hits = {"waldo": 0, "static": 0, "dyn": 0}
        for D in Ds:
            for key, sch in (("waldo", ident), ("static", static_opt), ("dyn", dyn)):
                lo, hi = sch.confidence_interval(
                    alpha,
                    np.asarray([D]),
                    model,
                    prior,
                    waldo,
                )
                if lo <= theta_true <= hi:
                    hits[key] += 1
        cov = {k: v / n_reps for k, v in hits.items()}

        # WALDO and dynamic-η: both calibrated; expect within MC noise of 0.95.
        # MC SE ~ sqrt(0.05 * 0.95 / 600) ≈ 0.009; allow 3σ.
        assert abs(cov["waldo"] - 0.95) < 0.027, cov
        assert abs(cov["dyn"] - 0.95) < 0.027, cov

        # Static η*-opt: empirically ~0.93 at θ=−2; pin a one-sided floor
        # well below nominal so the test asserts the *direction* of the
        # post-selection effect rather than a fragile point estimate.
        assert cov["static"] < 0.945, (
            f"static η*-opt selector unexpectedly hit nominal coverage "
            f"{cov['static']:.3f}; either MC sample landed above the mean "
            f"(rerun with a different seed) or the post-selection effect "
            f"has been corrected — investigate. Other cells: {cov}"
        )
        # And the gap should run the right way:
        assert cov["static"] < cov["dyn"], cov
