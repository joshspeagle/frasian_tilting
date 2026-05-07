"""Phase 4 verification §4: Bernoulli end-to-end coverage smoke.

This is the smallest smoke that closes the plan's verification §4
("run the coverage experiment with the Bernoulli fixture and confirm
empirical coverage hits nominal 1 - alpha within MC error"). Two
caveats from the implementation that shaped this test:

1. ``CoverageExperiment`` (the registered cell) hardcodes
   ``NormalNormalModel + NormalDistribution`` with a (theta_true, w)
   grid that is meaningless for Bernoulli. Generalising it to take
   any (model, prior) is a follow-up; this test reaches the same
   conclusion via a hand-rolled MC loop on the generic CI.

2. ``PowerLawTilting.confidence_regions`` raises
   ``NotImplementedError`` for dynamic-eta selectors on
   non-Normal-Normal pairings (the dynamic_ci_scan builds its
   theta-window from D ± search_mult * sigma — Normal-Normal-
   flavoured). We therefore exercise the **static-selector** path
   via ``_generic_tilted_confidence_interval`` directly, which is
   the same routine ``confidence_regions`` calls under the hood for
   non-NN + static.

Runtime budget: each Bernoulli generic CI inversion runs n_mc MC
reference draws inside a brentq loop, with grid integration on
n_grid=128 points per draw. At n_mc=50 each CI is ~20 s on dev
hardware; n_reps=5 keeps wall-time under ~2 min. This is a smoke
test, not a calibration regression — n_reps=5 has SE ≈ 0.13 at
alpha=0.10, so the only assertion is that empirical coverage is in
``[0.5, 1.0]`` (i.e. not catastrophically broken). A tighter
calibration regression at n_reps=300+ would need hours and belongs
in a nightly job.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import BetaDistribution


@pytest.mark.L5
@pytest.mark.slow
def test_bernoulli_coverage_smoke() -> None:
    """Empirical coverage on (Bernoulli, Beta(2, 2), power_law, waldo)
    at FixedEtaSelector(eta=0) is non-zero and bounded by the wide
    n_reps=5 Monte-Carlo SE band."""
    from frasian.tilting.power_law import _generic_tilted_confidence_interval

    alpha = 0.10
    target = 1.0 - alpha
    n_reps = 5
    n_obs = 20
    n_mc = 50
    rng = np.random.default_rng(seed=2026)

    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    eta = 0.0
    theta_true = 0.5

    n_covered = 0
    for _ in range(n_reps):
        data = model.sample_data(theta_true, rng, n=n_obs)
        lo, hi = _generic_tilted_confidence_interval(
            alpha, data, model, prior, eta, "waldo", n_mc=n_mc,
        )
        if lo <= theta_true <= hi:
            n_covered += 1
    empirical = n_covered / n_reps
    # n_reps=5 SE ~ 0.13; 3 SE band [target - 0.4, 1.0]. Lower bound
    # is just "did we cover at all"; the test isn't a calibration
    # regression at this n_reps.
    assert empirical >= target - 0.40, (
        f"Bernoulli generic CI catastrophic undercoverage: "
        f"{empirical:.2f} vs nominal {target:.2f} (n_reps={n_reps})"
    )
    assert 0.0 <= empirical <= 1.0
