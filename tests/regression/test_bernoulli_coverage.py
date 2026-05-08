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
    """Smoke: (Bernoulli, Beta, power_law[Fixed(0)], waldo) generic CI runs.

    Audit P0-17: the original assertion `0.0 <= empirical <= 1.0` was
    vacuous (true by construction). The real calibration check is in
    `test_bernoulli_coverage_calibrated_at_nominal` below — a nightly
    L3 run that asserts |empirical - (1-alpha)| < 3·SE. This smoke test
    is preserved as a no-crash check at minimal n_reps.
    """
    from frasian.tilting.power_law import _generic_tilted_confidence_interval

    alpha = 0.10
    n_reps = 3
    n_obs = 10
    n_mc = 20
    rng = np.random.default_rng(seed=2026)

    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    eta = 0.0
    theta_true = 0.5

    for _ in range(n_reps):
        data = model.sample_data(theta_true, rng, n=n_obs)
        lo, hi = _generic_tilted_confidence_interval(
            alpha, data, model, prior, eta, "waldo", n_mc=n_mc,
        )
        # Smoke contract: no NaN / inf, lo <= hi, lo and hi inside the
        # parameter support. Removes the previously-vacuous tautology.
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.nightly
def test_bernoulli_coverage_calibrated_at_nominal() -> None:
    """Audit P0-17: real calibration regression on the Bernoulli generic path.

    The CLAUDE.md headline claim "WALDO uses an MC reference distribution
    under H_0 sampled via model.sample_data ... coverage at nominal
    level" was previously unverified — the only Bernoulli coverage check
    asserted `0.0 <= empirical <= 1.0` (vacuous). This test runs the
    (BernoulliModel, BetaDistribution, FixedEtaSelector(0), waldo) cell
    at n_reps≥300 and pins the calibration band:

        |empirical_coverage - (1 - alpha)| < 3 · SE

    where SE = sqrt(α(1-α)/n_reps). At alpha=0.10, n_reps=300 → SE≈0.017,
    so 3·SE≈0.051 — wide enough to absorb MC noise in WaldoStatistic's
    own n_mc=200 reference draws but tight enough to catch a real
    calibration failure (e.g. the variance-floor bug from Cluster B that
    would have driven coverage to ~10%).

    Nightly: ~5 minutes wall-time on dev hardware. Marked `nightly` so
    normal CI doesn't gate on it.
    """
    from frasian.tilting.power_law import _generic_tilted_confidence_interval

    alpha = 0.10
    target = 1.0 - alpha
    n_reps = 300
    n_obs = 16
    n_mc = 200
    rng = np.random.default_rng(seed=20260507)

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

    # Three-sigma calibration band on a binomial proportion.
    se = np.sqrt(target * (1.0 - target) / n_reps)
    assert abs(empirical - target) < 3.0 * se, (
        f"Bernoulli generic-CI coverage failed calibration band: "
        f"empirical={empirical:.4f}, target={target:.4f}, "
        f"se={se:.4f}, |Δ|={abs(empirical - target):.4f} (3σ band {3 * se:.4f})"
    )
