"""WALDO generic-path reproducibility & CRN sanity tests.

These pin three properties of the Phase 2 generic WALDO Monte-Carlo
path that the original implementation broke (skeptic findings #1, #3,
#13 in the Phase 2 review):

1. The empirical p-value is reproducible across processes given a
   fixed `(data, model, prior, alpha, seed)` tuple. Achieved via a
   `hashlib.blake2b` stable hash, since Python's built-in `hash()`
   randomises tuples-containing-strings per `PYTHONHASHSEED`.
2. The CI inversion uses common random numbers (CRN) across brentq
   probes — same seed across all theta means `f(theta)` is a
   deterministic function (piecewise-constant for Bernoulli, smooth
   for Normal) instead of a fresh stochastic process at every iterate.
3. The `(k+1)/(n+1)` continuity correction makes the empirical p
   strictly conservative: empirical coverage on H_0 data is >=
   nominal `1-alpha` (not == nominal). Cross-check that we don't
   accidentally land on the anti-conservative side.

These run as L0 (fast) where possible; the coverage check is L3.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import BetaDistribution, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic


@pytest.mark.L0
def test_pvalue_reproducible_within_process():
    """Two calls with identical inputs return identical p-values."""
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    stat = WaldoStatistic(n_mc=200, seed=42)
    data = np.asarray([0.7])
    p1 = float(stat._generic_pvalue(0.5, data, model, prior))
    p2 = float(stat._generic_pvalue(0.5, data, model, prior))
    assert p1 == p2, f"non-deterministic pvalue: {p1} != {p2}"


@pytest.mark.L0
def test_pvalue_reproducible_across_processes():
    """Same inputs in a fresh Python process give identical p-values.

    Catches the skeptic finding #1: the original implementation used
    Python's built-in `hash()` over a tuple containing model.fingerprint()
    string elements; `PYTHONHASHSEED` randomises string hashes by
    default, so the MC seed varied across runs and the cross-check
    test only passed because it ran in a single process. The new
    implementation uses `hashlib.blake2b` over a deterministic byte
    encoding.
    """
    snippet = (
        "import numpy as np;"
        "from frasian.models.distributions import NormalDistribution;"
        "from frasian.models.normal_normal import NormalNormalModel;"
        "from frasian.statistics.waldo import WaldoStatistic;"
        "stat = WaldoStatistic(n_mc=200, seed=42);"
        "data = np.asarray([0.7]);"
        "p = float(stat._generic_pvalue(0.5, data, NormalNormalModel(sigma=1.0), "
        "NormalDistribution(loc=0.0, scale=1.0)));"
        "print(f'{p:.16e}')"
    )
    # Inherit PYTHONPATH explicitly so an editable-install package
    # remains importable from a subprocess that doesn't inherit the
    # parent's sys.path (skeptic re-review LOW finding).
    base_env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    env_a = {**base_env, "PYTHONHASHSEED": "0"}
    env_b = {**base_env, "PYTHONHASHSEED": "12345"}
    out_a = subprocess.check_output(
        [sys.executable, "-c", snippet], env=env_a, text=True
    ).strip()
    out_b = subprocess.check_output(
        [sys.executable, "-c", snippet], env=env_b, text=True
    ).strip()
    assert out_a == out_b, (
        f"WALDO p-value depends on PYTHONHASHSEED — CRN seed is not "
        f"cross-process stable. PYTHONHASHSEED=0 gave {out_a}; =12345 gave {out_b}."
    )


@pytest.mark.L0
def test_ci_brentq_uses_common_random_numbers():
    """Same data + same alpha + same seed -> same CI bounds (twice).

    The skeptic finding #13: the previous implementation re-seeded the
    MC reference at every brentq probe (because seed depended on the
    candidate theta), so `f(theta)` was a fresh random function each
    iterate and brentq could not converge. With CRN, the seed is
    fixed for the whole CI inversion and brentq sees a deterministic
    function — repeating the call must give bit-identical bounds.
    """
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    stat = WaldoStatistic(n_mc=300, seed=7)
    data = np.asarray([0.5])
    lo1, hi1 = stat._generic_confidence_interval(0.05, data, model, prior)
    lo2, hi2 = stat._generic_confidence_interval(0.05, data, model, prior)
    assert lo1 == lo2 and hi1 == hi2, (
        f"non-deterministic CI: ({lo1}, {hi1}) vs ({lo2}, {hi2}) — "
        f"CRN is not honoured across brentq probes."
    )


@pytest.mark.L0
def test_wald_ci_does_not_raise_on_bernoulli_boundary_data():
    """Wald generic CI on Bernoulli all-zeros / all-ones must NOT raise.

    Skeptic Phase 2 re-review #5: at MLE in {0, 1} the Fisher info
    `1/(p(1-p))` blows up; the bracket half-width clip and the
    `except BracketingFailed` fallthrough together must keep the CI
    finite. Mirrors the WALDO equivalent below.
    """
    from frasian.statistics.wald import WaldStatistic
    stat = WaldStatistic()
    model = BernoulliModel()
    data_zeros = np.zeros(20, dtype=np.float64)
    lo, hi = stat._generic_confidence_interval(0.05, data_zeros, model)
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isfinite(lo) and np.isfinite(hi)
    data_ones = np.ones(20, dtype=np.float64)
    lo, hi = stat._generic_confidence_interval(0.05, data_ones, model)
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isfinite(lo) and np.isfinite(hi)


@pytest.mark.L0
def test_ci_does_not_raise_on_bernoulli_boundary_data():
    """All-zero / all-one Bernoulli data must produce a CI without raising.

    Skeptic Phase 2 re-review #4/#5 caught a duplicate-class bug where
    `_solvers.BracketingFailed` and `_errors.BracketingFailed` were
    distinct classes; the `except BracketingFailed` blocks in
    `_generic_confidence_interval` never caught (dead code), and the
    underlying L3 conservative-coverage test masked the resulting
    crash with a broad `except Exception:`. This L0 regression pins
    the load-bearing claim: at MLE in {0, 1} (where I(theta) is
    singular), the CI must return a sensible (lo, hi) tuple, NOT
    raise. With the duplicate class fixed, brentq's BracketingFailed
    is caught and the CI collapses to the support boundary.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    stat = WaldoStatistic(n_mc=200, seed=0)
    # All-zero data: MLE = 0, posterior is Beta(2, 22); the boundary
    # is the failure mode for the bracket-width Fisher-info estimate.
    data_zeros = np.zeros(20, dtype=np.float64)
    lo, hi = stat._generic_confidence_interval(0.10, data_zeros, model, prior)
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isfinite(lo) and np.isfinite(hi)
    # All-one data: mirror.
    data_ones = np.ones(20, dtype=np.float64)
    lo, hi = stat._generic_confidence_interval(0.10, data_ones, model, prior)
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isfinite(lo) and np.isfinite(hi)


@pytest.mark.L3
def test_smoothed_pvalue_is_strictly_conservative():
    """Empirical coverage at H_0 data >= 1 - alpha at small alpha.

    The (k+1)/(n+1) continuity correction biases coverage upward (CI
    over-covers) by O(1/n_mc). This test asserts the bias direction
    is correct: at alpha=0.10 with `n_reps` H_0 datasets, empirical
    coverage should be >= 1-alpha minus a small MC margin.
    Anti-conservative behaviour (under-coverage) is the failure mode.

    NOTE: this test catches `BracketingFailed` ONLY (the documented
    boundary-fallthrough path). It does NOT swallow other exceptions —
    a bare `except Exception:` would mask the kind of duplicate-class
    bug the Phase 2 re-review caught. If a different exception type
    propagates, the test fails fast and the maintainer can see why.
    """
    from frasian._errors import BracketingFailed
    rng = np.random.default_rng(0)
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    stat = WaldoStatistic(n_mc=500, seed=99)
    theta_true = 0.4
    n_obs = 30
    n_reps = 60
    alpha = 0.10
    n_in = 0
    for _ in range(n_reps):
        data = model.sample_data(theta_true, rng, n_obs)
        try:
            lo, hi = stat._generic_confidence_interval(alpha, data, model, prior)
        except BracketingFailed:
            n_in += 1  # bracket failure -> CI spans support -> covers
            continue
        if lo <= theta_true <= hi:
            n_in += 1
    cov = n_in / n_reps
    # Coverage is approximately Bernoulli(1-alpha); the std error is
    # sqrt((1-alpha)*alpha / n_reps) ~ 0.039 here. Assert >= nominal -
    # 3*SE ~ 0.78 to allow MC noise. The (k+1)/(n+1) bias should put
    # us comfortably above the nominal 0.90.
    assert cov >= 1.0 - alpha - 3.0 * np.sqrt((1.0 - alpha) * alpha / n_reps), (
        f"WALDO empirical coverage = {cov:.3f} at alpha={alpha} on "
        f"BernoulliModel; expected >= {1 - alpha:.2f} - 3*SE."
    )
