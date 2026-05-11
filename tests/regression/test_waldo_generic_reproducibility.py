"""WALDO generic-path reproducibility & CRN sanity tests.

These pin two properties of the Phase 2 generic WALDO Monte-Carlo
path that the original implementation broke (skeptic findings #1, #13
in the Phase 2 review):

1. The empirical p-value is reproducible across processes given a
   fixed `(data, model, prior, alpha, seed)` tuple. Achieved via a
   `hashlib.blake2b` stable hash, since Python's built-in `hash()`
   randomises tuples-containing-strings per `PYTHONHASHSEED`.
2. The CI inversion uses common random numbers (CRN) across brentq
   probes — same seed across all theta means `f(theta)` is a
   deterministic function (smooth for Normal) instead of a fresh
   stochastic process at every iterate.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
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


