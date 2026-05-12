"""Timing-regression tests for the tilting layer (L0).

A previous JAX-port attempt of `src/frasian/tilting/` slowed
`tests/regression/test_dynamic_pvalue.py` from ~15 s to >30 s by
JAX-ifying scalar hot loops (each `float(jnp_array[i])` inside a Python
brentq loop costs ~10-100 µs of JAX dispatch). These three tests pin
the three workloads where that class-of-bug shows up first:

A. Full dynamic-CI inversion (5 D values via power_law[dynamic_numerical]).
B. Bulk vector tilted_pvalue on a 4001-point θ-grid.
C. 1000 scalar tilted_pvalue calls (mimics the Brent inner loop).

Budgets are intentionally generous (2.5-5× the dev-machine baseline)
so they do NOT flap on shared CI runners (typically 2-3× slower than
dev hardware), but tighten enough to catch the 100× regression we
previously hit.

These tests are L0 — they must pass on every commit. Captured dev
baselines at the time of writing (post-tilting-port + Phase A
review fixes, commit 0a0bad2):
- A: dynamic-CI inversion at 5 D values, ~6 s baseline; budget 20 s
  (3.3× headroom). Failure indicates per-CI runtime grew >2x.
- B: 10 bulk vector calls on 4001-pt grid, ~0.16 s baseline; budget
  0.5 s (3.1× headroom). Catches a per-call jit-cache regression.
- C: 1000 scalar tilted_pvalue calls, ~0.18 s baseline; budget 0.6 s
  (3.3× headroom). The load-bearing test for the seam that broke
  last time — generous-but-not-unbounded budget catches a >=3x
  per-call dispatch regression while tolerating CI variance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L0
def test_dynamic_ci_inversion_wall_time_budget():
    """Catch the 'jax-everywhere-including-scan-body' regression class.

    Dev baseline ~5-6 s for 5 D values via the dynamic-eta selector
    + WALDO (the dynamic-eta selector itself is the dominant cost — it
    pre-fits eta(|Delta|) on a coarse grid via scipy.optimize.minimize
    per CI call). Budget 30 s gives ~5x headroom on dev — bumped from
    20 s after 2026-05-12 CI flaked at 20.47 s on a slow shared runner.
    The 100x JAX-everywhere regression class lands at >100 s regardless,
    so the wider budget still catches it cleanly.
    """
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    scheme = PowerLawTilting(selector=DynamicNumericalEtaSelector())
    statistic = WaldoStatistic()
    Ds = [-2.0, -0.5, 0.0, 1.5, 4.0]
    t0 = time.perf_counter()
    for D in Ds:
        scheme.confidence_interval(0.05, np.asarray([D]), model, prior, statistic)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"dynamic CI inversion took {elapsed:.2f}s (budget 30.0s)"


@pytest.mark.L0
def test_bulk_tilted_pvalue_wall_time_budget():
    """Vector tilted_pvalue on a 4001-point grid stays <=500 ms over 10 calls.

    Dev baseline ~0.16 s after JIT warmup; budget 0.5 s gives 3.1x
    headroom. Catches a per-call dispatch regression (~50 ms/call
    pre-warmup or ~5 ms/call post-warmup); a 100x slowdown from a
    misplaced eager-mode hop would land us >>0.5 s.
    """
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    scheme = PowerLawTilting()
    theta = np.linspace(-8.0, 8.0, 4001)
    eta = np.full_like(theta, 0.3)
    # Warmup: amortize any first-call tracing/dispatch cost.
    _ = scheme.tilted_pvalue(theta, 1.5, model, prior, eta, "waldo")
    t0 = time.perf_counter()
    for _ in range(10):
        p = scheme.tilted_pvalue(theta, 1.5, model, prior, eta, "waldo")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.50, f"10 bulk tilted_pvalue calls took {elapsed:.2f}s (budget 0.50s)"
    assert np.asarray(p).shape == (4001,)


@pytest.mark.L0
def test_scalar_brentq_loop_dispatch_budget():
    """Mimic the brentq inner-loop access pattern: 1000 scalar
    tilted_pvalue calls + float() conversion.

    Dev baseline ~0.18 s after numpy fast-path dispatch (~180 us/call).
    Budget 0.6 s gives 3.3x headroom. The prior JAX-everywhere attempt
    blew this to ~2 s (2 ms/call) — any regression of that class still
    lands well above 0.6 s.
    THIS is the load-bearing test — it directly exercises the seam
    that broke last time. Failing this test almost always means a
    `jnp.asarray` slipped into a scalar/Python-loop boundary.
    """
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    scheme = PowerLawTilting()
    rng = np.random.default_rng(0)
    thetas = rng.uniform(-3, 3, size=1000)
    etas = rng.uniform(-0.4, 0.6, size=1000)
    t0 = time.perf_counter()
    s = 0.0
    for t, e in zip(thetas, etas):
        s += float(scheme.tilted_pvalue(float(t), 1.5, model, prior, float(e), "waldo"))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.60, f"1000 scalar tilted_pvalue calls took {elapsed:.2f}s (budget 0.60s)"
    assert np.isfinite(s)
