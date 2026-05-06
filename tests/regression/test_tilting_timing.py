"""Timing-regression tests for the tilting layer (L0).

A previous JAX-port attempt of `src/frasian/tilting/` slowed
`tests/regression/test_dynamic_pvalue.py` from ~15 s to >30 s by
JAX-ifying scalar hot loops (each `float(jnp_array[i])` inside a Python
brentq loop costs ~10-100 µs of JAX dispatch). These three tests pin
the three workloads where that class-of-bug shows up first:

A. Full dynamic-CI inversion (5 D values via power_law[dynamic_numerical]).
B. Bulk vector tilted_pvalue on a 4001-point θ-grid.
C. 1000 scalar tilted_pvalue calls (mimics the Brent inner loop).

Budgets are intentionally generous (2-5× the current numpy baseline)
so they do NOT flap on incidental drift, but tighten enough to catch
the 100× regression we previously hit.

These tests are L0 — they must pass on every commit. Captured numpy
baselines at the time of writing (commit 88de01c, pre-tilting-port):
- A: ~15 s for 9 dynamic_pvalue tests, of which 5 D values represent
     the most expensive subset → ~5-6 s baseline; cap at 10 s.
- B: ~5-10 ms per bulk call → 10 calls cap at 200 ms.
- C: ~10-50 µs per scalar call → 1000 calls cap at 250 ms.
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

    Numpy baseline ~5-6 s for 5 D values via the dynamic-eta selector
    + WALDO (the dynamic-eta selector itself is the dominant cost — it
    pre-fits eta(|Delta|) on a coarse grid via scipy.optimize.minimize
    per CI call). Budget 10.0 s allows up to ~2x drift and is comfortably
    below the 30+ s timeout that the prior JAX-everywhere attempt produced.
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
    assert elapsed < 10.0, f"dynamic CI inversion took {elapsed:.2f}s (budget 10.0s)"


@pytest.mark.L0
def test_bulk_tilted_pvalue_wall_time_budget():
    """Vector tilted_pvalue on a 4001-point grid stays <=200 ms over 10 calls.

    Numpy baseline ~5-10 ms/call. 200 ms allows for one cold JAX
    compile if any future port jit-traces this kernel.
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
    assert elapsed < 0.20, f"10 bulk tilted_pvalue calls took {elapsed:.2f}s (budget 0.20s)"
    assert np.asarray(p).shape == (4001,)


@pytest.mark.L0
def test_scalar_brentq_loop_dispatch_budget():
    """Mimic the brentq inner-loop access pattern: 1000 scalar
    tilted_pvalue calls + float() conversion.

    The prior JAX-everywhere attempt made each call ~50 us (vs. ~10 us
    numpy baseline). Budget 250 ms catches a >=5x regression on the
    scalar dispatch path. THIS is the load-bearing test — it directly
    exercises the seam that broke last time.
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
    assert elapsed < 0.25, f"1000 scalar tilted_pvalue calls took {elapsed:.2f}s (budget 0.25s)"
    assert np.isfinite(s)
