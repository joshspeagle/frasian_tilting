"""Phase 3 vectorisation regressions for `dynamic_ci_scan` and friends.

Closes Tier 1.3 N1 / N2 / N3:

- `test_dynamic_ci_scan_vectorised_matches_scalar_loop` — pins the
  vectorised scan body in `_dynamic.py` against the original Python
  loop on a fixed (D, prior, model, scheme, statistic, alpha) input
  at atol 1e-12. Augments
  `tests/regression/test_dynamic_ci_scan_refactor.py` (which compares
  against an inline reference) by directly opting back into the
  scalar-loop fallback inside `dynamic_ci_scan`.
- `test_dynamic_tilted_pvalue_vectorised_matches_loop` — pins the
  vectorised array path in `power_law.dynamic_tilted_pvalue` /
  `ot.dynamic_tilted_pvalue` against an explicit per-element loop
  reference at atol 1e-12.
- `test_compute_pvalues_per_sample_matches_scalar_loop` — Task B
  byte-identity check on a 10k LHS pool. The vectorised
  closed-form-pre-mask path returns the same array as the legacy
  per-sample try/except loop (within 1e-12; NaN positions match
  exactly).
- `test_dynamic_ci_scan_speedup` — wall-clock microbenchmark at
  ``n_grid=401``: vectorised vs scalar fallback. Marked `slow` so
  CI doesn't gate on timing flakiness.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting._dynamic import dynamic_ci_scan
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


def _scan_with_mode(scheme, model, prior, *, vectorised: bool, **kwargs):
    """Drive `dynamic_ci_scan` directly, choosing scalar-loop vs bulk path.

    The scheme's high-level `dynamic_tilted_confidence_interval` always
    plugs in the vectorised callback now; this helper bypasses that to
    feed `tilted_pvalue_vec_fn=None` (legacy scalar loop) for the
    regression baseline. The two modes must produce byte-identical
    output by construction.
    """
    sigma = float(model.sigma)
    mu0 = float(prior.loc)
    sigma0 = float(prior.scale)
    w = sigma0**2 / (sigma**2 + sigma0**2)
    statistic_name = kwargs.pop("statistic_name", "waldo")
    eta_selector = kwargs.pop("eta_selector")
    alpha = kwargs.pop("alpha", 0.05)
    D = kwargs.pop("D")

    def _scalar_fn(theta: float, eta: float) -> float:
        return float(
            scheme.tilted_pvalue(theta, D, model, prior, eta, statistic_name)
        )

    def _vec_fn(theta_arr, eta_arr):
        return np.asarray(
            scheme.tilted_pvalue(theta_arr, D, model, prior, eta_arr, statistic_name),
            dtype=np.float64,
        )

    return dynamic_ci_scan(
        tilted_pvalue_fn=_scalar_fn,
        tilted_pvalue_vec_fn=_vec_fn if vectorised else None,
        alpha=alpha,
        D=D,
        w=w,
        mu0=mu0,
        sigma=sigma,
        eta_selector=eta_selector,
        scheme=scheme,
        statistic_name=statistic_name,
        n_grid=kwargs.pop("n_grid", 401),
        coarse_n=kwargs.pop("coarse_n", 25),
        search_mult=kwargs.pop("search_mult", 8.0),
        model_fingerprint=model.fingerprint(),
        prior_fingerprint=prior.fingerprint(),
    )


@pytest.mark.L2
@pytest.mark.parametrize("scheme_factory", [PowerLawTilting, OTTilting])
@pytest.mark.parametrize("D", [-2.0, 0.0, 1.5, 4.0])
def test_dynamic_ci_scan_vectorised_matches_scalar_loop(scheme_factory, D):
    """Phase 3 N1: vectorised `dynamic_ci_scan` matches scalar loop at 1e-12."""
    scheme = scheme_factory()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    selector = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0)

    regions_v, total_v, n_v = _scan_with_mode(
        scheme, model, prior,
        vectorised=True,
        D=D, alpha=0.05, statistic_name="waldo", eta_selector=selector,
        n_grid=401, coarse_n=25, search_mult=8.0,
    )
    regions_s, total_s, n_s = _scan_with_mode(
        scheme, model, prior,
        vectorised=False,
        D=D, alpha=0.05, statistic_name="waldo", eta_selector=selector,
        n_grid=401, coarse_n=25, search_mult=8.0,
    )
    assert n_v == n_s
    assert len(regions_v) == len(regions_s)
    for (lo_v, hi_v), (lo_s, hi_s) in zip(regions_v, regions_s):
        np.testing.assert_allclose(lo_v, lo_s, atol=1e-12)
        np.testing.assert_allclose(hi_v, hi_s, atol=1e-12)
    np.testing.assert_allclose(total_v, total_s, atol=1e-12)


@pytest.mark.L2
@pytest.mark.parametrize("scheme_factory", [PowerLawTilting, OTTilting])
def test_dynamic_tilted_pvalue_vectorised_matches_loop(scheme_factory):
    """Phase 3 N3: `dynamic_tilted_pvalue` bulk call matches per-element loop."""
    scheme = scheme_factory()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    rng = np.random.default_rng(0)
    theta = np.linspace(-3.0, 5.0, 51)
    if scheme.name == "ot":
        eta = rng.uniform(0.05, 0.95, size=theta.size)
    else:  # power_law: w=0.5 → admissible eta < 2; sample inside.
        eta = rng.uniform(-0.4, 1.5, size=theta.size)
    bulk = scheme.dynamic_tilted_pvalue(theta, 1.5, model, prior, "waldo", eta)
    ref = np.array([
        float(np.asarray(
            scheme.tilted_pvalue(
                np.array([float(theta[i])]), 1.5, model, prior,
                float(eta[i]), "waldo",
            )
        ).reshape(-1)[0])
        for i in range(theta.size)
    ])
    np.testing.assert_allclose(bulk, ref, atol=1e-12)


@pytest.mark.L2
@pytest.mark.parametrize("scheme_factory", [PowerLawTilting, OTTilting])
def test_compute_pvalues_per_sample_matches_scalar_loop(scheme_factory):
    """Phase 3 N2: Task B byte-identity check on a 10k LHS pool.

    Compares the vectorised closed-form-mask path against the legacy
    per-sample try/except loop. Mix of valid + invalid η so both paths
    exercise NaN handling. NaN positions must match exactly; valid
    positions must agree to atol 1e-12.
    """
    from frasian.learned.training.validity import (
        _compute_pvalues_per_sample_loop,
        compute_pvalues_per_sample,
    )

    scheme = scheme_factory()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    rng = np.random.default_rng(7)
    n = 10_000
    theta = rng.uniform(-3.0, 3.0, size=n)
    D = rng.normal(loc=theta, scale=1.0)
    if scheme.name == "ot":
        # Mix of admissible (in [0,1]) and inadmissible η.
        eta = rng.uniform(-0.2, 1.2, size=n)
    else:  # power_law w=0.5: admissible eta < 2; sample slightly past.
        eta = rng.uniform(-0.5, 2.3, size=n)

    fast = compute_pvalues_per_sample(scheme, theta, D, model, prior, eta, "waldo")
    slow = _compute_pvalues_per_sample_loop(
        scheme, theta, D, model, prior, eta, "waldo"
    )
    # NaN positions must match exactly.
    np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
    # Finite positions agree to atol 1e-12.
    finite = ~np.isnan(slow)
    np.testing.assert_allclose(fast[finite], slow[finite], atol=1e-12)


@pytest.mark.slow
@pytest.mark.L2
def test_dynamic_ci_scan_speedup():
    """Microbenchmark: vectorised `dynamic_ci_scan` ≥ 5× faster than scalar.

    The audit (Tier 1.3 §1) projects ~50–100× on the kernel; we assert
    a conservative 5× to keep the test robust to platform variance.
    Marked ``slow`` so it doesn't run in normal CI.
    """
    scheme = PowerLawTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    selector = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0)

    def _run(vectorised: bool):
        return _scan_with_mode(
            scheme, model, prior,
            vectorised=vectorised,
            D=1.5, alpha=0.05, statistic_name="waldo", eta_selector=selector,
            n_grid=401, coarse_n=25, search_mult=8.0,
        )

    # Warmup (selector cache, scipy lazy imports).
    _run(False)
    _run(True)

    n_iter = 25
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _run(False)
    t_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _run(True)
    t_vec = time.perf_counter() - t0

    speedup = t_scalar / max(t_vec, 1e-9)
    assert speedup >= 5.0, (
        f"vectorised scan only {speedup:.1f}× faster than scalar "
        f"(scalar={t_scalar:.3f}s, vec={t_vec:.3f}s, n_iter={n_iter}); "
        f"audit projects ≥ 50× — investigate."
    )
