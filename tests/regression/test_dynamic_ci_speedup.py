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
def test_dynamic_tilted_pvalue_at_admissibility_boundary(scheme_factory):
    """Phase 3 review §5: bulk path agrees with scalar loop at η boundaries.

    OT admits η in [0, 1] — sweep includes the closed endpoints exactly.
    power_law admits η < 1/(1-w); at w=0.5 the strict bound is 2.0, so we
    probe just below it (1/(1-w) - 1e-9) to verify the bulk path agrees
    with the scalar loop at the edge. The legacy random sweep
    (`test_dynamic_tilted_pvalue_vectorised_matches_loop`) excludes
    boundaries; this pins the edge cases that a future broadcast bug
    would silently regress.
    """
    scheme = scheme_factory()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    sigma = float(model.sigma)
    sigma0 = float(prior.scale)
    w = sigma0**2 / (sigma**2 + sigma0**2)

    if scheme.name == "ot":
        # Inclusive endpoints + interior mix.
        eta = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    else:  # power_law
        # Just inside the strict raise bound 1/(1-w) at w=0.5 → 2.0.
        eta_high_inside = 1.0 / (1.0 - w) - 1e-9
        eta = np.array([
            -0.4, 0.0, 0.5, 1.0, eta_high_inside,
        ], dtype=np.float64)
    theta = np.linspace(-2.0, 2.0, eta.size)

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
    # All values must be finite at admissible boundaries (the buffer-band
    # near eta=1/(1-w) might overflow but we stay 1e-9 inside).
    assert np.all(np.isfinite(bulk)), f"non-finite at boundary: {bulk}"


@pytest.mark.L2
def test_power_law_tilted_pvalue_raises_on_nan_eta():
    """Phase 3 review §3: `dynamic_tilted_pvalue` raises TiltingDomainError
    on NaN η rather than silently returning NaN p-values.

    Mirrors `OTTilting`'s upfront finiteness guard. Without this check the
    NaN slot would propagate through `denom = 1 - eta*(1-w)` (NaN <= 0 is
    False) and surface as a NaN p-value — silently breaking the bulk
    training path's validity oracle.
    """
    from frasian._errors import TiltingDomainError

    scheme = PowerLawTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    theta = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    eta_at_theta = np.array([0.5, np.nan, 0.7], dtype=np.float64)
    with pytest.raises(TiltingDomainError, match=r"index 1"):
        scheme.dynamic_tilted_pvalue(theta, 1.5, model, prior, "waldo", eta_at_theta)


@pytest.mark.L2
def test_dynamic_ci_scan_scalar_fallback_against_production_scheme():
    """Phase 3 review §4: pin the `tilted_pvalue_vec_fn=None` fallback.

    The fallback branch in `_dynamic.py` (scalar Python loop over
    `tilted_pvalue_fn`) is dead in production — both `power_law.py` and
    `ot.py` plumb the bulk callback. A future scheme that omits the
    callback hits this branch; if a refactor silently changes the loop's
    behaviour, the divergence would only surface during that new
    scheme's training. This test exercises a real production scheme
    (`PowerLawTilting`) without the bulk callback and asserts byte-
    identity (atol 1e-12) against the bulk path.
    """
    scheme = PowerLawTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    selector = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0)
    sigma = float(model.sigma)
    mu0 = float(prior.loc)
    sigma0 = float(prior.scale)
    w = sigma0**2 / (sigma**2 + sigma0**2)
    D = 1.5

    def _scalar_fn(theta: float, eta: float) -> float:
        return float(scheme.tilted_pvalue(theta, D, model, prior, eta, "waldo"))

    def _vec_fn(theta_arr, eta_arr):
        return np.asarray(
            scheme.tilted_pvalue(theta_arr, D, model, prior, eta_arr, "waldo"),
            dtype=np.float64,
        )

    # Bulk path (production callers always plumb the vec callback).
    regions_v, total_v, n_v = dynamic_ci_scan(
        tilted_pvalue_fn=_scalar_fn,
        tilted_pvalue_vec_fn=_vec_fn,
        alpha=0.05, D=D, w=w, mu0=mu0, sigma=sigma,
        eta_selector=selector, scheme=scheme, statistic_name="waldo",
        n_grid=401, coarse_n=25, search_mult=8.0,
        model_fingerprint=model.fingerprint(),
        prior_fingerprint=prior.fingerprint(),
    )
    # Scalar fallback path (vec_fn=None): emulates a future scheme that
    # exposes only the scalar tilted_pvalue closure.
    regions_s, total_s, n_s = dynamic_ci_scan(
        tilted_pvalue_fn=_scalar_fn,
        tilted_pvalue_vec_fn=None,
        alpha=0.05, D=D, w=w, mu0=mu0, sigma=sigma,
        eta_selector=selector, scheme=scheme, statistic_name="waldo",
        n_grid=401, coarse_n=25, search_mult=8.0,
        model_fingerprint=model.fingerprint(),
        prior_fingerprint=prior.fingerprint(),
    )
    assert n_v == n_s
    assert len(regions_v) == len(regions_s)
    for (lo_v, hi_v), (lo_s, hi_s) in zip(regions_v, regions_s):
        np.testing.assert_allclose(lo_v, lo_s, atol=1e-12)
        np.testing.assert_allclose(hi_v, hi_s, atol=1e-12)
    np.testing.assert_allclose(total_v, total_s, atol=1e-12)


@pytest.mark.L2
def test_admissibility_mask_unknown_scheme_warns():
    """Phase 3 review §6: `_admissibility_mask` warns on unknown schemes.

    When mixture / fisher_rao land in Phase 6 without extending the
    closed-form predicate, the bulk call falls back to the per-sample
    loop — silently. This test pins the `RuntimeWarning` so the slow-path
    regression is discoverable when those schemes ship.
    """
    from frasian.learned.training.validity import _admissibility_mask

    class _UnknownScheme:
        name = "fisher_rao_stub"

    eta_arr = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    with pytest.warns(RuntimeWarning, match=r"no closed-form admissibility"):
        mask = _admissibility_mask(_UnknownScheme(), eta_arr, model=None, prior=None)
    # All-True (modulo NaN) — the per-sample loop handles validation.
    assert np.all(mask)


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
    """Microbenchmark: vectorised `dynamic_ci_scan` ≥ 10× faster than scalar.

    The audit (Tier 1.3 §1) projects ~50–100× on the kernel; we observe
    ~24× on a quiet runner at ``n_grid=401``. Tightened the assertion
    floor from 5× to 10× per Phase 3 review §7 — we have ~24× headroom
    so 10× is robust to CI-runner load while still catching genuine
    regressions. Full 50× is achieved at ``n_grid≥1001`` where brentq
    + selector grid lookup (O(1) fixed cost) stops dominating.
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
    assert speedup >= 10.0, (
        f"vectorised scan only {speedup:.1f}× faster than scalar "
        f"(scalar={t_scalar:.3f}s, vec={t_vec:.3f}s, n_iter={n_iter}); "
        f"audit projects ≥ 50× — investigate."
    )
