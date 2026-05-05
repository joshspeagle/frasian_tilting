"""Shared helper for dynamic-η-per-θ CI inversion (Normal-Normal sandbox).

The dynamic procedure: at each θ in a fine scan, look up
η = η*(|Δ_θ|) from a pre-computed coarse grid, evaluate the tilted
p-value at that θ with that θ-specific η, then find α-crossings to
delineate the CI regions. Calibration is automatic because η depends
only on θ (not on D).

The algorithm itself is scheme-agnostic *given the |Δ| coordinate is
the Normal-Normal scaled prior-data conflict* `|Δ| = (1-w)|μ₀-θ|/σ`;
only the `tilted_pvalue` callback is scheme-specific. Concrete
`TiltingScheme`s (`PowerLawTilting`, `OTTilting`, ...) delegate
`dynamic_tilted_confidence_interval` to `dynamic_ci_scan` here,
passing in their own `tilted_pvalue` closure.

Future non-Gaussian schemes (e.g. on a `BernoulliModel`) would need
either a generalisation of the `|Δ|`-formula (`abs_delta(theta)`
callback would have to be parametrised) or a sibling helper. The
current contract bakes in the Normal-Normal sandbox.

Originally duplicated across `power_law.py` and `ot.py`; extracted
here so the algorithm has a single source of truth and so future
Normal-Normal schemes (`fisher_rao`, `mixture`) plug in without
re-implementing the scan.

Phase 3 (Tier 1.3 N1) replaces the original scalar Python loop over
`theta_grid` with a single bulk call into the scheme's array-aware
`tilted_pvalue`. Speedup ≈ 50–100× on the kernel; per-cell experiment
runtime drops correspondingly. The brentq refinement still uses a
scalar closure — it's called only at sign-changes (typically 2 per
scan), so the loop overhead there is negligible.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import optimize

from .._errors import BracketingFailed


def dynamic_ci_scan(
    *,
    tilted_pvalue_fn: Callable[[float, float], float],
    alpha: float,
    D: float,
    w: float,
    mu0: float,
    sigma: float,
    eta_selector: Any,
    scheme: Any,
    statistic_name: str,
    n_grid: int = 401,
    coarse_n: int = 25,
    search_mult: float = 8.0,
    model_fingerprint: tuple | None = None,
    prior_fingerprint: tuple | None = None,
    tilted_pvalue_vec_fn: Callable[
        [np.ndarray, np.ndarray], np.ndarray
    ]
    | None = None,
) -> tuple[list[tuple[float, float]], float, int]:
    """Dynamic-η-per-θ CI inversion.

    Algorithm:
      1. Build a coarse |Δ| grid covering the search range.
      2. Compute η*(|Δ|) on the coarse grid via `eta_selector.select_grid`.
      3. Build a fine θ scan; interpolate η* per θ from the coarse grid.
      4. Compute the dynamic p-value at each θ via `tilted_pvalue_fn`.
      5. Find α-crossings; refine each via brentq.
      6. Stitch crossings into intervals (region count may be > 1: the
         dynamic p-value can be multimodal at low |Δ|).

    Parameters
    ----------
    tilted_pvalue_fn
        `(theta: float, eta: float) -> float`. The scheme-specific
        tilted-WALDO/Wald p-value, with `(D, model, prior, statistic_name)`
        already closed over by the caller. Used inside the brentq
        refinement at α-crossings.
    alpha, D
        Significance level and observed datum.
    w, mu0, sigma
        Normal-Normal weight and prior/likelihood parameters; used
        to derive `|Δ_θ|` and the search window.
    eta_selector
        Anything with `select_grid(coarse_grid, scheme, *, statistic, w, alpha)`.
    scheme, statistic_name
        Passed through to `eta_selector` for caching / dispatch.
    n_grid, coarse_n, search_mult
        Scan resolution and window half-width (in σ).
    tilted_pvalue_vec_fn
        Optional `(theta_arr, eta_arr) -> p_arr` bulk callback used for
        the fine-scan evaluation. When supplied, the scan replaces its
        scalar loop with a single vectorised call (≈ 50–100× speedup,
        Tier 1.3 N1). When ``None``, the scan falls back to a Python
        loop over ``tilted_pvalue_fn`` for backward compatibility with
        callers that don't expose an array-aware path.

    Returns
    -------
    (regions, total_width, n_regions)
        `regions` is a list of disjoint `(lo, hi)` intervals, sorted.
        `total_width` is `sum(hi - lo)` (union semantics, NOT convex hull).
    """
    from .eta_selectors import _NamedStatistic

    # First attempt at the requested search width; on boundary-hit we
    # auto-widen once before raising BracketingFailed (Tier 1.5-O6).
    regions, total, n_regions, hit_boundary = _run_dynamic_scan(
        tilted_pvalue_fn=tilted_pvalue_fn,
        tilted_pvalue_vec_fn=tilted_pvalue_vec_fn,
        alpha=alpha,
        D=D,
        w=w,
        mu0=mu0,
        sigma=sigma,
        eta_selector=eta_selector,
        scheme=scheme,
        statistic_name=statistic_name,
        n_grid=n_grid,
        coarse_n=coarse_n,
        search_mult=search_mult,
        model_fingerprint=model_fingerprint,
        prior_fingerprint=prior_fingerprint,
        named_statistic_cls=_NamedStatistic,
    )
    if hit_boundary:
        # Retry once at 2x search width. If the boundary is still hit,
        # something is genuinely off (unbounded CI region) — raise rather
        # than silently truncate.
        regions, total, n_regions, hit_boundary = _run_dynamic_scan(
            tilted_pvalue_fn=tilted_pvalue_fn,
            tilted_pvalue_vec_fn=tilted_pvalue_vec_fn,
            alpha=alpha,
            D=D,
            w=w,
            mu0=mu0,
            sigma=sigma,
            eta_selector=eta_selector,
            scheme=scheme,
            statistic_name=statistic_name,
            n_grid=n_grid,
            coarse_n=coarse_n,
            search_mult=2.0 * search_mult,
            model_fingerprint=model_fingerprint,
            prior_fingerprint=prior_fingerprint,
            named_statistic_cls=_NamedStatistic,
        )
        if hit_boundary:
            raise BracketingFailed(
                f"dynamic_ci_scan: CI extends past search box (±{2.0 * search_mult}·σ "
                f"around D={D!r}; sigma={sigma!r}). Increase search_mult or "
                f"check for an unbounded p-value tail at this θ."
            )
    return regions, total, n_regions


def _run_dynamic_scan(
    *,
    tilted_pvalue_fn: Callable[[float, float], float],
    tilted_pvalue_vec_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    alpha: float,
    D: float,
    w: float,
    mu0: float,
    sigma: float,
    eta_selector: Any,
    scheme: Any,
    statistic_name: str,
    n_grid: int,
    coarse_n: int,
    search_mult: float,
    model_fingerprint: tuple | None,
    prior_fingerprint: tuple | None,
    named_statistic_cls: Any,
) -> tuple[list[tuple[float, float]], float, int, bool]:
    """Single scan pass; returns ``(regions, total, n_regions, hit_boundary)``.

    ``hit_boundary`` is True iff the lower or upper search edge has
    p-value ≥ alpha (i.e. the accept-region extends past the box).
    """
    _NamedStatistic = named_statistic_cls

    search_half = search_mult * sigma
    theta_lo = D - search_half
    theta_hi = D + search_half
    theta_grid = np.linspace(theta_lo, theta_hi, n_grid)

    abs_delta_theta = np.abs((1.0 - w) * (mu0 - theta_grid) / sigma)

    ad_max = float(abs_delta_theta.max()) + 1e-6
    coarse_grid = np.linspace(0.0, ad_max, coarse_n)
    select_kwargs = dict(
        statistic=_NamedStatistic(statistic_name),
        w=w,
        alpha=alpha,
    )
    # Phase E selectors accept inference-time prior + model
    # fingerprints to enforce strict cross-experiment refusal.
    # Older selectors (Numerical/DynamicNumerical) don't have these
    # kwargs in their signature; introspect to dispatch cleanly.
    # Avoids try/except TypeError which would swallow real bugs in
    # the selector body (skeptic E pre-PR review #6).
    import inspect as _inspect
    from collections.abc import Mapping

    _params: Mapping[str, _inspect.Parameter]
    try:
        _sig = _inspect.signature(eta_selector.select_grid)
        _params = _sig.parameters
    except (TypeError, ValueError):
        _params = {}
    if model_fingerprint is not None and "model_fingerprint" in _params:
        select_kwargs["model_fingerprint"] = model_fingerprint
    if prior_fingerprint is not None and "prior_fingerprint" in _params:
        select_kwargs["prior_fingerprint"] = prior_fingerprint
    coarse_eta = eta_selector.select_grid(
        coarse_grid,
        scheme,
        **select_kwargs,
    )
    eta_at_theta = np.interp(abs_delta_theta, coarse_grid, coarse_eta)

    # Fine-scan p-values: prefer the bulk vectorised callback when the
    # caller supplied one (Tier 1.3 N1). The scalar loop fallback keeps
    # the door open for schemes that don't yet broadcast over array eta.
    if tilted_pvalue_vec_fn is not None:
        p_theta = np.asarray(
            tilted_pvalue_vec_fn(theta_grid, eta_at_theta),
            dtype=np.float64,
        )
        if p_theta.shape != theta_grid.shape:
            raise ValueError(
                f"tilted_pvalue_vec_fn returned shape {p_theta.shape!r}; "
                f"expected {theta_grid.shape!r}."
            )
    else:
        p_theta = np.empty_like(theta_grid)
        for i in range(theta_grid.size):
            p_theta[i] = float(
                tilted_pvalue_fn(float(theta_grid[i]), float(eta_at_theta[i]))
            )

    diff = p_theta - alpha
    crossings: list[float] = []
    for i in range(theta_grid.size - 1):
        if diff[i] * diff[i + 1] < 0.0:

            def _f(theta_val: float, _i=i) -> float:
                ad = abs((1.0 - w) * (mu0 - theta_val) / sigma)
                eta = float(np.interp(ad, coarse_grid, coarse_eta))
                return float(tilted_pvalue_fn(theta_val, eta)) - alpha

            try:
                cross = optimize.brentq(
                    _f,
                    theta_grid[i],
                    theta_grid[i + 1],
                    xtol=1e-9,
                )
                crossings.append(float(cross))
            except ValueError:
                t = diff[i] / (diff[i] - diff[i + 1])
                crossings.append(float(theta_grid[i] + t * (theta_grid[i + 1] - theta_grid[i])))

    regions: list[tuple[float, float]] = []
    hit_boundary = False
    if not crossings:
        if p_theta[len(p_theta) // 2] >= alpha:
            regions = [(float(theta_lo), float(theta_hi))]
            # Whole window is the accept region — unambiguous truncation.
            hit_boundary = True
    else:
        entries = list(crossings)
        if p_theta[0] >= alpha:
            entries = [float(theta_lo)] + entries
            hit_boundary = True
        if p_theta[-1] >= alpha:
            entries = entries + [float(theta_hi)]
            hit_boundary = True
        for i in range(0, len(entries) - 1, 2):
            regions.append((entries[i], entries[i + 1]))

    total = float(sum(hi - lo for lo, hi in regions))
    return regions, total, len(regions), hit_boundary
