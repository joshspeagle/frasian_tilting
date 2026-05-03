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
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import optimize


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
        already closed over by the caller.
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

    Returns
    -------
    (regions, total_width, n_regions)
        `regions` is a list of disjoint `(lo, hi)` intervals, sorted.
        `total_width` is `sum(hi - lo)` (union semantics, NOT convex hull).
    """
    from .eta_selectors import _NamedStatistic

    search_half = search_mult * sigma
    theta_lo = D - search_half
    theta_hi = D + search_half
    theta_grid = np.linspace(theta_lo, theta_hi, n_grid)

    abs_delta_theta = np.abs((1.0 - w) * (mu0 - theta_grid) / sigma)

    ad_max = float(abs_delta_theta.max()) + 1e-6
    coarse_grid = np.linspace(0.0, ad_max, coarse_n)
    select_kwargs = dict(
        statistic=_NamedStatistic(statistic_name), w=w, alpha=alpha,
    )
    # Phase E selectors require the inference-time prior + model
    # fingerprints to refuse cross-experiment use. Older selectors
    # (Numerical/DynamicNumerical, legacy v1 LearnedDynamic) ignore
    # these kwargs.
    if model_fingerprint is not None:
        select_kwargs["model_fingerprint"] = model_fingerprint
    if prior_fingerprint is not None:
        select_kwargs["prior_fingerprint"] = prior_fingerprint
    try:
        coarse_eta = eta_selector.select_grid(
            coarse_grid, scheme, **select_kwargs,
        )
    except TypeError:
        # Selector predates the fingerprint plumbing; fall back to the
        # original signature. The Phase E selector's _check_experiment
        # then uses w-only validation as a degraded check.
        coarse_eta = eta_selector.select_grid(
            coarse_grid, scheme,
            statistic=select_kwargs["statistic"],
            w=w, alpha=alpha,
        )
    eta_at_theta = np.interp(abs_delta_theta, coarse_grid, coarse_eta)

    p_theta = np.empty_like(theta_grid)
    for i in range(theta_grid.size):
        p_theta[i] = float(tilted_pvalue_fn(
            float(theta_grid[i]), float(eta_at_theta[i])
        ))

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
                    _f, theta_grid[i], theta_grid[i + 1], xtol=1e-9,
                )
                crossings.append(float(cross))
            except ValueError:
                t = diff[i] / (diff[i] - diff[i + 1])
                crossings.append(
                    float(theta_grid[i] + t * (theta_grid[i + 1]
                                                - theta_grid[i]))
                )

    regions: list[tuple[float, float]] = []
    if not crossings:
        if p_theta[len(p_theta) // 2] >= alpha:
            regions = [(float(theta_lo), float(theta_hi))]
    else:
        entries = list(crossings)
        if p_theta[0] >= alpha:
            entries = [float(theta_lo)] + entries
        if p_theta[-1] >= alpha:
            entries = entries + [float(theta_hi)]
        for i in range(0, len(entries) - 1, 2):
            regions.append((entries[i], entries[i + 1]))

    total = float(sum(hi - lo for lo, hi in regions))
    return regions, total, len(regions)
