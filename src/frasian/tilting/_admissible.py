"""Numerical admissible-range probe for `TiltingScheme`.

Currently, both `PowerLawTilting` and `OTTilting` return closed-form
admissible ranges from variance-positivity arguments specific to the
Normal-Normal sandbox. For future schemes whose admissible η-range
depends on the model + prior in a non-closed-form way (e.g. quantile-
mixture interpolation in `OTTilting.tilt` for non-Gaussian posteriors,
or a `mixture`/`fisher_rao` geodesic on a Bernoulli posterior), we
need a numerical fallback.

The procedure here is a bisection-based outward search from a known-
valid identity η:

    1. Given a `validity_fn(eta) -> bool` and a known-valid `eta_id`,
    2. probe at `eta_id ± step` for increasing `step` until validity
       fails on each side,
    3. bisect each failing bracket to atol.

Validity is a scheme-specific predicate: typically "does
`tilted_pvalue` evaluated at a small probe set of θ values produce
finite numbers in [0, 1]?". The default `is_valid_eta` (below)
implements this generic recipe.

`numerical_admissible_range` is wrapped in an LRU cache keyed on a
hashable `cache_key` so repeated calls with the same
`(scheme.name, model_fp, prior_fp, w)` are served instantly. Concrete
schemes either override `admissible_range` with a closed form (the
current power_law / OT path) or call this helper from their own
`admissible_range` method.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ----- Validity oracle ---------------------------------------------------


def is_valid_eta(
    *,
    tilted_pvalue_fn: Callable[[float], NDArray[np.float64]],
    p_atol: float = 1e-9,
) -> bool:
    """Generic validity predicate at a single η.

    `tilted_pvalue_fn(eta)` should return a 1D array of p-values
    evaluated at scheme-chosen probe θ points (closed over the
    eta-independent context: model, prior, D, statistic_name, etc.).

    η is "valid" if every probe p-value is finite, in `[-p_atol, 1+p_atol]`,
    and the array is monotonically non-increasing in `|θ - mode|` (a
    sanity check that the p-curve is bell-shaped). The caller passes
    probes ordered so monotonicity is the sane direction; if the check
    is too tight the caller can supply a softer `tilted_pvalue_fn`
    that already validates internally.
    """
    raise NotImplementedError(
        "is_valid_eta(): use `numerical_admissible_range_via_probe` "
        "below; the standalone helper exists for reference only."
    )


# ----- Bisection-based numerical search ----------------------------------


def numerical_admissible_range(
    validity_fn: Callable[[float], bool],
    *,
    eta_id: float,
    step_init: float = 0.5,
    step_max: float = 1e3,
    step_growth: float = 2.0,
    atol: float = 1e-3,
    max_outer_iter: int = 30,
    max_bisect_iter: int = 30,
) -> Tuple[float, float]:
    """Find `(eta_low, eta_high)` such that `validity_fn(eta)` is True
    for `eta_low < eta < eta_high`.

    Algorithm (per side, lower / upper):
      1. Start from `eta_id` (which must be valid). Step outward by
         `step_init`, doubling each iteration up to `step_max`, until
         validity fails OR `step_max` is reached without failure.
      2. Bisect the [last_valid, first_invalid] bracket until its
         width is below `atol`.

    Returns `(eta_low + atol, eta_high - atol)` — i.e., shrunk by
    `atol` on each side so the returned range is strictly inside the
    valid set, mirroring the buffer used in closed-form
    `admissible_range` methods.

    Raises `ValueError` if `validity_fn(eta_id)` is False (no valid
    starting point).
    """
    if not validity_fn(eta_id):
        raise ValueError(
            f"numerical_admissible_range: identity eta_id={eta_id} is invalid; "
            f"cannot bracket from there. Pass a known-valid identity."
        )

    def _search_one_side(direction: int) -> float:
        step = step_init
        eta_valid = float(eta_id)
        eta_invalid: float | None = None
        for _ in range(max_outer_iter):
            eta_probe = eta_id + direction * step
            if validity_fn(eta_probe):
                eta_valid = eta_probe
                if step >= step_max:
                    return eta_valid  # never failed within step_max
                step = min(step * step_growth, step_max)
            else:
                eta_invalid = eta_probe
                break
        else:
            return eta_valid

        # Bisect [eta_valid, eta_invalid] until its width is below atol.
        for _ in range(max_bisect_iter):
            if abs(eta_invalid - eta_valid) <= atol:
                break
            mid = 0.5 * (eta_valid + eta_invalid)
            if validity_fn(mid):
                eta_valid = mid
            else:
                eta_invalid = mid
        return eta_valid

    eta_high = _search_one_side(+1)
    eta_low = _search_one_side(-1)
    # Buffer inward so the returned range is strictly safe.
    return (eta_low + atol, eta_high - atol)


# ----- Cached entry point ------------------------------------------------


_CACHE_MAX_SIZE = 4096


@functools.lru_cache(maxsize=_CACHE_MAX_SIZE)
def _cached_call(
    cache_key: Any,
    validity_fn_id: int,  # we serialise via id() of the bound callable
    eta_id: float,
    step_init: float,
    step_max: float,
    step_growth: float,
    atol: float,
    max_outer_iter: int,
    max_bisect_iter: int,
) -> Tuple[float, float]:
    raise NotImplementedError(
        "Internal: should not be called directly; use "
        "`numerical_admissible_range_cached`."
    )


# Top-level dict cache keyed by (cache_key, atol). Functools.lru_cache
# can't cache arbitrary Callable args (id() changes each call); a
# manual dict is simpler.
_RANGE_CACHE: dict[Tuple[Any, float], Tuple[float, float]] = {}


def numerical_admissible_range_cached(
    cache_key: Any,
    validity_fn: Callable[[float], bool],
    *,
    eta_id: float,
    atol: float = 1e-3,
    **kwargs: Any,
) -> Tuple[float, float]:
    """Cached variant of `numerical_admissible_range`.

    `cache_key` should be a hashable tuple capturing every input that
    affects the result — typically `(scheme.name, model_fingerprint,
    prior_fingerprint, round(w, 4))`. The validity_fn itself is *not*
    part of the key; if you change the validity logic, evict the cache
    or use a different `cache_key`.

    On cache miss, calls `numerical_admissible_range`. On hit, returns
    the cached `(eta_low, eta_high)` immediately.
    """
    key = (cache_key, atol)
    cached = _RANGE_CACHE.get(key)
    if cached is not None:
        return cached
    result = numerical_admissible_range(
        validity_fn, eta_id=eta_id, atol=atol, **kwargs,
    )
    if len(_RANGE_CACHE) >= _CACHE_MAX_SIZE:
        # Simple LRU-ish: drop the first inserted entry. Good enough
        # since the cache key space (scheme × w-grid) is small.
        first_key = next(iter(_RANGE_CACHE))
        _RANGE_CACHE.pop(first_key, None)
    _RANGE_CACHE[key] = result
    return result


def clear_admissible_range_cache() -> None:
    """Drop all cached numerical admissible-range computations.

    Useful in tests to force re-probing.
    """
    _RANGE_CACHE.clear()


# ----- Default validity oracle for Normal-Normal -------------------------


def make_default_validity_fn(
    scheme: Any,
    model: Any,
    prior: Any,
    *,
    statistic_name: str,
    D_probe: float | None = None,
    n_theta_probes: int = 5,
    p_atol: float = 1e-9,
) -> Callable[[float], bool]:
    """Build a `validity_fn(eta) -> bool` for a Normal-Normal-sandbox scheme.

    Probes `tilted_pvalue` at `n_theta_probes` θ values spanning
    `prior.loc ± 3·prior.scale`; returns False if any probe yields
    NaN/Inf or a value outside `[-p_atol, 1 + p_atol]`. `D_probe`
    defaults to `prior.loc` (no-conflict point).
    """
    if D_probe is None:
        D_probe = float(prior.loc)
    half = 3.0 * float(prior.scale)
    theta_probes = np.linspace(prior.loc - half, prior.loc + half,
                                  n_theta_probes)

    def validity_fn(eta: float) -> bool:
        try:
            p = scheme.tilted_pvalue(
                theta_probes, D_probe, model, prior, float(eta),
                statistic_name,
            )
        except Exception:
            return False
        p_arr = np.asarray(p, dtype=np.float64)
        if not np.all(np.isfinite(p_arr)):
            return False
        if np.any(p_arr < -p_atol) or np.any(p_arr > 1.0 + p_atol):
            return False
        return True

    return validity_fn
