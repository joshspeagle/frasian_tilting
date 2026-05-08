"""The single root-finder shared by all tilting CIs.

The legacy code had three near-identical brentq blocks (`tilting.py:212`,
`:270`, `:508`) that each grew their own bracket-doubling and exception
handling. This module collapses them into one well-tested helper.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# scipy: brentq has no JAX equivalent we want yet; this module is the
# public CI-inversion boundary. Per `docs/jax_style.md`, scipy lives
# here and callers convert via `float(...)` before invoking the
# closure (see `tilting/power_law.py::tilted_confidence_interval`).
from scipy import optimize

# The canonical BracketingFailed lives in `frasian._errors`. We re-export
# here so legacy `from frasian.tilting._solvers import BracketingFailed`
# imports continue to resolve to the same class. There used to be a
# distinct `class BracketingFailed(RuntimeError)` defined in this module;
# the duplicate caused `except BracketingFailed` blocks in callers that
# imported from `_errors` to silently never catch the raised exception
# (Phase A skeptic re-review, finding #4/#5).
from .._errors import BracketingFailed  # noqa: F401  — public re-export


def brentq_with_doubling(
    f: Callable[[float], float],
    *,
    midpoint: float,
    initial_half_width: float,
    direction: int,
    max_doublings: int = 16,
    xtol: float = 1e-9,
    rtol: float = 1e-9,
    maxiter: int = 200,
) -> float:
    """Find a root of `f` near `midpoint` in the given `direction`.

    Parameters
    ----------
    f : callable
        Continuous function whose root is sought.
    midpoint : float
        One end of the search bracket; `f(midpoint)` should have known sign.
    initial_half_width : float
        Initial distance from midpoint to the other end of the bracket.
    direction : int
        +1 to search above midpoint, -1 to search below.
    max_doublings : int
        Maximum number of times to double the bracket if no sign change is
        detected. After the cap is hit, raises `BracketingFailed`.
    xtol, rtol, maxiter
        Forwarded to `scipy.optimize.brentq`.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1, got {direction!r}")
    if initial_half_width <= 0:
        raise ValueError(f"initial_half_width must be positive, got {initial_half_width!r}")

    f_mid = f(midpoint)
    # Audit P1 H.5: refuse non-finite f(midpoint) up front. Pre-fix the
    # loop would burn `max_doublings + 1` iterations before raising
    # BracketingFailed with a generic message; the actual problem
    # (a NaN/inf at the bracket midpoint, e.g. the user's f raises and
    # returns sentinel +inf, or the underlying p-value is ill-defined
    # at this θ) is much cheaper to surface here.
    if not np.isfinite(f_mid):
        raise BracketingFailed(
            f"f(midpoint) is non-finite ({f_mid!r}) at midpoint={midpoint!r}; "
            f"cannot bracket. Check that the inversion target is well-defined "
            f"at the midpoint (e.g. observed test statistic is finite, "
            f"posterior is non-degenerate)."
        )
    half = initial_half_width
    for _ in range(max_doublings + 1):
        endpoint = midpoint + direction * half
        f_end = f(endpoint)
        if np.isfinite(f_end) and f_mid * f_end <= 0.0:
            a, b = (endpoint, midpoint) if direction < 0 else (midpoint, endpoint)
            return float(optimize.brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter))
        half *= 2.0

    raise BracketingFailed(
        f"could not bracket a root within {max_doublings} doublings; "
        f"midpoint={midpoint!r} initial_half_width={initial_half_width!r} "
        f"direction={direction!r}"
    )
