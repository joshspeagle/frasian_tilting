"""The single root-finder shared by all tilting CIs.

The legacy code had three near-identical brentq blocks (`tilting.py:212`,
`:270`, `:508`) that each grew their own bracket-doubling and exception
handling. This module collapses them into one well-tested helper.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import optimize


class BracketingFailed(RuntimeError):
    """Raised when `brentq_with_doubling` cannot bracket a root."""


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
        raise ValueError(
            f"initial_half_width must be positive, got {initial_half_width!r}"
        )

    f_mid = f(midpoint)
    half = initial_half_width
    for _ in range(max_doublings + 1):
        endpoint = midpoint + direction * half
        f_end = f(endpoint)
        if np.isfinite(f_mid) and np.isfinite(f_end) and f_mid * f_end <= 0.0:
            a, b = (endpoint, midpoint) if direction < 0 else (midpoint, endpoint)
            return float(optimize.brentq(f, a, b, xtol=xtol, rtol=rtol,
                                         maxiter=maxiter))
        half *= 2.0

    raise BracketingFailed(
        f"could not bracket a root within {max_doublings} doublings; "
        f"midpoint={midpoint!r} initial_half_width={initial_half_width!r} "
        f"direction={direction!r}"
    )
