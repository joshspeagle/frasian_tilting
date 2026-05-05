"""Bracket-hardening for the ±search_mult·σ search box in dynamic CI scan.

Pins Tier 1.5-O6: previously, when the dynamic CI extended past the
fixed ±8σ search half-width, the boundary was used as the CI endpoint
silently. The new behaviour widens the box once on a boundary-hit and
raises ``BracketingFailed`` only if the second pass also fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian._errors import BracketingFailed
from frasian.tilting._dynamic import dynamic_ci_scan
from frasian.tilting.eta_selectors import _NamedStatistic


class _AllAcceptSelector:
    """Selector whose grid lookup returns η=0 everywhere — leaves the
    p-value uncontracted, used to force the search box to be the
    binding constraint.
    """

    name = "fixed_zero"
    is_dynamic = True

    def select_grid(self, abs_delta_grid, scheme, *, statistic, w, alpha):
        return np.zeros_like(np.asarray(abs_delta_grid, dtype=np.float64))


class _AlwaysAcceptScheme:
    """Stub scheme whose tilted_pvalue is identically 1.0 — the
    accept region is the entire real line. The dynamic CI will hit
    both boundaries, triggering the bracket-hardening path.
    """

    name = "always_accept"


def _always_accept_pvalue(theta: float, eta: float) -> float:
    return 1.0


def _always_accept_pvalue_vec(
    theta_arr: np.ndarray, eta_arr: np.ndarray
) -> np.ndarray:
    return np.ones_like(theta_arr, dtype=np.float64)


@pytest.mark.L2
def test_bracket_hardening_raises_on_unbounded_accept_region() -> None:
    """When the accept region exceeds even the auto-widened 2x search box,
    BracketingFailed must surface — not a silently-truncated interval.
    """
    with pytest.raises(BracketingFailed):
        dynamic_ci_scan(
            tilted_pvalue_fn=_always_accept_pvalue,
            tilted_pvalue_vec_fn=_always_accept_pvalue_vec,
            alpha=0.05,
            D=0.0,
            w=0.5,
            mu0=0.0,
            sigma=1.0,
            eta_selector=_AllAcceptSelector(),
            scheme=_AlwaysAcceptScheme(),
            statistic_name="waldo",
            n_grid=51,
            coarse_n=11,
            search_mult=8.0,
        )


@pytest.mark.L2
def test_bracket_hardening_message_mentions_search_box() -> None:
    """The error message must clearly mention the search-box bound, so a
    user can spot the issue without reading the source.
    """
    with pytest.raises(BracketingFailed, match="search box"):
        dynamic_ci_scan(
            tilted_pvalue_fn=_always_accept_pvalue,
            tilted_pvalue_vec_fn=_always_accept_pvalue_vec,
            alpha=0.05,
            D=0.0,
            w=0.5,
            mu0=0.0,
            sigma=1.0,
            eta_selector=_AllAcceptSelector(),
            scheme=_AlwaysAcceptScheme(),
            statistic_name="waldo",
            n_grid=51,
            coarse_n=11,
            search_mult=8.0,
        )
