"""Regression: extracted `dynamic_ci_scan` matches the per-scheme algorithm.

When `_dynamic.py` was extracted from `power_law.py` and `ot.py`, both
schemes' `dynamic_tilted_confidence_interval` became 5-line shims
delegating to `dynamic_ci_scan`. This test pins that the refactor is
behaviour-preserving on a representative grid by re-implementing the
old per-scheme inline algorithm and asserting byte-identical regions.

The reference implementation is the legacy inline body of
`PowerLawTilting.dynamic_tilted_confidence_interval` from before the
extraction (preserved verbatim in `_reference_inline_dynamic_ci`
below). The test runs both schemes (power_law + ot) across a
(D, w) grid and asserts the new helper's output equals the reference
to atol 1e-12 on every endpoint.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from scipy import optimize

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector, _NamedStatistic
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


def _reference_inline_dynamic_ci(
    *,
    scheme,
    alpha: float,
    D: float,
    model: NormalNormalModel,
    prior: NormalDistribution,
    statistic_name: str,
    eta_selector,
    n_grid: int,
    coarse_n: int,
    search_mult: float,
) -> tuple[list[tuple[float, float]], float, int]:
    """Reference: the pre-extraction inline algorithm, verbatim.

    Kept here as the source-of-truth for the refactor regression. If
    `dynamic_ci_scan` ever drifts from this, the test fails.
    """
    sigma = float(model.sigma)
    mu0 = float(prior.loc)
    sigma0 = float(prior.scale)
    w = sigma0**2 / (sigma**2 + sigma0**2)

    search_half = search_mult * sigma
    theta_lo = D - search_half
    theta_hi = D + search_half
    theta_grid = np.linspace(theta_lo, theta_hi, n_grid)

    # Phase 3a-1: reference now uses the θ-keyed coarse grid (the
    # selector's new signature). The pre-3a-1 |Δ|-keyed reference is
    # gone; tests compare two equivalent θ-grid implementations.
    coarse_theta_grid = np.linspace(theta_lo, theta_hi, coarse_n)
    coarse_eta = eta_selector.select_grid(
        coarse_theta_grid,
        scheme,
        statistic=_NamedStatistic(statistic_name),
        model=model,
        prior=prior,
        alpha=alpha,
    )
    eta_at_theta = np.interp(theta_grid, coarse_theta_grid, coarse_eta)

    p_theta = np.empty_like(theta_grid)
    for i in range(theta_grid.size):
        p_theta[i] = float(
            scheme.tilted_pvalue(
                float(theta_grid[i]),
                D,
                model,
                prior,
                float(eta_at_theta[i]),
                statistic_name,
            )
        )

    diff = p_theta - alpha
    crossings: list[float] = []
    for i in range(theta_grid.size - 1):
        if diff[i] * diff[i + 1] < 0.0:

            def _f(theta_val: float, _i=i) -> float:
                eta = float(np.interp(theta_val, coarse_theta_grid, coarse_eta))
                return (
                    float(
                        scheme.tilted_pvalue(
                            theta_val,
                            D,
                            model,
                            prior,
                            eta,
                            statistic_name,
                        )
                    )
                    - alpha
                )

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


@pytest.mark.L2
@pytest.mark.parametrize("scheme_factory", [PowerLawTilting, OTTilting])
@pytest.mark.parametrize(
    "D, sigma0",
    [
        # Standard sweep
        (-2.0, 0.5),
        (-2.0, 1.0),
        (-2.0, 2.0),
        (0.0, 0.5),
        (0.0, 1.0),
        (0.0, 2.0),
        (1.5, 0.5),
        (1.5, 1.0),
        (1.5, 2.0),
        (4.0, 0.5),
        (4.0, 1.0),
        (4.0, 2.0),
        # Extreme conflict — exercises multi-region branch when applicable.
        # At (D=8, σ0=2.0): w=0.8, |Δ|=0.2·8=1.6; high-conflict territory
        # where the dynamic p-value can become multimodal.
        (8.0, 2.0),
        # At (D=6, σ0=0.5): w=0.2, |Δ|=0.8·6=4.8; very high conflict.
        (6.0, 0.5),
    ],
)
def test_extracted_helper_matches_inline(scheme_factory, D, sigma0):
    """`dynamic_ci_scan` produces byte-identical regions to the reference."""
    sigma, mu0, alpha = 1.0, 0.0, 0.05
    n_grid, coarse_n, search_mult = 401, 25, 8.0

    scheme = scheme_factory()
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    selector = DynamicNumericalEtaSelector(sigma=sigma, mu0=mu0)

    new_regions, new_total, new_n = scheme.dynamic_tilted_confidence_interval(
        alpha=alpha,
        D=D,
        model=model,
        prior=prior,
        statistic_name="waldo",
        eta_selector=selector,
        n_grid=n_grid,
        coarse_n=coarse_n,
        search_mult=search_mult,
    )
    ref_regions, ref_total, ref_n = _reference_inline_dynamic_ci(
        scheme=scheme,
        alpha=alpha,
        D=D,
        model=model,
        prior=prior,
        statistic_name="waldo",
        eta_selector=selector,
        n_grid=n_grid,
        coarse_n=coarse_n,
        search_mult=search_mult,
    )

    assert new_n == ref_n, f"region count differs: new={new_n}, ref={ref_n}"
    assert len(new_regions) == len(ref_regions)
    for (n_lo, n_hi), (r_lo, r_hi) in zip(new_regions, ref_regions):
        np.testing.assert_allclose(n_lo, r_lo, atol=1e-12)
        np.testing.assert_allclose(n_hi, r_hi, atol=1e-12)
    np.testing.assert_allclose(new_total, ref_total, atol=1e-12)
