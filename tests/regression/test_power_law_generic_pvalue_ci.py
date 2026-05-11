"""Phase 3c regression tests for PowerLawTilting generic tilted CI.

Pins: cross-path agreement on Normal-Normal (L3, MC tolerance) — the
generic MC path's CI matches the closed-form CI within ~3 * SE.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import FixedEtaSelector
from frasian.tilting.power_law import (
    PowerLawTilting,
    _generic_tilted_confidence_interval,
)


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("eta", [0.0, 0.3])
@pytest.mark.parametrize("D", [-0.5, 1.0])
def test_generic_ci_matches_closed_form_normal_normal(eta, D):
    """Generic vs closed-form CI on Normal-Normal at static η.

    Generic: MC with n_mc=500 + brentq inversion. Tolerance ~3 *
    sigma_post / sqrt(n_mc) plus a coarse-grid floor — enough to
    flag O(1) regressions in the generic path; not a precision pin.

    Trimmed D grid: drop D=0 (no information beyond non-conflict
    D=-0.5 + conflict D=1.0). Marked ``@slow``: cross-path agreement
    is a full-tier concern.
    """
    sigma, sigma0 = 1.0, 1.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=sigma0)
    data = np.asarray([float(D)])
    alpha = 0.10
    scheme = PowerLawTilting(selector=FixedEtaSelector(eta=eta))
    statistic = WaldoStatistic()

    n_mc = 500
    cf_lo, cf_hi = scheme.tilted_confidence_interval(
        alpha, float(D), model, prior, eta, statistic.name
    )
    gn_lo, gn_hi = _generic_tilted_confidence_interval(
        alpha, data, model, prior, eta, statistic.name, n_mc=n_mc,
    )

    sigma_post = float(np.sqrt(model.posterior(data, prior).var()))
    tol = 3.0 * sigma_post / np.sqrt(n_mc) + 0.10
    assert abs(cf_lo - gn_lo) < tol, (
        f"CI lower disagreement: closed={cf_lo:.4f} generic={gn_lo:.4f} (tol={tol:.4f})"
    )
    assert abs(cf_hi - gn_hi) < tol, (
        f"CI upper disagreement: closed={cf_hi:.4f} generic={gn_hi:.4f} (tol={tol:.4f})"
    )
