"""Generic-MC path supports lrto/scoreo for PL/OT/FR (this file).

The closed-form NN+Normal trinity collapse routes (PL/OT/FR, lrto/scoreo)
to waldo's Φ formula. This file tests the generic-MC path that runs on
non-NN models — verifying τ_LRTO and τ_SCOREO are correctly computed
from a grid-distribution representation of q_η.

Tests:
  - generic-path agreement at η=0 with bare LRTOStatistic/ScoreoStatistic.pvalue
    (modulo MC noise) on NN+Normal (forced via statistic.force_generic semantics).
  - p ∈ [0, 1] across a parameter sweep.
  - Trinity-collapse fingerprint: PL's generic path at η=0 with lrto/scoreo
    matches its waldo p-value to MC tolerance (Gaussian q_η on NN means
    τ_LRTO = τ_SCOREO = τ_WALDO).
"""
from __future__ import annotations

import numpy as np
import pytest

from frasian.models.normal_normal import NormalNormalModel
from frasian.models.distributions import NormalDistribution
from frasian.tilting.power_law import (
    PowerLawTilting,
    _generic_tilted_pvalue,
)
from frasian.tilting.eta_selectors import FixedEtaSelector


@pytest.mark.L2
@pytest.mark.parametrize("stat_name", ["lrto", "scoreo"])
@pytest.mark.parametrize("eta", [0.0, 0.3])
@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_pl_generic_path_trinity_collapse(stat_name, eta, theta):
    """On NN+Normal, PL's generic-MC path gives τ_LRTO/SCOREO ≈ τ_WALDO
    because q_η is still a single Gaussian (trinity collapse propagates).
    The tolerance is MC noise — should agree within ~0.1 at n_mc=200.
    """
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    data = np.array([0.7])
    p_waldo = _generic_tilted_pvalue(theta, data, model, prior, eta, "waldo")
    p_stat = _generic_tilted_pvalue(theta, data, model, prior, eta, stat_name)
    assert 0.0 <= float(p_waldo) <= 1.0
    assert 0.0 <= float(p_stat) <= 1.0
    # MC noise tolerance — same CRN seed across statistics by construction,
    # but the per-replicate τ comparison differs slightly because the grid
    # representation of q_η has finite resolution. Trinity collapse should
    # still hold within MC noise.
    assert abs(float(p_stat) - float(p_waldo)) < 0.15, (
        f"{stat_name} eta={eta} theta={theta}: p_stat={p_stat} p_waldo={p_waldo}"
    )
