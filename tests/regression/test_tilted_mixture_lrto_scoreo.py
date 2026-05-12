"""Mixture tilting + (lrto, scoreo): no closed-form Φ shortcut.

q_η,mix is a 2-Gaussian mixture; LRTO/SCOREO probe its shape, not just
its moments. The accept set in D'-space is non-quadratic, so the
closed-form path computes τ analytically per replicate and integrates
the H₀ reference via MC. See docs/notes/2026-05-12-tilted-trinity-
derivation.md.

Tests:
  - η=0 recovery: at η=0 the mixture collapses onto the prior-tilted
    component (lrto/scoreo coincide with the closed-form NN bare-LRTO /
    bare-SCOREO, modulo MC noise).
  - Uniformity under H₀: empirical p-values at the true θ are
    approximately U[0,1] (KS test, n_reps=200).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from frasian.models.normal_normal import NormalNormalModel
from frasian.models.distributions import NormalDistribution
from frasian.statistics.lrto import LRTOStatistic
from frasian.statistics.scoreo import ScoreoStatistic
from frasian.tilting.mixture import MixtureTilting
from frasian.tilting.eta_selectors import FixedEtaSelector


@pytest.mark.L2
@pytest.mark.parametrize("stat_name", ["lrto", "scoreo"])
def test_mixture_eta_zero_matches_bare_stat(stat_name):
    """At η=0, mixture tilt is identity → matches bare lrto/scoreo p-value."""
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    data = np.array([0.7])
    tilt = MixtureTilting(selector=FixedEtaSelector(eta=0.0))
    p_mix = float(np.asarray(tilt.tilted_pvalue(
        0.5, data, model, prior, 0.0, statistic_name=stat_name
    )))
    stat = (LRTOStatistic if stat_name == "lrto" else ScoreoStatistic)()
    p_bare = float(np.asarray(stat.pvalue(0.5, data, model, prior)))
    # MC noise tolerance — η=0 mixture-path is closed-form-equivalent
    # but the implementation still goes through the per-replicate path,
    # so use a generous tolerance.
    assert abs(p_mix - p_bare) < 0.05, f"{stat_name}: p_mix={p_mix} p_bare={p_bare}"


@pytest.mark.L3
@pytest.mark.parametrize("stat_name", ["lrto", "scoreo"])
def test_mixture_pvalue_uniform_under_H0(stat_name):
    """KS test: under D' ~ N(θ_true, σ²), p(θ_true; D') ~ U[0,1]."""
    rng = np.random.default_rng(20260512)
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    theta_true = 0.3
    tilt = MixtureTilting(selector=FixedEtaSelector(eta=0.3))
    n_reps = 200
    p_vals = np.empty(n_reps)
    for i in range(n_reps):
        D = rng.normal(theta_true, model.sigma, size=1)
        p_vals[i] = float(np.asarray(tilt.tilted_pvalue(
            theta_true, D, model, prior, 0.3, statistic_name=stat_name
        )))
    ks_stat, ks_p = kstest(p_vals, "uniform")
    assert ks_p > 0.01, f"{stat_name}: KS p={ks_p} (stat={ks_stat})"
