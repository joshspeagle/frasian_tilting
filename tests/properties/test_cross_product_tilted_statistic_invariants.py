"""Cross-product invariants for (TiltingScheme × {lrto, scoreo}) cells.

Covers the matrix `{PL, OT, MX, FR} × {lrto, scoreo} = 8 cells` with
shared protocol invariants. Earlier files test specific paths (trinity
collapse on closed-form / generic-MC; mixture-specific KS uniformity).
This file is the cross-product sweep.

Tests per cell:
  L1: ``p ∈ [0, 1]`` (smoke)
  L1: continuity in θ (no jumps > 0.1 for Δθ ≈ 1e-3)
  L1: η=0 collapse to bare ``LRTOStatistic`` / ``ScoreoStatistic.pvalue``
  L3: KS uniformity under H₀ at θ_true (n_reps=200, KS p > 0.01)

Markers chosen to allow targeted test selection: ``pytest -m L1`` skips
the slow KS sweep. KS uniformity restricted to PL/OT/FR (closed-form
Gaussian fast paths) to keep wall-clock tractable; MX has its own
KS uniformity test in ``tests/regression/test_tilted_mixture_lrto_scoreo.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.lrto import LRTOStatistic
from frasian.statistics.scoreo import ScoreoStatistic
from frasian.tilting.eta_selectors import FixedEtaSelector
from frasian.tilting.fisher_rao import FisherRaoTilting
from frasian.tilting.mixture import MixtureTilting
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting

_CELLS = [
    (PowerLawTilting, "power_law"),
    (OTTilting, "ot"),
    (MixtureTilting, "mixture"),
    (FisherRaoTilting, "fisher_rao"),
]

# KS-uniformity tests: only PL/OT/FR (Gaussian q_η → exact closed-form
# via trinity collapse → no MC discretization). MX has its own KS test
# in tests/regression/test_tilted_mixture_lrto_scoreo.py and per-CI cost
# is prohibitive for an n_reps=200 sweep.
_PL_OT_FR_CELLS = [c for c in _CELLS if c[1] != "mixture"]


def _build_tilt(tilt_cls, eta):
    return tilt_cls(selector=FixedEtaSelector(eta=eta))


# Smaller (θ, η) grid for MX to keep wall-clock under control —
# MX per-call cost dominates the other schemes by an order of magnitude.
_PL_OT_FR_THETA_GRID = [-0.5, 0.0, 0.7]
_PL_OT_FR_ETA_GRID = [0.0, 0.3]
_MX_THETA_GRID = [0.0]
_MX_ETA_GRID = [0.0, 0.3]


def _pvalue_grid_params():
    params = []
    for tilt_cls, scheme in _CELLS:
        thetas = _MX_THETA_GRID if scheme == "mixture" else _PL_OT_FR_THETA_GRID
        etas = _MX_ETA_GRID if scheme == "mixture" else _PL_OT_FR_ETA_GRID
        for stat_name in ("lrto", "scoreo"):
            for eta in etas:
                for theta in thetas:
                    params.append((tilt_cls, scheme, stat_name, eta, theta))
    return params


@pytest.mark.L1
@pytest.mark.properties
@pytest.mark.parametrize(
    "tilt_cls,scheme,stat_name,eta,theta", _pvalue_grid_params()
)
def test_pvalue_in_unit_interval(tilt_cls, scheme, stat_name, eta, theta):
    """``p ∈ [0, 1]`` across (scheme, statistic, η, θ)."""
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    data = np.array([0.7])
    tilt = _build_tilt(tilt_cls, eta)
    p = float(
        np.asarray(
            tilt.tilted_pvalue(
                theta, data, model, prior, eta, statistic_name=stat_name
            )
        )
    )
    assert 0.0 <= p <= 1.0, (
        f"{scheme} {stat_name} eta={eta} theta={theta}: p={p}"
    )


@pytest.mark.L1
@pytest.mark.properties
@pytest.mark.parametrize("tilt_cls,scheme", _CELLS)
@pytest.mark.parametrize("stat_name", ["lrto", "scoreo"])
def test_pvalue_continuous_in_theta(tilt_cls, scheme, stat_name):
    """Small Δθ → small Δp. Catches discretization / branching errors."""
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    data = np.array([0.7])
    tilt = _build_tilt(tilt_cls, 0.3)
    theta_a = 0.4
    theta_b = 0.4 + 1e-3
    p_a = float(
        np.asarray(
            tilt.tilted_pvalue(
                theta_a, data, model, prior, 0.3, statistic_name=stat_name
            )
        )
    )
    p_b = float(
        np.asarray(
            tilt.tilted_pvalue(
                theta_b, data, model, prior, 0.3, statistic_name=stat_name
            )
        )
    )
    assert abs(p_a - p_b) < 0.1, (
        f"{scheme} {stat_name}: |p(θ={theta_a}) - p(θ={theta_b})| = "
        f"{abs(p_a - p_b)} (>0.1 jump suggests a branching error)"
    )


@pytest.mark.L1
@pytest.mark.properties
@pytest.mark.parametrize("tilt_cls,scheme", _CELLS)
@pytest.mark.parametrize("stat_name", ["lrto", "scoreo"])
def test_eta_zero_collapses_to_bare(tilt_cls, scheme, stat_name):
    """At η=0, the tilted p-value coincides with the bare statistic's.

    Tolerance 0.05 absorbs MC noise on the mixture-generic path and
    captures the trinity-collapse exact equality on PL/OT/FR
    (where the residual is zero).
    """
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    data = np.array([0.7])
    tilt = _build_tilt(tilt_cls, 0.0)
    p_tilt = float(
        np.asarray(
            tilt.tilted_pvalue(
                0.5, data, model, prior, 0.0, statistic_name=stat_name
            )
        )
    )
    bare = (LRTOStatistic if stat_name == "lrto" else ScoreoStatistic)()
    p_bare = float(np.asarray(bare.pvalue(0.5, data, model, prior)))
    assert abs(p_tilt - p_bare) < 0.05, (
        f"{scheme} {stat_name}: p_tilt={p_tilt} vs p_bare={p_bare}"
    )


@pytest.mark.L3
@pytest.mark.parametrize("tilt_cls,scheme", _PL_OT_FR_CELLS)
@pytest.mark.parametrize("stat_name", ["lrto", "scoreo"])
def test_pvalue_uniform_under_H0(tilt_cls, scheme, stat_name):
    """KS test: under ``D' ~ N(θ_true, σ²)``, ``p(θ_true; D') ~ U[0, 1]``.

    The strongest calibration check across the 6 (PL/OT/FR × {lrto, scoreo})
    cells; a regression here is a real calibration bug.
    """
    rng = np.random.default_rng(20260512)
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    theta_true = 0.3
    tilt = _build_tilt(tilt_cls, 0.3)
    n_reps = 200
    p_vals = np.empty(n_reps)
    for i in range(n_reps):
        D = rng.normal(theta_true, model.sigma, size=1)
        p_vals[i] = float(
            np.asarray(
                tilt.tilted_pvalue(
                    theta_true, D, model, prior, 0.3, statistic_name=stat_name
                )
            )
        )
    ks_stat, ks_p = kstest(p_vals, "uniform")
    assert ks_p > 0.01, (
        f"{scheme} {stat_name}: KS p={ks_p} (stat={ks_stat})"
    )
