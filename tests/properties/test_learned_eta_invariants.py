"""L1 invariant: trained MonotonicEtaNet produces η non-decreasing in |Δ|.

The architecture enforces `∂η*/∂|Δ| ≥ 0` structurally (positive-weight
ReLU pathway over `|Δ'|`, then sigmoid). This test pins that
guarantee as a regression: load each shipped checkpoint, scan a dense
|Δ| grid at several w values, assert non-decreasing.

If this test fails, the architecture has been changed in a way that
broke the structural monotonicity — investigate
`src/frasian/learned/training/architecture.py:MonotonicEtaNet.forward`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.learned.monotonic_eta import MonotonicEtaArtifact
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


_CHECKPOINTS = [
    ("power_law", Path("artifacts/learned_eta_power_law_v0_smoke.pt"),
     PowerLawTilting),
    ("ot", Path("artifacts/learned_eta_ot_v0_smoke.pt"), OTTilting),
]


def _available_checkpoints():
    return [(name, p, cls) for (name, p, cls) in _CHECKPOINTS if p.exists()]


@pytest.mark.L1
@pytest.mark.properties
@pytest.mark.parametrize("scheme_name,path,scheme_cls", _available_checkpoints())
@pytest.mark.parametrize("w_val", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_eta_non_decreasing_in_abs_delta(scheme_name, path, scheme_cls, w_val):
    """`∂η/∂|Δ| ≥ 0` from the trained MLP at every w on a dense scan."""
    artifact = MonotonicEtaArtifact(artifact_path=path)
    selector = LearnedDynamicEtaSelector(artifact=artifact)
    scheme = scheme_cls()

    ad_grid = np.linspace(0.0, 6.0, 601)
    eta = selector.select_grid(
        ad_grid, scheme, statistic=WaldoStatistic(),
        w=w_val, alpha=0.05,
    )
    diffs = np.diff(eta)
    # Allow a tiny float-noise tolerance; the architecture guarantees
    # exact non-decreasing under infinite precision.
    assert np.all(diffs >= -1e-6), (
        f"η is not non-decreasing in |Δ| at w={w_val} for "
        f"{scheme_name}: min diff = {diffs.min():.2e} "
        f"at |Δ|≈{ad_grid[int(np.argmin(diffs))]:.3f}"
    )


@pytest.mark.L1
@pytest.mark.properties
@pytest.mark.parametrize("scheme_name,path,scheme_cls", _available_checkpoints())
def test_eta_in_admissible_range(scheme_name, path, scheme_cls):
    """Trained η stays inside the scheme's admissible range at every (|Δ|, w).

    For `power_law`: η ∈ (η_min(w), 1) where η_min(w) = -w/(1-w).
    For `ot`:        η ∈ (0, 1).

    The architecture's bounded-sigmoid output `0.01 + 0.98·sigmoid(...)`
    keeps η' ∈ [0.01, 0.99]; per-scheme `eta_inverse` maps this to a
    safe interior of the admissible range.
    """
    artifact = MonotonicEtaArtifact(artifact_path=path)
    selector = LearnedDynamicEtaSelector(artifact=artifact)
    scheme = scheme_cls()

    ad_grid = np.linspace(0.0, 6.0, 121)
    for w_val in (0.05, 0.2, 0.5, 0.8, 0.95):
        eta = selector.select_grid(
            ad_grid, scheme, statistic=WaldoStatistic(),
            w=w_val, alpha=0.05,
        )
        if scheme_name == "power_law":
            eta_min = -w_val / (1.0 - w_val)
            assert np.all(eta > eta_min), (
                f"power_law η below η_min(w={w_val})={eta_min}: "
                f"min η = {eta.min()}"
            )
            assert np.all(eta < 1.0), f"power_law η ≥ 1: max = {eta.max()}"
        elif scheme_name == "ot":
            assert np.all(eta > 0.0), f"ot η ≤ 0: min = {eta.min()}"
            assert np.all(eta < 1.0), f"ot η ≥ 1: max = {eta.max()}"
        else:
            pytest.skip(f"no admissible-range check for scheme {scheme_name}")
