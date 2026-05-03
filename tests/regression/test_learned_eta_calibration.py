"""Headline test 1: empirical calibration of the Phase E learned-η selector.

For each θ_true at the trained checkpoint's w, simulate ``n_reps`` D
draws, build the dynamic CI via ``(power_law[learned], waldo)``, and
check empirical coverage hits the nominal level within MC noise.

Calibration is automatic (η depends only on θ, not D), but this is
the empirical safety net that catches any "validity vs distribution
properness" gap in the trained checkpoint (see
``docs/methods/learned_eta.md``).

Phase E checkpoints are per-experiment: the test runs at the trained
``w`` only (the canonical configs train at w=0.5). Other w values
need their own trained checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.learned.eta_artifact import EtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.power_law import PowerLawTilting


_CHECKPOINT_CANDIDATES = [
    Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v1.pt"),
    Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v0_smoke.pt"),
]


def _checkpoint_path() -> Path:
    for c in _CHECKPOINT_CANDIDATES:
        if c.exists():
            return c
    pytest.skip("no Phase E learned-eta checkpoint available; train one first")


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("theta_true", [-2.0, 0.0, 2.0])
def test_calibration_at_alpha_05(theta_true):
    """At α=0.05 and the trained w (=0.5 for canonical configs),
    empirical coverage matches nominal 0.95 within 3·MC_SE."""
    alpha = 0.05
    n_reps = 300

    artifact = EtaArtifact(artifact_path=_checkpoint_path())
    artifact.load()
    cfg = artifact.metadata["experiment_config"]
    sigma = float(cfg["model_fingerprint"][1])
    mu0 = float(cfg["prior_fingerprint"][1])
    sigma0 = float(cfg["prior_fingerprint"][2])

    selector = LearnedDynamicEtaSelector(
        artifact=artifact, sigma=sigma, mu0=mu0,
    )
    scheme = PowerLawTilting(selector=selector)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    model = NormalNormalModel(sigma=sigma)

    rng = np.random.default_rng(seed=42 + int(theta_true))
    covered = 0
    for _ in range(n_reps):
        D = rng.normal(theta_true, sigma)
        regions = scheme.confidence_regions(
            alpha, np.asarray([D]), model, prior, WaldoStatistic(),
        )
        in_ci = any(lo <= theta_true <= hi for lo, hi in regions)
        covered += int(in_ci)
    coverage = covered / n_reps

    target = 1.0 - alpha
    se = float(np.sqrt(target * (1.0 - target) / n_reps))
    # Allow 3·SE at α=0.05 (n=300 SE ≈ 0.0126; 3·SE ≈ 0.038).
    assert abs(coverage - target) < 3.0 * se, (
        f"coverage {coverage:.3f} differs from {target} by "
        f"{abs(coverage-target):.3f} > 3*SE={3*se:.3f} "
        f"at θ_true={theta_true}"
    )


@pytest.mark.L3
@pytest.mark.slow
def test_eta_pred_valid_rate_in_checkpoint():
    """The per-checkpoint ``final_eta_pred_valid_rate`` is high.

    Replaces the legacy ``calibration_report`` in-checkpoint check.
    Phase E checkpoints record this scalar instead of a 5x5 grid.
    A low rate signals an undertrained checkpoint that will trip
    the runtime safety clamp.
    """
    artifact = EtaArtifact(artifact_path=_checkpoint_path())
    artifact.load()
    rate = float(artifact.metadata["final_eta_pred_valid_rate"])
    # 0.95 is a soft floor — production checkpoints should be ≥0.99.
    assert rate >= 0.95, (
        f"final_eta_pred_valid_rate={rate:.3f} < 0.95 — checkpoint is "
        f"undertrained; the runtime clamp will fire and calibration "
        f"may fail. Train longer."
    )
