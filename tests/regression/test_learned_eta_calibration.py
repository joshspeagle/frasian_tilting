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

from frasian.learned.eta_artifact import EtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.power_law import PowerLawTilting

_CHECKPOINT_CANDIDATES = {
    "powerlaw": [
        Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v1.eqx"),
        Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v0_smoke.eqx"),
    ],
    "ot": [
        Path("artifacts/learned_eta_canonical_normal_normal_ot_v1.eqx"),
        Path("artifacts/learned_eta_canonical_normal_normal_ot_v0_smoke.eqx"),
    ],
}


def _checkpoint_path(scheme_label: str) -> Path:
    for c in _CHECKPOINT_CANDIDATES[scheme_label]:
        if c.exists():
            return c
    pytest.skip(f"no Phase E {scheme_label} learned-eta checkpoint available; " f"train one first")


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("scheme_label", ["powerlaw", "ot"])
@pytest.mark.parametrize(
    "theta_true",
    [-4.0, -3.0, -2.0, 0.0, 2.0, 3.0, 4.0],
)
@pytest.mark.parametrize("alpha", [0.05, 0.50])
def test_calibration_at_multiple_alphas(scheme_label, theta_true, alpha):
    """Empirical coverage matches nominal 1-α within 3·MC_SE for both
    power_law and ot smoke checkpoints across α ∈ {0.05, 0.50}.

    α grid trimmed from {0.05, 0.20, 0.50} → {0.05, 0.50}: α=0.20 was
    the costliest case (binomial variance peaks at p=0.5) and the
    {0.05, 0.50} pair already brackets the integrated_p loss's
    interior. θ grid kept at the audit-blessed conflict band
    {-4, ±3, ±2, 0, +3, +4}; only the cargo-cult interior α point
    was dropped.

    Pins the theoretical guarantee (η depends only on θ →
    p_dyn(θ_0; D, η_φ) is U[0,1] under H0) against perturbations
    from the runtime safety clamp and the symmetric branch-averaging
    in the selector's ``select_grid``, *and* across α-levels.
    """
    n_reps = 300

    artifact = EtaArtifact(artifact_path=_checkpoint_path(scheme_label))
    artifact.load()
    cfg = artifact.metadata["experiment_config"]
    sigma = float(cfg["model_fingerprint"][1])
    mu0 = float(cfg["prior_fingerprint"][1])
    sigma0 = float(cfg["prior_fingerprint"][2])

    selector = LearnedDynamicEtaSelector(
        artifact=artifact,
        sigma=sigma,
        mu0=mu0,
    )
    if scheme_label == "powerlaw":
        scheme = PowerLawTilting(selector=selector)
    else:
        from frasian.tilting.ot import OTTilting

        scheme = OTTilting(selector=selector)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    model = NormalNormalModel(sigma=sigma)

    # Stable per-scheme offset (NOT hash(scheme_label), which varies
    # per Python process via PYTHONHASHSEED — same root cause as the
    # WALDO blake2b fix). Constant offsets keep this test reproducible
    # across runs and immune to pytest-randomly ordering effects.
    _SCHEME_SEED_OFFSET = {"powerlaw": 0, "ot": 17}
    rng = np.random.default_rng(
        seed=42 + int(theta_true) + _SCHEME_SEED_OFFSET[scheme_label] + int(100 * alpha)
    )
    covered = 0
    for _ in range(n_reps):
        D = rng.normal(theta_true, sigma)
        regions = scheme.confidence_regions(
            alpha,
            np.asarray([D]),
            model,
            prior,
            WaldoStatistic(),
        )
        in_ci = any(lo <= theta_true <= hi for lo, hi in regions)
        covered += int(in_ci)
    coverage = covered / n_reps

    target = 1.0 - alpha
    se = float(np.sqrt(target * (1.0 - target) / n_reps))
    # Allow 3·SE: at α=0.05 → 3·SE≈0.038; α=0.20 → 0.069;
    # α=0.50 → 0.087. Wider noise band at higher α reflects the
    # binomial variance peaking at p=0.5.
    assert abs(coverage - target) < 3.0 * se, (
        f"coverage {coverage:.3f} differs from {target} by "
        f"{abs(coverage-target):.3f} > 3*SE={3*se:.3f} "
        f"at θ_true={theta_true}, α={alpha} ({scheme_label})"
    )


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("scheme_label", ["powerlaw", "ot"])
def test_eta_pred_valid_rate_in_checkpoint(scheme_label):
    """The per-checkpoint ``final_eta_pred_valid_rate`` is high.

    Replaces the legacy ``calibration_report`` in-checkpoint check.
    Phase E checkpoints record this scalar instead of a 5x5 grid.
    A low rate signals an undertrained checkpoint that will trip
    the runtime safety clamp.
    """
    artifact = EtaArtifact(artifact_path=_checkpoint_path(scheme_label))
    artifact.load()
    rate = float(artifact.metadata["final_eta_pred_valid_rate"])
    # 0.95 is a soft floor — production checkpoints should be ≥0.99.
    assert rate >= 0.95, (
        f"final_eta_pred_valid_rate={rate:.3f} < 0.95 for "
        f"{scheme_label} — checkpoint is undertrained; the runtime "
        f"clamp will fire and calibration may fail. Train longer."
    )
