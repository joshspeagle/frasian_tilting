"""Headline test 1: empirical calibration of the learned dynamic-η selector.

For each (θ_true, w, α) cell, simulate `n_reps` D draws, build the
dynamic CI via `(power_law[learned], waldo).confidence_regions`, and
check empirical coverage hits the nominal level within MC noise.

Calibration is automatic (η depends only on θ, not D), but this is
the empirical safety net. Loads the v0_smoke checkpoint as a CI
fixture; for finer regression evidence, run with the v1 checkpoint
loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.learned.monotonic_eta import MonotonicEtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.power_law import PowerLawTilting


_CHECKPOINT_CANDIDATES = [
    Path("artifacts/learned_eta_power_law_v1.pt"),
    Path("artifacts/learned_eta_power_law_v0_smoke.pt"),
]


def _checkpoint_path() -> Path:
    for c in _CHECKPOINT_CANDIDATES:
        if c.exists():
            return c
    pytest.skip("no learned-eta checkpoint available; train one first")


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("theta_true", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("w_val", [0.3, 0.5, 0.7])
def test_calibration_at_alpha_05(theta_true, w_val):
    """At α=0.05, empirical coverage matches nominal 0.95 within 3·MC_SE."""
    sigma, mu0, alpha = 1.0, 0.0, 0.05
    n_reps = 300

    artifact = MonotonicEtaArtifact(artifact_path=_checkpoint_path())
    selector = LearnedDynamicEtaSelector(artifact=artifact)
    scheme = PowerLawTilting(selector=selector)

    sigma0 = float(np.sqrt(w_val / max(1.0 - w_val, 1e-9)) * sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    model = NormalNormalModel(sigma=sigma)

    rng = np.random.default_rng(seed=42 + int(100 * w_val) + int(theta_true))
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
        f"at θ_true={theta_true}, w={w_val}"
    )


@pytest.mark.L3
@pytest.mark.slow
def test_calibration_report_in_checkpoint_within_noise():
    """The pre-computed calibration_report in the checkpoint passes."""
    artifact = MonotonicEtaArtifact(artifact_path=_checkpoint_path())
    artifact.load()
    report = artifact.metadata["calibration_report"]
    cov = np.array(report["coverage"])           # (n_θ, n_w, n_α)
    alphas = report["alphas"]
    n_reps = report["n_reps"]
    for k, alpha in enumerate(alphas):
        target = 1.0 - alpha
        se = float(np.sqrt(target * (1.0 - target) / n_reps))
        max_err = float(np.abs(cov[..., k] - target).max())
        # Each cell within 4·SE (loose; slow tests below pin 3·SE per cell).
        assert max_err < 4.0 * se, (
            f"alpha={alpha}: max |cov-target|={max_err:.4f} > "
            f"4·SE={4*se:.4f}"
        )
