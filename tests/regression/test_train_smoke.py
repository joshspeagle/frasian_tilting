"""Smoke test: training loop runs end-to-end and produces a usable artifact.

Tiny budgets (n_lhs=64, n_epochs=5). Marked `slow`. Skipped if torch
unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.learned.monotonic_eta import MonotonicEtaArtifact
from frasian.learned.training.sampling import TrainingDistribution
from frasian.learned.training.train import fit_monotonic_eta_artifact
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L2
@pytest.mark.slow
def test_train_smoke_round_trip(tmp_path):
    """Train a tiny MLP, save, reload, verify η values are sane."""
    out_path = tmp_path / "smoke.pt"
    result = fit_monotonic_eta_artifact(
        scheme_name="power_law",
        statistic_name="waldo",
        loss_kind="integrated_p",
        n_lhs=64,
        n_mc=2,
        n_epochs=5,
        batch_size=16,
        theta_grid_n=51,
        out_path=out_path,
        seed=42,
        verbose=False,
    )
    assert result.artifact_path == out_path
    assert out_path.exists()
    assert len(result.train_losses) == 5
    assert len(result.val_losses) == 5
    # Loss should at least not blow up.
    assert np.isfinite(result.final_val_loss)
    assert result.final_val_loss < 100.0

    # Reload via artifact, verify metadata.
    artifact = MonotonicEtaArtifact(artifact_path=out_path)
    artifact.load()
    meta = artifact.metadata
    assert meta["scheme"] == "power_law"
    assert meta["loss"] == "integrated_p"
    assert meta["alpha_mode"] == "marginalised"
    assert meta["checkpoint_format_version"] == 1

    # Predict via selector, verify η values are in admissible range
    # for power_law: η_min(w) = -w/(1-w), η_max = 1.
    selector = LearnedDynamicEtaSelector(artifact=artifact)
    scheme = PowerLawTilting()
    ad_grid = np.linspace(0.0, 5.0, 11)
    eta = selector.select_grid(
        ad_grid, scheme, statistic=WaldoStatistic(), w=0.5, alpha=0.05,
    )
    assert eta.shape == (11,)
    # η in (-1, 1) at w=0.5 since η_min(0.5) = -1; bounded sigmoid keeps
    # η' ∈ [0.01, 0.99] which maps to η ∈ (-0.98, 0.98).
    assert np.all(eta > -1.0), f"η below η_min: {eta}"
    assert np.all(eta < 1.0), f"η above 1: {eta}"


@pytest.mark.L2
@pytest.mark.slow
def test_train_smoke_static_width_alpha_required(tmp_path):
    """static_width loss requires --alpha; integrated_p forbids it."""
    with pytest.raises(ValueError, match="static_width"):
        fit_monotonic_eta_artifact(
            scheme_name="power_law", loss_kind="static_width", alpha=None,
            n_lhs=8, n_mc=1, n_epochs=1, theta_grid_n=11,
            out_path=tmp_path / "x.pt", verbose=False,
        )
    with pytest.raises(ValueError, match="α-marginalised"):
        fit_monotonic_eta_artifact(
            scheme_name="power_law", loss_kind="integrated_p", alpha=0.05,
            n_lhs=8, n_mc=1, n_epochs=1, theta_grid_n=11,
            out_path=tmp_path / "y.pt", verbose=False,
        )


@pytest.mark.L2
@pytest.mark.slow
@pytest.mark.parametrize("scheme,loss,alpha", [
    ("power_law", "integrated_p", None),
    ("power_law", "cd_variance",  None),
    ("power_law", "static_width", 0.05),
    ("ot",        "integrated_p", None),
    ("ot",        "cd_variance",  None),
])
def test_train_smoke_all_loss_scheme_combos(tmp_path, scheme, loss, alpha):
    """Each (scheme, loss) combination trains end-to-end at tiny budgets.

    Adds breadth that the original `test_train_smoke_round_trip` (only
    `(power_law, integrated_p)`) lacked. Phase-C skeptic finding #6.
    """
    out_path = tmp_path / f"smoke_{scheme}_{loss}.pt"
    result = fit_monotonic_eta_artifact(
        scheme_name=scheme,
        statistic_name="waldo",
        loss_kind=loss,
        alpha=alpha,
        n_lhs=64,
        n_mc=2,
        n_epochs=3,
        batch_size=16,
        theta_grid_n=51,
        out_path=out_path,
        seed=42,
        verbose=False,
    )
    assert out_path.exists()
    assert np.isfinite(result.final_val_loss)
    # Verify metadata records the right combo.
    artifact = MonotonicEtaArtifact(artifact_path=out_path)
    artifact.load()
    meta = artifact.metadata
    assert meta["scheme"] == scheme
    assert meta["loss"] == loss
    if alpha is None:
        assert meta["alpha_mode"] == "marginalised"
    else:
        assert meta["alpha_mode"] == f"fixed_{alpha:.6g}"


@pytest.mark.L2
@pytest.mark.slow
def test_calibration_report_present_in_checkpoint(tmp_path):
    """Phase-C skeptic block #1: calibration_report must be in checkpoint."""
    out_path = tmp_path / "calib_check.pt"
    fit_monotonic_eta_artifact(
        scheme_name="power_law", loss_kind="integrated_p",
        n_lhs=64, n_mc=2, n_epochs=3, batch_size=16, theta_grid_n=51,
        out_path=out_path, seed=42, verbose=False,
    )
    artifact = MonotonicEtaArtifact(artifact_path=out_path)
    artifact.load()
    report = artifact.metadata["calibration_report"]
    assert "alphas" in report and len(report["alphas"]) > 0
    assert "theta_true_grid" in report
    assert "w_grid" in report
    assert "coverage" in report
    cov = np.array(report["coverage"])
    n_theta = len(report["theta_true_grid"])
    n_w = len(report["w_grid"])
    n_alpha = len(report["alphas"])
    assert cov.shape == (n_theta, n_w, n_alpha), cov.shape
    # All coverage values in [0, 1].
    assert np.all((cov >= 0.0) & (cov <= 1.0))
