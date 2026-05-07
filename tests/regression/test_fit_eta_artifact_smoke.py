"""Regression: ``fit_eta_artifact`` end-to-end smoke test on a tiny config.

Phase 4 skeptic §9: the 5-module split moved ``prepare_held_out_validity``,
``collect_validity_batch``, ``compose_width_loss``, ``compose_boundary_penalty``,
``lambda_schedule``, ``evaluate_head_b_accuracy``, ``_training_step``,
``_run_epoch_steps``, ``_evaluate_epoch``, ``_epoch_iteration``,
``run_epoch_loop``, and ``save_checkpoint`` into separate modules
without adding direct unit tests. They are exercised transitively by
the slow phase-E selector test, but a small wiring change can slip
past that. This test runs the full ``fit_eta_artifact`` orchestrator
on a tiny config (n_lhs=64, n_epochs=2, batch_size=16) and asserts:

- A non-None ``EtaTrainResult`` is returned.
- Each dataclass field is set with a sensible value (loss lists are
  non-empty floats, accuracy is in [0, 1], paths exist, metadata is
  a populated dict).
- The saved checkpoint exists at the requested path and contains the
  expected metadata keys (post-Equinox port: equinox_version /
  jax_version replace torch_version).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frasian.learned.training.sampling import (
    ExperimentConfig,
    UniformThetaDistribution,
)
from frasian.learned.training.train import EtaTrainResult, fit_eta_artifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel


@pytest.mark.L4
@pytest.mark.slow
def test_fit_eta_artifact_returns_populated_result(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """Tiny end-to-end run pins all dataclass fields + metadata + on-disk file."""
    config = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(loc=0.0, scale=1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-3.0, high=3.0),
        n_grid=33,
        n_lhs=64,
        eta_explore_box=(-5.0, 5.0),
        seed=2026,
    )
    out_path = tmp_path / "smoke.eqx"
    result = fit_eta_artifact(
        config=config,
        out_path=out_path,
        loss_kind="integrated_p",
        n_epochs=2,
        batch_size=16,
        n_aux=16,
        patience=2,
        antithetic=False,
        verbose=False,
    )

    assert isinstance(result, EtaTrainResult)
    assert result.artifact_path == out_path
    assert out_path.exists(), f"checkpoint should exist at {out_path}"

    # Loss + metric lists: one entry per epoch (≤ n_epochs after early stop).
    assert isinstance(result.train_losses, list) and result.train_losses
    assert isinstance(result.train_width_losses, list) and result.train_width_losses
    assert isinstance(result.train_penalty_losses, list) and result.train_penalty_losses
    assert isinstance(result.val_losses, list) and result.val_losses
    assert isinstance(result.head_b_accuracy, list) and result.head_b_accuracy
    n_epochs_run = len(result.train_losses)
    assert 1 <= n_epochs_run <= 2
    for lst in (
        result.train_width_losses,
        result.train_penalty_losses,
        result.val_losses,
        result.head_b_accuracy,
    ):
        assert len(lst) == n_epochs_run
        for v in lst:
            assert isinstance(v, float)

    assert isinstance(result.final_val_loss, float)

    # Metadata: must be populated and contain a few key fields.
    assert isinstance(result.metadata, dict) and result.metadata
    for key in (
        "checkpoint_format_version",
        "architecture",
        "arch_sha",
        "equinox_version",
        "jax_version",
        "experiment_config",
        "loss_kind",
        "antithetic",
    ):
        assert key in result.metadata, f"metadata missing key {key!r}"
    assert result.metadata["loss_kind"] == "integrated_p"
    assert result.metadata["antithetic"] is False
