"""Phase 4d L4 smoke: end-to-end ``fit_eta_artifact`` on Bernoulli + Beta.

Marquee Phase 4 deliverable — train a learned-η selector for the
**non-Normal-Normal** ``(BernoulliModel, BetaDistribution, power_law,
waldo)`` cell. Exercises:

- ``ExperimentConfig.n_data > 1`` plumbing through
  ``sample_data_per_theta`` + ``compute_pvalues_per_sample``
  (Phase 4c-1).
- ``_admissibility_mask``'s Bernoulli branch + per-sample loop
  routing to ``power_law._generic_tilted_pvalue`` (Phase 4c-2).
- The width-loss generic-grid dispatch
  ``WIDTH_LOSS_DISPATCH[("power_law", "generic")]`` (Phase 4b).

We don't pin numerical convergence on a 2-epoch smoke — only that
``fit_eta_artifact`` runs end-to-end without raising and produces a
populated ``EtaTrainResult`` plus an on-disk checkpoint that round-
trips through ``EtaArtifact.load`` + ``LearnedDynamicEtaSelector``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frasian.learned.training.sampling import (
    ExperimentConfig,
    UniformThetaDistribution,
)
from frasian.learned.training.train import EtaTrainResult, fit_eta_artifact
from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import BetaDistribution


@pytest.mark.L4
@pytest.mark.slow
def test_fit_eta_artifact_bernoulli_smoke(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """Tiny Bernoulli + Beta end-to-end run: training completes and a
    saved checkpoint is produced."""
    config = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=BetaDistribution(alpha=2.0, beta=2.0),
        model=BernoulliModel(),
        theta_distribution=UniformThetaDistribution(low=0.05, high=0.95),
        n_grid=33,
        n_lhs=64,
        n_data=8,
        eta_explore_box=(-2.0, 2.0),
        seed=2026,
    )
    out_path = tmp_path / "bernoulli_smoke.eqx"
    result = fit_eta_artifact(
        config=config,
        out_path=out_path,
        loss_kind="integrated_p",
        n_epochs=2,
        batch_size=8,
        n_aux=8,
        patience=2,
        antithetic=False,
        verbose=False,
    )

    assert isinstance(result, EtaTrainResult)
    assert result.artifact_path == out_path
    assert out_path.exists()

    # Loss lists: one float per epoch.
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
            assert np.isfinite(v), f"non-finite metric {v!r}"

    assert isinstance(result.final_val_loss, float)
    assert np.isfinite(result.final_val_loss)

    # Metadata persists model + prior fingerprints (Bernoulli + Beta).
    md = result.metadata
    cfg_md = md["experiment_config"]
    assert tuple(cfg_md["model_fingerprint"]) == ("bernoulli",)
    assert tuple(cfg_md["prior_fingerprint"]) == ("beta", 2.0, 2.0)
    assert int(cfg_md["n_data"]) == 8


@pytest.mark.L4
@pytest.mark.slow
def test_fit_eta_artifact_bernoulli_round_trip_through_selector(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """Trained Bernoulli checkpoint loads through
    ``LearnedDynamicEtaSelector`` (Phase 4c-3) and predicts η on a
    θ-grid without raising."""
    from frasian.learned.eta_artifact import EtaArtifact
    from frasian.statistics.waldo import WaldoStatistic
    from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
    from frasian.tilting.power_law import PowerLawTilting

    config = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=BetaDistribution(alpha=2.0, beta=2.0),
        model=BernoulliModel(),
        theta_distribution=UniformThetaDistribution(low=0.05, high=0.95),
        n_grid=33,
        n_lhs=64,
        n_data=8,
        eta_explore_box=(-2.0, 2.0),
        seed=2026,
    )
    out_path = tmp_path / "bernoulli_smoke_loaded.eqx"
    fit_eta_artifact(
        config=config,
        out_path=out_path,
        loss_kind="integrated_p",
        n_epochs=2,
        batch_size=8,
        n_aux=8,
        patience=2,
        antithetic=False,
        verbose=False,
    )

    artifact = EtaArtifact(name="test", artifact_path=out_path)
    selector = LearnedDynamicEtaSelector(artifact=artifact)
    scheme = PowerLawTilting()
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)

    theta_grid = np.linspace(0.1, 0.9, 11)
    eta = selector.select_grid(
        theta_grid,
        scheme,
        statistic=WaldoStatistic(),
        model=model,
        prior=prior,
        alpha=0.05,
    )
    assert eta.shape == (11,)
    assert np.all(np.isfinite(eta))
