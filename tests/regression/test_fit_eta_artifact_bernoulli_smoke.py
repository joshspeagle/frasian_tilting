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

Both test bodies share a single trained checkpoint via the
``_bernoulli_smoke_artifact`` module-scoped fixture — running two
independent trainings here is wasteful (each takes ~30 s post-Phase-4
validity-fast-path) and was the dominant cost of the L4 sweep.
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


def _make_smoke_config() -> ExperimentConfig:
    return ExperimentConfig(
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


@pytest.fixture(scope="module")
def _bernoulli_smoke_artifact(tmp_path_factory):
    """Train one Bernoulli checkpoint and reuse across both L4 tests.

    Module-scoped so the heavy fit_eta_artifact only runs once even
    though two tests in this module use it. Saves ~30 s of L4 wall-time.

    The registry must be bootstrapped at module-fixture build time
    because ``fit_eta_artifact`` looks up the scheme + statistic via
    ``_registry``. ``conftest.py``'s ``bootstrapped_registry`` is
    function-scoped (autouse ``_isolated_registry`` clears between
    tests), so we can't request it here — we bootstrap inline. The
    trained checkpoint persists to disk; subsequent tests load via
    ``EtaArtifact`` which doesn't touch the registry.
    """
    from frasian._registry_bootstrap import bootstrap

    bootstrap()
    out_dir = tmp_path_factory.mktemp("bernoulli_smoke")
    out_path = out_dir / "bernoulli_smoke.eqx"
    config = _make_smoke_config()
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
    return out_path, result


@pytest.mark.L4
@pytest.mark.slow
def test_fit_eta_artifact_bernoulli_smoke(
    _bernoulli_smoke_artifact: tuple[Path, EtaTrainResult],
) -> None:
    """Tiny Bernoulli + Beta end-to-end run: training completes and a
    saved checkpoint is produced."""
    out_path, result = _bernoulli_smoke_artifact

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
    _bernoulli_smoke_artifact: tuple[Path, EtaTrainResult],
) -> None:
    """Trained Bernoulli checkpoint loads through
    ``LearnedDynamicEtaSelector`` (Phase 4c-3) and predicts η on a
    θ-grid without raising."""
    from frasian.learned.eta_artifact import EtaArtifact
    from frasian.statistics.waldo import WaldoStatistic
    from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
    from frasian.tilting.power_law import PowerLawTilting

    out_path, _ = _bernoulli_smoke_artifact

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
