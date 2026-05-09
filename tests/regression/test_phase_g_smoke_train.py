"""Phase G smoke train: end-to-end with conditional architecture."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.training.hyperparam_distribution import (
    HyperparamDistribution, ScalarDist,
)
from frasian.learned.training.sampling import (
    ExperimentConfig, UniformThetaDistribution,
)
from frasian.learned.training.train import fit_eta_artifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel


@pytest.mark.L2
@pytest.mark.usefixtures("bootstrapped_registry")
def test_smoke_train_conditional_nn_powerlaw():
    config = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior_cls=NormalDistribution,
        model_cls=NormalNormalModel,
        hyperparam_distribution=HyperparamDistribution(
            prior_specs={
                "loc":   ScalarDist("uniform", -2.0, 2.0),
                "scale": ScalarDist("loguniform", 0.2, 5.0),
            },
            lik_specs={
                "sigma": ScalarDist("loguniform", 0.5, 2.0),
            },
        ),
        theta_distribution=UniformThetaDistribution(low=-5.0, high=5.0),
        n_lhs=512, n_data=1, seed=0, name="smoke",
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "smoke.eqx"
        res = fit_eta_artifact(
            config=config, out_path=out, loss_kind="integrated_p",
            n_epochs=3, batch_size=64, verbose=False,
        )
    assert all(0.0 < l < 100.0 for l in res.train_losses)
    assert res.metadata["final_head_b_accuracy"] > 0.5
