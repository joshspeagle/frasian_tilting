"""L4 integration test: fit_eta_artifact diagnostics sidecar.

End-to-end check that ``fit_eta_artifact(diagnostics_out=...)`` writes
a JSON sidecar with the expected structure. Catches future regressions
in the wire-up between train.py, the per-epoch diagnostics block, and
the JSON serialization.
"""

from __future__ import annotations

import json
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


_EXPECTED_CONFIG_KEYS = {
    "scheme", "statistic", "prior_class", "model_class", "version",
    "n_epochs", "batch_size", "loss_kind", "alpha",
    "lr_a", "lr_b", "weight_decay",
    "lambda_max", "lambda_warmup_frac",
    "anti_wald_max", "anti_collapse_max", "anti_decay_frac",
    "eta_hidden_sizes", "validity_hidden_sizes", "normalize_inputs",
    "patience", "min_delta",
    "seed", "probe_n",
}

_EXPECTED_EPOCH_KEY_PREFIXES = ("d1_", "d2_", "d3_", "d4_")


@pytest.mark.L4
@pytest.mark.usefixtures("bootstrapped_registry")
def test_fit_eta_artifact_diagnostics_sidecar_smoke():
    """Run 2 epochs on a tiny config; verify sidecar shape + key set."""
    n_epochs = 2
    probe_n = 8
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
        n_lhs=128, n_grid=51, n_data=1, seed=0, name="diag_smoke",
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "diag.eqx"
        diag = Path(td) / "diag.json"
        fit_eta_artifact(
            config=config, out_path=out, loss_kind="integrated_p",
            n_epochs=n_epochs, batch_size=32, verbose=False,
            diagnostics_out=diag, probe_batch_size=probe_n,
        )
        assert diag.exists(), f"diagnostics sidecar not written: {diag}"
        with diag.open() as f:
            payload = json.load(f)

    assert set(payload.keys()) == {"config", "epochs"}
    cfg = payload["config"]
    missing = _EXPECTED_CONFIG_KEYS - set(cfg)
    assert not missing, f"missing config keys: {missing}"
    # spot-check populated values
    assert cfg["scheme"] == "power_law"
    assert cfg["statistic"] == "waldo"
    assert cfg["prior_class"] == "NormalDistribution"
    assert cfg["model_class"] == "NormalNormalModel"
    assert cfg["n_epochs"] == n_epochs
    assert cfg["probe_n"] == probe_n
    assert cfg["seed"] == 0

    epochs = payload["epochs"]
    assert len(epochs) == n_epochs, (
        f"expected {n_epochs} epoch records, got {len(epochs)}"
    )
    # Each record carries D1-D4 keyed prefixes plus the meta fields.
    for i, ep in enumerate(epochs):
        assert ep["epoch"] == i + 1
        for prefix in _EXPECTED_EPOCH_KEY_PREFIXES:
            assert any(k.startswith(prefix) for k in ep), (
                f"epoch {i+1} missing any key with prefix {prefix!r}: "
                f"{sorted(ep)}"
            )
