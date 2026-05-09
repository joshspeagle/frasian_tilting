"""L2 regression: ExperimentConfig v4 schema (Phase G).

Pins the new YAML keys (prior_class, model_class, hyperparam_distribution)
and the required-field discipline.
"""

from __future__ import annotations

import pytest

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.training.sampling import ExperimentConfig
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel


@pytest.mark.L2
@pytest.mark.usefixtures("bootstrapped_registry")
class TestExperimentConfigV4FromDict:
    def _example_dict(self) -> dict:
        return {
            "prior_class": "normal",
            "model_class": "normal_normal",
            "hyperparam_distribution": {
                "prior": {
                    "loc":   {"dist": "uniform",    "low": -2.0, "high": 2.0},
                    "scale": {"dist": "loguniform", "low": 0.2,  "high": 5.0},
                },
                "lik": {
                    "sigma": {"dist": "loguniform", "low": 0.5, "high": 2.0},
                },
            },
            "theta_distribution": {
                "type": "uniform", "low": -5.0, "high": 5.0,
            },
            "scheme": "power_law",
            "statistic": "waldo",
            "n_grid": 401,
            "n_lhs": 10000,
            "seed": 42,
            "name": "test_v4",
            "description": "test",
            "n_data": 1,
        }

    def test_from_dict_constructs(self):
        cfg = ExperimentConfig.from_dict(self._example_dict())
        assert cfg.prior_cls is NormalDistribution
        assert cfg.model_cls is NormalNormalModel
        assert cfg.hyperparam_distribution.prior_specs.keys() == {"loc", "scale"}
        assert cfg.hyperparam_distribution.lik_specs.keys() == {"sigma"}

    def test_old_v3_keys_rejected(self):
        bad = self._example_dict()
        bad["prior"] = {"type": "normal", "loc": 0.0, "scale": 1.0}  # v3 key
        del bad["prior_class"]
        with pytest.raises((KeyError, ValueError)):
            ExperimentConfig.from_dict(bad)

    def test_unknown_prior_class_raises(self):
        bad = self._example_dict()
        bad["prior_class"] = "not_a_real_prior"
        with pytest.raises(ValueError, match="prior_class"):
            ExperimentConfig.from_dict(bad)

    def test_to_dict_roundtrip(self):
        d = self._example_dict()
        cfg = ExperimentConfig.from_dict(d)
        d2 = cfg.to_dict()
        assert d2["prior_class"] == "normal"
        assert d2["model_class"] == "normal_normal"
        assert "hyperparam_distribution" in d2
        cfg2 = ExperimentConfig.from_dict(d2)
        assert cfg2.fingerprint() == cfg.fingerprint()
