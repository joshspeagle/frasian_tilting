"""End-to-end test for DynamicCIExperiment on a tiny grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from frasian import Config, registry, run_experiment
from frasian.config import GridSpec
from frasian.simulation.storage import load_result


def _tiny_config() -> Config:
    return Config.fast().from_overrides(
        n_reps=4,
        theta_grid=GridSpec("theta", -1.0, 1.0, 2),
        w_grid=GridSpec("w", 0.5, 0.5, 1),
    )


@pytest.mark.L4
class TestDynamicCIExperimentEndToEnd:
    def test_runs_and_produces_manifest(self, tmp_path: Path,
                                          bootstrapped_registry):
        # Override n_grid/coarse_n to keep the test fast.
        experiment = registry.experiments["dynamic_ci"](
            n_grid=81, coarse_n=9,
        )
        run_experiment(
            experiment=experiment,
            tiltings=registry.tiltings.implemented(),
            statistics=registry.statistics.implemented(),
            config=_tiny_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["experiment"] == "dynamic_ci"
        assert len(manifest["cells"]) == 2
        assert "dynamic_ci" in manifest["diagnostics"]
        assert (tmp_path / "figures" / "dynamic_ci.png").exists()
        assert (tmp_path / "dynamic_ci.csv").exists()

    def test_wald_cell_gives_static_wald_results(self, tmp_path: Path,
                                                    bootstrapped_registry):
        experiment = registry.experiments["dynamic_ci"](
            n_grid=81, coarse_n=9,
        )
        run_experiment(
            experiment=experiment,
            tiltings=registry.tiltings.implemented(),
            statistics=[registry.statistics["wald"]],
            config=_tiny_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        cell = manifest["cells"][0]
        result = load_result(tmp_path / cell["cache_path"])
        widths = result.arrays["mean_width"]
        regions = result.arrays["mean_regions"]
        z = stats.norm.ppf(0.975)
        # Wald width must equal 2*z*sigma exactly (not approximately).
        finite = widths[np.isfinite(widths)]
        np.testing.assert_allclose(finite, 2 * z, atol=0.03)
        # All regions == 1.
        np.testing.assert_array_equal(regions, np.ones_like(regions))

    def test_csv_columns(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["dynamic_ci"](
            n_grid=81, coarse_n=9,
        )
        run_experiment(
            experiment=experiment,
            tiltings=registry.tiltings.implemented(),
            statistics=[registry.statistics["wald"]],
            config=_tiny_config(),
            out_dir=tmp_path,
        )
        df = pd.read_csv(tmp_path / "dynamic_ci.csv")
        for col in ("coverage", "coverage_se", "mean_width",
                     "width_se", "mean_regions"):
            assert col in df.columns, f"missing {col}"
