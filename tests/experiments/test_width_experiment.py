"""End-to-end tests for `WidthExperiment` on a small grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from frasian import Config, registry, run_experiment
from frasian.config import GridSpec
from frasian.simulation.storage import load_result


def _small_config() -> Config:
    return Config.fast().from_overrides(
        n_reps=40,
        theta_grid=GridSpec("theta", -2.0, 2.0, 4),
        w_grid=GridSpec("w", 0.3, 0.7, 3),
    )


@pytest.mark.L4
class TestWidthExperimentEndToEnd:
    def test_wald_width_is_constant(self, tmp_path: Path,
                                       bootstrapped_registry):
        experiment = registry.experiments["width"]()
        run_experiment(
            experiment=experiment,
            tiltings=registry.tiltings.implemented(),
            statistics=[registry.statistics["wald"]],
            config=_small_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        wald_cell = manifest["cells"][0]
        result = load_result(tmp_path / wald_cell["cache_path"])
        width = result.arrays["mean_width"]
        # Wald: 2 * z_{1-alpha/2} * sigma = 2 * 1.96 * 1 ≈ 3.92.
        expected = 2 * stats.norm.ppf(0.975)
        np.testing.assert_allclose(width, expected, atol=1e-9)

    def test_waldo_width_varies_with_conflict(self, tmp_path: Path,
                                                  bootstrapped_registry):
        experiment = registry.experiments["width"]()
        run_experiment(
            experiment=experiment,
            tiltings=registry.tiltings.implemented(),
            statistics=[registry.statistics["waldo"]],
            config=_small_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        cell = manifest["cells"][0]
        result = load_result(tmp_path / cell["cache_path"])
        width = result.arrays["mean_width"]
        # WALDO width should *vary* across the grid (depends on Delta and w).
        finite = width[np.isfinite(width)]
        assert finite.std() > 1e-3, "WALDO mean width should vary with grid"
