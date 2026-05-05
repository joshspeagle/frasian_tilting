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
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


def _small_config() -> Config:
    return Config.fast().from_overrides(
        n_reps=40,
        theta_grid=GridSpec("theta", -2.0, 2.0, 4),
        w_grid=GridSpec("w", 0.3, 0.7, 3),
    )


@pytest.mark.L4
class TestWidthExperimentEndToEnd:
    def test_wald_width_is_constant(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["width"]()
        run_experiment(
            experiment=experiment,
            tiltings=[IdentityTilting()],
            statistics=[WaldStatistic()],
            config=_small_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        ok_cells = [c for c in manifest["cells"] if c["status"] == "ok"]
        result = load_result(tmp_path / ok_cells[0]["cache_path"])
        width = result.arrays["mean_width"]
        # Wald: 2 * z_{1-alpha/2} * sigma = 2 * 1.96 * 1 ≈ 3.92.
        expected = 2 * stats.norm.ppf(0.975)
        np.testing.assert_allclose(width, expected, atol=1e-9)

    def test_waldo_width_varies_with_conflict(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["width"]()
        run_experiment(
            experiment=experiment,
            tiltings=[IdentityTilting()],
            statistics=[WaldoStatistic()],
            config=_small_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        cell = [c for c in manifest["cells"] if c["status"] == "ok"][0]
        result = load_result(tmp_path / cell["cache_path"])
        width = result.arrays["mean_width"]
        # WALDO width should *vary* across the grid (depends on Delta and w).
        finite = width[np.isfinite(width)]
        assert finite.std() > 1e-3, "WALDO mean width should vary with grid"

    def test_dynamic_waldo_cell_runs(self, tmp_path: Path, bootstrapped_registry):
        """The (power_law[dynamic_numerical], waldo) cell — i.e.
        Dynamic-WALDO — runs end-to-end through the uniform CI interface
        and produces a finite-width surface. This is the regression test
        that the Phase-4 dynamic_ci removal preserved the Dynamic-WALDO
        measurement on the coverage/width surface.
        """
        cfg = Config.fast().from_overrides(
            n_reps=4,
            theta_grid=GridSpec("theta", -1.0, 1.0, 2),
            w_grid=GridSpec("w", 0.5, 0.5, 1),
        )
        dyn = PowerLawTilting(
            selector=DynamicNumericalEtaSelector(n_grid=81, coarse_n=9),
        )
        run_experiment(
            experiment=registry.experiments["width"](),
            tiltings=[dyn],
            statistics=[WaldoStatistic()],
            config=cfg,
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        cell = [c for c in manifest["cells"] if c["status"] == "ok"][0]
        assert cell["tilting"] == "power_law[dynamic_numerical]"
        result = load_result(tmp_path / cell["cache_path"])
        widths = result.arrays["mean_width"]
        finite = widths[np.isfinite(widths)]
        # Dynamic CI must produce positive widths.
        assert finite.size > 0
        assert (finite > 0).all()
        # Multi-region uplift: `mean_n_regions` is now a stored array.
        # On this tiny grid at α=0.05 we expect single-region everywhere
        # (the bimodal regime is at high α; see test_confidence_regions).
        assert "mean_n_regions" in result.arrays
        n_reg = result.arrays["mean_n_regions"]
        n_reg_finite = n_reg[np.isfinite(n_reg)]
        assert (n_reg_finite >= 1.0).all()
