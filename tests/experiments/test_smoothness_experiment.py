"""End-to-end tests for SmoothnessExperiment.

Verifies the load-bearing claim of Step 5: that for the (power_law, waldo)
cell, the diagnostic *detects* the discontinuity (positive Lipschitz, TV,
discontinuity count) while for (power_law, wald) it is approximately zero.
This is the framework's "is the central pathology actually visible?"
gating test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frasian import Config, default_smoothness_tiltings, registry, run_experiment
from frasian.config import GridSpec
from frasian.simulation.storage import load_result
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic


def _smoothness_config() -> Config:
    return Config.fast().from_overrides(
        delta_grid=GridSpec("abs_delta", 0.0, 5.0, 51),
    )


@pytest.mark.L4
class TestSmoothnessExperimentEndToEnd:
    def test_runs_and_produces_manifest(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["smoothness"]()
        run_experiment(
            experiment=experiment,
            tiltings=default_smoothness_tiltings(),
            statistics=[WaldStatistic(), WaldoStatistic()],
            config=_smoothness_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["experiment"] == "smoothness"
        # 3 tiltings × 2 statistics = 6 cells; (wald × power_law) and
        # (wald × ot) gated out as incompatible.
        ok = [c for c in manifest["cells"] if c["status"] == "ok"]
        skipped = [c for c in manifest["cells"] if c["status"] == "incompatible"]
        assert len(ok) == 4
        assert len(skipped) == 2
        assert "smoothness" in manifest["diagnostics"]
        assert (tmp_path / "figures" / "smoothness_metrics.png").exists()
        assert (tmp_path / "smoothness.csv").exists()

    def test_waldo_cell_detects_discontinuity(self, tmp_path: Path, bootstrapped_registry):
        """The framework's central diagnostic: (power_law, waldo) must show
        a high Lipschitz, positive TV, and >0 discontinuity count."""
        experiment = registry.experiments["smoothness"]()
        run_experiment(
            experiment=experiment,
            tiltings=default_smoothness_tiltings(),
            statistics=[WaldoStatistic()],
            config=_smoothness_config(),
            out_dir=tmp_path,
        )
        df = pd.read_csv(tmp_path / "smoothness.csv")
        # Two waldo rows now: identity (constant η=0) and power_law (the
        # one that should detect the discontinuity). Filter on power_law.
        sub = df[(df["statistic"] == "waldo") & (df["tilting"] == "power_law")]

        lip = float(sub[sub["metric"] == "lipschitz_eta"]["value"].iloc[0])
        tv = float(sub[sub["metric"] == "total_variation_eta"]["value"].iloc[0])
        disc = float(sub[sub["metric"] == "discontinuity_count_eta"]["value"].iloc[0])

        # The kink near |Delta| ~ 0.3-0.7 should produce a Lipschitz spike;
        # the empirical demo run shows ~17 at 51 grid points.
        assert lip > 1.0, f"Lipschitz too small: {lip}"
        # Total variation: eta* ranges from ~-0.998 (clamp) to ~+0.99 (Wald),
        # so TV >~ 2.0 with some additional jitter.
        assert tv > 1.5, f"TV too small: {tv}"
        # At least one outlier 2nd-difference at the kink.
        assert disc >= 1, f"discontinuity_count too low: {disc}"

    def test_identity_wald_cell_is_smooth_baseline(self, tmp_path: Path, bootstrapped_registry):
        """The (identity, wald) cell is the smoothness floor: ~0 Lipschitz,
        ~0 TV, 0 discontinuities — η is constant 0 by construction."""
        experiment = registry.experiments["smoothness"]()
        from frasian.tilting.identity import IdentityTilting

        run_experiment(
            experiment=experiment,
            tiltings=[IdentityTilting()],
            statistics=[WaldStatistic()],
            config=_smoothness_config(),
            out_dir=tmp_path,
        )
        df = pd.read_csv(tmp_path / "smoothness.csv")
        sub = df[df["statistic"] == "wald"]
        lip = float(sub[sub["metric"] == "lipschitz_eta"]["value"].iloc[0])
        tv = float(sub[sub["metric"] == "total_variation_eta"]["value"].iloc[0])
        disc = float(sub[sub["metric"] == "discontinuity_count_eta"]["value"].iloc[0])
        assert lip < 1e-3
        assert tv < 1e-3
        assert disc == 0

    def test_eta_star_within_admissible_range(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["smoothness"]()
        run_experiment(
            experiment=experiment,
            tiltings=default_smoothness_tiltings(),
            statistics=[WaldoStatistic()],
            config=_smoothness_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        # Pick the power_law cell — identity records constant eta=0.
        cell = next(
            c
            for c in manifest["cells"]
            if c["status"] == "ok" and c["tilting"].startswith("power_law")
        )
        result = load_result(tmp_path / cell["cache_path"])
        eta = result.arrays["eta_star"]
        finite = eta[np.isfinite(eta)]
        assert finite.size > 0
        # admissible_range for w=0.5: (-1 + buffer, 2 - buffer) approximately.
        assert finite.min() > -1.0
        assert finite.max() < 1.001  # capped at 1 - buffer in the selector
