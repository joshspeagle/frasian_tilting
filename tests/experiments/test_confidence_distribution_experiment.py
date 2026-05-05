"""End-to-end tests for `ConfidenceDistributionExperiment`.

L4 tests: structural correctness on a tiny grid, not statistical
convergence. The L0/L1/L2 layers (cd module + constructor + distances)
already pin the underlying numerics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frasian import Config, default_cells, registry, run_experiment
from frasian.config import GridSpec
from frasian.simulation.storage import load_result
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


def _tiny_config() -> Config:
    """Tiny grid + few reps; the dynamic cell's CD construction is the
    bottleneck, so we keep n_reps small."""
    return Config.fast().from_overrides(
        n_reps=4,
        theta_grid=GridSpec("theta", -1.0, 1.0, 2),
        w_grid=GridSpec("w", 0.5, 0.5, 1),
    )


@pytest.mark.L4
class TestConfidenceDistributionExperimentEndToEnd:
    def test_runs_and_produces_manifest(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["confidence_distribution"](
            n_grid_cd=201,
        )
        tiltings, statistics = default_cells(
            experiment="confidence_distribution",
            n_grid=201,
            coarse_n=11,
        )
        cfg = _tiny_config()
        run_experiment(
            experiment=experiment,
            tiltings=tiltings,
            statistics=statistics,
            config=cfg,
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["experiment"] == "confidence_distribution"
        # 6 cells in the cross product (3 tiltings × 2 statistics);
        # the two (wald × non-identity) cells gate out as incompatible.
        ok = [c for c in manifest["cells"] if c["status"] == "ok"]
        skipped = [c for c in manifest["cells"] if c["status"] == "incompatible"]
        assert len(ok) == 4
        assert len(skipped) == 2
        for sk in skipped:
            assert sk["statistic"] == "wald"
            assert sk["tilting"].startswith("power_law") or sk["tilting"].startswith("ot")
        assert "cd_summary" in manifest["diagnostics"]
        # Diagnostic outputs.
        assert (tmp_path / "figures" / "cd_summary.png").exists()
        assert (tmp_path / "cd_summary.csv").exists()

    def test_identity_wald_cell_has_zero_w1_to_wald(self, tmp_path: Path, bootstrapped_registry):
        """The (identity, wald) cell IS the Wald CD reference. So
        W₁ to Wald is 0 (modulo numerical precision)."""
        experiment = registry.experiments["confidence_distribution"](
            n_grid_cd=201,
        )
        run_experiment(
            experiment=experiment,
            tiltings=[IdentityTilting()],
            statistics=[WaldStatistic()],
            config=_tiny_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        cell = next(
            c
            for c in manifest["cells"]
            if c["status"] == "ok" and c["statistic"] == "wald" and c["tilting"] == "identity"
        )
        result = load_result(tmp_path / cell["cache_path"])
        w1 = result.arrays["w1_to_wald_cd"]
        finite = w1[np.isfinite(w1)]
        assert finite.size > 0
        # 5e-3 since both sides go through finite-diff with kink-handling.
        assert finite.max() < 5e-3, f"W₁(Wald, Wald) should be ~0, got {finite}"

    def test_dyn_waldo_cell_records_nonmonotone_at_conflict(
        self, tmp_path: Path, bootstrapped_registry
    ):
        """The (power_law[dyn], waldo) cell at θ=±2 (so |Δ| ≈ 1) and
        w=0.5 should record some non-monotone replicates as the dynamic
        p-value's bimodal regime kicks in."""
        # Wider θ_grid to span the bimodal regime more strongly.
        cfg = Config.fast().from_overrides(
            n_reps=8,
            theta_grid=GridSpec("theta", -3.0, 3.0, 3),
            w_grid=GridSpec("w", 0.5, 0.5, 1),
        )
        dyn = PowerLawTilting(
            selector=DynamicNumericalEtaSelector(
                sigma=1.0,
                mu0=0.0,
                n_grid=201,
                coarse_n=11,
            ),
        )
        run_experiment(
            experiment=registry.experiments["confidence_distribution"](
                n_grid_cd=401,
            ),
            tiltings=[dyn],
            statistics=[WaldoStatistic()],
            config=cfg,
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        cell = next(c for c in manifest["cells"] if c["status"] == "ok")
        assert cell["tilting"] == "power_law[dynamic_numerical]"
        result = load_result(tmp_path / cell["cache_path"])
        nm = result.arrays["nonmonotone_fraction"]
        finite = nm[np.isfinite(nm)]
        # We expect ≥ one (theta_true, w) cell to record some non-monotone
        # replicates — the smoothness pathology must be detectable.
        assert (finite > 0.0).any(), (
            f"Dyn-WALDO should produce some non-monotone CDs at conflict; "
            f"got nonmonotone_fraction={nm}"
        )

    def test_all_summaries_finite_on_supported_cells(self, tmp_path: Path, bootstrapped_registry):
        """Every supported cell produces finite values for all four
        headline metrics."""
        experiment = registry.experiments["confidence_distribution"](
            n_grid_cd=201,
        )
        tiltings, statistics = default_cells(
            experiment="confidence_distribution",
            n_grid=201,
            coarse_n=11,
        )
        run_experiment(
            experiment=experiment,
            tiltings=tiltings,
            statistics=statistics,
            config=_tiny_config(),
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        for cell in manifest["cells"]:
            if cell["status"] != "ok":
                continue
            result = load_result(tmp_path / cell["cache_path"])
            for col in ("cd_median", "cd_width_95", "w1_to_wald_cd", "nonmonotone_fraction"):
                arr = result.arrays[col]
                assert np.all(np.isfinite(arr)), (
                    f"{col} has non-finite entries in {cell['tilting']} × "
                    f"{cell['statistic']}: {arr}"
                )
