"""End-to-end tests for `CoverageExperiment` on a small grid.

These are L4 tests: structural correctness on a tiny grid, not
statistical convergence. The L3 calibration check (Wald uniform p-values
under H0) lives in `tests/properties/test_wald_invariants.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frasian import Config, default_cells, registry, run_experiment
from frasian.config import GridSpec
from frasian.statistics.wald import WaldStatistic
from frasian.tilting.identity import IdentityTilting


def _small_config() -> Config:
    """Used by the identity-only Wald sub-test (cheap CI path)."""
    return Config.fast().from_overrides(
        n_reps=80,
        theta_grid=GridSpec("theta", -2.0, 2.0, 5),
        w_grid=GridSpec("w", 0.2, 0.8, 4),
    )


def _dynamic_friendly_config() -> Config:
    """For tests that exercise the (power_law[dynamic], waldo) cell.

    Dynamic CI inversion runs an η* lookup + scan per D sample, which is
    hundreds of times more expensive than the static path; we shrink the
    grid + n_reps so the test stays within a few seconds.
    """
    return Config.fast().from_overrides(
        n_reps=4,
        theta_grid=GridSpec("theta", -1.0, 1.0, 2),
        w_grid=GridSpec("w", 0.5, 0.5, 1),
    )


@pytest.mark.L4
class TestCoverageExperimentEndToEnd:
    def test_runs_and_produces_manifest(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["coverage"]()
        tiltings, statistics = default_cells(n_grid=81, coarse_n=9)
        cfg = _dynamic_friendly_config()
        run_experiment(
            experiment=experiment,
            tiltings=tiltings,
            statistics=statistics,
            config=cfg,
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["experiment"] == "coverage"
        assert manifest["config_fingerprint"] == cfg.fingerprint()
        # 6 cells in the cross-product (3 tiltings × 2 statistics); the two
        # (wald × non-identity) cells gate out as incompatible. Four run.
        ok = [c for c in manifest["cells"] if c["status"] == "ok"]
        skipped = [c for c in manifest["cells"] if c["status"] == "incompatible"]
        assert len(ok) == 4
        assert len(skipped) == 2
        for sk in skipped:
            assert sk["statistic"] == "wald"
            assert sk["tilting"].startswith("power_law") or sk["tilting"].startswith("ot")
        assert "coverage_rate" in manifest["diagnostics"]
        # Cache paths are relative to the manifest directory.
        for cell in ok:
            cache_path = tmp_path / cell["cache_path"]
            assert cache_path.exists(), cache_path
        # Figure file produced.
        fig_path = tmp_path / "figures" / "coverage_rate.png"
        assert fig_path.exists()
        # Tidy CSV produced.
        csv_path = tmp_path / "coverage_rate.csv"
        assert csv_path.exists()

    def test_wald_cell_has_full_grid_coverage(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["coverage"]()
        run_experiment(
            experiment=experiment,
            tiltings=[IdentityTilting()],
            statistics=[WaldStatistic()],
            config=_small_config(),
            out_dir=tmp_path,
        )
        # Load the Wald cell's arrays directly.
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        wald_cell = next(
            c for c in manifest["cells"] if c["statistic"] == "wald" and c["status"] == "ok"
        )
        from frasian.simulation.storage import load_result

        result = load_result(tmp_path / wald_cell["cache_path"])
        cov = result.arrays["coverage"]
        assert cov.shape == (5, 4)
        # Wald is independent of w: each row should have nearly constant
        # coverage across columns (tolerance proportional to MC noise).
        for i in range(cov.shape[0]):
            assert (
                cov[i].max() - cov[i].min()
            ) <= 0.25, f"wald coverage varies with w at row {i}: {cov[i]}"
        # Empirical coverage near 95% (n_reps=80, expect ~5% noise).
        np.testing.assert_allclose(cov.mean(), 0.95, atol=0.1)

    def test_byte_reproducible_at_same_inputs(self, tmp_path: Path, bootstrapped_registry):
        """Two runs with the same Config + same git-sha produce identical
        cells and manifests *modulo* the figures (matplotlib timestamps)."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        cfg = _dynamic_friendly_config()
        experiment = registry.experiments["coverage"]()
        tiltings, statistics = default_cells(n_grid=81, coarse_n=9)
        run_experiment(
            experiment=experiment, tiltings=tiltings, statistics=statistics, config=cfg, out_dir=a
        )
        # Reconstruct fresh instances (frozen dataclasses are hashable but
        # we want a clean state for the second run).
        tiltings2, statistics2 = default_cells(n_grid=81, coarse_n=9)
        run_experiment(
            experiment=experiment, tiltings=tiltings2, statistics=statistics2, config=cfg, out_dir=b
        )
        m_a = json.loads((a / "manifest.json").read_text())
        m_b = json.loads((b / "manifest.json").read_text())
        # Strip the figures field (matplotlib PNGs are non-deterministic byte-
        # for-byte due to embedded timestamps). Cells use relative paths so
        # they should match.
        m_a.pop("figures")
        m_b.pop("figures")
        assert m_a == m_b
