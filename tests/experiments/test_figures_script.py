"""End-to-end test for `scripts/figures.py`: regenerate from results dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frasian import default_cells, registry, run_experiment
from frasian.config import Config, GridSpec


def _small_config() -> Config:
    """Smallest config that still produces a figure-able coverage table.

    n_reps=3, 2×1 grid → ~6 CIs per cell × ~5 cells = ~30 CIs total.
    The test only checks that ``regenerate`` produces a PNG and round-
    trips the manifest; it doesn't pin coverage values or dimensions
    beyond the file existing, so the smallest grid that still yields a
    non-empty table is correct here.
    """
    return Config.fast().from_overrides(
        n_reps=3,
        theta_grid=GridSpec("theta", -1.0, 1.0, 2),
        w_grid=GridSpec("w", 0.5, 0.5, 1),
    )


@pytest.mark.L4
class TestFiguresScript:
    def test_regenerate_from_results(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["coverage"]()
        # Smaller dynamic-CI scan grids (defaults are 81/9). The figures
        # smoke doesn't need the production-grade resolution; it only
        # needs the cells to populate so the table + PNG can be drawn.
        tiltings, statistics = default_cells(n_grid=21, coarse_n=5)
        run_experiment(
            experiment=experiment,
            tiltings=tiltings,
            statistics=statistics,
            config=_small_config(),
            out_dir=tmp_path,
        )
        # Delete the figures dir to verify regenerate actually does work.
        fig_path = tmp_path / "figures" / "coverage_rate.png"
        assert fig_path.exists()
        fig_path.unlink()

        from scripts.figures import regenerate

        out = regenerate(tmp_path)
        assert any(p.name == "coverage_rate.png" for p in out)
        assert fig_path.exists()
