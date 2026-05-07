"""End-to-end test for `scripts/figures.py`: regenerate from results dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frasian import default_cells, registry, run_experiment
from frasian.config import Config, GridSpec


def _small_config() -> Config:
    """Smallest config that still produces a figure-able coverage table.

    n_reps=10, 3×2 grid → ~60 CIs per cell × ~5 cells = ~300 CIs total.
    Wall-time ~30 s on dev hardware (was ~3 min at the previous
    n_reps=30 / 4×3 grid budget). The test only checks that
    `regenerate` produces a PNG; it doesn't pin coverage values, so
    a smaller budget is fine.
    """
    return Config.fast().from_overrides(
        n_reps=10,
        theta_grid=GridSpec("theta", -1.5, 1.5, 3),
        w_grid=GridSpec("w", 0.3, 0.7, 2),
    )


@pytest.mark.L4
class TestFiguresScript:
    def test_regenerate_from_results(self, tmp_path: Path, bootstrapped_registry):
        experiment = registry.experiments["coverage"]()
        tiltings, statistics = default_cells(n_grid=81, coarse_n=9)
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
