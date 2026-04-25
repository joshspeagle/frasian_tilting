"""End-to-end test for `scripts/figures.py`: regenerate from results dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frasian import registry, run_experiment
from frasian.config import Config, GridSpec


def _small_config() -> Config:
    return Config.fast().from_overrides(
        n_reps=30,
        theta_grid=GridSpec("theta", -1.5, 1.5, 4),
        w_grid=GridSpec("w", 0.3, 0.7, 3),
    )


@pytest.mark.L4
class TestFiguresScript:
    def test_regenerate_from_results(self, tmp_path: Path,
                                        bootstrapped_registry):
        experiment = registry.experiments["coverage"]()
        run_experiment(
            experiment=experiment,
            tiltings=registry.tiltings.implemented(),
            statistics=registry.statistics.implemented(),
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
