"""L4: end-to-end coverage experiment with `LearnedDynamicEtaSelector`.

Mirrors `tests/experiments/test_coverage_experiment.py` but explicitly
plugs in a `LearnedDynamicEtaSelector` (loaded from the shipped
v0_smoke checkpoint) into the cell list. Asserts:

  - run_experiment completes without error,
  - the manifest records the learned cell with status='ok',
  - the diagnostics array contains coverage values in [0, 1],
  - the cache + figure + CSV outputs are written.

Skipped if torch isn't available or if the v0_smoke checkpoint is
missing. Marked L4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian import Config, registry, run_experiment
from frasian.config import GridSpec
from frasian.learned.monotonic_eta import MonotonicEtaArtifact
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


_CKPT = Path("artifacts/learned_eta_power_law_v0_smoke.pt")


def _config() -> Config:
    return Config.fast().from_overrides(
        n_reps=60,
        theta_grid=GridSpec("theta", -2.0, 2.0, 3),
        w_grid=GridSpec("w", 0.3, 0.7, 3),
    )


@pytest.mark.L4
class TestLearnedEtaPipelineEndToEnd:
    def test_runs_through_runner(self, tmp_path: Path, bootstrapped_registry):
        if not _CKPT.exists():
            pytest.skip(f"checkpoint missing at {_CKPT}; train first")
        artifact = MonotonicEtaArtifact(artifact_path=_CKPT)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        tiltings = [
            IdentityTilting(),
            PowerLawTilting(selector=selector),
        ]
        statistics = [WaldoStatistic()]

        experiment = registry.experiments["coverage"]()
        cfg = _config()
        run_experiment(
            experiment=experiment,
            tiltings=tiltings, statistics=statistics,
            config=cfg, out_dir=tmp_path,
        )

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["experiment"] == "coverage"
        ok = [c for c in manifest["cells"] if c["status"] == "ok"]
        assert len(ok) == 2  # identity, power_law[learned_dynamic]
        learned_cell = next(
            c for c in ok if c["tilting"].startswith("power_law")
        )
        assert "learned_dynamic" in learned_cell["tilting"]

        # Coverage array shape + range.
        from frasian.simulation.storage import load_result
        result = load_result(tmp_path / learned_cell["cache_path"])
        cov = result.arrays["coverage"]
        assert cov.shape == (cfg.theta_grid.n_points, cfg.w_grid.n_points)
        assert np.all((cov >= 0.0) & (cov <= 1.0))

        # Figure and CSV outputs.
        assert (tmp_path / "figures" / "coverage_rate.png").exists()
        assert (tmp_path / "coverage_rate.csv").exists()

    def test_env_var_default_switch(self, tmp_path: Path, monkeypatch,
                                       bootstrapped_registry):
        """`FRASIAN_DEFAULT_DYNAMIC_ETA=learned` selects LearnedDynamicEtaSelector."""
        if not _CKPT.exists():
            pytest.skip(f"checkpoint missing at {_CKPT}; train first")
        if not Path("artifacts/learned_eta_ot_v0_smoke.pt").exists():
            pytest.skip("ot smoke checkpoint missing; train first")
        from frasian._default_cells import default_tiltings
        monkeypatch.setenv("FRASIAN_DEFAULT_DYNAMIC_ETA", "learned")
        tiltings = default_tiltings()
        # identity + power_law[learned_dynamic] + ot[learned_dynamic]
        assert len(tiltings) == 3
        cell_names = [getattr(t, "cell_name", t.name) for t in tiltings]
        assert any("learned_dynamic" in n for n in cell_names), cell_names
