"""Runner resilience: a single bad cell must not abort the experiment.

Pins Tier 1.7-C3: previously, ``experiment.run_cell`` was called inside
the runner loop with no try/except, so any cell that raised propagated
out and the manifest was never written. Per-cell try/except with
``status="error"`` recovery now keeps the manifest write intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frasian import Config, run_experiment
from frasian.diagnostics.base import Diagnostic, DiagnosticTable
from frasian.experiments.base import ExperimentContext, RawResult
from frasian.statistics.base import TestStatistic
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.base import TiltingScheme
from frasian.tilting.identity import IdentityTilting


class _NoOpDiagnostic:
    """Trivial diagnostic — returns one row per RawResult, no figure write."""

    name = "noop"

    def compute(self, raw):
        import pandas as pd

        return DiagnosticTable(
            name=self.name,
            table=pd.DataFrame({"tilting": [raw.tilting], "value": [0.0]}),
            units={"value": "dimensionless"},
            metadata={},
        )

    def render(self, table, fig_dir: Path) -> Path:
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / f"{self.name}.png"
        out.write_bytes(b"")  # empty file is fine for the test
        return out


class _FlakyExperiment:
    """One cell raises mid-loop; the others succeed.

    The runner gates `(wald, identity)` in (so that's the OK cell);
    we use a custom statistic to inject the failing cell so the runner
    actually tries to run it (rather than gating it out as incompatible).
    """

    name = "flaky"

    def setup(self, config: Config) -> ExperimentContext:
        return ExperimentContext(
            config=config,
            grid={"theta": np.array([0.0])},
            rng_seed=0,
        )

    def run_cell(
        self, ctx: ExperimentContext, tilting: TiltingScheme, statistic: TestStatistic
    ) -> RawResult:
        # Fail iff the cell's statistic is the "boom" statistic.
        if getattr(statistic, "name", "") == "boom":
            raise RuntimeError("intentional failure for resilience test")
        return RawResult(
            experiment=self.name,
            tilting=getattr(tilting, "cell_name", tilting.name),
            statistic=statistic.name,
            arrays={"x": np.array([1.0, 2.0, 3.0])},
            metadata={"raw_fingerprint": ""},
        )

    def diagnostics(self) -> list[Diagnostic]:
        return [_NoOpDiagnostic()]  # type: ignore[list-item]


class _BoomStatistic:
    """Test-only statistic that the FlakyExperiment causes to raise."""

    name = "boom"

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def pvalue(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def confidence_interval(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def accepts_tilting(self, tilting: Any) -> bool:
        # Accept identity so the runner tries to run the cell (and
        # then the FlakyExperiment raises).
        return True


@pytest.mark.L4
def test_runner_persists_manifest_when_a_cell_raises(tmp_path: Path) -> None:
    """When a cell raises, the runner records status="error" and continues.

    Verifies (a) no exception propagates out of run_experiment, (b) the
    manifest is still written, (c) the failing cell has status="error"
    with a non-empty reason, (d) the OK cell has status="ok", and (e)
    a ``RuntimeWarning`` fires so the failure is visible at runtime
    (skeptic Phase 5 vector #6).
    """
    with pytest.warns(RuntimeWarning, match=r"cell .*/boom raised RuntimeError"):
        summary = run_experiment(
            experiment=_FlakyExperiment(),  # type: ignore[arg-type]
            tiltings=[IdentityTilting()],
            statistics=[WaldStatistic(), _BoomStatistic()],  # type: ignore[list-item]
            config=Config.fast(),
            out_dir=tmp_path,
        )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["experiment"] == "flaky"

    by_stat = {c["statistic"]: c for c in manifest["cells"]}
    # Wald cell must succeed.
    assert "wald" in by_stat
    assert by_stat["wald"]["status"] == "ok"
    assert by_stat["wald"]["cache_path"]  # non-empty

    # Boom cell must be marked status="error" with a clear reason.
    assert "boom" in by_stat
    assert by_stat["boom"]["status"] == "error"
    assert "intentional failure for resilience test" in by_stat["boom"]["reason"]
    assert by_stat["boom"]["cache_path"] == ""

    # Summary mirrors the manifest cells.
    assert any(c.status == "error" for c in summary.cells)
    assert any(c.status == "ok" for c in summary.cells)
