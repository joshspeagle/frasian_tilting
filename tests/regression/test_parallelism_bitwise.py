"""L4 regression: per-replicate parallelism is byte-identical to serial.

Pins the contract documented in `_parallel.py` and the `n_jobs`-
fingerprint comment in `Config`: switching `Config.n_jobs` between
`1` and `>1` reorders only the *evaluation* of replicates, not their
*seeding* — the per-replicate `D` values come from a pre-generated
`generate_normal_D_samples` stream. Result arrays must therefore agree
to floating-point exactness.

Runs `coverage` + `width` + `smoothness` + `confidence_distribution`
through `run_experiment` at `n_jobs=1` and `n_jobs=2`, compares every
result-array key. Uses a tiny grid + cheap closed-form Wald to keep
the test under ~30 s on this 12-core box.

Why use `n_jobs=2` rather than `n_jobs=-1`: minimum exercise of the
parallel branch (proves the worker dispatch + result-aggregation
plumbing works) without paying the import-warmup cost on every core.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frasian import Config, registry, run_experiment
from frasian._registry_bootstrap import bootstrap
from frasian.simulation.storage import load_result
from frasian.statistics.wald import WaldStatistic
from frasian.tilting.identity import IdentityTilting


def _tiny_config(n_jobs: int) -> Config:
    """Smaller-than-fast config so the test stays cheap. Keeps n_reps
    high enough that the parallel branch actually fans out (>=2)."""
    base = Config.fast()
    return base.from_overrides(n_jobs=n_jobs, n_reps=12)


def _run_one(experiment_name: str, n_jobs: int, out_dir: Path) -> dict[str, np.ndarray]:
    bootstrap()
    summary = run_experiment(
        experiment=registry.experiments[experiment_name](),
        tiltings=[IdentityTilting()],
        statistics=[WaldStatistic()],
        config=_tiny_config(n_jobs),
        out_dir=out_dir,
    )
    # Identify the (only) cell in this run and load its on-disk arrays
    # back; comparison is array-vs-array, not summary-vs-summary.
    [cell] = [c for c in summary.cells if c.status == "ok"]
    res = load_result(out_dir / cell.cache_path)
    return dict(res.arrays)


@pytest.mark.L4
@pytest.mark.parametrize(
    "experiment_name",
    ["coverage", "width", "smoothness", "confidence_distribution"],
)
def test_parallelism_byte_identical(tmp_path, experiment_name, bootstrapped_registry):
    serial = _run_one(experiment_name, n_jobs=1, out_dir=tmp_path / "serial")
    parallel = _run_one(experiment_name, n_jobs=2, out_dir=tmp_path / "parallel")
    # Same set of array keys.
    assert set(serial.keys()) == set(parallel.keys()), (
        f"key mismatch: serial={set(serial)} parallel={set(parallel)}"
    )
    # Each array byte-identical (NaN-aware comparison: equal_nan=True).
    for key in serial:
        np.testing.assert_array_equal(
            serial[key],
            parallel[key],
            err_msg=f"{experiment_name} array {key!r} differs at n_jobs=1 vs n_jobs=2",
        )
