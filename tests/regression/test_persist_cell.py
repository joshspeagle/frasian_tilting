"""Regression tests for `simulation.runner.persist_cell` and `learned.null`.

Verifies that the cross-product runner can persist and reuse cell results
through the full cache pipeline, and that a `LearnedArtifact` stub works
the way the framework needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frasian import Config, MissingArtifactError
from frasian.experiments.base import RawResult
from frasian.learned.null import NullArtifact
from frasian.simulation.cache import git_sha
from frasian.simulation.runner import persist_cell
from frasian.simulation.storage import load_result


def _raw(value=1.0):
    return RawResult(
        experiment="demo",
        tilting="power_law",
        statistic="waldo",
        arrays={"x": np.full(4, value)},
        metadata={"value": value},
    )


@pytest.mark.L0
class TestPersistCell:
    def test_persists_and_returns_path(self, tmp_path: Path):
        path = persist_cell(
            raw_result=_raw(2.0),
            config=Config.fast(),
            cache_root=tmp_path,
        )
        assert path.exists()
        result = load_result(path)
        np.testing.assert_array_equal(result.arrays["x"], np.full(4, 2.0))

    def test_cache_key_recorded_in_metadata(self, tmp_path: Path):
        path = persist_cell(
            raw_result=_raw(3.0),
            config=Config.fast(),
            cache_root=tmp_path,
        )
        result = load_result(path)
        meta = result.metadata["_cache_key"]
        assert meta["experiment"] == "demo"
        assert meta["tilting"] == "power_law"
        assert meta["statistic"] == "waldo"
        assert meta["config_fingerprint"] == Config.fast().fingerprint()

    def test_disabled_cache_via_config(self, tmp_path: Path):
        cfg = Config.fast().from_overrides(cache_enabled=False)
        # Even with cache_enabled=False at config level, persist_cell still
        # writes (so a later enabled run can hit the cache); behaviour matches
        # the documented design choice for `enabled=False` in get_or_compute.
        path = persist_cell(raw_result=_raw(4.0), config=cfg,
                             cache_root=tmp_path)
        assert path.exists()


@pytest.mark.L0
class TestNullArtifact:
    def test_predict_before_load_raises(self):
        a = NullArtifact(value=5.0)
        with pytest.raises(MissingArtifactError):
            a.predict(np.zeros(3))

    def test_predict_after_load_returns_constant(self):
        a = NullArtifact(value=5.0)
        a.load()
        np.testing.assert_array_equal(a.predict(np.zeros(3)), np.full(3, 5.0))

    def test_fingerprint_changes_with_value(self):
        a = NullArtifact(value=1.0)
        b = NullArtifact(value=2.0)
        assert a.fingerprint() != b.fingerprint()

    def test_load_is_idempotent(self):
        a = NullArtifact(value=1.0)
        a.load()
        a.load()
        a.predict(np.zeros(2))  # no error


@pytest.mark.L0
class TestGitSha:
    def test_returns_string(self):
        # In our git repo this returns either a real sha or 'dirty:<hash>'.
        sha = git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0
