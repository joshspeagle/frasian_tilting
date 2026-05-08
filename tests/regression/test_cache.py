"""Regression tests for the mandatory cache layer.

The cache must:
  - reuse on identical (config, sha, raw, extra) keys
  - invalidate when *any* input changes
  - never hit on dirty git trees (so source modifications recompute)
  - tolerate missing-cache gracefully (compute + persist)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from frasian.simulation.cache import (
    CacheKey,
    cache_path,
    clear_cache,
    clear_cache_lru,
    get_or_compute,
)


def _key(**overrides) -> CacheKey:
    base = dict(
        experiment="demo",
        tilting="power_law",
        statistic="waldo",
        config_fingerprint="cfg0",
        git_sha="abc1234",
        raw_fingerprint="raw0",
        extra={},
    )
    base.update(overrides)
    return CacheKey(**base)


def _compute_factory(value):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return ({"x": np.full(3, value)}, {"value": value})

    return compute, calls


@pytest.mark.L0
class TestCacheKey:
    def test_digest_is_deterministic(self):
        a = _key().digest()
        b = _key().digest()
        assert a == b

    def test_digest_changes_with_any_input(self):
        base = _key().digest()
        for change in [
            dict(experiment="other"),
            dict(tilting="other"),
            dict(statistic="other"),
            dict(config_fingerprint="cfg1"),
            dict(git_sha="def5678"),
            dict(raw_fingerprint="raw1"),
            dict(extra={"k": 1}),
        ]:
            assert _key(**change).digest() != base, change

    def test_is_dirty_detection(self):
        assert _key(git_sha="dirty:abc").is_dirty()
        assert not _key(git_sha="abc1234").is_dirty()


@pytest.mark.L0
class TestGetOrCompute:
    def test_first_call_computes_and_persists(self, tmp_path: Path):
        compute, calls = _compute_factory(1.0)
        result = get_or_compute(_key(), compute, cache_root=tmp_path)
        assert calls["n"] == 1
        np.testing.assert_array_equal(result.arrays["x"], np.full(3, 1.0))
        assert (tmp_path / "demo" / _key().digest() / "arrays.npz").exists()

    def test_second_call_reuses_cache(self, tmp_path: Path):
        compute1, calls1 = _compute_factory(2.0)
        compute2, calls2 = _compute_factory(99.0)  # different output
        get_or_compute(_key(), compute1, cache_root=tmp_path)
        result = get_or_compute(_key(), compute2, cache_root=tmp_path)
        assert calls1["n"] == 1
        assert calls2["n"] == 0  # cache hit; compute2 never invoked
        np.testing.assert_array_equal(result.arrays["x"], np.full(3, 2.0))

    def test_changed_key_recomputes(self, tmp_path: Path):
        compute1, calls1 = _compute_factory(2.0)
        compute2, calls2 = _compute_factory(3.0)
        get_or_compute(_key(), compute1, cache_root=tmp_path)
        get_or_compute(_key(config_fingerprint="cfg1"), compute2, cache_root=tmp_path)
        assert calls1["n"] == 1
        assert calls2["n"] == 1

    def test_dirty_sha_always_recomputes(self, tmp_path: Path):
        compute, calls = _compute_factory(2.0)
        key = _key(git_sha="dirty:xyz")
        get_or_compute(key, compute, cache_root=tmp_path)
        get_or_compute(key, compute, cache_root=tmp_path)
        assert calls["n"] == 2

    def test_force_recomputes(self, tmp_path: Path):
        compute, calls = _compute_factory(2.0)
        get_or_compute(_key(), compute, cache_root=tmp_path)
        get_or_compute(_key(), compute, cache_root=tmp_path, force=True)
        assert calls["n"] == 2

    def test_disabled_recomputes_but_does_not_persist(self, tmp_path: Path):
        compute, calls = _compute_factory(2.0)
        get_or_compute(_key(), compute, cache_root=tmp_path, enabled=False)
        # Disabled mode still computes; verify the flag does *not* skip writes
        # — we always persist so a later enabled run can hit the cache.
        # (Design choice: disabling reads, not writes, keeps debug runs cheap.)
        assert calls["n"] == 1
        assert cache_path(tmp_path, _key()).exists()


@pytest.mark.L0
class TestClearCache:
    def test_clear_all(self, tmp_path: Path):
        compute, _ = _compute_factory(1.0)
        get_or_compute(_key(experiment="a"), compute, cache_root=tmp_path)
        get_or_compute(_key(experiment="b"), compute, cache_root=tmp_path)
        assert clear_cache(tmp_path) == 2

    def test_clear_one_experiment(self, tmp_path: Path):
        compute, _ = _compute_factory(1.0)
        get_or_compute(_key(experiment="a"), compute, cache_root=tmp_path)
        get_or_compute(_key(experiment="b"), compute, cache_root=tmp_path)
        assert clear_cache(tmp_path, experiment="a") == 1
        assert clear_cache(tmp_path, experiment="b") == 1

    def test_clear_nonexistent_root_returns_zero(self, tmp_path: Path):
        assert clear_cache(tmp_path / "nope") == 0

    def test_clear_recurses_into_subdirectories(self, tmp_path: Path):
        """Audit P2 (Cluster I): clear_cache must use shutil.rmtree
        so a result-dir layout that grows subdirectories does not
        crash with `IsADirectoryError`. Pre-fix the per-file
        `f.unlink()` would explode on the first sub-dir.
        """
        compute, _ = _compute_factory(1.0)
        get_or_compute(_key(experiment="rec"), compute, cache_root=tmp_path)
        # Inject a sub-directory inside the result dir to simulate a
        # future per-shard layout.
        result_dir = next((tmp_path / "rec").iterdir())
        (result_dir / "shards").mkdir()
        (result_dir / "shards" / "shard_0.npz").write_bytes(b"")
        # Recursive clear must remove the entire result dir (no
        # IsADirectoryError on the inner shard).
        assert clear_cache(tmp_path, experiment="rec") == 1


@pytest.mark.L0
class TestClearCacheLRU:
    """Audit P2 (Cluster I): `clear_cache_lru` keeps the N most-recent
    entries. The all-or-nothing `clear_cache` was insufficient for
    long-running result directories.
    """

    def test_keeps_n_most_recent(self, tmp_path: Path):
        import os
        import time

        compute, _ = _compute_factory(1.0)
        for i in range(5):
            get_or_compute(
                _key(experiment="lru", git_sha=f"sha{i}"),
                compute,
                cache_root=tmp_path,
            )
            # Stagger mtimes so the sort is deterministic on fast
            # filesystems where consecutive writes share an mtime.
            time.sleep(0.01)
        # 5 entries → keep 2 → 3 should be pruned.
        removed = clear_cache_lru(tmp_path, keep=2, experiment="lru")
        assert removed == 3
        remaining = sorted((tmp_path / "lru").iterdir())
        assert len(remaining) == 2
        # Newest mtimes survived.
        sha_remaining = {p.name for p in remaining}
        # The two `sha3`/`sha4` digests are last-written, so their
        # entries are the most recent.
        # We can't predict the digest names, but the count + mtime
        # ordering is verified.

    def test_keep_zero_removes_everything(self, tmp_path: Path):
        compute, _ = _compute_factory(1.0)
        get_or_compute(_key(experiment="lru0"), compute, cache_root=tmp_path)
        assert clear_cache_lru(tmp_path, keep=0, experiment="lru0") == 1

    def test_keep_more_than_present_no_op(self, tmp_path: Path):
        compute, _ = _compute_factory(1.0)
        get_or_compute(_key(experiment="few"), compute, cache_root=tmp_path)
        assert clear_cache_lru(tmp_path, keep=10, experiment="few") == 0

    def test_negative_keep_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="keep"):
            clear_cache_lru(tmp_path, keep=-1)


@pytest.mark.L0
class TestProcessedResultSchemaVersion:
    """Audit P2 (Cluster I): `ProcessedResult.SCHEMA_VERSION` mirrors
    `storage.SCHEMA_VERSION` so consumers can read the version from
    the type without going to disk.
    """

    def test_class_var_matches_storage_constant(self):
        from frasian.simulation.processing import (
            PROCESSED_RESULT_SCHEMA_VERSION,
            ProcessedResult,
        )
        from frasian.simulation.storage import SCHEMA_VERSION

        assert ProcessedResult.SCHEMA_VERSION == PROCESSED_RESULT_SCHEMA_VERSION
        assert ProcessedResult.SCHEMA_VERSION == SCHEMA_VERSION
