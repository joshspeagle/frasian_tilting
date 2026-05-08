"""L1/L2 regression: Cluster L — cache / reproducibility.

Pins the audit P1 fixes:

  L.2 — `_MetadataJSONEncoder` handles numpy scalars / arrays / `Path`
        instances cleanly. Pre-fix `json.dumps(metadata)` raised
        `TypeError` whenever a metadata field carried any of these.

  L.3 — `save_result` uses `tempfile.mkdtemp` for the staging
        directory (per-call unique name), so two concurrent writers
        do not collide on the fixed `<path>.tmp` name.

  L.4 — `git_sha` no longer caches via `lru_cache`. Long-running
        sessions that transition clean → dirty → committed get the
        honest sha each call, instead of a stale memoised value.

  L.6 — `Config.fingerprint()` hashes numpy + jax major.minor
        versions. An environment upgrade (e.g. jax 0.4.x → 0.5.x,
        which changed PRNG semantics) invalidates the cache so a
        re-run in the new environment recomputes.

  L.7 — `RawSamples.fingerprint` includes shape + dtype explicitly.
        Pre-fix `tobytes()` collisions across reshape were silent.

  L.8 — `get_or_compute` validates `_cache_key.digest` on load and
        refuses on mismatch (digest collision, relocated cache dir,
        schema drift).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frasian.config import Config
from frasian.simulation.cache import (
    CacheKey,
    get_or_compute,
    git_sha,
)
from frasian.simulation.raw import RawSamples
from frasian.simulation.storage import _MetadataJSONEncoder, save_result, load_result


# --- L.2 JSON encoder ----------------------------------------------------


@pytest.mark.L0
class TestMetadataJSONEncoder:
    """The encoder makes `json.dumps` tolerant of numpy + Path."""

    def test_numpy_int_serialises(self):
        out = json.dumps({"k": np.int64(42)}, cls=_MetadataJSONEncoder)
        assert json.loads(out)["k"] == 42

    def test_numpy_float_serialises(self):
        out = json.dumps({"k": np.float64(3.14)}, cls=_MetadataJSONEncoder)
        assert json.loads(out)["k"] == pytest.approx(3.14)

    def test_numpy_bool_serialises(self):
        out = json.dumps({"k": np.bool_(True)}, cls=_MetadataJSONEncoder)
        assert json.loads(out)["k"] is True

    def test_numpy_array_serialises_to_list(self):
        out = json.dumps({"k": np.array([1, 2, 3])}, cls=_MetadataJSONEncoder)
        assert json.loads(out)["k"] == [1, 2, 3]

    def test_path_serialises(self):
        out = json.dumps({"k": Path("/tmp/foo")}, cls=_MetadataJSONEncoder)
        assert json.loads(out)["k"] == "/tmp/foo"


# --- L.3 concurrency-safe write ------------------------------------------


@pytest.mark.L1
class TestSaveResultUsesMkdtemp:
    """The save_result routine uses `tempfile.mkdtemp` (per-call unique
    staging dir) so concurrent writers do not collide."""

    def test_save_result_round_trips_with_numpy_metadata(self, tmp_path):
        # Round-trip with numpy + Path metadata fields — pre-fix this
        # would raise TypeError at the `json.dumps` call.
        path = tmp_path / "result"
        save_result(
            path,
            arrays={"x": np.arange(5)},
            metadata={
                "config_fingerprint": "abc",
                "alpha": np.float64(0.05),
                "n_reps": np.int64(100),
                "out_dir": tmp_path,
            },
        )
        loaded = load_result(path)
        assert loaded.metadata["alpha"] == pytest.approx(0.05)
        assert loaded.metadata["n_reps"] == 100
        assert "tmp_path" not in str(loaded.metadata)  # path serialised as posix

    def test_no_orphan_tmp_dirs_after_clean_save(self, tmp_path):
        path = tmp_path / "result"
        save_result(
            path, arrays={"x": np.zeros(3)}, metadata={"config_fingerprint": "abc"}
        )
        # No `.tmp.*` or `.backup.*` dirs left in the parent.
        leftovers = [
            p
            for p in tmp_path.iterdir()
            if p.name.startswith(".tmp.") or p.name.startswith(".backup.")
        ]
        assert leftovers == [], f"orphan tmp/backup dirs: {leftovers}"


# --- L.4 git_sha no longer cached ----------------------------------------


@pytest.mark.L0
class TestGitShaNotCached:
    """The `git_sha` function should not have an `lru_cache`."""

    def test_no_cache_clear_attribute(self):
        # `lru_cache`-wrapped functions expose `cache_clear`. Without
        # the wrapper, the attribute is absent.
        assert not hasattr(git_sha, "cache_clear"), (
            "git_sha should not be lru_cache-wrapped (audit P1 L.4)"
        )


# --- L.6 Config.fingerprint includes numpy/jax versions -----------------


@pytest.mark.L0
class TestConfigFingerprintIncludesVersions:
    """The Config fingerprint should change when numpy/jax major.minor
    changes (we can't actually swap versions in a test, so we just pin
    that the version fields are in `_serializable`)."""

    def test_serialisable_includes_numpy_version(self):
        s = Config.fast()._serializable()
        assert "_numpy_version" in s

    def test_serialisable_includes_jax_version(self):
        s = Config.fast()._serializable()
        assert "_jax_version" in s

    def test_versions_are_major_minor(self):
        s = Config.fast()._serializable()
        # major.minor format: "X.Y" (no patch).
        np_ver = s["_numpy_version"]
        if np_ver != "unknown":
            parts = np_ver.split(".")
            assert len(parts) == 2, f"expected major.minor, got {np_ver}"


# --- L.7 RawSamples.fingerprint includes shape + dtype -------------------


@pytest.mark.L1
class TestRawSamplesFingerprintShape:
    """Two RawSamples with same byte content but different shapes should
    have different fingerprints."""

    def test_reshape_changes_fingerprint(self):
        # 12-element float64 array, two distinct shapes.
        flat = np.arange(12, dtype=np.float64)
        a = flat.reshape(3, 4)
        b = flat.reshape(4, 3)
        # Both have same `tobytes()` (numpy is C-contiguous), so
        # the pre-fix fingerprint would collide.
        assert a.tobytes() == b.tobytes()
        rs_a = RawSamples(
            name="t",
            D=a,
            theta_grid=np.arange(3, dtype=np.float64),
            sigma=1.0,
            seed=0,
        )
        rs_b = RawSamples(
            name="t",
            D=b,
            theta_grid=np.arange(4, dtype=np.float64),
            sigma=1.0,
            seed=0,
        )
        assert rs_a.fingerprint() != rs_b.fingerprint(), (
            "pre-fix collision: reshape with same bytes but different "
            "shapes shouldn't share a fingerprint"
        )


# --- L.8 cache key validation on load -----------------------------------


@pytest.mark.L1
class TestGetOrComputeValidatesCacheKey:
    """`get_or_compute` raises if the loaded `_cache_key.digest` differs
    from the requested key's digest (relocation / collision / drift)."""

    def test_corrupted_digest_raises(self, tmp_path):
        cache_root = tmp_path / "cache"

        def compute():
            return ({"x": np.arange(3)}, {"flag": "ok"})

        key = CacheKey(
            experiment="exp",
            tilting="t",
            statistic="s",
            config_fingerprint="abc",
            git_sha="def",
            raw_fingerprint="raw",
        )
        # First call populates.
        get_or_compute(key, compute, cache_root=cache_root, enabled=True)

        # Tamper with the stored _cache_key.digest.
        from frasian.simulation.cache import cache_path
        path = cache_path(cache_root, key)
        meta = json.loads((path / "metadata.json").read_text())
        meta["_cache_key"]["digest"] = "wrong_digest_value"
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Second call: load + validate → raise.
        with pytest.raises(RuntimeError, match="Cache integrity"):
            get_or_compute(key, compute, cache_root=cache_root, enabled=True)
