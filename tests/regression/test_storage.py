"""Regression tests for the npz+json result-directory storage layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frasian.simulation.storage import (
    SCHEMA_VERSION,
    load_metadata,
    load_result,
    result_exists,
    save_result,
)


@pytest.mark.L0
class TestStorageRoundTrip:
    def test_save_then_load(self, tmp_path: Path):
        path = tmp_path / "demo"
        arrays = {
            "x": np.linspace(0, 1, 5),
            "y": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        meta = {"experiment": "t", "n": 5}
        save_result(path, arrays, meta)
        assert result_exists(path)

        result = load_result(path)
        np.testing.assert_array_equal(result.arrays["x"], arrays["x"])
        np.testing.assert_array_equal(result.arrays["y"], arrays["y"])
        assert result.metadata["experiment"] == "t"
        assert result.metadata["n"] == 5
        assert result.metadata["_schema_version"] == SCHEMA_VERSION
        assert "_saved_at" in result.metadata

    def test_load_metadata_only(self, tmp_path: Path):
        path = tmp_path / "demo"
        save_result(path, {"x": np.zeros(3)}, {"hello": "world"})
        meta = load_metadata(path)
        assert meta["hello"] == "world"

    def test_overwrites_existing(self, tmp_path: Path):
        path = tmp_path / "demo"
        save_result(path, {"x": np.zeros(3)}, {"v": 1})
        save_result(path, {"x": np.ones(2)}, {"v": 2})
        result = load_result(path)
        assert result.arrays["x"].shape == (2,)
        assert result.metadata["v"] == 2

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_result(tmp_path / "nope")
        with pytest.raises(FileNotFoundError):
            load_metadata(tmp_path / "nope")

    def test_result_exists_partial_dir_returns_false(self, tmp_path: Path):
        partial = tmp_path / "partial"
        partial.mkdir()
        (partial / "arrays.npz").write_bytes(b"")  # missing metadata.json
        assert result_exists(partial) is False
