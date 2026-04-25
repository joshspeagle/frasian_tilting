"""Result-directory I/O.

A "result" is a directory containing:
  arrays.npz     — `np.savez_compressed` of all numpy arrays
  metadata.json  — config fingerprint, git sha, schema version, timestamps

This is simpler than HDF5 (no h5py dep), debuggable with `unzip` / `cat`, and
sufficient for the framework's needs. The legacy `simulations/storage.py` used
HDF5; that complexity is a future extension if/when datasets get large.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredResult:
    """In-memory representation of a persisted result directory."""

    arrays: Mapping[str, NDArray]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def save_result(path: Path, arrays: Mapping[str, NDArray],
                metadata: Mapping[str, Any]) -> None:
    """Atomically persist a result to `path`.

    `path` is a *directory*. Existing contents are overwritten via the
    rename-from-temp pattern: write to `path.with_suffix(".tmp")` then rename.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        for child in tmp.iterdir():
            child.unlink()
        tmp.rmdir()
    tmp.mkdir(parents=True)

    np.savez_compressed(tmp / "arrays.npz", **dict(arrays))
    full_meta = {
        "_schema_version": SCHEMA_VERSION,
        "_saved_at": time.time(),
        **dict(metadata),
    }
    (tmp / "metadata.json").write_text(json.dumps(full_meta, indent=2,
                                                    sort_keys=True))
    if path.exists():
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    tmp.rename(path)


def load_result(path: Path) -> StoredResult:
    """Load a result directory previously written by `save_result`."""
    path = Path(path)
    if not result_exists(path):
        raise FileNotFoundError(f"no result at {path!r}")
    with np.load(path / "arrays.npz") as npz:
        arrays = {key: npz[key].copy() for key in npz.files}
    metadata = json.loads((path / "metadata.json").read_text())
    return StoredResult(arrays=arrays, metadata=metadata)


def load_metadata(path: Path) -> dict[str, Any]:
    """Load only the metadata.json sidecar (cheap; no array decompression)."""
    path = Path(path)
    if not (path / "metadata.json").exists():
        raise FileNotFoundError(f"no metadata at {path!r}")
    return json.loads((path / "metadata.json").read_text())


def result_exists(path: Path) -> bool:
    path = Path(path)
    return (path / "arrays.npz").exists() and (path / "metadata.json").exists()
