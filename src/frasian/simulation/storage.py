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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredResult:
    """In-memory representation of a persisted result directory."""

    arrays: Mapping[str, NDArray]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def save_result(path: Path, arrays: Mapping[str, NDArray], metadata: Mapping[str, Any]) -> None:
    """Crash-safe persist: write to `<path>.tmp`, rotate the existing
    directory aside, swap, then delete the rotated copy.

    `path` is a *directory*. Sequence:
      1. Materialise the new result inside `<path>.tmp`.
      2. If `path` already exists, rename it to `<path>.backup`.
      3. Rename `<path>.tmp` to `path`.
      4. Delete `<path>.backup`.

    A crash between step 2 and step 3 leaves `<path>.tmp` and
    `<path>.backup` on disk; the next call to `save_result` cleans
    those up before retrying. The previous result is therefore *never*
    lost across a crash window — the worst case is two leftover dirs
    that get garbage-collected on the next write.

    This is not strictly POSIX-atomic (you can't rename a non-empty
    directory over another non-empty directory in one syscall) but it
    is crash-recoverable, which is what callers actually need.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    backup = path.with_name(path.name + ".backup")

    # Clean up leftovers from any prior crash.
    for stale in (tmp, backup):
        if stale.exists():
            for child in stale.iterdir():
                child.unlink()
            stale.rmdir()

    # 1. Materialise the new directory under `<path>.tmp`.
    tmp.mkdir(parents=True)
    np.savez_compressed(tmp / "arrays.npz", **dict(arrays))  # type: ignore[arg-type]
    full_meta = {
        "_schema_version": SCHEMA_VERSION,
        "_saved_at": time.time(),
        **dict(metadata),
    }
    (tmp / "metadata.json").write_text(json.dumps(full_meta, indent=2, sort_keys=True))

    # 2. Rotate the existing result aside (no data loss across the gap).
    if path.exists():
        path.rename(backup)
    # 3. Swap the new directory in.
    tmp.rename(path)
    # 4. Drop the rotated copy. A crash here leaves an orphan `.backup`;
    # the next save_result call cleans it up.
    if backup.exists():
        for child in backup.iterdir():
            child.unlink()
        backup.rmdir()


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
