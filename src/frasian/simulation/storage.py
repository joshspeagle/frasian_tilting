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
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = 1


class _MetadataJSONEncoder(json.JSONEncoder):
    """Audit P1 L.2: tolerant encoder for numpy + Path types.

    Pre-fix `json.dumps(metadata)` raised `TypeError` whenever a
    metadata field carried a numpy scalar (`np.float64`, `np.int64`)
    or a `pathlib.Path`. The error surfaced at cache-write time as
    a hard crash with no recovery path. The encoder converts:

    * numpy integer scalars → Python int
    * numpy floating scalars → Python float (NaN / inf preserved)
    * numpy boolean scalars → Python bool
    * numpy arrays → list (small arrays only — large arrays should
      live in `arrays.npz`, not the metadata sidecar)
    * `pathlib.Path` → str (POSIX, for cross-platform stability)
    * `bytes` → hex (rare; rejected by default JSON)

    Anything still unencodable falls back to the default raise so
    we don't silently lose information.
    """

    def default(self, obj):  # noqa: D401  (json convention)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, bytes):
            return obj.hex()
        return super().default(obj)


@dataclass(frozen=True)
class StoredResult:
    """In-memory representation of a persisted result directory."""

    arrays: Mapping[str, NDArray]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def save_result(path: Path, arrays: Mapping[str, NDArray], metadata: Mapping[str, Any]) -> None:
    """Crash-safe + concurrency-safe persist (audit P1 L.2 + L.3).

    `path` is a *directory*. Sequence:
      1. Materialise the new result inside a per-call `mkdtemp`
         directory (e.g. `<parent>/.tmp.<random>`). Two concurrent
         writers get distinct names so neither blocks the other.
      2. If `path` already exists, rotate it to a per-call backup
         (`<parent>/.backup.<random>`).
      3. Rename `mkdtemp` dir to `path` (atomic within a filesystem).
      4. Delete the rotated backup.

    A crash mid-sequence leaves orphan `.tmp.*` / `.backup.*` dirs in
    the parent. They are NOT auto-cleaned (we can't tell whose they
    are without flock) but a follow-up `save_result` to a different
    path is unaffected. Operators can `rm -rf .tmp.* .backup.*` at
    leisure.

    Audit P1 L.2: metadata.json uses `_MetadataJSONEncoder` so
    numpy scalars and `Path` instances serialise cleanly instead of
    raising `TypeError` at write time.

    Audit P1 L.3: per-call mkdtemp avoids the fixed `<path>.tmp`
    name collision when two pytest-xdist workers write concurrently.
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # 1. Materialise the new result inside a unique tmp directory.
    tmp = Path(tempfile.mkdtemp(prefix=".tmp." + path.name + ".", dir=parent))
    np.savez_compressed(tmp / "arrays.npz", **dict(arrays))  # type: ignore[arg-type]
    full_meta = {
        "_schema_version": SCHEMA_VERSION,
        "_saved_at": time.time(),
        **dict(metadata),
    }
    (tmp / "metadata.json").write_text(
        json.dumps(full_meta, indent=2, sort_keys=True, cls=_MetadataJSONEncoder)
    )

    # 2. Rotate the existing result aside (no data loss across the gap).
    backup: Path | None = None
    if path.exists():
        backup = Path(tempfile.mkdtemp(prefix=".backup." + path.name + ".", dir=parent))
        # mkdtemp created an empty dir; remove it so the rename works.
        backup.rmdir()
        path.rename(backup)
    # 3. Swap the new directory in.
    tmp.rename(path)
    # 4. Drop the rotated copy.
    if backup is not None and backup.exists():
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
