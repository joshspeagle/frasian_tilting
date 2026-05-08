"""Mandatory cache for processed simulation results.

Cache keys are deterministic: a SHA-256 of (experiment, tilting, statistic,
config fingerprint, git sha, raw fingerprint, extra). The legacy code keyed
on file mtime, which is fragile under reorganisation; this version keys on
content fingerprints, so a result is reused only when *every input that
could affect it* matches.

Dirty git trees (uncommitted changes) get a sha of `dirty:<hash>` where
`<hash>` is a short SHA-256 of the `git status --porcelain` output.  This
differs from a git tree-object hash: it reflects only which paths have
changed, not the content of those changes. The guarantee is that any
uncommitted modification will produce a different `<hash>` than a clean tree,
so dirty trees always recompute — committing the source is still the hard
prerequisite for byte-reproducible results.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from numpy.typing import NDArray

from .storage import StoredResult, load_result, result_exists, save_result


@dataclass(frozen=True)
class CacheKey:
    """Deterministic cache key. The hex digest is the on-disk directory name."""

    experiment: str
    tilting: str
    statistic: str
    config_fingerprint: str
    git_sha: str
    raw_fingerprint: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        payload = json.dumps(
            {
                "experiment": self.experiment,
                "tilting": self.tilting,
                "statistic": self.statistic,
                "config": self.config_fingerprint,
                "git": self.git_sha,
                "raw": self.raw_fingerprint,
                "extra": dict(self.extra),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:24]

    def is_dirty(self) -> bool:
        return self.git_sha.startswith("dirty:")


def git_sha(repo_root: Path | None = None) -> str:
    """Current git sha or `dirty:<hash>` if there are uncommitted changes.

    On a dirty tree, `<hash>` is a short SHA-256 of the `git status
    --porcelain` output — not a git tree-object id.  The returned value
    always changes when any tracked file is modified or staged, guaranteeing
    that dirty-tree cache keys never collide with clean-tree keys.

    Audit P1 L.4: dropped `@lru_cache(maxsize=1)`. The legacy cache
    was per-process-lifetime, so a long-running session that
    transitioned from clean → dirty → committed via interactive
    edits would keep the stale sha and either pollute the cache
    with a wrong sha or, worse, treat dirty results as clean.
    Tests previously had to call `git_sha.cache_clear()`; the
    new contract is "always re-shells, always honest." The two
    `subprocess.run` calls cost ~2 ms total; the cache pipeline
    is dominated by other work.
    """
    cwd = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # Detect dirty tree: any unstaged or staged changes.
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status.strip():
            tree = hashlib.sha256(status.encode()).hexdigest()[:8]
            return f"dirty:{tree}"
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repo (e.g. installed package). Treat as dirty.
        return "dirty:nogit"


def cache_path(root: Path, key: CacheKey) -> Path:
    """Where the cached result lives on disk."""
    return Path(root) / key.experiment / key.digest()


def get_or_compute(
    key: CacheKey,
    compute: Callable[[], tuple[Mapping[str, NDArray], Mapping[str, Any]]],
    *,
    cache_root: Path,
    enabled: bool = True,
    force: bool = False,
) -> StoredResult:
    """Return the cached result for `key`, computing if missing or dirty.

    `compute` returns a `(arrays, metadata)` pair which is then persisted.
    Dirty git trees never hit the cache (we always recompute and overwrite),
    so committing the source is a hard prerequisite for reproducibility.

    Audit P1 L.8: when loading from disk we validate that the stored
    ``_cache_key`` matches the requested ``key``. Pre-fix the loader
    trusted the directory name (digest) without reading the embedded
    metadata; a hash collision (24-char SHA-256 prefix is ~ 10⁻¹⁴
    bits of entropy collision risk) or a manually-relocated cache
    dir would silently return the wrong result. Loud refusal here.
    """
    path = cache_path(cache_root, key)
    if enabled and not force and not key.is_dirty() and result_exists(path):
        loaded = load_result(path)
        # Audit P1 L.8: validate the loaded `_cache_key` matches the
        # requested key. Mismatch signals a digest collision, a
        # relocated cache dir, or schema drift; refuse loudly.
        stored_key = dict(loaded.metadata).get("_cache_key", {})
        expected_digest = key.digest()
        actual_digest = stored_key.get("digest")
        if actual_digest is not None and actual_digest != expected_digest:
            raise RuntimeError(
                f"Cache integrity error at {path}: stored "
                f"_cache_key.digest={actual_digest!r} but requested "
                f"key.digest()={expected_digest!r}. Either delete the "
                f"cache directory or pass force=True to recompute."
            )
        return loaded
    arrays, metadata = compute()
    full_meta = {
        "_cache_key": {
            "experiment": key.experiment,
            "tilting": key.tilting,
            "statistic": key.statistic,
            "config_fingerprint": key.config_fingerprint,
            "git_sha": key.git_sha,
            "raw_fingerprint": key.raw_fingerprint,
            "extra": dict(key.extra),
            "digest": key.digest(),
        },
        **dict(metadata),
    }
    save_result(path, arrays, full_meta)
    return load_result(path)


def clear_cache(cache_root: Path, *, experiment: str | None = None) -> int:
    """Delete cached results. Returns count removed."""
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return 0
    targets = [cache_root / experiment] if experiment is not None else list(cache_root.iterdir())
    n = 0
    for target in targets:
        if not target.exists() or not target.is_dir():
            continue
        for child in target.iterdir():
            if child.is_dir() and result_exists(child):
                for f in child.iterdir():
                    f.unlink()
                child.rmdir()
                n += 1
    return n
