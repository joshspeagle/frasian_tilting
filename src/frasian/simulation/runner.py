"""Wires the cross-product runner to the cache layer.

`persist_cell` takes one cell's `RawResult` plus the inputs that produced
it, derives a `CacheKey`, and stores or reuses via `cache.get_or_compute`.
The framework's experiments (`coverage`, `width`, `smoothness`,
`confidence_distribution`) all route through this helper.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from ..config import Config
from ..experiments.base import RawResult
from .cache import CacheKey, get_or_compute, git_sha


def persist_cell(
    *,
    raw_result: RawResult,
    config: Config,
    cache_root: Path,
    raw_fingerprint: str = "",
    enabled: bool = True,
    force: bool = False,
    tilting: Any = None,
) -> Path:
    """Persist `raw_result` under a deterministic cache directory.

    Returns the on-disk path. The result is loadable later via
    `simulation.storage.load_result(path)`.

    When ``tilting`` is provided, its selector is inspected for a
    ``artifact.fingerprint()`` (Phase E learned-η selectors only); the
    digest is folded into ``CacheKey.extra`` under
    ``"selector_artifact_fingerprint"`` so swapping a learned checkpoint
    invalidates the cache even at the same git sha + config.
    """
    extra = dict(raw_result.metadata)
    if tilting is not None:
        try:
            selector = getattr(tilting, "selector", None)
            artifact = getattr(selector, "artifact", None) if selector is not None else None
            fp_fn = getattr(artifact, "fingerprint", None) if artifact is not None else None
            if callable(fp_fn):
                extra["selector_artifact_fingerprint"] = str(fp_fn())
        except (AttributeError, TypeError):
            # Narrow scope: only the "no fingerprint method" cases (a bare
            # selector or an artifact whose fingerprint signature mismatches)
            # are silently skipped. Real failures — OSError when the
            # checkpoint is missing on disk, MissingArtifactError, etc. —
            # propagate so the runner fails loudly rather than collapsing
            # the cache key onto a stale entry.
            pass

    key = CacheKey(
        experiment=raw_result.experiment,
        tilting=raw_result.tilting,
        statistic=raw_result.statistic,
        config_fingerprint=config.fingerprint(),
        git_sha=git_sha(),
        raw_fingerprint=raw_fingerprint,
        extra=extra,
    )

    def compute() -> tuple[Mapping[str, NDArray], Mapping[str, Any]]:
        return raw_result.arrays, dict(raw_result.metadata)

    get_or_compute(
        key, compute, cache_root=cache_root, enabled=enabled and config.cache_enabled, force=force
    )
    from .cache import cache_path

    return cache_path(cache_root, key)
