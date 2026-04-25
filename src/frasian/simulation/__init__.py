"""Three-layer simulation infrastructure.

  Layer 0 (raw):       generate D ~ N(theta, sigma) samples (raw.py)
  Layer 1 (processed): deterministic transforms (processing.py)
  Layer 1.5 (cache):   mandatory persistent cache, content-keyed (cache.py)

Storage uses npz + json sidecars (see storage.py); the cache is keyed on
config fingerprint, git sha, and raw fingerprint, so a result is reused
only when every input that could affect it matches.
"""

from .cache import CacheKey, cache_path, clear_cache, get_or_compute, git_sha
from .processing import ProcessedResult, smoke_process
from .raw import RawSamples, generate_normal_D_samples
from .runner import persist_cell
from .storage import (
    SCHEMA_VERSION,
    StoredResult,
    load_metadata,
    load_result,
    result_exists,
    save_result,
)

__all__ = [
    "CacheKey",
    "ProcessedResult",
    "RawSamples",
    "SCHEMA_VERSION",
    "StoredResult",
    "cache_path",
    "clear_cache",
    "generate_normal_D_samples",
    "get_or_compute",
    "git_sha",
    "load_metadata",
    "load_result",
    "persist_cell",
    "result_exists",
    "save_result",
    "smoke_process",
]
