"""Trivial LearnedArtifact used to test the framework without any real model.

Used by the cache and runner tests where we want to inject an artifact of
known behavior. Concrete artifacts (the monotonic eta MLP) land in a
later step alongside the experiments that use them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .._errors import MissingArtifactError


@dataclass
class NullArtifact:
    """Predicts the constant `value` after `load()`. Stateful by design."""

    name: str = "null"
    version: str = "v0"
    artifact_path: Path = Path("/dev/null")
    value: float = 0.0
    _loaded: bool = False

    def load(self) -> None:
        self._loaded = True

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded; call .load()")
        return np.full_like(np.asarray(x, dtype=np.float64), self.value)

    def fingerprint(self) -> str:
        h = hashlib.sha256(f"{self.name}:{self.version}:{self.value}".encode())
        return h.hexdigest()[:16]
