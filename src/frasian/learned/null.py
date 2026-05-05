"""Trivial LearnedArtifact used to test the framework without any real model.

Used by the cache and runner tests where we want to inject an artifact of
known behavior. Concrete artifacts (the monotonic eta MLP) land in a
later step alongside the experiments that use them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._errors import MissingArtifactError


@dataclass
class NullArtifact:
    """Predicts the constant `value` after `load()`. Stateful by design.

    Implements the full ``LearnedArtifact`` Protocol surface (incl.
    Phase E ``predict_eta`` / ``predict_validity`` / ``metadata``) by
    returning trivial constants. The legacy ``predict`` method is
    preserved for tests written against the pre-Phase-E surface.
    """

    name: str = "null"
    version: str = "v0"
    artifact_path: Path = Path("/dev/null")
    value: float = 0.0
    _loaded: bool = False
    _metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def load(self) -> None:
        self._loaded = True

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Legacy pre-Phase-E predict; returns the constant ``value``."""
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded; call .load()")
        return np.full_like(np.asarray(x, dtype=np.float64), self.value)

    def predict_eta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Phase E surface: constant η = ``value`` everywhere."""
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded; call .load()")
        theta_arr = np.asarray(theta, dtype=np.float64)
        return np.full(theta_arr.shape, self.value, dtype=np.float64)

    def predict_validity(
        self,
        theta: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Phase E surface: every (θ, η) pair is treated as valid (P=1)."""
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded; call .load()")
        out_shape = np.broadcast(theta, eta).shape
        return np.ones(out_shape, dtype=np.float64)

    @property
    def metadata(self) -> dict[str, Any]:
        """Read-only view of the stub metadata; empty before ``load()``."""
        return dict(self._metadata)

    def fingerprint(self) -> str:
        h = hashlib.sha256(f"{self.name}:{self.version}:{self.value}".encode())
        return h.hexdigest()[:16]
