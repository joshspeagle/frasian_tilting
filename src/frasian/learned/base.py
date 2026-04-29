"""LearnedArtifact protocol — replaces the legacy global MLP cache.

The legacy code held a module-level `_optimal_eta_predictor` that lazily
loaded a MLP from disk. That global broke testability and parallel safety.
Here, a `LearnedArtifact` is an explicit dependency injected into whatever
`EtaSelector` (or other consumer) needs it. Tests inject stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class LearnedArtifact(Protocol):
    """An offline-trained model with an explicit lifecycle.

    Invariants:
        - `load()` is idempotent.
        - After `load()`, `predict` is callable; before, it raises
          `MissingArtifactError`.
        - `version` and `artifact_path` uniquely identify the artifact and
          are recorded in any `RawResult` whose computation depends on it.
    """

    name: str
    version: str
    artifact_path: Path

    def load(self) -> None: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fingerprint(self) -> str: ...
