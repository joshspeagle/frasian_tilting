"""LearnedArtifact protocol — replaces the legacy global MLP cache.

The legacy code held a module-level `_optimal_eta_predictor` that lazily
loaded a MLP from disk. That global broke testability and parallel safety.
Here, a `LearnedArtifact` is an explicit dependency injected into whatever
`EtaSelector` (or other consumer) needs it. Tests inject stubs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class LearnedArtifact(Protocol):
    """An offline-trained model with an explicit lifecycle.

    Invariants:
        - `load()` is idempotent.
        - After `load()`, `predict_eta` / `predict_validity` are callable;
          before, they raise `MissingArtifactError`.
        - `version` and `artifact_path` uniquely identify the artifact and
          are recorded in any `RawResult` whose computation depends on it.
        - `metadata` exposes a read-only view of the loaded checkpoint's
          metadata (e.g. ``checkpoint_format_version``,
          ``experiment_config``); empty before ``load()``.

    The Phase E surface (``predict_eta`` / ``predict_validity`` /
    ``metadata``) is part of the Protocol so that
    ``LearnedDynamicEtaSelector`` can accept any structurally
    compatible artifact (the concrete ``EtaArtifact``, the
    ``NullArtifact`` test stub, or a future ``WeightedEtaArtifact``).
    """

    name: str
    version: str
    artifact_path: Path

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def load(self) -> None: ...
    def predict_eta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_validity(
        self,
        theta: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    def fingerprint(self) -> str: ...
