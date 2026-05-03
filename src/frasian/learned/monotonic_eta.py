"""MonotonicEtaArtifact — a `LearnedArtifact` wrapping a trained MLP.

The artifact holds a trained `MonotonicEtaNet` (defined in
`frasian.learned.training.architecture`) plus metadata describing
which scheme it was trained against, which loss, and which α mode.

Lifecycle:
  - Construct with the path to a `.pt` checkpoint produced by
    `scripts/train_learned_eta.py`.
  - Call `.load()` to read the checkpoint (lazy: imports torch only
    here). After load, `.predict(x)` is callable.
  - `.predict(x)` takes a numpy `(N, 2)` array of `[w, |Δ'|]` rows and
    returns a numpy `(N,)` array of `η'` values in `(0, 1)`. The
    selector applies `eta_inverse` per-scheme to recover η.

Self-describing checkpoints: every checkpoint records the scheme it
was trained for, the loss, the α mode (marginalised vs fixed), and
the training distribution. A consumer (the selector) verifies these
match its inference context at `load()` time and raises
`MissingArtifactError` on mismatch.

Checkpoint format (v1; documented in `docs/methods/learned_eta.md`):

    REQUIRED keys:
      "checkpoint_format_version": int       (= 1 for this format)
      "architecture":               str      (= "MonotonicEtaNet")
      "architecture_kwargs":        dict     (kwargs to MonotonicEtaNet.__init__,
                                               required even if empty so the
                                               trained weights match the
                                               re-instantiated module)
      "model_state_dict":           torch state dict
      "scheme":                     str      ("power_law", "ot", ...)
      "loss":                       str      ("integrated_p" | "cd_variance"
                                               | "static_width")
      "alpha_mode":                 str      ("marginalised" or
                                               f"fixed_{alpha:.6g}")

    OPTIONAL keys (recommended):
      "training_distribution":      dict     (serialised TrainingDistribution)
      "n_lhs", "n_mc", "n_epochs":  int      (training budgets)
      "seed":                       int
      "version":                    str      (artifact version, e.g. "v0_smoke")
      "calibration_report":         dict     (5x5 (θ_true, w) coverage table)
      "training_finished_at":       str      (ISO timestamp)

`load()` validates the REQUIRED keys explicitly and raises
`MissingArtifactError` with a precise message on missing or wrong-typed
fields. Format-version mismatch also raises (no silent migration).

The architecture and training loop live under
`frasian.learned.training`; this module is the inference-only
boundary so the rest of the framework can use trained models
without ever importing torch from a hot path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._errors import MissingArtifactError

CHECKPOINT_FORMAT_VERSION = 1
_REQUIRED_KEYS = (
    "checkpoint_format_version",
    "architecture",
    "architecture_kwargs",
    "model_state_dict",
    "scheme",
    "loss",
    "alpha_mode",
)


@dataclass
class MonotonicEtaArtifact:
    """Wraps a trained `MonotonicEtaNet` for use by `LearnedDynamicEtaSelector`.

    Attributes
    ----------
    artifact_path
        Path to the `.pt` checkpoint.
    name
        Identifier (default `"monotonic_eta"`).
    version
        Recorded in the checkpoint and on reproducibility metadata.

    Lazily-populated state (all begin `None`; `load()` fills them):
        - `_model`: the torch nn.Module
        - `_metadata`: dict with `scheme`, `loss`, `alpha_mode`, ...
        - `_loaded`: True after a successful `load()`
    """

    artifact_path: Path = Path("artifacts/learned_eta_v0_smoke.pt")
    name: str = "monotonic_eta"
    version: str = "v0"
    device: str = "auto"

    _model: Any = field(default=None, init=False, repr=False, compare=False)
    _metadata: dict[str, Any] = field(default_factory=dict, init=False,
                                        repr=False, compare=False)
    _loaded: bool = field(default=False, init=False, repr=False, compare=False)
    _device: str = field(default="cpu", init=False, repr=False, compare=False)

    def load(self) -> None:
        """Read the checkpoint, instantiate the model, validate metadata.

        Idempotent: subsequent calls are no-ops.
        """
        if self._loaded:
            return

        path = Path(self.artifact_path)
        if not path.exists():
            raise MissingArtifactError(
                f"MonotonicEtaArtifact: checkpoint not found at {path}. "
                f"Train via `python -m scripts.train_learned_eta ...` "
                f"or point `artifact_path` at an existing `.pt` file."
            )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded path
            raise ImportError(
                "MonotonicEtaArtifact.load requires torch; install via "
                "`pip install -e \".[ml]\"`."
            ) from exc

        from .training.architecture import MonotonicEtaNet

        device = self._resolve_device()
        state = torch.load(str(path), map_location=device, weights_only=False)

        # Validate required keys (see module docstring for format spec).
        missing = [k for k in _REQUIRED_KEYS if k not in state]
        if missing:
            raise MissingArtifactError(
                f"MonotonicEtaArtifact: checkpoint at {path} missing "
                f"required keys {missing}; got keys {sorted(state)}."
            )
        version = state["checkpoint_format_version"]
        if version != CHECKPOINT_FORMAT_VERSION:
            raise MissingArtifactError(
                f"MonotonicEtaArtifact: checkpoint format version "
                f"{version} != expected {CHECKPOINT_FORMAT_VERSION}; "
                f"re-train or migrate."
            )
        if state["architecture"] != "MonotonicEtaNet":
            raise MissingArtifactError(
                f"MonotonicEtaArtifact: expected architecture "
                f"'MonotonicEtaNet', got {state['architecture']!r}."
            )
        if not isinstance(state["architecture_kwargs"], dict):
            raise MissingArtifactError(
                f"MonotonicEtaArtifact: architecture_kwargs must be a dict; "
                f"got {type(state['architecture_kwargs']).__name__!r}."
            )

        net = MonotonicEtaNet(**state["architecture_kwargs"])
        net.load_state_dict(state["model_state_dict"])
        net.to(device)
        net.eval()

        self._model = net
        self._metadata = {
            k: v for k, v in state.items() if k != "model_state_dict"
        }
        self._device = device
        self._loaded = True

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict η' on a `(N, 2)` `[w, |Δ'|]` feature matrix.

        Returns a `(N,)` array in `(0, 1)`. The selector applies
        `eta_inverse(scheme_name, η', w)` to recover η.
        """
        if not self._loaded:
            raise MissingArtifactError(
                f"{self.name} not loaded; call .load()"
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("predict requires torch") from exc

        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[1] != 2:
            raise ValueError(
                f"predict expects a (N, 2) array of [w, |Δ'|]; got "
                f"shape {x_arr.shape!r}."
            )

        with torch.no_grad():
            x_t = torch.as_tensor(x_arr, dtype=torch.float32,
                                    device=self._device)
            y_t = self._model(x_t)
        return np.asarray(y_t.cpu().numpy(), dtype=np.float64).reshape(-1)

    def fingerprint(self) -> str:
        """Hash of `(name, version, sha256(checkpoint_bytes))`.

        Recorded by the cache so a different artifact triggers
        recomputation. Does not require the model to be loaded — just
        the file on disk.
        """
        path = Path(self.artifact_path)
        h = hashlib.sha256(f"{self.name}:{self.version}:".encode())
        if path.exists():
            h.update(path.read_bytes())
        else:
            h.update(b"<missing>")
        return h.hexdigest()[:16]

    @property
    def metadata(self) -> dict[str, Any]:
        """Read-only view of the loaded checkpoint's metadata.

        Empty until `load()` is called. After load, includes at least
        `scheme`, `loss`, `alpha_mode`.
        """
        return dict(self._metadata)

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
