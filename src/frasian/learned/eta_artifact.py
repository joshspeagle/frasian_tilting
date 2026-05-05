"""EtaArtifact — wraps a Phase E checkpoint (EtaNet + ValidityNet + config).

The Phase E checkpoint has format version 2 and contains:

- ``eta_state_dict`` — trained ``EtaNet`` weights (θ → η).
- ``validity_state_dict`` — trained ``ValidityNet`` weights
  ((θ, η) → logit). Loaded but not used at inference; kept for
  introspection / illustrations of the learned boundary.
- ``experiment_config`` — round-tripped ``ExperimentConfig.to_dict()``
  with prior / model / theta_distribution fingerprints. The selector
  compares these against the inference-time objects to refuse cross-
  experiment use.
- λ schedule + training metrics + ``final_head_b_accuracy`` etc.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._errors import MissingArtifactError

CHECKPOINT_FORMAT_VERSION = 2
_REQUIRED_KEYS = (
    "checkpoint_format_version",
    "architecture",
    "eta_architecture_kwargs",
    "validity_architecture_kwargs",
    "eta_state_dict",
    "validity_state_dict",
    "experiment_config",
)


@dataclass
class EtaArtifact:
    """Wraps a trained ``EtaNet`` + ``ValidityNet`` pair (Phase E v2).

    Attributes
    ----------
    artifact_path
        Path to the ``.pt`` checkpoint produced by
        ``fit_eta_artifact``.
    name
        Identifier (default ``"learned_eta"``).
    version
        Recorded in the checkpoint's metadata.

    Lazily populated by ``load()``:
        - ``_eta_net``: the EtaNet
        - ``_validity_net``: the ValidityNet
        - ``_metadata``: full checkpoint dict minus the state dicts
    """

    artifact_path: Path = Path("artifacts/learned_eta_v0_smoke.pt")
    name: str = "learned_eta"
    version: str = "v0"
    device: str = "auto"

    _eta_net: Any = field(default=None, init=False, repr=False, compare=False)
    _validity_net: Any = field(default=None, init=False, repr=False, compare=False)
    _metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _loaded: bool = field(default=False, init=False, repr=False, compare=False)
    _device: str = field(default="cpu", init=False, repr=False, compare=False)

    def load(self) -> None:
        """Read the checkpoint, instantiate both nets, validate required keys.

        Idempotent.
        """
        if self._loaded:
            return

        path = Path(self.artifact_path)
        if not path.exists():
            raise MissingArtifactError(
                f"EtaArtifact: checkpoint not found at {path}. "
                f"Train via `python -m scripts.train_learned_eta "
                f"--config <experiment.yaml> --out {path}`."
            )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("EtaArtifact.load requires torch.") from exc

        from .training._checkpoint import warn_on_metadata_mismatch
        from .training.architecture import EtaNet, ValidityNet

        device = self._resolve_device()
        state = torch.load(str(path), map_location=device, weights_only=False)

        missing = [k for k in _REQUIRED_KEYS if k not in state]
        if missing:
            raise MissingArtifactError(
                f"EtaArtifact: checkpoint at {path} missing required "
                f"keys {missing}; got keys {sorted(state)}."
            )
        v = state["checkpoint_format_version"]
        if v != CHECKPOINT_FORMAT_VERSION:
            raise MissingArtifactError(
                f"EtaArtifact: checkpoint format version {v} != "
                f"expected {CHECKPOINT_FORMAT_VERSION}; re-train via "
                f"`python -m scripts.train_learned_eta --config "
                f"<experiment.yaml>`."
            )
        if state["architecture"] != "EtaNet+ValidityNet":
            raise MissingArtifactError(
                f"EtaArtifact: expected architecture "
                f"'EtaNet+ValidityNet', got {state['architecture']!r}."
            )

        # 1.4-S3 / 1.2-NN3: torch version + architecture-shape compat
        # diagnostic. Both are warnings (not raises) so a user with a
        # slightly different torch can still load and decide whether
        # to retrain.
        warn_on_metadata_mismatch(state, artifact_path=path)

        eta_net = EtaNet(**state["eta_architecture_kwargs"])
        eta_net.load_state_dict(state["eta_state_dict"])
        eta_net.to(device)
        eta_net.eval()

        val_net = ValidityNet(**state["validity_architecture_kwargs"])
        val_net.load_state_dict(state["validity_state_dict"])
        val_net.to(device)
        val_net.eval()

        self._eta_net = eta_net
        self._validity_net = val_net
        self._metadata = {
            k: v for k, v in state.items() if k not in ("eta_state_dict", "validity_state_dict")
        }
        self._device = device
        self._loaded = True

    def predict_eta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict η on a (N,) θ array. Returns a (N,) array."""
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded; call .load()")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("predict_eta requires torch") from exc

        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.ndim != 1:
            raise ValueError(f"predict_eta expects 1D θ; got shape {theta_arr.shape!r}")
        with torch.no_grad():
            theta_t = torch.as_tensor(theta_arr, dtype=torch.float32, device=self._device)
            eta_t = self._eta_net(theta_t)
        return np.asarray(eta_t.cpu().numpy(), dtype=np.float64).reshape(-1)

    def predict_validity(
        self,
        theta: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict P(valid | θ, η) on broadcast-compatible (θ, η) arrays.

        Used by the illustration to plot the learned boundary; not
        used at inference time (the trained η is already inside the
        valid region by construction).
        """
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded; call .load()")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("predict_validity requires torch") from exc

        theta_arr = np.broadcast_to(
            np.asarray(theta, dtype=np.float64),
            np.broadcast(theta, eta).shape,
        ).ravel()
        eta_arr = np.broadcast_to(
            np.asarray(eta, dtype=np.float64),
            np.broadcast(theta, eta).shape,
        ).ravel()
        out_shape = np.broadcast(theta, eta).shape
        with torch.no_grad():
            inputs = torch.as_tensor(
                np.column_stack([theta_arr, eta_arr]),
                dtype=torch.float32,
                device=self._device,
            )
            logits = self._validity_net(inputs)
            p = torch.sigmoid(logits)
        return np.asarray(p.cpu().numpy(), dtype=np.float64).reshape(out_shape)

    def fingerprint(self) -> str:
        """Hash of (name, version, sha256(checkpoint_bytes)). 16 hex chars."""
        path = Path(self.artifact_path)
        h = hashlib.sha256(f"{self.name}:{self.version}:".encode())
        if path.exists():
            h.update(path.read_bytes())
        else:
            h.update(b"<missing>")
        return h.hexdigest()[:16]

    @property
    def metadata(self) -> dict[str, Any]:
        """Read-only view of the loaded checkpoint's metadata."""
        return dict(self._metadata)

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
