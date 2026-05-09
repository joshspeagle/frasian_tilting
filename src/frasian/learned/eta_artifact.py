"""EtaArtifact — wraps a Phase E checkpoint (EtaNet + ValidityNet + config).

The Phase E checkpoint format v3 (post-Equinox port) is a self-
describing binary container; see ``training/_checkpoint.py`` for the
on-disk layout. It carries:

- ``EtaNet`` weights (θ → η).
- ``ValidityNet`` weights ((θ, η) → logit). Loaded but not used at
  inference; kept for introspection / illustrations of the learned
  boundary.
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

CHECKPOINT_FORMAT_VERSION = 4
_REQUIRED_KEYS = (
    "checkpoint_format_version",
    "architecture",
    "eta_architecture_kwargs",
    "validity_architecture_kwargs",
    "experiment_config",
)


@dataclass
class EtaArtifact:
    """Wraps a trained ``EtaNet`` + ``ValidityNet`` pair (Phase E v3 / Equinox port).

    Attributes
    ----------
    artifact_path
        Path to the ``.eqx`` checkpoint produced by
        ``fit_eta_artifact``.
    name
        Identifier (default ``"learned_eta"``).
    version
        Recorded in the checkpoint's metadata.

    Lazily populated by ``load()``:
        - ``_eta_net``: the EtaNet (an ``equinox.Module``)
        - ``_validity_net``: the ValidityNet
        - ``_metadata``: full checkpoint metadata dict
    """

    artifact_path: Path = Path("artifacts/learned_eta_v0_smoke.eqx")
    name: str = "learned_eta"
    version: str = "v0"
    device: str = "auto"

    _eta_net: Any = field(default=None, init=False, repr=False, compare=False)
    _validity_net: Any = field(default=None, init=False, repr=False, compare=False)
    _metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _loaded: bool = field(default=False, init=False, repr=False, compare=False)
    _eta_predict_jit: Any = field(default=None, init=False, repr=False, compare=False)
    _validity_predict_jit: Any = field(default=None, init=False, repr=False, compare=False)

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
            import jax
        except ImportError as exc:  # pragma: no cover
            raise ImportError("EtaArtifact.load requires jax.") from exc

        from .training._checkpoint import read_eqx_file, warn_on_metadata_mismatch
        from .training.architecture import EtaNet, ValidityNet

        # We need a peek at the metadata to know which architecture
        # kwargs to use for the skeleton; ``read_eqx_file`` reads the
        # metadata first, then the leaves. Build skeletons in two
        # steps: first read metadata via a bare-minimum skeleton path
        # (since we need shapes to construct skeletons), we instead
        # parse metadata directly from the file header.
        metadata = _read_metadata_only(path)
        missing = [k for k in _REQUIRED_KEYS if k not in metadata]
        if missing:
            raise MissingArtifactError(
                f"EtaArtifact: checkpoint at {path} missing required "
                f"keys {missing}; got keys {sorted(metadata)}."
            )
        v = metadata["checkpoint_format_version"]
        if v == 3:
            raise MissingArtifactError(
                f"EtaArtifact: checkpoint at {path} is format version 3 "
                f"(fixed-prior architecture, Phase B/C). v4 introduces the "
                f"conditional architecture (prior_hp + lik_hp inputs). "
                f"Re-train via `python -m scripts.train_learned_eta "
                f"--config <updated_v4.yaml>`. See "
                f"`docs/methods/learned_eta.md` migration section for the "
                f"YAML schema change."
            )
        if v != CHECKPOINT_FORMAT_VERSION:
            raise MissingArtifactError(
                f"EtaArtifact: checkpoint format version {v} != "
                f"expected {CHECKPOINT_FORMAT_VERSION}; re-train via "
                f"`python -m scripts.train_learned_eta --config "
                f"<experiment.yaml>`."
            )
        if metadata["architecture"] != "EtaNet+ValidityNet":
            raise MissingArtifactError(
                f"EtaArtifact: expected architecture "
                f"'EtaNet+ValidityNet', got {metadata['architecture']!r}."
            )

        # Equinox version + arch_sha compat warnings.
        warn_on_metadata_mismatch(metadata, artifact_path=path)

        # Build skeletons with a fresh PRNG key — the leaves will be
        # overwritten by deserialisation. Splitting from PRNGKey(0)
        # deterministically yields the same bytes every load (the
        # init isn't used).
        eta_kwargs = metadata["eta_architecture_kwargs"]
        val_kwargs = metadata["validity_architecture_kwargs"]
        skel_key_a, skel_key_b = jax.random.split(jax.random.PRNGKey(0))
        eta_skeleton = EtaNet(**eta_kwargs, key=skel_key_a)
        val_skeleton = ValidityNet(**val_kwargs, key=skel_key_b)

        _, eta_net, val_net = read_eqx_file(path, eta_skeleton, val_skeleton)

        self._eta_net = eta_net
        self._validity_net = val_net
        self._metadata = metadata
        self._loaded = True

    def predict_eta(
        self,
        theta: NDArray[np.float64],
        prior_hp: NDArray[np.float64],
        lik_hp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict η at (θ, prior_hp, lik_hp) (Phase G conditional).

        theta:    `(N,)` array of θ values to predict η at.
        prior_hp: `(prior_dim,)` 1-D vector of the prior's hyperparams,
                  broadcast across all N θ values.
        lik_hp:   `(lik_dim,)` 1-D vector of the model's hyperparams,
                  broadcast across all N θ values.
        """
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded; call .load()")
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover
            raise ImportError("predict_eta requires jax") from exc

        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.ndim != 1:
            raise ValueError(f"predict_eta expects 1D theta; got shape {theta_arr.shape!r}")
        prior_arr = np.asarray(prior_hp, dtype=np.float64).reshape(-1)
        lik_arr = np.asarray(lik_hp, dtype=np.float64).reshape(-1)
        N = theta_arr.shape[0]
        prior_b = np.broadcast_to(prior_arr[None, :], (N, prior_arr.size)).copy()
        lik_b = np.broadcast_to(lik_arr[None, :], (N, lik_arr.size)).copy()

        if self._eta_predict_jit is None:
            import equinox as eqx
            net = self._eta_net
            self._eta_predict_jit = eqx.filter_jit(
                lambda t, p, l: net(t, p, l)
            )
        eta_out = self._eta_predict_jit(
            jnp.asarray(theta_arr),
            jnp.asarray(prior_b),
            jnp.asarray(lik_b),
        )
        return np.asarray(eta_out, dtype=np.float64).reshape(-1)

    def predict_validity(
        self,
        theta: NDArray[np.float64],
        prior_hp: NDArray[np.float64],
        lik_hp: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict P(valid | θ, prior_hp, lik_hp, η) (Phase G conditional).

        ``theta`` and ``eta`` are broadcast-compatible arrays; ``prior_hp``
        and ``lik_hp`` are 1-D vectors broadcast across the broadcast
        shape.
        """
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded; call .load()")
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp
            import jax.nn as jnn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("predict_validity requires jax") from exc

        out_shape = np.broadcast(theta, eta).shape
        theta_arr = np.broadcast_to(
            np.asarray(theta, dtype=np.float64), out_shape,
        ).ravel()
        eta_arr = np.broadcast_to(
            np.asarray(eta, dtype=np.float64), out_shape,
        ).ravel()
        prior_arr = np.asarray(prior_hp, dtype=np.float64).reshape(-1)
        lik_arr = np.asarray(lik_hp, dtype=np.float64).reshape(-1)
        N = theta_arr.shape[0]
        prior_b = np.broadcast_to(prior_arr[None, :], (N, prior_arr.size)).copy()
        lik_b = np.broadcast_to(lik_arr[None, :], (N, lik_arr.size)).copy()

        if self._validity_predict_jit is None:
            import equinox as eqx
            net = self._validity_net

            def _forward(t, p, l, e):
                return jnn.sigmoid(net(t, p, l, e))

            self._validity_predict_jit = eqx.filter_jit(_forward)

        p = self._validity_predict_jit(
            jnp.asarray(theta_arr),
            jnp.asarray(prior_b),
            jnp.asarray(lik_b),
            jnp.asarray(eta_arr),
        )
        return np.asarray(p, dtype=np.float64).reshape(out_shape)

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


_MAX_METADATA_BYTES: int = 16 * 1024 * 1024  # 16 MiB hard cap


def _read_metadata_only(path: Path) -> dict[str, Any]:
    """Read just the JSON metadata header of a .eqx file.

    Layout: 4-byte BE length prefix, then JSON metadata bytes, then
    the equinox-serialised leaves (which we ignore here).

    Audit P2 (Cluster F): the 4-byte length prefix can declare up to
    ~4 GiB. A maliciously-crafted .eqx file could request a multi-GiB
    allocation before any JSON parsing happens. Real headers are tens
    of KiB; we cap at 16 MiB and refuse anything larger with a
    `MissingArtifactError`. The cap also bounds the file-size sanity
    check (raises if the on-disk file is smaller than the declared
    header, which would be `read()` returning truncated bytes that
    `json.loads` would fail on with a less clear error).
    """
    import json
    import struct

    file_size = path.stat().st_size
    if file_size < 4:
        raise MissingArtifactError(
            f"EtaArtifact: {path} is too small ({file_size} bytes) "
            f"to contain a metadata length prefix; checkpoint is "
            f"corrupt."
        )
    with open(path, "rb") as fh:
        (meta_len,) = struct.unpack(">I", fh.read(4))
        if meta_len > _MAX_METADATA_BYTES:
            raise MissingArtifactError(
                f"EtaArtifact: {path} declares a {meta_len}-byte metadata "
                f"header (cap is {_MAX_METADATA_BYTES} bytes / "
                f"{_MAX_METADATA_BYTES // (1024*1024)} MiB). Refusing to "
                f"read; checkpoint is either corrupt or adversarial."
            )
        if 4 + meta_len > file_size:
            raise MissingArtifactError(
                f"EtaArtifact: {path} declares a {meta_len}-byte metadata "
                f"header but the file is only {file_size} bytes; "
                f"checkpoint is truncated."
            )
        return json.loads(fh.read(meta_len).decode("utf-8"))
