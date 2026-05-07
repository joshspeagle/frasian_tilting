"""Checkpoint save/load helpers for the Phase E learned-η selector.

Two concerns:

1. ``save_checkpoint``: assemble the metadata dict, serialise both
   nets via ``equinox.tree_serialise_leaves`` into a single ``.eqx``
   file with a length-prefixed JSON metadata blob followed by the
   net leaves; write atomically via ``.tmp`` + ``os.replace`` so a
   crashed training run never leaves a corrupt file on disk.

2. ``arch_spec_sha``: SHA-256 over the architecture spec
   (hidden widths, depth, theta_dim, output dim). Recorded in the
   checkpoint and re-checked at load time; mismatch → RuntimeWarning.

The Equinox version is also recorded at save time. Both checks are
**warnings, not errors**: the user gets a heads-up but can still
override and load.

Format
------
``.eqx`` files written here are a thin custom container, not raw
``equinox.tree_serialise_leaves`` output (which only carries leaves,
no metadata). Layout:

    [4 bytes BE uint32: len(meta_json)]
    [meta_json bytes]                          # JSON metadata blob
    [equinox.tree_serialise_leaves(eta_net)]   # variable length
    [equinox.tree_serialise_leaves(val_net)]   # variable length

The eqx-leaves boundary is determined by the eta_net's PyTree
structure (passed in at load time as a fresh skeleton). Equinox's
serialiser writes one length-prefixed block per leaf and stops at
the right offset, so reading proceeds in order.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os as _os
import struct
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .architecture import EtaNet, ValidityNet


CHECKPOINT_FORMAT_VERSION = 3  # bumped from 2 for the Equinox port.


def _architecture_version() -> str:
    """Return the architecture-spec version string.

    Lives here so ``arch_spec_sha`` stays import-light; bumping the
    string flips the sha and triggers ``warn_on_metadata_mismatch``
    even when parameter shapes are unchanged (e.g., on a future
    architecture refactor that swaps the activation).
    """
    return "1.0"


def _equinox_version() -> str | None:
    """Return ``equinox.__version__`` if equinox is importable, else None."""
    try:
        import equinox as _eqx
    except ImportError:  # pragma: no cover
        return None
    return str(_eqx.__version__)


def _jax_version() -> str | None:
    """Return ``jax.__version__`` if jax is importable, else None."""
    try:
        import jax as _jax
    except ImportError:  # pragma: no cover
        return None
    return str(_jax.__version__)


def arch_spec_sha(
    eta_kwargs: dict[str, Any],
    validity_kwargs: dict[str, Any],
) -> str:
    """SHA-256 (24 hex chars) over a stable JSON of the architecture spec.

    The spec covers ``theta_dim`` and ``hidden_sizes`` for both nets
    plus ``architecture.__version__`` — together this identifies the
    parameter-tensor shapes AND the activation / init / layer types.
    Used to detect architecture drift between training time and
    inference time.
    """
    blob = {
        "eta": _stable_kwargs(eta_kwargs),
        "validity": _stable_kwargs(validity_kwargs),
        "arch_version": _architecture_version(),
    }
    payload = json.dumps(blob, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _stable_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    """Coerce kwargs to JSON-serialisable, deterministic-key form."""
    out: dict[str, Any] = {}
    for k, v in sorted(kw.items()):
        if isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _write_eqx_file(
    out_path: Path,
    metadata: dict[str, Any],
    eta_net: "EtaNet",
    val_net: "ValidityNet",
) -> None:
    """Atomically write the (metadata + eta_net + val_net) .eqx container.

    Layout described in the module docstring. Atomic via ``.tmp`` +
    ``os.replace`` so a crashed run never leaves a corrupt checkpoint.
    """
    import equinox as eqx

    meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        fh.write(struct.pack(">I", len(meta_bytes)))
        fh.write(meta_bytes)
        eqx.tree_serialise_leaves(fh, eta_net)
        eqx.tree_serialise_leaves(fh, val_net)
    _os.replace(tmp_path, out_path)


def read_eqx_file(
    in_path: Path,
    eta_skeleton: "EtaNet",
    val_skeleton: "ValidityNet",
) -> tuple[dict[str, Any], "EtaNet", "ValidityNet"]:
    """Read a .eqx container; return (metadata, eta_net, val_net).

    The skeletons are required by ``equinox.tree_deserialise_leaves`` —
    they describe the PyTree structure into which the leaves are
    deserialised. Construct them with the same ``architecture_kwargs``
    the checkpoint was saved with.
    """
    import equinox as eqx

    with open(in_path, "rb") as fh:
        (meta_len,) = struct.unpack(">I", fh.read(4))
        metadata_bytes = fh.read(meta_len)
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        eta_net = eqx.tree_deserialise_leaves(fh, eta_skeleton)
        val_net = eqx.tree_deserialise_leaves(fh, val_skeleton)
    return metadata, eta_net, val_net


def save_checkpoint(
    *,
    out_path: Path,
    eta_net: "EtaNet",
    val_net: "ValidityNet",
    config: Any,  # ExperimentConfig
    loss_kind: str,
    alpha: float | None,
    lambda_max: float,
    lambda_warmup_frac: float,
    n_aux: int,
    lr_a: float,
    lr_b: float,
    weight_decay: float,
    n_epochs: int,
    epochs_run: int,
    stopped_early: bool,
    best_epoch: int,
    patience: int,
    min_delta: float,
    batch_size: int,
    seed: int,
    version: str,
    train_losses: list[float],
    train_width_losses: list[float],
    train_penalty_losses: list[float],
    val_losses: list[float],
    head_b_accuracies: list[float],
    final_val_loss: float,
    final_head_b_accuracy: float,
    final_eta_pred_valid_rate: float,
    antithetic: bool = False,
) -> dict[str, Any]:
    """Assemble + atomically write the checkpoint. Returns the metadata dict.

    The two nets are serialised via ``equinox.tree_serialise_leaves``;
    the metadata dict carries everything else (loss schedules,
    fingerprints, training stats, framework versions). Returns the
    metadata dict for the caller's snapshot.
    """
    eta_kwargs = eta_net.architecture_kwargs()
    validity_kwargs = val_net.architecture_kwargs()
    sha = arch_spec_sha(eta_kwargs, validity_kwargs)

    metadata: dict[str, Any] = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": "EtaNet+ValidityNet",
        "arch_sha": sha,
        "equinox_version": _equinox_version(),
        "jax_version": _jax_version(),
        "eta_architecture_kwargs": eta_kwargs,
        "validity_architecture_kwargs": validity_kwargs,
        "experiment_config": config.to_dict(),
        "loss_kind": loss_kind,
        "alpha": alpha,
        "lambda_max": lambda_max,
        "lambda_warmup_frac": lambda_warmup_frac,
        "n_aux": n_aux,
        "lr_a": lr_a,
        "lr_b": lr_b,
        "weight_decay": weight_decay,
        "n_epochs": n_epochs,
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "patience": patience,
        "min_delta": min_delta,
        "batch_size": batch_size,
        "seed": seed,
        "version": version,
        "antithetic": antithetic,
        "training_finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "train_losses": train_losses,
        "train_width_losses": train_width_losses,
        "train_penalty_losses": train_penalty_losses,
        "val_losses": val_losses,
        "head_b_accuracies": head_b_accuracies,
        "final_val_loss": final_val_loss,
        "final_head_b_accuracy": final_head_b_accuracy,
        "final_eta_pred_valid_rate": final_eta_pred_valid_rate,
    }

    _write_eqx_file(Path(out_path), metadata, eta_net, val_net)
    return metadata


def warn_on_metadata_mismatch(
    state: dict[str, Any],
    *,
    artifact_path: Path | str = "",
) -> None:
    """Emit a ``RuntimeWarning`` if the checkpoint's framework versions
    or arch_sha differ from the current environment.

    Both checks are warnings (not raises) so the user can still load
    the checkpoint and decide whether to retrain. A *missing*
    framework-version or arch_sha key is also warned about, once.

    Parameters
    ----------
    state
        The loaded checkpoint metadata dict.
    artifact_path
        Path to display in the warning message; purely cosmetic.
    """
    cur_eqx = _equinox_version()
    saved_eqx = state.get("equinox_version")
    if saved_eqx is None:
        warnings.warn(
            f"EtaArtifact at {artifact_path}: checkpoint has no "
            f"`equinox_version` field (pre-port checkpoint). Loading "
            f"with current equinox={cur_eqx}; semantics may have "
            f"drifted if the saved equinox was very different.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif saved_eqx != cur_eqx:
        warnings.warn(
            f"EtaArtifact at {artifact_path}: equinox version mismatch. "
            f"Saved with equinox={saved_eqx}, loading with "
            f"equinox={cur_eqx}. Loading anyway; if you hit a "
            f"deserialisation error, retrain via "
            f"`scripts.train_learned_eta`.",
            RuntimeWarning,
            stacklevel=3,
        )

    saved_sha = state.get("arch_sha")
    if saved_sha is None:
        warnings.warn(
            f"EtaArtifact at {artifact_path}: checkpoint has no "
            f"`arch_sha` field (pre-port checkpoint). Cannot verify "
            f"that the architecture spec matches the current "
            f"`architecture.py`.",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    eta_kwargs = state.get("eta_architecture_kwargs") or {}
    val_kwargs = state.get("validity_architecture_kwargs") or {}
    cur_sha = arch_spec_sha(eta_kwargs, val_kwargs)
    if cur_sha != saved_sha:
        warnings.warn(
            f"EtaArtifact at {artifact_path}: arch_sha mismatch. "
            f"Saved sha={saved_sha}, current sha={cur_sha}. The "
            f"checkpoint's recorded architecture kwargs and the "
            f"`arch_spec_sha` algorithm disagree; the checkpoint may "
            f"have been edited or `architecture.py` changed shape.",
            RuntimeWarning,
            stacklevel=3,
        )
