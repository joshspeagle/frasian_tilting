"""Checkpoint save/load helpers for the Phase E learned-η selector.

Extracted from ``train.py`` (Tier 1.2 §7 split) and
``eta_artifact.py`` (Tier 1.4 S3). Two concerns:

1. ``save_checkpoint``: assemble the metadata dict, write atomically
   via ``.tmp`` + ``os.replace`` so a crashed training run never
   leaves a corrupt ``.pt`` on disk.

2. ``arch_spec_sha``: SHA-256 over the architecture spec
   (hidden widths, depth, theta_dim, output dim). Recorded in the
   checkpoint and re-checked at load time; mismatch → RuntimeWarning.

The torch version is also recorded at save time. Both checks are
**warnings, not errors**: the user gets a heads-up but can still
override and load.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os as _os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .architecture import EtaNet, ValidityNet


def _torch_version() -> str | None:
    """Return ``torch.__version__`` if torch is importable, else None.

    Local-import keeps ``arch_spec_sha`` and
    ``warn_on_metadata_mismatch`` torch-free so they can be unit-
    tested without torch installed (the audit env is torch-free).
    """
    try:
        import torch as _torch
    except ImportError:  # pragma: no cover
        return None
    return str(_torch.__version__)


def arch_spec_sha(
    eta_kwargs: dict[str, Any],
    validity_kwargs: dict[str, Any],
) -> str:
    """SHA-256 (24 hex chars) over a stable JSON of the architecture spec.

    The spec covers ``theta_dim`` and ``hidden_sizes`` for both nets —
    everything that uniquely identifies the parameter-tensor shapes.
    Used to detect architecture drift between training time and
    inference time (e.g., training on a future torch where
    ``architecture.py`` has been modified).
    """
    blob = {
        "eta": _stable_kwargs(eta_kwargs),
        "validity": _stable_kwargs(validity_kwargs),
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


def save_checkpoint(
    *,
    out_path: Path,
    eta_net: EtaNet,
    val_net: ValidityNet,
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

    The full state dict is in ``state["eta_state_dict"]`` /
    ``["validity_state_dict"]``; the returned dict strips those for
    the caller's metadata snapshot.
    """
    import torch

    eta_kwargs = eta_net.architecture_kwargs()
    validity_kwargs = val_net.architecture_kwargs()
    sha = arch_spec_sha(eta_kwargs, validity_kwargs)

    state: dict[str, Any] = {
        "checkpoint_format_version": 2,  # E.2 bump
        "architecture": "EtaNet+ValidityNet",
        "arch_sha": sha,
        "torch_version": torch.__version__,
        "eta_architecture_kwargs": eta_kwargs,
        "validity_architecture_kwargs": validity_kwargs,
        "eta_state_dict": eta_net.state_dict(),
        "validity_state_dict": val_net.state_dict(),
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

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Skeptic block #8: atomic write.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(state, str(tmp_path))
    _os.replace(tmp_path, out_path)  # atomic

    return {
        k: v for k, v in state.items() if k not in ("eta_state_dict", "validity_state_dict")
    }


def warn_on_metadata_mismatch(
    state: dict[str, Any],
    *,
    artifact_path: Path | str = "",
) -> None:
    """Emit a ``RuntimeWarning`` if the checkpoint's torch_version /
    arch_sha differ from the current environment.

    Both checks are warnings (not raises) so the user can still load
    the checkpoint and decide whether to retrain. A *missing* torch
    version or arch_sha key (from a pre-1.4-S3 checkpoint) is also
    warned about, once.

    Parameters
    ----------
    state
        The loaded checkpoint dict.
    artifact_path
        Path to display in the warning message; purely cosmetic.
    """
    cur_torch = _torch_version()
    saved_torch = state.get("torch_version")
    if saved_torch is None:
        warnings.warn(
            f"EtaArtifact at {artifact_path}: checkpoint has no "
            f"`torch_version` field (pre-1.4-S3 checkpoint). Loading "
            f"with current torch={cur_torch}; semantics may have "
            f"drifted if the saved torch was very different.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif saved_torch != cur_torch:
        warnings.warn(
            f"EtaArtifact at {artifact_path}: torch version mismatch. "
            f"Saved with torch={saved_torch}, loading with "
            f"torch={cur_torch}. Loading anyway; if you hit a "
            f"`Missing key(s)` or shape-mismatch error, retrain via "
            f"`scripts.train_learned_eta`.",
            RuntimeWarning,
            stacklevel=3,
        )

    saved_sha = state.get("arch_sha")
    if saved_sha is None:
        warnings.warn(
            f"EtaArtifact at {artifact_path}: checkpoint has no "
            f"`arch_sha` field (pre-1.4-S3 checkpoint). Cannot verify "
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
