"""Pre-flight helpers for ``fit_eta_artifact`` (Phase 4 skeptic §8 split).

These four functions resolve the runtime environment before any
training-side state is created:

- ``resolve_device`` — ``"auto"`` → ``"cuda"``/``"cpu"``.
- ``enable_determinism`` — ``torch.use_deterministic_algorithms`` +
  ``cudnn.deterministic`` + seed numpy and torch.
- ``validate_loss_kind`` — refuse unknown ``loss_kind`` and the
  ``static_width`` requires-α / ``integrated_p`` forbids-α matrix.
- ``spawn_rngs`` — sub-spawn 4 independent ``np.random.Generator``s.

They are ``train.py``-internal and not part of the public surface;
the orchestrator calls them in order at the top of
``fit_eta_artifact``.
"""

from __future__ import annotations

import os as _os

import numpy as np
import torch

_LOSS_KINDS = ("integrated_p", "cd_variance", "static_width")


def resolve_device(device: str) -> str:
    """``"auto"`` → ``"cuda"`` if available else ``"cpu"``; pass through otherwise."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def enable_determinism(seed: int) -> None:
    """Enable deterministic torch + numpy paths.

    Called at the top of ``fit_eta_artifact`` before any torch-side
    state is created. CUBLAS deterministic mode requires the
    workspace config env var; we set it lazily so users who don't
    care about CUDA are unaffected.
    """
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def validate_loss_kind(loss_kind: str, alpha: float | None) -> None:
    """Refuse unknown ``loss_kind`` and enforce the alpha-presence matrix.

    - ``static_width`` *requires* a finite α ∈ (0, 1).
    - ``integrated_p`` / ``cd_variance`` are α-marginalised; recording
      a non-None α at training time would lock the checkpoint to that
      one α at inference, defeating the marginalisation. Refuse loudly.
    """
    if loss_kind not in _LOSS_KINDS:
        raise ValueError(f"loss_kind must be one of {_LOSS_KINDS}; got {loss_kind!r}")
    if loss_kind == "static_width":
        if alpha is None:
            raise ValueError("loss_kind=static_width requires alpha")
        if not (np.isfinite(alpha) and 0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be finite and in (0, 1); got {alpha!r}")
    elif alpha is not None:
        raise ValueError(
            f"alpha={alpha} given but loss_kind={loss_kind!r} is "
            f"α-marginalised; pass alpha=None."
        )


def spawn_rngs(
    seed: int,
) -> tuple[
    np.random.Generator, np.random.Generator, np.random.Generator, np.random.Generator
]:
    """Sub-spawn 4 independent RNGs (skeptic block #11)."""
    base_rng = np.random.default_rng(seed)
    if hasattr(base_rng, "spawn"):
        rngs = [np.random.default_rng(s) for s in base_rng.spawn(4) if s is not None]
    else:
        rngs = [np.random.default_rng(seed + i) for i in range(4)]
    return rngs[0], rngs[1], rngs[2], rngs[3]
