"""Pre-flight helpers for ``fit_eta_artifact``.

These four functions resolve the runtime environment before any
training-side state is created:

- ``resolve_device`` — ``"auto"`` → ``"gpu"`` if a JAX GPU device is
  visible else ``"cpu"``; pass-through otherwise.
- ``enable_determinism`` — JAX is bit-deterministic on CPU at a fixed
  PRNG key, so this is a near no-op: it seeds the global numpy RNG
  for any numpy-side randomness and returns the JAX root key.
- ``validate_loss_kind`` — refuse unknown ``loss_kind`` and the
  ``static_width`` requires-α / ``integrated_p`` forbids-α matrix.
- ``spawn_rngs`` — sub-spawn 4 independent ``np.random.Generator``s.

They are ``train.py``-internal and not part of the public surface;
the orchestrator calls them in order at the top of
``fit_eta_artifact``.
"""

from __future__ import annotations

import jax
import numpy as np

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active

_FORCE_X64 = _x64  # keep static-analysis from stripping the import

_LOSS_KINDS = ("integrated_p", "cd_variance", "static_width")


def resolve_device(device: str) -> str:
    """``"auto"`` → ``"gpu"`` if a JAX GPU device is visible else ``"cpu"``.

    ``"cuda"`` is accepted as a backward-compat alias for ``"gpu"`` (audit
    P0-14: the CLI advertises ``cuda`` but JAX's device platform name is
    ``gpu``; without remapping the metadata field claimed ``cuda`` while
    JAX silently fell back to CPU).

    Pass-through for any non-``"auto"`` non-``"cuda"`` value (the caller
    may force ``"cpu"`` even on a GPU host).
    """
    if device == "cuda":
        device = "gpu"
    if device != "auto":
        return device
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    return "gpu" if gpus else "cpu"


def enable_determinism(seed: int) -> jax.Array:
    """Return a deterministic JAX root key + seed numpy.

    JAX is bit-deterministic on CPU at a fixed ``jax.random.PRNGKey``
    (no global state, no nondeterministic kernels). We do not need
    the legacy torch ``use_deterministic_algorithms`` /
    ``CUBLAS_WORKSPACE_CONFIG`` ceremony.

    Numpy's global RNG is still seeded for any numpy-side randomness
    that isn't routed through ``np.random.default_rng(seed)`` (most
    of the loop is, but a stray ``np.random.*`` call from a third-
    party dependency would otherwise drift across runs).
    """
    np.random.seed(int(seed))
    return jax.random.PRNGKey(int(seed))


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
