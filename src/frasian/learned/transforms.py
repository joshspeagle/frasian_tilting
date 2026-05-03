"""Coordinate transforms for the learned-η selector input/output space.

Maps the unbounded (|Δ|, η) space to the bounded `(|Δ'|, η')` space
that the MLP operates on. Numpy versions are used at inference (and
in tests); torch versions (guarded by `try: import torch`) are used in
training only.

Transforms (ported verbatim from
`legacy/src/frasian/simulations/mlp_data.py:23-62`):

    Δ' = Δ / (1 + Δ)            maps [0, ∞)        → [0, 1)
    η' = η · (1 - w) + w         maps [η_min(w), 1] → [0, 1]
    η_min(w) = -w / (1 - w)      lower clamp from variance-positivity

Inverses are exact algebraic inverses with the same domain as their
forward maps. Used by:
- `MonotonicEtaArtifact.predict` (numpy) — preprocesses |Δ| inputs and
  postprocesses η' outputs.
- training/sampling.py (torch) — same transforms applied in the data
  pipeline; the MLP sees `(|Δ'|, w)` and outputs `η'`.

Note: the **forward `η_transform` is power-law-specific** (the
`η_min(w) = -w/(1-w)` lower clamp is power_law's admissible-range
floor). For OT, η ∈ [0, 1] strictly so `η_transform_ot(η) = η`. The
generic `eta_transform` / `eta_inverse` helpers below take the
admissible range explicitly; per-scheme dispatch is handled by callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ----- numpy versions (always available) -----


# Boundary-safety constants. The transforms divide by (1-x); we clamp x
# strictly below 1 to keep the result finite for any grid the framework
# might generate (coverage corners can hit |Δ| = O(50), which bumps Δ'
# against 1.0 within float64 resolution).
_DELTA_PRIME_MAX = 1.0 - 1e-12
_W_MIN = 1e-6
_W_MAX = 1.0 - 1e-6


def _validate_w(w: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip `w` to `[_W_MIN, _W_MAX]` to keep `1-w` strictly positive."""
    return np.clip(w, _W_MIN, _W_MAX)


def delta_transform(delta: ArrayLike) -> NDArray[np.float64]:
    """Δ → Δ' = Δ/(1+Δ). Maps `[0, ∞]` → `[0, 1]` (image clipped at
    `1 - 1e-12` to keep the inverse bounded).

    Handles `delta = np.inf` cleanly (returns `1 - 1e-12`).
    """
    delta = np.asarray(delta, dtype=np.float64)
    # delta/(1+delta) returns NaN for inf+inf in numpy; rewrite as 1/(1 + 1/delta).
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            np.isinf(delta),
            np.full_like(delta, _DELTA_PRIME_MAX),
            delta / (1.0 + delta),
        )
    return np.clip(out, 0.0, _DELTA_PRIME_MAX)


def delta_inverse(delta_prime: ArrayLike) -> NDArray[np.float64]:
    """Δ' → Δ = Δ'/(1-Δ'). Maps `[0, 1)` → `[0, ∞)`.

    Input is clamped to `[0, 1 - 1e-12]` to keep the result finite.
    """
    delta_prime = np.asarray(delta_prime, dtype=np.float64)
    delta_prime = np.clip(delta_prime, 0.0, _DELTA_PRIME_MAX)
    return delta_prime / (1.0 - delta_prime)


def eta_min_powerlaw(w: ArrayLike) -> NDArray[np.float64]:
    """Lower bound on η for `power_law`: η_min(w) = -w/(1-w).

    Below this, the tilted variance becomes non-positive
    (`denom = 1 - η(1-w) ≤ 0`) and `tilt` raises `TiltingDomainError`.
    `w` is clamped to `[1e-6, 1-1e-6]` to keep the result finite.
    """
    w = _validate_w(np.asarray(w, dtype=np.float64))
    return -w / (1.0 - w)


def eta_transform_powerlaw(eta: ArrayLike, w: ArrayLike
                            ) -> NDArray[np.float64]:
    """η → η' = η(1-w) + w for `power_law`. Maps [η_min(w), 1] → [0, 1].

    η_min(w) = -w/(1-w) ↦ 0; η = 1 ↦ 1. `w` clamped to `[1e-6, 1-1e-6]`.
    """
    eta = np.asarray(eta, dtype=np.float64)
    w = _validate_w(np.asarray(w, dtype=np.float64))
    return eta * (1.0 - w) + w


def eta_inverse_powerlaw(eta_prime: ArrayLike, w: ArrayLike
                          ) -> NDArray[np.float64]:
    """η' → η = (η' - w)/(1-w) for `power_law`. Maps [0, 1] → [η_min(w), 1].

    `w` clamped to `[1e-6, 1-1e-6]` to keep `1-w` strictly positive.
    """
    eta_prime = np.asarray(eta_prime, dtype=np.float64)
    w = _validate_w(np.asarray(w, dtype=np.float64))
    return (eta_prime - w) / (1.0 - w)


def eta_transform_ot(eta: ArrayLike, w: ArrayLike = None  # noqa: ARG001
                      ) -> NDArray[np.float64]:
    """η → η' = η for `ot` (admissible range is already [0, 1])."""
    return np.asarray(eta, dtype=np.float64)


def eta_inverse_ot(eta_prime: ArrayLike, w: ArrayLike = None  # noqa: ARG001
                    ) -> NDArray[np.float64]:
    """η' → η = η' for `ot`."""
    return np.asarray(eta_prime, dtype=np.float64)


# Per-scheme dispatch; keyed on scheme.name.
_ETA_TRANSFORMS: dict[str, tuple[Any, Any]] = {
    "power_law": (eta_transform_powerlaw, eta_inverse_powerlaw),
    "ot": (eta_transform_ot, eta_inverse_ot),
}


def eta_transform(scheme_name: str, eta: ArrayLike, w: ArrayLike
                   ) -> NDArray[np.float64]:
    """Forward η-transform for the given scheme name."""
    if scheme_name not in _ETA_TRANSFORMS:
        raise NotImplementedError(
            f"No eta_transform registered for scheme {scheme_name!r}. "
            f"Available: {sorted(_ETA_TRANSFORMS)}."
        )
    return _ETA_TRANSFORMS[scheme_name][0](eta, w)


def eta_inverse(scheme_name: str, eta_prime: ArrayLike, w: ArrayLike
                 ) -> NDArray[np.float64]:
    """Inverse η-transform for the given scheme name."""
    if scheme_name not in _ETA_TRANSFORMS:
        raise NotImplementedError(
            f"No eta_inverse registered for scheme {scheme_name!r}. "
            f"Available: {sorted(_ETA_TRANSFORMS)}."
        )
    return _ETA_TRANSFORMS[scheme_name][1](eta_prime, w)


# ----- torch versions (optional, used only inside training) -----


def delta_transform_torch(delta):
    """torch version of `delta_transform`. Requires `torch` installed."""
    if not _HAS_TORCH:
        raise ImportError(
            "delta_transform_torch requires torch; install via `pip install -e \".[ml]\"`"
        )
    return delta / (1.0 + delta)


def delta_inverse_torch(delta_prime):
    if not _HAS_TORCH:
        raise ImportError(
            "delta_inverse_torch requires torch; install via `pip install -e \".[ml]\"`"
        )
    return delta_prime / (1.0 - delta_prime)


def eta_transform_powerlaw_torch(eta, w):
    if not _HAS_TORCH:
        raise ImportError(
            "eta_transform_powerlaw_torch requires torch"
        )
    return eta * (1.0 - w) + w


def eta_inverse_powerlaw_torch(eta_prime, w):
    if not _HAS_TORCH:
        raise ImportError(
            "eta_inverse_powerlaw_torch requires torch"
        )
    return (eta_prime - w) / (1.0 - w)


def eta_transform_ot_torch(eta, w=None):  # noqa: ARG001
    return eta


def eta_inverse_ot_torch(eta_prime, w=None):  # noqa: ARG001
    return eta_prime
