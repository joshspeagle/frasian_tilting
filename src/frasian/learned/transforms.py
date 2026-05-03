"""Coordinate transforms for the learned-η selector input/output space.

Maps the unbounded (|Δ|, η) space to the bounded `(|Δ'|, η')` space
that the MLP operates on. Numpy versions are used at inference (and
in tests); torch versions (guarded by `try: import torch`) are used in
training only.

|Δ| transform (universal; not scheme-specific):

    Δ' = Δ / (1 + Δ)            maps [0, ∞)        → [0, 1)

η transform (general affine map between any admissible η range and
the bounded sigmoid output space):

    η' = (η - η_low) / (η_high - η_low)        maps [η_low, η_high] → [0, 1]
    η  = η_low + η' (η_high - η_low)           inverse

The admissible η range `(η_low, η_high)` is scheme- and `w`-dependent.
For closed-form schemes on the Normal-Normal sandbox we hard-code:
  - power_law: (-w/(1-w), 1)         (variance-positivity floor; Wald cap)
  - ot:        (0, 1)                 (W2 geodesic endpoints)
For future schemes whose admissible range isn't closed-form (e.g.
`fisher_rao`, `mixture`, or any non-Gaussian-sandbox extension), the
caller can probe via
`frasian.tilting._admissible.numerical_admissible_range_cached` and
pass the resulting `(η_low, η_high)` into the general transform.

Used by:
- `MonotonicEtaArtifact.predict` (numpy) — preprocesses |Δ| inputs and
  postprocesses η' outputs via the general transform.
- `training/train._eta_from_mlp` (torch) — same general transform on
  per-sample (η_low(w), η_high(w)) tensors.

The legacy per-scheme functions (`eta_transform_powerlaw`,
`eta_transform_ot`, etc.) remain as backwards-compatibility shims
that compute their `(η_low, η_high)` and call the general form. New
code should prefer `eta_transform_general` /
`eta_inverse_general` directly.
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


# ----- General affine transform (scheme-agnostic) -----


def eta_transform_general(eta: ArrayLike, eta_low: ArrayLike,
                            eta_high: ArrayLike) -> NDArray[np.float64]:
    """η → η' = (η - η_low) / (η_high - η_low). Maps [η_low, η_high] → [0, 1].

    The general affine map between any admissible η range and the
    bounded sigmoid output space. Works for any scheme as long as the
    caller provides the correct `(eta_low, eta_high)`.
    """
    eta = np.asarray(eta, dtype=np.float64)
    eta_low = np.asarray(eta_low, dtype=np.float64)
    eta_high = np.asarray(eta_high, dtype=np.float64)
    return (eta - eta_low) / (eta_high - eta_low)


def eta_inverse_general(eta_prime: ArrayLike, eta_low: ArrayLike,
                          eta_high: ArrayLike) -> NDArray[np.float64]:
    """η' → η = η_low + η' (η_high - η_low). Maps [0, 1] → [η_low, η_high]."""
    eta_prime = np.asarray(eta_prime, dtype=np.float64)
    eta_low = np.asarray(eta_low, dtype=np.float64)
    eta_high = np.asarray(eta_high, dtype=np.float64)
    return eta_low + eta_prime * (eta_high - eta_low)


# ----- Per-scheme admissible-range registry (closed-form for now) -----


def admissible_range_powerlaw(w: ArrayLike) -> tuple:
    """power_law: (η_low, η_high) = (-w/(1-w), 1). Wald-capped at +1."""
    w = _validate_w(np.asarray(w, dtype=np.float64))
    return -w / (1.0 - w), np.ones_like(w)


def admissible_range_ot(w: ArrayLike) -> tuple:
    """ot: (η_low, η_high) = (0, 1). W2 geodesic from posterior to data."""
    w = np.asarray(w, dtype=np.float64)
    return np.zeros_like(w), np.ones_like(w)


# Registry keyed on `scheme.name`. Schemes whose admissible range
# isn't closed-form omit themselves here; callers fall back to
# `frasian.tilting._admissible.numerical_admissible_range_cached`.
_ADMISSIBLE_RANGES: dict[str, Any] = {
    "power_law": admissible_range_powerlaw,
    "ot":        admissible_range_ot,
}


def admissible_range_for_scheme(scheme_name: str, w: ArrayLike) -> tuple:
    """Look up `(η_low, η_high)` for the learned-η transform.

    Closed-form for `power_law` and `ot`. For unregistered schemes,
    raises NotImplementedError; the caller should use
    `numerical_admissible_range_cached` from
    `frasian.tilting._admissible` and pass the result into
    `eta_transform_general` / `eta_inverse_general` directly.
    """
    if scheme_name in _ADMISSIBLE_RANGES:
        return _ADMISSIBLE_RANGES[scheme_name](w)
    raise NotImplementedError(
        f"No closed-form admissible_range registered for scheme "
        f"{scheme_name!r}. Use `numerical_admissible_range_cached` "
        f"from `frasian.tilting._admissible` and pass the result "
        f"into `eta_transform_general` / `eta_inverse_general` directly."
    )


# ----- Backwards-compat per-scheme functions (now thin wrappers) -----


def eta_min_powerlaw(w: ArrayLike) -> NDArray[np.float64]:
    """Lower bound on η for `power_law`: η_min(w) = -w/(1-w).

    Below this, the tilted variance becomes non-positive
    (`denom = 1 - η(1-w) ≤ 0`) and `tilt` raises `TiltingDomainError`.
    `w` is clamped to `[1e-6, 1-1e-6]` to keep the result finite.
    """
    eta_low, _ = admissible_range_powerlaw(w)
    return eta_low


def eta_transform_powerlaw(eta: ArrayLike, w: ArrayLike
                            ) -> NDArray[np.float64]:
    """η → η' = η(1-w) + w for `power_law`. Maps [η_min(w), 1] → [0, 1].

    Backwards-compat thin wrapper around `eta_transform_general` with
    the closed-form power_law admissible range.
    """
    eta_low, eta_high = admissible_range_powerlaw(w)
    return eta_transform_general(eta, eta_low, eta_high)


def eta_inverse_powerlaw(eta_prime: ArrayLike, w: ArrayLike
                          ) -> NDArray[np.float64]:
    """η' → η for `power_law`. Maps [0, 1] → [η_min(w), 1]."""
    eta_low, eta_high = admissible_range_powerlaw(w)
    return eta_inverse_general(eta_prime, eta_low, eta_high)


def eta_transform_ot(eta: ArrayLike, w: ArrayLike = None  # noqa: ARG001
                      ) -> NDArray[np.float64]:
    """η → η' = η for `ot` (admissible range is [0, 1] so transform is identity).

    Backwards-compat thin wrapper around `eta_transform_general`. The
    `w` argument is ignored (OT's admissible range is w-independent).
    """
    eta_low, eta_high = admissible_range_ot(np.broadcast_to(0.5, np.shape(eta)))
    return eta_transform_general(eta, eta_low, eta_high)


def eta_inverse_ot(eta_prime: ArrayLike, w: ArrayLike = None  # noqa: ARG001
                    ) -> NDArray[np.float64]:
    """η' → η = η' for `ot`."""
    eta_low, eta_high = admissible_range_ot(
        np.broadcast_to(0.5, np.shape(eta_prime))
    )
    return eta_inverse_general(eta_prime, eta_low, eta_high)


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


# ----- General torch transform (scheme-agnostic) -----


def eta_transform_general_torch(eta, eta_low, eta_high):
    """torch: η → η' = (η - η_low) / (η_high - η_low)."""
    if not _HAS_TORCH:
        raise ImportError("eta_transform_general_torch requires torch")
    return (eta - eta_low) / (eta_high - eta_low)


def eta_inverse_general_torch(eta_prime, eta_low, eta_high):
    """torch: η' → η = η_low + η' (η_high - η_low)."""
    if not _HAS_TORCH:
        raise ImportError("eta_inverse_general_torch requires torch")
    return eta_low + eta_prime * (eta_high - eta_low)


# ----- Per-scheme torch admissible-range registry -----


def admissible_range_powerlaw_torch(w):
    """torch: power_law admissible range (-w/(1-w), 1)."""
    if not _HAS_TORCH:
        raise ImportError("admissible_range_powerlaw_torch requires torch")
    return -w / (1.0 - w), torch.ones_like(w)


def admissible_range_ot_torch(w):
    """torch: ot admissible range (0, 1)."""
    if not _HAS_TORCH:
        raise ImportError("admissible_range_ot_torch requires torch")
    return torch.zeros_like(w), torch.ones_like(w)


_ADMISSIBLE_RANGES_TORCH: dict[str, Any] = {
    "power_law": admissible_range_powerlaw_torch,
    "ot":        admissible_range_ot_torch,
}


def admissible_range_for_scheme_torch(scheme_name: str, w):
    """torch lookup: `(η_low, η_high)` for the learned-η transform.

    Closed-form for `power_law` and `ot`. For an unregistered scheme,
    the caller must compute the admissible range upstream (e.g. via
    `numerical_admissible_range_cached` on a w-grid pre-training)
    and pass `(eta_low, eta_high)` tensors directly into
    `eta_inverse_general_torch`.
    """
    if scheme_name not in _ADMISSIBLE_RANGES_TORCH:
        raise NotImplementedError(
            f"No closed-form torch admissible_range registered for "
            f"{scheme_name!r}. Compute it upstream and call "
            f"`eta_inverse_general_torch(eta_prime, eta_low, eta_high)` "
            f"directly."
        )
    return _ADMISSIBLE_RANGES_TORCH[scheme_name](w)


# ----- Backwards-compat per-scheme torch wrappers -----


def eta_transform_powerlaw_torch(eta, w):
    if not _HAS_TORCH:
        raise ImportError(
            "eta_transform_powerlaw_torch requires torch"
        )
    eta_low, eta_high = admissible_range_powerlaw_torch(w)
    return eta_transform_general_torch(eta, eta_low, eta_high)


def eta_inverse_powerlaw_torch(eta_prime, w):
    if not _HAS_TORCH:
        raise ImportError(
            "eta_inverse_powerlaw_torch requires torch"
        )
    eta_low, eta_high = admissible_range_powerlaw_torch(w)
    return eta_inverse_general_torch(eta_prime, eta_low, eta_high)


def eta_transform_ot_torch(eta, w=None):
    if not _HAS_TORCH:
        raise ImportError("eta_transform_ot_torch requires torch")
    if w is None:
        w_arr = torch.full_like(eta, 0.5)
    else:
        w_arr = w
    eta_low, eta_high = admissible_range_ot_torch(w_arr)
    return eta_transform_general_torch(eta, eta_low, eta_high)


def eta_inverse_ot_torch(eta_prime, w=None):
    if not _HAS_TORCH:
        raise ImportError("eta_inverse_ot_torch requires torch")
    if w is None:
        w_arr = torch.full_like(eta_prime, 0.5)
    else:
        w_arr = w
    eta_low, eta_high = admissible_range_ot_torch(w_arr)
    return eta_inverse_general_torch(eta_prime, eta_low, eta_high)


# Per-scheme torch dispatch (mirrors the numpy `_ETA_TRANSFORMS` registry).
_ETA_TRANSFORMS_TORCH: dict[str, tuple[Any, Any]] = {
    "power_law": (eta_transform_powerlaw_torch, eta_inverse_powerlaw_torch),
    "ot":        (eta_transform_ot_torch,        eta_inverse_ot_torch),
}


def eta_transform_torch(scheme_name: str, eta, w):
    """Forward η-transform (torch) for the given scheme name."""
    if not _HAS_TORCH:
        raise ImportError("eta_transform_torch requires torch")
    if scheme_name not in _ETA_TRANSFORMS_TORCH:
        raise NotImplementedError(
            f"No eta_transform_torch registered for scheme {scheme_name!r}. "
            f"Available: {sorted(_ETA_TRANSFORMS_TORCH)}."
        )
    return _ETA_TRANSFORMS_TORCH[scheme_name][0](eta, w)


def eta_inverse_torch(scheme_name: str, eta_prime, w):
    """Inverse η-transform (torch) for the given scheme name.

    For a clamping-aware version that handles `w → 1` gracefully, the
    caller should pre-clamp `w` (training samples already do via the
    `w_range` in `TrainingDistribution`).
    """
    if not _HAS_TORCH:
        raise ImportError("eta_inverse_torch requires torch")
    if scheme_name not in _ETA_TRANSFORMS_TORCH:
        raise NotImplementedError(
            f"No eta_inverse_torch registered for scheme {scheme_name!r}. "
            f"Available: {sorted(_ETA_TRANSFORMS_TORCH)}."
        )
    return _ETA_TRANSFORMS_TORCH[scheme_name][1](eta_prime, w)
