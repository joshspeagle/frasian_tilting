"""Per-sample validity helpers for the Phase E dual-head training loop.

Validity at a single ``(θ, η)`` query is whether the **scalar p-value**
returned by ``scheme.tilted_pvalue(np.array([θ]), D, model, prior, η,
statistic_name)`` is finite and in ``[0, 1]`` (with FP slack). Head B
(``ValidityNet``) learns this per-point predicate; Head A's boundary
penalty pushes ``η_pred`` into the predicted-valid region.

This module contains:

- ``is_pair_valid(p_scalar)`` — per-sample predicate.
- ``validity_mask(p_array)`` — vectorised version returning bool ndarray.
- ``compute_pvalues_per_sample(scheme, theta, D, model, prior, eta,
  statistic_name)`` — runs ``tilted_pvalue`` once per sample and
  returns a NaN-on-failure array (catches ``TiltingDomainError`` /
  ``ValueError`` / ``RuntimeError`` and yields NaN in that slot, so
  downstream ``validity_mask`` marks the slot invalid).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..._errors import TiltingDomainError

_FP_SLACK = 1e-9


def is_pair_valid(p_scalar: float) -> bool:
    """Per-(θ, η) validity predicate on the scalar p-value.

    Valid iff the p-value is finite and inside ``[-1e-9, 1+1e-9]``
    (the small slack handles FP noise from e.g. ``norm.cdf``
    rounding at ±5σ).
    """
    return (
        np.isfinite(p_scalar)
        and -_FP_SLACK <= p_scalar <= 1.0 + _FP_SLACK
    )


def validity_mask(p_array: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Vectorised ``is_pair_valid`` over a (N,) p-value array.

    NaN → False (catches the typical "improper distribution surfaces
    as NaN" path).
    """
    arr = np.asarray(p_array, dtype=np.float64)
    return (
        np.isfinite(arr)
        & (arr >= -_FP_SLACK)
        & (arr <= 1.0 + _FP_SLACK)
    )


def compute_pvalues_per_sample(
    scheme,
    theta: NDArray[np.float64],
    D: NDArray[np.float64],
    model,
    prior,
    eta: NDArray[np.float64],
    statistic_name: str,
) -> NDArray[np.float64]:
    """Per-sample ``scheme.tilted_pvalue`` lookup; NaN on failure.

    Each ``(θ_i, D_i, η_i)`` triple is fed into
    ``scheme.tilted_pvalue(np.array([θ_i]), D_i, model, prior, η_i,
    statistic_name)`` and the scalar output is collected. Exceptions
    that the framework conventionally raises for invalid η (i.e.,
    ``TiltingDomainError``, ``ValueError``, ``RuntimeError``) are
    caught and converted to NaN — ``validity_mask`` then marks the
    slot invalid.

    All arrays must share the same shape ``(N,)``.
    """
    if not hasattr(scheme, "tilted_pvalue"):
        raise AttributeError(
            f"{type(scheme).__name__} does not implement `tilted_pvalue`; "
            f"compute_pvalues_per_sample requires it. Stub schemes "
            f"(fisher_rao, mixture) need a torch port and a numpy "
            f"implementation before they can be trained."
        )

    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    D_arr = np.atleast_1d(np.asarray(D, dtype=np.float64))
    eta_arr = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    if theta_arr.shape != D_arr.shape or theta_arr.shape != eta_arr.shape:
        raise ValueError(
            "theta, D, and eta must share shape; got "
            f"{theta_arr.shape}, {D_arr.shape}, {eta_arr.shape}."
        )

    out = np.empty(theta_arr.shape, dtype=np.float64)
    for i in range(theta_arr.size):
        try:
            p = scheme.tilted_pvalue(
                np.array([theta_arr[i]]),
                float(D_arr[i]),
                model,
                prior,
                float(eta_arr[i]),
                statistic_name,
            )
            out[i] = float(np.asarray(p).reshape(-1)[0])
        except (TiltingDomainError, ValueError, RuntimeError,
                NotImplementedError, FloatingPointError):
            # Deliberately NOT catching AttributeError here — a typo
            # like `self.foo` inside a scheme's `tilted_pvalue` body
            # should surface as a stack trace, not silently turn every
            # sample into NaN. The hasattr() pre-check above covers
            # the "scheme has no method at all" case loudly.
            out[i] = np.nan
    return out


__all__ = [
    "is_pair_valid",
    "validity_mask",
    "compute_pvalues_per_sample",
]
