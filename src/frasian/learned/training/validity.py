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
  statistic_name)`` — runs ``tilted_pvalue`` per sample and returns a
  NaN-on-failure array. Phase 3 (Tier 1.3 N2) pre-masks invalid η via
  closed-form admissibility, then issues one bulk ``tilted_pvalue``
  call per ``D`` value across the surviving (θ, η) pairs (D varies
  per sample so the call is grouped by unique D). Catches
  ``TiltingDomainError`` / ``ValueError`` / ``RuntimeError`` /
  ``NotImplementedError`` / ``ArithmeticError`` and yields NaN in
  any slot that escapes the pre-mask, so downstream
  ``validity_mask`` still marks it invalid.
"""

from __future__ import annotations

import warnings
from typing import Any

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
    return np.isfinite(p_scalar) and -_FP_SLACK <= p_scalar <= 1.0 + _FP_SLACK


def validity_mask(p_array: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Vectorised ``is_pair_valid`` over a (N,) p-value array.

    NaN → False (catches the typical "improper distribution surfaces
    as NaN" path).
    """
    arr = np.asarray(p_array, dtype=np.float64)
    return np.isfinite(arr) & (arr >= -_FP_SLACK) & (arr <= 1.0 + _FP_SLACK)


def _admissibility_mask(
    scheme: Any,
    eta_arr: NDArray[np.float64],
    model: Any,
    prior: Any,
) -> NDArray[np.bool_]:
    """Closed-form per-element admissibility predicate for the trained schemes.

    - power_law: denom = 1 - eta(1-w) > 0  ⟺  eta < 1/(1-w).
    - ot:        eta in [0, 1].

    For unrecognised schemes returns an all-True mask, falling back
    fully to the per-element exception path. The float check on η
    finiteness is shared so both schemes treat NaN/Inf as invalid.
    """
    finite = np.isfinite(eta_arr)
    name = getattr(scheme, "name", "")
    if name == "power_law":
        # w depends only on (model.sigma, prior.scale), constant across
        # the per-sample loop in the training pipeline.
        sigma = float(getattr(model, "sigma", float("nan")))
        sigma0 = float(getattr(prior, "scale", float("nan")))
        if not (np.isfinite(sigma) and np.isfinite(sigma0)):
            return finite  # fall back to per-element exception path
        w = sigma0**2 / (sigma**2 + sigma0**2)
        # Strict raise-bound: power_law.tilted_pvalue raises iff
        # `denom = 1 - eta*(1-w) <= 0`, i.e. `eta >= 1/(1-w)`. We allow
        # up to but excluding that bound. Note: this is broader than
        # the buffered η bracket `NumericalEtaSelector` uses for its
        # minimization (see `NumericalEtaSelector._eta_bounds`). The
        # mask matches the strict raise bound (what `tilted_pvalue`
        # actually enforces), NOT the buffered selector bound.
        return finite & (eta_arr < 1.0 / (1.0 - w))
    if name == "ot":
        return finite & (eta_arr >= 0.0) & (eta_arr <= 1.0)
    # TODO Phase 6: extend _admissibility_mask for mixture (η ∈ [0,1])
    # and fisher_rao (open half-plane).
    # Unknown scheme: don't pre-mask; let the per-element fallback handle it.
    # Warn so the slow-path regression is discoverable when a future scheme
    # ships without a closed-form predicate here.
    warnings.warn(
        f"_admissibility_mask: no closed-form admissibility for scheme "
        f"{name!r}, falling back to per-sample exception loop",
        RuntimeWarning,
        stacklevel=2,
    )
    return finite


def compute_pvalues_per_sample(
    scheme: Any,
    theta: NDArray[np.float64],
    D: NDArray[np.float64],
    model: Any,
    prior: Any,
    eta: NDArray[np.float64],
    statistic_name: str,
) -> NDArray[np.float64]:
    """Per-sample ``scheme.tilted_pvalue`` lookup; NaN on failure.

    Phase 3 (Tier 1.3 N2) replaces the original Python loop with a
    closed-form admissibility pre-mask + a single vectorised
    ``tilted_pvalue`` call across the surviving samples. The output is
    bytewise-identical to the scalar reference for valid samples and
    NaN for samples that fail either the pre-mask or the residual
    post-call validation (NaN/Inf p-value, or rare exceptions that
    slip through the closed-form predicate). Downstream
    ``validity_mask`` marks NaN slots invalid as before.

    Exceptions raised by ``tilted_pvalue`` on the bulk path
    (``TiltingDomainError`` / ``ValueError`` / ``RuntimeError`` /
    ``NotImplementedError`` / ``ArithmeticError``) trigger a fallback
    to the per-sample loop so a single bad sample doesn't fail the
    whole batch — preserving the original "NaN per slot" semantics.

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

    out = np.full(theta_arr.shape, np.nan, dtype=np.float64)
    admissible = _admissibility_mask(scheme, eta_arr, model, prior)
    if not np.any(admissible):
        return out

    # Fast path: bulk vectorised call across the admissible subset.
    # tilted_pvalue broadcasts over (theta, eta) of the same shape;
    # D broadcasts naturally as another array of the same shape (it
    # only enters the formula via element-wise arithmetic). On any
    # exception we fall back to the legacy per-sample loop so a rare
    # bad-sample doesn't poison the whole batch.
    try:
        p_bulk = np.asarray(
            scheme.tilted_pvalue(
                theta_arr[admissible],
                D_arr[admissible],
                model,
                prior,
                eta_arr[admissible],
                statistic_name,
            ),
            dtype=np.float64,
        )
        # Any non-finite slot from the closed-form ufuncs (e.g., a
        # numerical blow-up that escapes the closed-form mask) gets
        # left as NaN downstream of validity_mask.
        out[admissible] = p_bulk
        return out
    except (TiltingDomainError, ValueError, RuntimeError, NotImplementedError, ArithmeticError):
        # Fallback: per-sample loop with try/except — preserves the
        # "NaN per offending slot" output semantics of the legacy path.
        return _compute_pvalues_per_sample_loop(
            scheme, theta_arr, D_arr, model, prior, eta_arr, statistic_name
        )


def _compute_pvalues_per_sample_loop(
    scheme: Any,
    theta_arr: NDArray[np.float64],
    D_arr: NDArray[np.float64],
    model: Any,
    prior: Any,
    eta_arr: NDArray[np.float64],
    statistic_name: str,
) -> NDArray[np.float64]:
    """Legacy per-sample loop kept as the exception-safe fallback.

    Used when the closed-form admissibility mask is unavailable for
    a scheme (unknown ``scheme.name``) or when the bulk call raises.
    """
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
        except (TiltingDomainError, ValueError, RuntimeError, NotImplementedError, ArithmeticError):
            # ArithmeticError is the parent of FloatingPointError /
            # OverflowError / ZeroDivisionError — catches numerical
            # blowups across schemes uniformly. Deliberately NOT
            # catching AttributeError: a typo like `self.foo` inside
            # a scheme's `tilted_pvalue` body should surface as a
            # stack trace, not silently turn every sample into NaN.
            # The hasattr() pre-check above covers the "scheme has
            # no method at all" case loudly.
            out[i] = np.nan
    return out


__all__ = [
    "is_pair_valid",
    "validity_mask",
    "compute_pvalues_per_sample",
]
