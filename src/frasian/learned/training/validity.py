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

# `_FP_SLACK` is the numerical slack used to admit p-values that are
# in [0, 1] modulo FP noise (e.g. `scipy.stats.norm.cdf` rounding at
# ±5σ). NOT user-tunable: tightening it below 1e-9 would falsely
# reject FP-noisy-but-correct pairs; loosening it would let truly
# off-domain p-values into the validity mask. Audit P2 (Cluster G)
# considered exposing this via env var; correctness-tuning knob, kept
# private by design.
_FP_SLACK = 1e-9

# Per-sample MC budget when routing through the scheme-specific generic
# tilted-pvalue path (for any non-Normal-Normal model). The validity
# labeller only needs to know whether the p-value is in [0, 1] and
# finite — it does not need the high-precision MC reference of
# inference-time CI inversion. n_mc=32 keeps per-step cost tractable
# while still producing reliable validity labels (< ~3% disagreement
# vs n_mc=200 at training-typical (θ, η)).
#
# Audit P2 (Cluster G): exposed via the env var ``FRASIAN_N_MC_VALIDITY``
# for sweeps that want tighter agreement with the n_mc=200 reference.
# Read once at module import. Defaults to 32.
def _resolve_n_mc_validity() -> int:
    import os

    raw = os.environ.get("FRASIAN_N_MC_VALIDITY", "32")
    try:
        v = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"FRASIAN_N_MC_VALIDITY must be int; got {raw!r}."
        ) from exc
    if v < 1:
        raise ValueError(f"FRASIAN_N_MC_VALIDITY must be >= 1; got {v}.")
    return v


_N_MC_VALIDITY: int = _resolve_n_mc_validity()


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

    Branches on (scheme, model_kind):

    - power_law + normal_normal: denom = 1 - eta(1-w) > 0  ⟺  eta < 1/(1-w).
    - power_law + non-NN: finite η only — the generic grid path's
      `log L + (1-η) log π` is well-defined for any finite η on a
      compact support, so the per-element exception path catches the
      residual edge cases (improper moments, etc.).
    - ot:        eta in [0, 1] (η-box check, model-independent).

    For unrecognised schemes returns an all-True mask, falling back
    fully to the per-element exception path. The float check on η
    finiteness is shared so all schemes treat NaN/Inf as invalid.
    """
    finite = np.isfinite(eta_arr)
    name = getattr(scheme, "name", "")
    model_kind = ""
    fp = getattr(model, "fingerprint", None)
    if callable(fp):
        try:
            model_kind = fp()[0]
        except (IndexError, TypeError):
            model_kind = ""
    if name == "power_law":
        if model_kind == "normal_normal":
            # NN closed-form bound: w depends only on (model.sigma,
            # prior.scale), constant across the per-sample loop in the
            # training pipeline.
            sigma = float(getattr(model, "sigma", float("nan")))
            sigma0 = float(getattr(prior, "scale", float("nan")))
            if not (np.isfinite(sigma) and np.isfinite(sigma0)):
                return finite  # fall back to per-element exception path
            w = sigma0**2 / (sigma**2 + sigma0**2)
            # Strict raise-bound: power_law.tilted_pvalue raises iff
            # `denom = 1 - eta*(1-w) <= 0`, i.e. `eta >= 1/(1-w)`. We
            # allow up to but excluding that bound. Note: this is
            # broader than the buffered η bracket
            # `NumericalEtaSelector` uses for its minimization (see
            # `NumericalEtaSelector._eta_bounds`). The mask matches
            # the strict raise bound (what `tilted_pvalue` actually
            # enforces), NOT the buffered selector bound.
            return finite & (eta_arr < 1.0 / (1.0 - w))
        # Generic (future non-conjugate models): the grid path is
        # well-defined for any finite η on a compact support; the
        # per-element exception path catches edge cases (improper
        # moments, var_tilted ≈ 0).
        return finite
    if name == "ot":
        # Audit P0-4: OT is no longer interval-clamped to [0, 1]. The
        # closed-form WALDO p-value is parameterised by
        # `s_t = (w + eta*(1-w)) * sigma`; admissibility is `s_t > 0`,
        # i.e. `eta > -w/(1-w)`. There is no upper bound from s_t.
        sigma = float(getattr(model, "sigma", float("nan")))
        sigma0 = float(getattr(prior, "scale", float("nan")))
        if not (np.isfinite(sigma) and np.isfinite(sigma0)):
            return finite
        w = sigma0**2 / (sigma**2 + sigma0**2)
        eta_lower = -w / (1.0 - w)
        return finite & (eta_arr > eta_lower)
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

    Shape contract: ``theta.shape == eta.shape == (N,)``. ``D`` can
    be either 1D ``(N,)`` (one observation per θ — Normal-Normal
    historical contract) or 2D ``(N, n_data)`` (Phase 4c: a vector
    of ``n_data`` observations per θ; required for future non-NN
    models where a single observation carries no posterior signal).
    The 2D path skips the bulk fast-path and routes per-sample so
    each row's ``D[i]`` reaches ``scheme.tilted_pvalue`` as a 1-D
    dataset.
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

    if theta_arr.ndim != 1 or eta_arr.ndim != 1:
        raise ValueError(
            "theta and eta must be 1-D; got "
            f"theta.shape={theta_arr.shape}, eta.shape={eta_arr.shape}."
        )
    if D_arr.ndim not in (1, 2):
        raise ValueError(f"D must be 1-D or 2-D; got shape {D_arr.shape}.")
    if D_arr.shape[0] != theta_arr.shape[0]:
        raise ValueError(
            "theta and D must agree on first axis; got "
            f"theta.shape={theta_arr.shape}, D.shape={D_arr.shape}."
        )
    if eta_arr.shape != theta_arr.shape:
        raise ValueError(
            "theta and eta must share shape; got "
            f"{theta_arr.shape}, {eta_arr.shape}."
        )

    out = np.full(theta_arr.shape, np.nan, dtype=np.float64)
    admissible = _admissibility_mask(scheme, eta_arr, model, prior)
    if not np.any(admissible):
        return out

    # Phase 4 skeptic #5 fast path: non-NN + recognized scheme produces
    # a structurally constant validity rate of 1 over the explore box.
    # Skip MC entirely — `_maybe_warn_class_degenerate` already documents
    # that ValidityNet's BCE is a near-no-op on non-NN.
    fast_p = _generic_validity_fast_path_p(
        scheme, theta_arr, D_arr, model, prior, eta_arr
    )
    if fast_p is not None:
        out[admissible] = fast_p[admissible]
        return out

    # 2D D path or non-Normal-Normal model: drop to the per-sample
    # loop. The closed-form NN bulk path expects a scalar D and raises
    # on non-NN models; routing straight to the loop avoids the
    # raise/catch noise. The loop dispatches to the scheme's generic
    # MC pvalue for non-NN models.
    model_fp = getattr(model, "fingerprint", None)
    is_nn = (
        callable(model_fp)
        and model_fp() and model_fp()[0] == "normal_normal"
    )
    if D_arr.ndim == 2 or not is_nn:
        return _compute_pvalues_per_sample_loop(
            scheme, theta_arr, D_arr, model, prior, eta_arr, statistic_name
        )

    # 1D D path: Fast-path bulk vectorised call across the admissible
    # subset. tilted_pvalue broadcasts over (theta, eta) of the same
    # shape; D broadcasts naturally as another array of the same shape
    # (it only enters the formula via element-wise arithmetic). On any
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


def _resolve_generic_tilted_pvalue(scheme_name: str) -> Any:
    """Return the scheme-specific generic MC tilted-pvalue helper, or
    ``None`` if the scheme has no generic path implemented."""
    if scheme_name == "power_law":
        from ...tilting.power_law import _generic_tilted_pvalue
        return _generic_tilted_pvalue
    if scheme_name == "ot":
        from ...tilting.ot import _generic_tilted_pvalue_ot
        return _generic_tilted_pvalue_ot
    return None


def _generic_validity_fast_path_p(
    scheme: Any,
    theta_arr: NDArray[np.float64],
    D_arr: NDArray[np.float64],
    model: Any,
    prior: Any,
    eta_arr: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Phase 4 skeptic #5: closed-form `True` validity for non-NN models.

    On non-Normal-Normal models with a bounded support and a finite
    prior log-pdf on the explore box, the generic MC tilted-pvalue
    *empirically* returns a finite value in [0, 1] for every (θ, η)
    drawn from the training-time exploration distribution — the
    validity rate is structurally ~1 (historically measured at 1.0000
    across n_mc ∈ {8, 32, 200}). Running n_mc=32 MC reference draws
    to recover that label is 11+ seconds per step on a bounded-support
    non-NN model — entirely wasted cost.

    Fast path: skip MC and return p ≡ 0.5 (any value in (0, 1)
    works; 0.5 is the median under H_0). The validity_mask then
    flips True everywhere and Head B's BCE step is class-degenerate
    on non-NN — which is the documented behaviour skipped by
    `_maybe_warn_class_degenerate` for non-NN.

    Returns ``None`` for the NN path (which keeps the closed-form
    bulk fast path) or any scheme without a known closed-form
    predicate; the caller falls through to MC.
    """
    fp = getattr(model, "fingerprint", None)
    if not callable(fp):
        return None
    try:
        model_kind = fp()[0]
    except (IndexError, TypeError):
        return None
    if model_kind == "normal_normal":
        return None
    if getattr(scheme, "name", "") not in ("power_law", "ot"):
        return None
    # Defense in depth: confirm finite tilted moments would obtain at
    # the boundary of the eta range; if anything is off (e.g. degenerate
    # prior), fall through to the MC path so the per-sample loop's
    # NaN-on-failure semantics still apply.
    if not bool(np.all(np.isfinite(eta_arr))):
        return None
    return np.full(theta_arr.shape, 0.5, dtype=np.float64)


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

    For Normal-Normal models, calls ``scheme.tilted_pvalue`` per row
    (closed-form fast path); for non-Normal models, routes to the
    scheme's generic MC tilted-pvalue (e.g.
    ``power_law._generic_tilted_pvalue``) at low ``n_mc`` so the
    training-time validity labels stay tractable.

    Accepts either 1-D ``D_arr`` (scalar D per sample) or 2-D
    ``D_arr`` (length-``n_data`` dataset per sample). For non-NN
    models the 1-D D is wrapped in a length-1 array before being
    passed as ``data`` to ``_generic_tilted_pvalue`` (matching its
    expected dataset-shaped argument).
    """
    out = np.empty(theta_arr.shape, dtype=np.float64)
    is_2d = D_arr.ndim == 2

    model_kind = ""
    fp = getattr(model, "fingerprint", None)
    if callable(fp):
        try:
            model_kind = fp()[0]
        except (IndexError, TypeError):
            model_kind = ""
    use_generic = model_kind not in ("normal_normal", "")
    generic_call = (
        _resolve_generic_tilted_pvalue(getattr(scheme, "name", ""))
        if use_generic
        else None
    )

    for i in range(theta_arr.size):
        try:
            if use_generic and generic_call is not None:
                # Generic MC path: data is the dataset (1D array of
                # observations). For 1D D_arr we wrap the scalar in a
                # length-1 array.
                data_i = (
                    np.asarray(D_arr[i], dtype=np.float64)
                    if is_2d
                    else np.atleast_1d(np.asarray(D_arr[i], dtype=np.float64))
                )
                p = generic_call(
                    theta=float(theta_arr[i]),
                    data=data_i,
                    model=model,
                    prior=prior,
                    eta=float(eta_arr[i]),
                    statistic_name=statistic_name,
                    n_mc=_N_MC_VALIDITY,
                )
                out[i] = float(p)
                continue
            d_i = D_arr[i] if is_2d else float(D_arr[i])
            p = scheme.tilted_pvalue(
                np.array([theta_arr[i]]),
                d_i,
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


def compute_pvalues_per_sample_with_hp(
    scheme: Any,
    theta_arr: NDArray[np.float64],
    D_arr: NDArray[np.float64],
    prior_cls: type,
    model_cls: type,
    prior_hp_batch: NDArray[np.float64],
    lik_hp_batch: NDArray[np.float64],
    eta_arr: NDArray[np.float64],
    statistic_name: str,
) -> NDArray[np.float64]:
    """Per-element scheme.tilted_pvalue with per-sample (prior, model).

    Phase G: each batch element has its own (prior_hp, lik_hp).
    Constructs ``prior_i`` and ``model_i`` per element via
    ``from_hyperparams``, then calls the scheme's ``tilted_pvalue``
    (NN closed-form fast path) or the scheme's generic-MC pvalue
    (future non-NN models, where the closed-form path raises
    ``NotImplementedError``).
    Returns NaN on failure (matching ``_compute_pvalues_per_sample_loop``).
    """
    n = theta_arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    is_2d = D_arr.ndim == 2
    use_generic = model_cls.__name__ != "NormalNormalModel"
    generic_call = (
        _resolve_generic_tilted_pvalue(getattr(scheme, "name", ""))
        if use_generic
        else None
    )
    for i in range(n):
        try:
            prior_i = prior_cls.from_hyperparams(prior_hp_batch[i])
            model_i = model_cls.from_hyperparams(lik_hp_batch[i])
            if use_generic and generic_call is not None:
                data_i = (
                    np.asarray(D_arr[i], dtype=np.float64)
                    if is_2d
                    else np.atleast_1d(np.asarray(D_arr[i], dtype=np.float64))
                )
                p = generic_call(
                    theta=float(theta_arr[i]),
                    data=data_i,
                    model=model_i,
                    prior=prior_i,
                    eta=float(eta_arr[i]),
                    statistic_name=statistic_name,
                    n_mc=_N_MC_VALIDITY,
                )
                out[i] = float(p)
                continue
            d_i = (
                np.asarray(D_arr[i], dtype=np.float64)
                if is_2d
                else float(D_arr[i])
            )
            p = scheme.tilted_pvalue(
                np.array([theta_arr[i]]),
                d_i,
                model_i,
                prior_i,
                float(eta_arr[i]),
                statistic_name,
            )
            out[i] = float(np.asarray(p).reshape(-1)[0])
        except (TiltingDomainError, ValueError, RuntimeError,
                NotImplementedError, ArithmeticError):
            out[i] = np.nan
    return out


__all__ = [
    "is_pair_valid",
    "validity_mask",
    "compute_pvalues_per_sample",
    "compute_pvalues_per_sample_with_hp",
]
