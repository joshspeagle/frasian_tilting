"""Differentiable torch implementations of `tilted_pvalue` per scheme.

A registry `TORCH_TILTED_PVALUE` keyed on `scheme.name` returning a
function that takes broadcastable tensors and produces a `(N, n_theta)`
p-value tensor. Used inside the training loss; not used at inference
(production code uses the numpy `tilted_pvalue` on each scheme).

Each torch implementation is a direct port of its numpy counterpart:
- power_law: `src/frasian/tilting/power_law.py:172-220`
- ot:        `src/frasian/tilting/ot.py:143-188`

Tested against the numpy versions to atol 1e-10 in
`tests/regression/test_torch_pvalue_matches_numpy.py`.

Φ(x) implementation
-------------------
`Φ(x) = 0.5 * (1 + torch.erf(x / sqrt(2)))`

This is the standard differentiable form. `torch.erf` is fully
differentiable and numerically stable across the float32 range.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch


_SQRT2 = math.sqrt(2.0)


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Standard-normal CDF Φ(x) via torch.erf."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def power_law_tilted_pvalue_torch(
    theta: torch.Tensor,
    D: torch.Tensor,
    w: torch.Tensor,
    mu0: torch.Tensor,
    sigma: torch.Tensor,
    eta: torch.Tensor,
    statistic_name: str,
) -> torch.Tensor:
    """Torch port of `PowerLawTilting.tilted_pvalue` for (power_law, waldo|wald).

    All tensor inputs are expected to broadcast to the desired output
    shape. Typical use: `theta` is `(B, N)`, `D, w, mu0, eta` are
    `(B, 1)` or `(B, N)`, `sigma` is scalar.

    Returns a tensor of the same broadcast shape as `theta`.
    """
    if statistic_name == "wald":
        z = torch.abs(D - theta) / sigma
        return 2.0 * (1.0 - _phi(z))

    if statistic_name == "waldo":
        # denom = 1 - eta(1 - w); clamp to avoid divide-by-zero in pathological η.
        denom = torch.clamp(1.0 - eta * (1.0 - w), min=1e-6)
        mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * mu0) / denom
        norm_factor = w * sigma / denom
        a = torch.abs(mu_eta - theta) / norm_factor
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / (denom * norm_factor)
        return _phi(b - a) + _phi(-a - b)

    raise NotImplementedError(
        f"power_law_tilted_pvalue_torch: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo')."
    )


def ot_tilted_pvalue_torch(
    theta: torch.Tensor,
    D: torch.Tensor,
    w: torch.Tensor,
    mu0: torch.Tensor,
    sigma: torch.Tensor,
    eta: torch.Tensor,
    statistic_name: str,
) -> torch.Tensor:
    """Torch port of `OTTilting.tilted_pvalue` for (ot, waldo|wald).

    Mirrors the numpy `OTTilting.tilted_pvalue` admissible-range
    enforcement (η ∈ [0, 1]). Outside the admissible range the torch
    port emits NaN per element so the loss masks downstream
    (``_masked_mean``) drop those samples instead of training Head A
    on a meaningless surface. This closes the gap between numpy
    (raises) and torch (silently computed) behaviours flagged in
    the E.2 skeptic review.

    See `power_law_tilted_pvalue_torch` for input/output shape
    conventions; signatures match for registry uniformity.
    """
    if statistic_name == "wald":
        z = torch.abs(D - theta) / sigma
        return 2.0 * (1.0 - _phi(z))

    if statistic_name == "waldo":
        # mu_t = (1 - eta)*mu_n + eta*D, with mu_n = w*D + (1-w)*mu0.
        mu_n = w * D + (1.0 - w) * mu0
        mu_t = (1.0 - eta) * mu_n + eta * D
        # Standard error of mu_t under repeated D ~ N(theta, sigma^2):
        s_t = (w + eta * (1.0 - w)) * sigma
        a = torch.abs(mu_t - theta) / s_t
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / s_t
        p = _phi(b - a) + _phi(-a - b)
        # Mask out elements with η outside [0, 1] (admissible range).
        # `torch.where(cond, NaN, p)` keeps the in-range subgraph
        # differentiable; `_masked_mean` drops the NaN samples.
        out_of_range = (eta < 0.0) | (eta > 1.0)
        nan_value = torch.full_like(p, float("nan"))
        return torch.where(out_of_range, nan_value, p)

    raise NotImplementedError(
        f"ot_tilted_pvalue_torch: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo')."
    )


# Registry keyed on scheme.name. Add new schemes by registering here.
TORCH_TILTED_PVALUE: Dict[str, Callable[..., torch.Tensor]] = {
    "power_law": power_law_tilted_pvalue_torch,
    "ot": ot_tilted_pvalue_torch,
}


def get_torch_tilted_pvalue(scheme_name: str) -> Callable[..., torch.Tensor]:
    """Look up the torch tilted-p-value function for a scheme.

    Raises `NotImplementedError` if the scheme has no registered torch
    p-value (training is gated on this; the non-torch numpy `tilted_pvalue`
    on the scheme still works at inference).
    """
    if scheme_name not in TORCH_TILTED_PVALUE:
        raise NotImplementedError(
            f"No torch tilted_pvalue registered for scheme {scheme_name!r}. "
            f"Available: {sorted(TORCH_TILTED_PVALUE)}. "
            f"To train against a new scheme, register a torch p-value here."
        )
    return TORCH_TILTED_PVALUE[scheme_name]
