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
from typing import Callable

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

    The torch port runs in two regimes:
      - Inside the admissible range (``η < 1/(1-w)``): exact numpy
        behaviour up to float32 precision.
      - Outside: ``denom.clamp(min=1e-6)`` keeps the algebra finite
        and produces a smooth surface that Head A's width loss can
        descend toward valid η. The validity helper (numpy path)
        independently raises ``TiltingDomainError`` for invalid η,
        so Head B's BCE labels are correct regardless.

    Returns a tensor of the same broadcast shape as `theta`.
    """
    if statistic_name == "wald":
        z = torch.abs(D - theta) / sigma
        return 2.0 * (1.0 - _phi(z))

    if statistic_name == "waldo":
        # denom = 1 - eta(1 - w); clamp to avoid divide-by-zero. The
        # clamped surface is smooth so Head A's width loss has a
        # gradient even when EtaNet predicts η outside the admissible
        # range — letting the boundary penalty + width signal jointly
        # push η back without masking the gradient out entirely.
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

    Like ``power_law_tilted_pvalue_torch``, the torch surface stays
    smooth and gradient-bearing for invalid η via
    ``s_t.clamp(min=1e-6)`` rather than NaN-masking — Head A's
    width loss can descend toward valid η even when EtaNet drifts
    outside [0, 1]. The numpy ``OTTilting.tilted_pvalue`` raises
    ``TiltingDomainError`` for invalid η, so the validity helper
    (numpy-driven) labels Head B's BCE correctly regardless of
    what the torch port returns. An earlier round NaN-masked here
    too, which broke OT training entirely (every aux sample masked
    out of the boundary-penalty signal).

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
        # s_t = (w + eta*(1-w))*sigma; admissible iff > 0. We clamp to
        # keep the gradient alive even at slightly-invalid η so Head A
        # can move out of the bad region under the joint width +
        # boundary signal. The validity helper (numpy path) raises
        # `TiltingDomainError` for η outside [0, 1], so Head B's
        # labels remain correct.
        s_t = torch.clamp((w + eta * (1.0 - w)) * sigma, min=1e-6)
        a = torch.abs(mu_t - theta) / s_t
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / s_t
        return _phi(b - a) + _phi(-a - b)

    raise NotImplementedError(
        f"ot_tilted_pvalue_torch: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo')."
    )


# Registry keyed on scheme.name. Add new schemes by registering here.
TORCH_TILTED_PVALUE: dict[str, Callable[..., torch.Tensor]] = {
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
