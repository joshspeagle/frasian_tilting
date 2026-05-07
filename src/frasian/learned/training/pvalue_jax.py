"""Differentiable JAX implementations of `tilted_pvalue` per scheme.

A registry `JAX_TILTED_PVALUE` keyed on `scheme.name` returning a
function that takes broadcastable JAX arrays and produces a `(N, n_theta)`
p-value tensor. Used inside the training loss; not used at inference
(production code uses the `tilted_pvalue` on each scheme, which has
its own JAX kernel + numpy scalar fast path; see `tilting/power_law.py`).

Each JAX implementation is a direct port of its numpy counterpart:
- power_law: `src/frasian/tilting/power_law.py`
- ot:        `src/frasian/tilting/ot.py`

Tested against the numpy reference to atol 1e-10 in
`tests/regression/test_jax_pvalue_matches_numpy.py`.

Why this is separate from `tilting/power_law.py::_tilted_pvalue_kernel`
-----------------------------------------------------------------------
The kernel in `tilting/` is the inference-time implementation: it
raises ``TiltingDomainError`` on invalid eta. The training-time
implementation here CLAMPS divisor-bearing intermediates so the loss
surface stays smooth and gradient-bearing for invalid eta — Head A's
width loss can descend toward valid eta even when EtaNet drifts
outside the admissible range. The validity helper (numpy path,
`validity.py`) labels Head B's BCE correctly regardless of what the
JAX surface returns. This is intentional and documented in the legacy
torch implementation (preserved verbatim in the port).
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active

_FORCE_X64 = _x64  # keep static-analysis from stripping the import

_SQRT2 = math.sqrt(2.0)


def _phi(x: jax.Array) -> jax.Array:
    """Standard-normal CDF via jax.scipy.stats.norm.cdf."""
    return jsp_stats.norm.cdf(x)


def power_law_tilted_pvalue_jax(
    theta: jax.Array,
    D: jax.Array,
    w: jax.Array,
    mu0: jax.Array,
    sigma: jax.Array,
    eta: jax.Array,
    statistic_name: str,
) -> jax.Array:
    """JAX port of `PowerLawTilting.tilted_pvalue` for (power_law, waldo|wald).

    All array inputs broadcast to the desired output shape. Typical use:
    ``theta`` is ``(B, N)``, ``D, w, mu0, eta`` are ``(B, 1)`` or ``(B, N)``,
    ``sigma`` is scalar.

    Two regimes:
      - Inside the admissible range (``eta < 1/(1-w)``): exact numpy
        behaviour up to float64 precision.
      - Outside: ``jnp.maximum(denom, 1e-6)`` keeps the algebra finite
        and produces a smooth surface that Head A's width loss can
        descend toward valid eta. The validity helper raises
        ``TiltingDomainError`` for invalid eta independently, so Head B's
        BCE labels are correct regardless.

    Returns an array broadcast-shaped from the inputs.
    """
    if statistic_name == "wald":
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - _phi(z))

    if statistic_name == "waldo":
        # denom = 1 - eta(1 - w); clamp to avoid divide-by-zero. The
        # clamped surface is smooth so Head A's width loss has a
        # gradient even when EtaNet predicts eta outside the admissible
        # range — letting the boundary penalty + width signal jointly
        # push eta back without masking the gradient out entirely.
        denom = jnp.maximum(1.0 - eta * (1.0 - w), 1e-6)
        mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * mu0) / denom
        norm_factor = w * sigma / denom
        a = jnp.abs(mu_eta - theta) / norm_factor
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / (denom * norm_factor)
        return _phi(b - a) + _phi(-a - b)

    raise NotImplementedError(
        f"power_law_tilted_pvalue_jax: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo')."
    )


def ot_tilted_pvalue_jax(
    theta: jax.Array,
    D: jax.Array,
    w: jax.Array,
    mu0: jax.Array,
    sigma: jax.Array,
    eta: jax.Array,
    statistic_name: str,
) -> jax.Array:
    """JAX port of `OTTilting.tilted_pvalue` for (ot, waldo|wald).

    Like ``power_law_tilted_pvalue_jax``, the JAX surface stays smooth
    and gradient-bearing for invalid eta via ``jnp.maximum(s_t, 1e-6)``
    rather than NaN-masking — Head A's width loss can descend toward
    valid eta even when EtaNet drifts outside [0, 1]. The numpy
    ``OTTilting.tilted_pvalue`` raises ``TiltingDomainError`` for
    invalid eta independently, so the validity helper (numpy-driven)
    labels Head B's BCE correctly regardless of what the JAX surface
    returns. An earlier round NaN-masked here too, which broke OT
    training entirely (every aux sample masked out of the boundary-
    penalty signal).

    See ``power_law_tilted_pvalue_jax`` for input/output shape conventions;
    signatures match for registry uniformity.
    """
    if statistic_name == "wald":
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - _phi(z))

    if statistic_name == "waldo":
        # mu_t = (1 - eta)*mu_n + eta*D, with mu_n = w*D + (1-w)*mu0.
        mu_n = w * D + (1.0 - w) * mu0
        mu_t = (1.0 - eta) * mu_n + eta * D
        # s_t = (w + eta*(1-w))*sigma; admissible iff > 0. We clamp to
        # keep the gradient alive even at slightly-invalid eta so Head A
        # can move out of the bad region under the joint width +
        # boundary signal. The validity helper (numpy path) raises
        # `TiltingDomainError` for eta outside [0, 1], so Head B's
        # labels remain correct.
        s_t = jnp.maximum((w + eta * (1.0 - w)) * sigma, 1e-6)
        a = jnp.abs(mu_t - theta) / s_t
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / s_t
        return _phi(b - a) + _phi(-a - b)

    raise NotImplementedError(
        f"ot_tilted_pvalue_jax: statistic={statistic_name!r} "
        f"not supported (expected 'wald' or 'waldo')."
    )


# Registry keyed on scheme.name. Add new schemes by registering here.
JAX_TILTED_PVALUE: dict[str, Callable[..., jax.Array]] = {
    "power_law": power_law_tilted_pvalue_jax,
    "ot": ot_tilted_pvalue_jax,
}


def get_jax_tilted_pvalue(scheme_name: str) -> Callable[..., jax.Array]:
    """Look up the JAX tilted-p-value function for a scheme.

    Raises ``NotImplementedError`` if the scheme has no registered JAX
    p-value (training is gated on this; the inference-time
    `tilted_pvalue` on the scheme still works via its own kernel +
    scalar fast path).
    """
    if scheme_name not in JAX_TILTED_PVALUE:
        raise NotImplementedError(
            f"No JAX tilted_pvalue registered for scheme {scheme_name!r}. "
            f"Available: {sorted(JAX_TILTED_PVALUE)}. "
            f"To train against a new scheme, register a JAX p-value here."
        )
    return JAX_TILTED_PVALUE[scheme_name]
