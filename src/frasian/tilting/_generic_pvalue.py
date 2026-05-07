"""Shared helpers for generic tilted-pvalue paths.

Both `PowerLawTilting` and `OTTilting` implement model-agnostic
numerical paths (Phase 3 of the master plan). The MC reference
distribution under H_0 is identical in algorithm — only the per-D
moment-extraction differs (log-density combo for PowerLaw via
`log L + (1-η) log π`; quantile-mixture for OT via the W2 geodesic
between posterior and likelihood-as-distribution).

This module hosts the **scheme-agnostic** plumbing:
- `_resolve_support(model, data)`: support inference with
  Distribution-protocol fallback to likelihood-class isinstance.
- `_stable_tilted_pvalue_seed(...)`: cross-process stable
  `hashlib.blake2b` seed for CRN-seeded MC reference.
- `likelihood_as_distribution(model, data, support, n_grid)`: build
  a `GridDistribution` representing the likelihood as a function of
  θ (normalised), used by `OTTilting._generic_tilt` as the second
  endpoint of the W2 geodesic.

`PowerLawTilting`-specific and `OTTilting`-specific logic (the
moment-extraction kernels, the public dispatch, the MC inner loop
with its scheme-specific `t_at_d` callback) stays in the respective
scheme modules.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._grid_distribution import GridDistribution, grid_distribution_from_log_density


def _resolve_support(
    model: object, data: NDArray[np.float64]
) -> tuple[float, float]:
    """Resolve `(support_lo, support_hi)` from a `Model` (preferred) or by
    falling back to inspecting the likelihood's class.

    The `Model` protocol declares `support()` so the `hasattr` guard is
    defensive; the likelihood-class fallback is retained for the rare
    `Model`-protocol partial conformer (test fixtures that mock just
    enough). Centralised here to avoid the same dispatch tree being
    inlined in PowerLaw / OT generic paths.
    """
    from ..models.distributions import BernoulliLikelihood, GaussianLikelihood

    if hasattr(model, "support"):
        lo, hi = tuple(model.support())
        return float(lo), float(hi)
    likelihood = model.likelihood(data)
    if isinstance(likelihood, BernoulliLikelihood):
        return (0.0, 1.0)
    if isinstance(likelihood, GaussianLikelihood):
        return (-float("inf"), float("inf"))
    return (-float("inf"), float("inf"))


def _stable_tilted_pvalue_seed(
    data: NDArray[np.float64],
    model: object,
    prior: Any,
    eta: float,
    alpha: float,
    base_seed: int,
) -> int:
    """Cross-process-stable 32-bit seed for MC reference draws.

    Hashlib.blake2b over a deterministic byte encoding so the seed is
    reproducible across Python processes regardless of PYTHONHASHSEED.
    Critically the seed is INDEPENDENT of theta — that's what makes
    common random numbers across brentq probes possible (skeptic
    finding from Phase 2). Used by both PowerLaw and OT generic paths.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(data, dtype=np.float64).tobytes())
    h.update(repr(model.fingerprint()).encode("utf-8"))
    h.update(repr(prior.fingerprint()).encode("utf-8"))
    h.update(np.float64(eta).tobytes())
    h.update(np.float64(alpha).tobytes())
    h.update(np.int64(base_seed).tobytes())
    return int.from_bytes(h.digest()[:4], "little", signed=False)


def likelihood_as_distribution(
    model: object,
    data: NDArray[np.float64],
    support: tuple[float, float],
    n_grid: int = 1024,
) -> GridDistribution:
    """Build a `Distribution`-conforming view of the likelihood.

    Used as the second endpoint of OTTilting's W2 geodesic when the
    likelihood is not natively a Distribution (e.g. BernoulliLikelihood
    only has `loglik`, not `pdf`/`cdf`/`quantile`). The likelihood
    treated as a function of θ — `L(θ) = exp(loglik(θ))` — is
    integrable on the model support for any common likelihood class
    (Beta-shaped on [0,1] for Bernoulli, Gaussian-shaped on R for
    Normal location). Normalisation is via trapezoidal Z on a fine
    θ-grid; bounded supports use the support window directly,
    unbounded supports use a heuristic mean ± 6·σ window centred on
    the MLE.
    """
    from ..models.distributions import GaussianLikelihood

    lo, hi = float(support[0]), float(support[1])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        # Unbounded: use MLE ± 6σ. For the Normal sandbox this matches
        # the framework's default "mean of data" + "model.sigma" widths;
        # other models with unbounded supports need their own heuristic
        # but are not currently exercised.
        try:
            mle = float(np.asarray(model.mle(data)))
        except (TypeError, AttributeError, ValueError):
            mle = 0.0
        likelihood = model.likelihood(data)
        if isinstance(likelihood, GaussianLikelihood):
            sigma = float(likelihood.sigma)
        else:
            sigma = 1.0
        lo_eff, hi_eff = mle - 6.0 * sigma, mle + 6.0 * sigma
    else:
        lo_eff, hi_eff = lo, hi

    theta_grid = np.linspace(lo_eff, hi_eff, n_grid)
    likelihood = model.likelihood(data)
    log_lik = np.asarray(likelihood.loglik(theta_grid), dtype=np.float64)
    log_lik = np.where(np.isfinite(log_lik), log_lik, -1e300)

    return grid_distribution_from_log_density(
        theta_grid,
        log_lik,
        metadata={"role": "likelihood_as_distribution", "n_grid": int(n_grid)},
    )
