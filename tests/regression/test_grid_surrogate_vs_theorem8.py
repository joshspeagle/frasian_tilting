"""Phase 4 skeptic #1: document the grid surrogate's bias vs Theorem 8.

The grid kernel ``pvalue_jax.generic_grid_tilted_pvalue`` (the
training-time surrogate) returns the symmetric normal-approximation
p-value ``2(1 - Φ(|μ - θ|/σ_tilted))`` rather than Theorem 8's
asymmetric ``Φ(b - a) + Φ(-a - b)``. Two algebraic differences between
the surrogate and Theorem 8:

1. **Different scale.** Theorem 8 uses ``ν = w·σ/denom`` (the standard
   error of ``μ_eta`` under H_0), while the surrogate uses ``σ_tilted
   = σ·√(w/denom)`` (the standard deviation of the tilted posterior
   itself). They differ by ``ν/σ_tilted = √(w/denom)``.

2. **Missing prior-conflict term ``b``.** Theorem 8's ``b =
   (1-η)(1-w)(μ₀-θ)/(denom·ν)`` carries the prior-data conflict
   asymmetry; the surrogate drops it entirely.

These are not bugs that can be fixed in the generic grid kernel:
``b`` requires NN-specific ``w, μ₀, denom``, and ``ν`` requires the
marginal sampling distribution of ``μ_tilted`` under H_0 — neither
is available from the tilted posterior moments alone. On a generic
model only Monte-Carlo over D' under H_0 recovers ``ν`` and ``b``,
which is what ``power_law._generic_tilted_pvalue`` does at inference
time (n_mc=200) — at training time it is too expensive.

**Important:** the surrogate is used ONLY for non-NN training
(``WIDTH_LOSS_DISPATCH[("power_law", "generic")]``). NN training
routes through ``_call_normal_normal_pvalue`` → ``power_law_tilted_pvalue_jax``,
which IS Theorem 8 exactly. So the bias documented here only
affects any future non-NN model. For such models the inference-time
path would use the MC reference (correct), and the trained
selector's calibration would be verified against that — not against
the surrogate.

These tests therefore PIN the bias as a regression: a future
implementation that closes the gap should tighten the bounds and
update the docstring; an implementation that widens the gap should
fail loudly. They are NOT a calibration claim.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from frasian.learned.training.pvalue_jax import (
    generic_grid_tilted_pvalue,
    power_law_tilted_pvalue_jax,
)
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel


def _build_nn_grids(D: float, mu0: float, sigma: float, sigma0: float, n_grid: int = 1024):
    """Build (theta_grid, log_p_lik_grid_b, log_p_prior_grid) for one NN sample."""
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    posterior = model.posterior(np.asarray([D]), prior)
    post_mu, post_sigma = float(posterior.mean()), float(np.sqrt(posterior.var()))
    k = 8.0
    lo = min(post_mu - k * post_sigma, mu0 - k * sigma0)
    hi = max(post_mu + k * post_sigma, mu0 + k * sigma0)
    theta_grid = jnp.linspace(lo, hi, n_grid)
    likelihood = model.likelihood(np.asarray([D]))
    log_p_lik_grid = likelihood.loglik(theta_grid)
    log_p_prior_grid = prior.logpdf(theta_grid)
    return theta_grid, log_p_lik_grid[None, :], log_p_prior_grid


def _theorem8_p(D, theta_test, eta, mu0, sigma, sigma0):
    w = sigma0**2 / (sigma**2 + sigma0**2)
    G = theta_test.shape[0]
    return power_law_tilted_pvalue_jax(
        theta_test,
        jnp.full((G,), float(D)),
        jnp.asarray(w),
        jnp.asarray(float(mu0)),
        jnp.asarray(float(sigma)),
        jnp.full((G,), float(eta)),
        "waldo",
    )


def _grid_p(theta_test, eta, theta_grid, log_lik_b, log_prior):
    G = theta_test.shape[0]
    eta_b = jnp.full((1, G), float(eta))
    out = generic_grid_tilted_pvalue(
        theta_test[None, :], eta_b, log_lik_b, log_prior, theta_grid, "waldo",
    )
    return out[0]


def _argmin_eta(loss_fn, eta_grid: np.ndarray) -> float:
    losses = np.asarray([float(loss_fn(eta)) for eta in eta_grid])
    return float(eta_grid[int(np.argmin(losses))])


# Documented bias bounds. These are REGRESSION pins: the values were
# measured at the time of skeptic Phase 4 #1, with sigma = sigma0 = 1
# (w = 0.5). A future fix that closes the gap should tighten these.
_DOCUMENTED_NO_CONFLICT_PVALUE_DRIFT = 0.20  # at b = 0
_DOCUMENTED_ARGMIN_ETA_DRIFT_AT_CONFLICT = 1.5  # |Δ| ≥ 1.5


@pytest.mark.L2
def test_documented_pvalue_drift_at_no_conflict() -> None:
    """At b = 0 (μ₀ = θ_test) the surrogate's scale-inflation bias.

    Even though Theorem 8's ``b`` term vanishes at θ_test = μ₀, the
    surrogate uses ``σ_tilted`` while Theorem 8 uses ``ν = w·σ/denom``,
    which differ by ``√(w/denom)``. Pin the worst-case p-value drift
    at b = 0 across (D, η) so a future kernel change can't silently
    increase it.
    """
    sigma, sigma0 = 1.0, 1.0
    mu0 = 0.5
    D = 0.0
    theta_grid, log_lik_b, log_prior = _build_nn_grids(D, mu0, sigma, sigma0)
    theta_test = jnp.asarray([mu0])
    max_drift = 0.0
    for eta in (-0.5, 0.0, 0.3, 0.7):
        p_grid = float(_grid_p(theta_test, eta, theta_grid, log_lik_b, log_prior)[0])
        p_t8 = float(_theorem8_p(D, theta_test, eta, mu0, sigma, sigma0)[0])
        drift = abs(p_grid - p_t8)
        max_drift = max(max_drift, drift)
        assert 0.0 <= p_grid <= 1.0 + 1e-9
        assert 0.0 <= p_t8 <= 1.0 + 1e-9
    assert max_drift <= _DOCUMENTED_NO_CONFLICT_PVALUE_DRIFT, (
        f"Surrogate-vs-Theorem 8 b=0 p-value drift exceeds documented "
        f"bound: {max_drift:.4f} > {_DOCUMENTED_NO_CONFLICT_PVALUE_DRIFT}. "
        f"The scale-inflation bias may have widened; investigate."
    )


@pytest.mark.L2
@pytest.mark.parametrize("D", [-1.5, 0.0, 1.5])
def test_documented_argmin_eta_drift_at_conflict(D: float) -> None:
    """argmin_η drift between surrogate and Theorem 8 at conflict.

    This pins how far the trained η would drift from the
    Theorem-8-optimal η on Normal-Normal IF the surrogate were used
    for NN training. (In production, NN training uses Theorem 8
    directly via `_call_normal_normal_pvalue`; this test is a
    regression on the surrogate's quality, not a calibration claim.)

    The bound 1.5 covers the worst-case observed drift across the
    (D, η) sweep; a future kernel that closes the bias should
    tighten this.
    """
    sigma, sigma0 = 1.0, 1.0
    mu0 = 0.0
    theta_grid, log_lik_b, log_prior = _build_nn_grids(D, mu0, sigma, sigma0)
    post_mu = sigma0**2 / (sigma**2 + sigma0**2) * D
    post_sigma = sigma * np.sqrt(sigma0**2 / (sigma**2 + sigma0**2))
    theta_test = jnp.linspace(post_mu - 3 * post_sigma, post_mu + 3 * post_sigma, 21)

    eta_grid = np.linspace(-1.0, 0.8, 19)

    def loss_grid(eta):
        p = _grid_p(theta_test, eta, theta_grid, log_lik_b, log_prior)
        return float(jnp.trapezoid(p, theta_test))

    def loss_t8(eta):
        p = _theorem8_p(D, theta_test, eta, mu0, sigma, sigma0)
        return float(jnp.trapezoid(p, theta_test))

    eta_grid_argmin = _argmin_eta(loss_grid, eta_grid)
    eta_t8_argmin = _argmin_eta(loss_t8, eta_grid)
    drift = abs(eta_grid_argmin - eta_t8_argmin)
    assert drift <= _DOCUMENTED_ARGMIN_ETA_DRIFT_AT_CONFLICT, (
        f"argmin_η drift {drift:.3f} > documented bound "
        f"{_DOCUMENTED_ARGMIN_ETA_DRIFT_AT_CONFLICT} at D={D}: "
        f"surrogate picks η={eta_grid_argmin:.3f}, Theorem 8 picks "
        f"η={eta_t8_argmin:.3f}. The surrogate's bias has widened; "
        f"either the kernel changed or this regression bound needs review."
    )


@pytest.mark.L2
def test_surrogate_finite_and_bounded_in_admissible_range() -> None:
    """Sanity: the surrogate produces a finite p-value in [0, 1] across
    the η sweep that triggered the failure mode in skeptic Phase 4 #1.

    This confirms that even though the surrogate is biased, it is at
    least a proper p-value (no NaN, no overflow, no excursion outside
    [0, 1]) — so training won't blow up on the surrogate alone.
    """
    sigma, sigma0 = 1.0, 1.0
    mu0 = 0.0
    for D in (-2.0, 0.0, 2.0):
        theta_grid, log_lik_b, log_prior = _build_nn_grids(D, mu0, sigma, sigma0)
        post_mu = sigma0**2 / (sigma**2 + sigma0**2) * D
        post_sigma = sigma * np.sqrt(sigma0**2 / (sigma**2 + sigma0**2))
        theta_test = jnp.linspace(post_mu - 3 * post_sigma, post_mu + 3 * post_sigma, 21)
        for eta in np.linspace(-1.0, 0.8, 11):
            p = np.asarray(_grid_p(theta_test, float(eta), theta_grid, log_lik_b, log_prior))
            assert np.all(np.isfinite(p)), f"non-finite p at D={D}, η={eta}"
            assert np.all(p >= 0.0) and np.all(p <= 1.0 + 1e-9), (
                f"p outside [0, 1] at D={D}, η={eta}: min={p.min()}, max={p.max()}"
            )
