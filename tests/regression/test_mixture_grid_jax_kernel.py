"""Generic-grid JAX kernel for MixtureTilting (m-geodesic).

Companion to ``test_mixture_jax_kernel.py`` (which tests the
NN-channel closed-form). The generic grid kernel is for non-NN models
(Bernoulli + future); analogous to ``generic_grid_tilted_pvalue`` for
power_law.

Endpoint sanity:
- η=0 → mixture moments = posterior moments → bare WALDO normal-approx
- η=1 → mixture moments = likelihood-as-dist moments → likelihood-only
  normal-approx

Numerical-agreement test against the numpy generic mixture moments
(``mixture._generic_tilted_moments_mixture``) is implicit via the
endpoint-collapse tests; the closed-form NN agreement at η=0 / η=1 is
already covered by the NN test file.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.L1
def test_mixture_grid_endpoint_eta_zero_collapses_to_posterior():
    """At η=0, mixture moments must equal posterior moments → p-value
    matches the posterior-only WALDO surrogate."""
    import jax.numpy as jnp

    from frasian.learned.training.pvalue_jax import (
        _mixture_grid_component_moments,
        mixture_grid_tilted_pvalue,
    )

    # Build a Normal-Normal-like log-density grid: log L = -(θ-D)²/(2σ²),
    # log π = -(θ-μ₀)²/(2σ₀²), with broadcast-friendly shape.
    theta_grid = jnp.linspace(-10.0, 10.0, 401, dtype=jnp.float64)
    D_b = jnp.asarray([0.5, 1.0, -0.5], dtype=jnp.float64)
    sigma = 1.0
    mu0, sigma0 = 0.0, 2.0
    log_p_lik_grid = -((theta_grid[None, :] - D_b[:, None]) ** 2) / (2.0 * sigma**2)
    log_p_prior_grid = -((theta_grid - mu0) ** 2) / (2.0 * sigma0**2)

    mu_post, var_post, mu_lik, var_lik = _mixture_grid_component_moments(
        log_p_lik_grid, log_p_prior_grid, theta_grid
    )
    mu_post_np = np.asarray(mu_post)
    mu_lik_np = np.asarray(mu_lik)

    # Closed-form posterior on Normal-Normal: μ_post = wD + (1−w)μ₀.
    w = sigma0**2 / (sigma**2 + sigma0**2)
    expected_mu_post = w * np.asarray(D_b) + (1.0 - w) * mu0
    expected_mu_lik = np.asarray(D_b)

    assert np.allclose(mu_post_np, expected_mu_post, atol=1e-6), (
        f"posterior μ on grid: got {mu_post_np}, expected {expected_mu_post}"
    )
    assert np.allclose(mu_lik_np, expected_mu_lik, atol=1e-6), (
        f"likelihood-as-dist μ on grid: got {mu_lik_np}, expected {expected_mu_lik}"
    )

    # At η=0, the mixture p-value should match the bare-posterior p-value.
    eta = jnp.zeros((3, 5), dtype=jnp.float64)
    theta_test = jnp.broadcast_to(
        jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float64), (3, 5)
    )
    p_mix = mixture_grid_tilted_pvalue(
        theta_test=theta_test,
        eta=eta,
        log_p_lik_grid=log_p_lik_grid,
        log_p_prior_grid=log_p_prior_grid,
        theta_grid=theta_grid,
        statistic_name="waldo",
    )
    # Manual posterior-only WALDO surrogate: 2(1 − Φ(|μ_post − θ|/√var_post)).
    from jax.scipy.stats import norm as _jsn

    sigma_post = jnp.sqrt(var_post)[:, None]
    z = jnp.abs(mu_post[:, None] - theta_test) / sigma_post
    p_expected = 2.0 * (1.0 - _jsn.cdf(z))
    assert np.allclose(np.asarray(p_mix), np.asarray(p_expected), atol=1e-10)


@pytest.mark.L1
def test_mixture_grid_endpoint_eta_one_collapses_to_likelihood():
    """At η=1, mixture moments must equal likelihood-as-dist moments →
    p-value matches the likelihood-only normal-approx."""
    import jax.numpy as jnp

    from frasian.learned.training.pvalue_jax import (
        _mixture_grid_component_moments,
        mixture_grid_tilted_pvalue,
    )

    theta_grid = jnp.linspace(-10.0, 10.0, 401, dtype=jnp.float64)
    D_b = jnp.asarray([0.5, 1.0, -0.5], dtype=jnp.float64)
    sigma = 1.0
    log_p_lik_grid = -((theta_grid[None, :] - D_b[:, None]) ** 2) / (2.0 * sigma**2)
    log_p_prior_grid = -((theta_grid - 0.0) ** 2) / (2.0 * 4.0)

    _, _, mu_lik, var_lik = _mixture_grid_component_moments(
        log_p_lik_grid, log_p_prior_grid, theta_grid
    )

    eta = jnp.ones((3, 5), dtype=jnp.float64)
    theta_test = jnp.broadcast_to(
        jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float64), (3, 5)
    )
    p_mix = mixture_grid_tilted_pvalue(
        theta_test=theta_test,
        eta=eta,
        log_p_lik_grid=log_p_lik_grid,
        log_p_prior_grid=log_p_prior_grid,
        theta_grid=theta_grid,
        statistic_name="waldo",
    )
    from jax.scipy.stats import norm as _jsn

    sigma_lik = jnp.sqrt(var_lik)[:, None]
    z = jnp.abs(mu_lik[:, None] - theta_test) / sigma_lik
    p_expected = 2.0 * (1.0 - _jsn.cdf(z))
    assert np.allclose(np.asarray(p_mix), np.asarray(p_expected), atol=1e-10)


@pytest.mark.L1
def test_mixture_grid_vmap_jit_compatible():
    """JIT + vmap-compatible across batch and θ_test axes."""
    import jax
    import jax.numpy as jnp

    from frasian.learned.training.pvalue_jax import mixture_grid_tilted_pvalue

    theta_grid = jnp.linspace(-10.0, 10.0, 401, dtype=jnp.float64)
    D_b = jnp.asarray([0.5, 1.0, -0.5, 0.0], dtype=jnp.float64)
    log_p_lik_grid = -((theta_grid[None, :] - D_b[:, None]) ** 2) / 2.0
    log_p_prior_grid = -((theta_grid - 0.0) ** 2) / (2.0 * 4.0)

    @jax.jit
    def fn(theta_test, eta):
        return mixture_grid_tilted_pvalue(
            theta_test=theta_test,
            eta=eta,
            log_p_lik_grid=log_p_lik_grid,
            log_p_prior_grid=log_p_prior_grid,
            theta_grid=theta_grid,
            statistic_name="waldo",
        )

    theta_test = jnp.broadcast_to(
        jnp.linspace(-2.0, 2.0, 7, dtype=jnp.float64), (4, 7)
    )
    eta = jnp.full((4, 7), 0.3, dtype=jnp.float64)
    p = fn(theta_test, eta)
    assert p.shape == (4, 7)
    arr = np.asarray(p)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)


@pytest.mark.L1
def test_mixture_grid_registered_in_jax_tilted_pvalue():
    """The (mixture, generic) cell must be registered so the runner
    can route bernoulli + mixture training through the JAX path."""
    from frasian.learned.training.pvalue_jax import (
        JAX_TILTED_PVALUE,
        get_jax_tilted_pvalue,
        mixture_grid_tilted_pvalue,
    )

    assert ("mixture", "generic") in JAX_TILTED_PVALUE
    assert JAX_TILTED_PVALUE[("mixture", "generic")] is mixture_grid_tilted_pvalue
    # Resolver fallback path: an unknown model_kind on mixture should
    # land on the generic kernel.
    assert get_jax_tilted_pvalue("mixture", "bernoulli") is mixture_grid_tilted_pvalue
