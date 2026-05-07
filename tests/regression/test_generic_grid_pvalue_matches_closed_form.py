"""Phase 4a regression: generic-grid kernel correctness on Normal-Normal.

The Phase 4 generic-grid kernel computes:
1. Tilted moments `(μ_tilted, σ²_tilted)` from `log L + (1-η) log π`
   on a θ-grid (deriver-verified to reduce to PowerLawTilting's
   Theorem 6 closed form on Normal-Normal at atol 1e-7 with N=1024).
2. WALDO p-value via the **simple symmetric normal approximation**
   `2(1 - Φ(|μ - θ|/σ))`.

Important: the kernel's p-value formula is NOT the asymmetric
`Φ(b-a) + Φ(-a-b)` two-term Theorem 8 form (which `power_law_tilted_pvalue_jax`
returns). The symmetric form is the deliberate training surrogate
choice — it has cleaner gradients through η and is sufficient as a
differentiable loss target. Production CI uses MC over D' for the
exact reference. This test pins MOMENT agreement (Theorem 6), not
p-value agreement (Theorem 8 differs).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from frasian.learned.training.pvalue_jax import (
    _generic_grid_tilted_moments,
    generic_grid_tilted_pvalue,
)
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import (
    NormalNormalModel,
    posterior_params,
    weight,
)


def _build_grid_inputs(
    D: float,
    model: NormalNormalModel,
    prior: NormalDistribution,
    n_grid: int = 1024,
    k: float = 8.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build (theta_grid, log_p_lik_grid_b, log_p_prior_grid) for the
    generic kernel. Single-batch (B=1)."""
    posterior = model.posterior(np.asarray([D]), prior)
    post_mu, post_sigma = float(posterior.mean()), float(np.sqrt(posterior.var()))
    prior_mu, prior_sigma = float(prior.mean()), float(np.sqrt(prior.var()))
    lo = min(post_mu - k * post_sigma, prior_mu - k * prior_sigma)
    hi = max(post_mu + k * post_sigma, prior_mu + k * prior_sigma)
    theta_grid = jnp.linspace(lo, hi, n_grid)
    likelihood = model.likelihood(np.asarray([D]))
    log_p_lik_grid = likelihood.loglik(theta_grid)
    log_p_prior_grid = prior.logpdf(theta_grid)
    return theta_grid, log_p_lik_grid[None, :], log_p_prior_grid


def _theorem6_moments(
    D: float, eta: float, mu0: float, sigma: float, sigma0: float
) -> tuple[float, float]:
    """Closed-form Theorem 6 (mu_eta, sigma_eta^2) for the tilted Normal-Normal posterior."""
    w = weight(sigma, sigma0)
    denom = 1.0 - eta * (1.0 - w)
    mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * mu0) / denom
    sigma_eta_sq = w * sigma**2 / denom
    return float(mu_eta), float(sigma_eta_sq)


@pytest.mark.L2
@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("D", [-1.0, 0.0, 0.5, 1.5])
@pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
def test_generic_grid_moments_match_theorem6(eta, D, sigma0):
    """Tilted moments from the grid kernel match Theorem 6 closed form.

    Deriver verified atol 1e-7 at N=1024 in standalone tests; the L2
    tolerance here is 1e-3 to absorb residual grid quantisation.
    """
    sigma, mu0 = 1.0, 0.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)

    theta_grid, log_p_lik_grid_b, log_p_prior_grid = _build_grid_inputs(
        D, model, prior, n_grid=1024
    )
    # eta is per (sample, theta_test); use a single theta_test point so
    # we read the moments for that (B=1, G_test=1) cell.
    eta_b = jnp.full((1, 1), float(eta))

    mu_grid, var_grid = _generic_grid_tilted_moments(
        eta_b, log_p_lik_grid_b, log_p_prior_grid, theta_grid
    )
    mu_grid_f = float(np.asarray(mu_grid)[0, 0])
    var_grid_f = float(np.asarray(var_grid)[0, 0])

    mu_t6, sigma2_t6 = _theorem6_moments(D, eta, mu0, sigma, sigma0)

    np.testing.assert_allclose(mu_grid_f, mu_t6, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(var_grid_f, sigma2_t6, atol=1e-3, rtol=1e-3)


@pytest.mark.L0
def test_generic_grid_pvalue_shape_and_finite():
    """End-to-end smoke: p-value finite, shape (B, G_test), in [0, 1]."""
    sigma, mu0, sigma0 = 1.0, 0.0, 1.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    theta_grid, log_p_lik_grid_b, log_p_prior_grid = _build_grid_inputs(
        D=0.5, model=model, prior=prior, n_grid=512
    )
    theta_test_b = jnp.linspace(-2.0, 2.0, 11)[None, :]
    eta_b = jnp.full(theta_test_b.shape, 0.3)

    p = generic_grid_tilted_pvalue(
        theta_test_b, eta_b, log_p_lik_grid_b, log_p_prior_grid, theta_grid, "waldo"
    )
    p_arr = np.asarray(p)
    assert p_arr.shape == (1, 11)
    assert np.all(np.isfinite(p_arr))
    assert np.all(p_arr >= 0.0) and np.all(p_arr <= 1.0)


@pytest.mark.L0
def test_generic_grid_pvalue_monotone_in_distance():
    """`p(θ_test)` decreases monotonically as |θ_test - μ_tilted| grows.

    Sanity check on the symmetric-normal-approx WALDO kernel: at fixed
    moments, p(θ_test) = 2(1 - Φ(|μ-θ|/σ)) IS monotone in |θ - μ|.
    """
    sigma, mu0, sigma0 = 1.0, 0.0, 1.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    theta_grid, log_p_lik_grid_b, log_p_prior_grid = _build_grid_inputs(
        D=0.5, model=model, prior=prior, n_grid=512
    )
    # μ_tilted ≈ 0.25 (posterior mean for D=0.5, mu0=0, w=0.5).
    theta_test_b = jnp.array([[0.25, 0.25 + 0.5, 0.25 + 1.5, 0.25 + 3.0]])
    eta_b = jnp.full(theta_test_b.shape, 0.0)
    p = generic_grid_tilted_pvalue(
        theta_test_b, eta_b, log_p_lik_grid_b, log_p_prior_grid, theta_grid, "waldo"
    )
    p_arr = np.asarray(p)[0]
    assert p_arr[0] > p_arr[1] > p_arr[2] > p_arr[3], (
        f"p-value not monotone in |θ_test - μ|: {p_arr}"
    )


@pytest.mark.L0
def test_generic_grid_grad_through_eta_finite():
    """`jax.grad` through the grid kernel via η is finite — the load-
    bearing autodiff property for Phase 4 learned-η training."""
    import jax

    sigma, mu0, sigma0 = 1.0, 0.0, 1.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    theta_grid, log_p_lik_grid_b, log_p_prior_grid = _build_grid_inputs(
        D=0.5, model=model, prior=prior, n_grid=512
    )
    theta_test_b = jnp.linspace(-2.0, 2.0, 11)[None, :]

    def loss(eta_scalar):
        eta_b = jnp.full(theta_test_b.shape, eta_scalar)
        p = generic_grid_tilted_pvalue(
            theta_test_b, eta_b, log_p_lik_grid_b, log_p_prior_grid, theta_grid, "waldo"
        )
        return jnp.sum(p)

    for eta_val in [0.0, 0.3, 0.7]:
        g = float(jax.grad(loss)(jnp.asarray(eta_val)))
        assert np.isfinite(g), (
            f"jax.grad(generic_grid_tilted_pvalue) NaN/Inf at eta={eta_val}"
        )


@pytest.mark.L0
def test_generic_grid_jit_traceable():
    """`@jax.jit` traces through the kernel without errors."""
    import jax

    sigma, mu0, sigma0 = 1.0, 0.0, 1.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    theta_grid, log_p_lik_grid_b, log_p_prior_grid = _build_grid_inputs(
        D=0.5, model=model, prior=prior, n_grid=256
    )
    theta_test_b = jnp.linspace(-2.0, 2.0, 11)[None, :]
    eta_b = jnp.full(theta_test_b.shape, 0.3)

    jit_kernel = jax.jit(generic_grid_tilted_pvalue, static_argnames=("statistic_name",))
    p = jit_kernel(theta_test_b, eta_b, log_p_lik_grid_b, log_p_prior_grid, theta_grid, "waldo")
    assert np.all(np.isfinite(np.asarray(p)))
    assert np.asarray(p).shape == (1, 11)


@pytest.mark.L0
def test_generic_grid_bernoulli_smoke():
    """Smoke: kernel runs end-to-end on Bernoulli + Beta inputs.

    Verifies the marquee Phase 4 capability — JAX-traceable tilted
    pvalue against a non-Normal-Normal pair.
    """
    from frasian.models.bernoulli import BernoulliModel
    from frasian.models.distributions import BetaDistribution

    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0])
    n_grid = 512
    theta_grid = jnp.linspace(0.01, 0.99, n_grid)  # avoid exact boundary
    likelihood = model.likelihood(data)
    log_p_lik_grid = likelihood.loglik(theta_grid)[None, :]  # (1, N_grid)
    log_p_prior_grid = prior.logpdf(theta_grid)

    theta_test = jnp.linspace(0.1, 0.9, 11)[None, :]
    eta_b = jnp.full(theta_test.shape, 0.3)
    p = generic_grid_tilted_pvalue(
        theta_test, eta_b, log_p_lik_grid, log_p_prior_grid, theta_grid, "waldo"
    )
    p_arr = np.asarray(p)
    assert p_arr.shape == (1, 11)
    assert np.all(np.isfinite(p_arr))
    assert np.all(p_arr >= 0.0) and np.all(p_arr <= 1.0)
