"""Differentiability: gradient checks on integrated_p / cd_variance / static_width losses.

Phase F port commit 2: the losses are JAX kernels; the gradcheck is
JAX's centred-finite-difference comparison against ``jax.grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frasian.learned.training.losses import (
    cd_variance_loss,
    integrated_pvalue_loss,
    static_width_loss,
)


def _random_p_theta(B: int = 2, N: int = 17, seed: int = 0):
    """Generate a random p-value array (B, N) ∈ (0, 1)."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((B, N)).astype(np.float64)
    # Pass through sigmoid to get values in (0, 1).
    p = 1.0 / (1.0 + np.exp(-raw))
    return jnp.asarray(p)


def _gradcheck(f, x, eps=1e-6, atol=1e-4):
    """Centred-finite-difference gradient check vs ``jax.grad``."""
    g_auto = np.asarray(jax.grad(lambda y: f(y).sum())(x))
    flat_x = np.asarray(x).ravel().copy()
    g_fd = np.zeros_like(flat_x)
    for i in range(flat_x.size):
        flat_p = flat_x.copy()
        flat_p[i] += eps
        flat_m = flat_x.copy()
        flat_m[i] -= eps
        xp = jnp.asarray(flat_p.reshape(x.shape))
        xm = jnp.asarray(flat_m.reshape(x.shape))
        g_fd[i] = (float(f(xp).sum()) - float(f(xm).sum())) / (2.0 * eps)
    np.testing.assert_allclose(g_auto.ravel(), g_fd, atol=atol)


@pytest.mark.L1
@pytest.mark.properties
def test_integrated_pvalue_loss_gradcheck():
    """∫p dθ is differentiable w.r.t. p."""
    p = _random_p_theta()
    theta = jnp.linspace(-2.0, 2.0, 17)

    def f(pp):
        return integrated_pvalue_loss(pp, theta)

    _gradcheck(f, p, eps=1e-6, atol=1e-4)


@pytest.mark.L1
@pytest.mark.properties
def test_cd_variance_loss_gradcheck():
    """CD variance loss is differentiable w.r.t. p."""
    p = _random_p_theta(N=21)
    theta = jnp.linspace(-2.0, 2.0, 21)

    def f(pp):
        return cd_variance_loss(pp, theta)

    _gradcheck(f, p, eps=1e-6, atol=1e-3)


@pytest.mark.L1
@pytest.mark.properties
def test_static_width_loss_gradcheck():
    """Sigmoid-relaxed static-width loss is differentiable w.r.t. p."""
    p = _random_p_theta(N=21)
    theta = jnp.linspace(-2.0, 2.0, 21)

    def f(pp):
        return static_width_loss(pp, theta, alpha=0.1, sharpness=10.0)

    _gradcheck(f, p, eps=1e-6, atol=1e-4)


@pytest.mark.L1
def test_static_width_loss_invalid_alpha():
    """alpha out of (0, 1) raises."""
    p = _random_p_theta()
    theta = jnp.linspace(-2.0, 2.0, 17)
    with pytest.raises(ValueError):
        static_width_loss(p, theta, alpha=0.0)
    with pytest.raises(ValueError):
        static_width_loss(p, theta, alpha=1.0)
