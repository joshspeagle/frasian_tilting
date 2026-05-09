"""L2 regression: conditional EtaNet/ValidityNet input shapes (Phase G).

Pins the new 3-block (EtaNet) / 4-block (ValidityNet) input layout.
Old fixed-prior single-input forward is intentionally removed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from frasian.learned.training.architecture import EtaNet, ValidityNet


@pytest.mark.L2
class TestConditionalEtaNetShapes:
    def test_forward_three_block_input(self):
        key = jax.random.PRNGKey(0)
        net = EtaNet(theta_dim=1, prior_dim=2, lik_dim=1, key=key)
        N = 16
        theta = jnp.zeros((N,))
        prior_hp = jnp.tile(jnp.array([0.0, 1.0]), (N, 1))   # (N, 2)
        lik_hp = jnp.tile(jnp.array([1.0]), (N, 1))           # (N, 1)
        out = net(theta, prior_hp, lik_hp)
        assert out.shape == (N,)
        assert jnp.all(jnp.isfinite(out))

    def test_default_constructor_requires_explicit_dims(self):
        """Phase G hard-replace: no defaults for prior_dim/lik_dim."""
        key = jax.random.PRNGKey(0)
        with pytest.raises(TypeError):
            EtaNet(theta_dim=1, key=key)   # missing prior_dim, lik_dim

    def test_zero_lik_dim_works(self):
        """Bernoulli has hyperparam_dim=0 — net must accept lik_dim=0."""
        key = jax.random.PRNGKey(0)
        net = EtaNet(theta_dim=1, prior_dim=2, lik_dim=0, key=key)
        N = 4
        theta = jnp.zeros((N,))
        prior_hp = jnp.tile(jnp.array([2.0, 3.0]), (N, 1))
        lik_hp = jnp.empty((N, 0))
        out = net(theta, prior_hp, lik_hp)
        assert out.shape == (N,)


@pytest.mark.L2
class TestConditionalValidityNetShapes:
    def test_forward_four_block_input(self):
        key = jax.random.PRNGKey(0)
        net = ValidityNet(theta_dim=1, prior_dim=2, lik_dim=1, key=key)
        N = 16
        theta = jnp.zeros((N,))
        prior_hp = jnp.tile(jnp.array([0.0, 1.0]), (N, 1))
        lik_hp = jnp.tile(jnp.array([1.0]), (N, 1))
        eta = jnp.zeros((N,))
        logits = net(theta, prior_hp, lik_hp, eta)
        assert logits.shape == (N,)
