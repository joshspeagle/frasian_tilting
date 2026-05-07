"""JAX/Equinox ``EtaNet`` invariants (Phase F port commit 2).

Pinned invariants:

- Forward output shape matches input batch shape.
- Gradient through θ is finite (autodiff-clean across the network).
- Gradient through parameters is finite.
- Round-trip via ``eqx.tree_serialise_leaves`` / ``tree_deserialise_leaves``
  recovers byte-equal weights.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from frasian.learned.training.architecture import EtaNet


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_forward_shape_matches_input():
    key = jax.random.PRNGKey(0)
    net = EtaNet(theta_dim=1, key=key)
    for n in (1, 5, 11):
        theta = jnp.linspace(-3.0, 3.0, n)
        out = net(theta)
        assert out.shape == theta.shape


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_forward_2d_input_shape_matches_leading_axis():
    key = jax.random.PRNGKey(1)
    net = EtaNet(theta_dim=3, key=key)
    theta = jax.random.normal(jax.random.PRNGKey(2), (8, 3))
    out = net(theta)
    assert out.shape == (8,)


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_grad_through_theta_is_finite():
    """∂(sum out)/∂θ is finite for a smooth-MLP forward."""
    key = jax.random.PRNGKey(0)
    net = EtaNet(theta_dim=1, key=key)
    theta = jnp.linspace(-3.0, 3.0, 11)

    def scalar(t):
        return net(t).sum()

    g = jax.grad(scalar)(theta)
    assert g.shape == theta.shape
    assert bool(jnp.isfinite(g).all())


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_grad_through_params_is_finite():
    """∂(sum out)/∂θ across EtaNet params yields finite values."""
    key = jax.random.PRNGKey(0)
    net = EtaNet(theta_dim=1, key=key)
    theta = jnp.linspace(-3.0, 3.0, 11)

    params, static = eqx.partition(net, eqx.is_array)

    def loss(p):
        return eqx.combine(p, static)(theta).sum()

    g = jax.grad(loss)(params)
    leaves = jtu.tree_leaves(g)
    assert leaves, "expected at least one parameter leaf"
    for leaf in leaves:
        assert bool(jnp.isfinite(leaf).all())


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_serialise_round_trip_byte_equal():
    """``eqx.tree_serialise_leaves`` round-trips weights byte-equal."""
    key = jax.random.PRNGKey(7)
    net = EtaNet(theta_dim=1, key=key)
    theta = jnp.linspace(-3.0, 3.0, 11)
    out_before = net(theta)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "etanet.eqx"
        eqx.tree_serialise_leaves(str(path), net)
        # Reconstruct an equally-shaped skeleton and load.
        skeleton = EtaNet(theta_dim=1, key=jax.random.PRNGKey(999))
        loaded = eqx.tree_deserialise_leaves(str(path), skeleton)

    out_after = loaded(theta)
    np.testing.assert_array_equal(np.asarray(out_before), np.asarray(out_after))

    # Leaf-by-leaf byte equality.
    leaves_before = jtu.tree_leaves(eqx.partition(net, eqx.is_array)[0])
    leaves_after = jtu.tree_leaves(eqx.partition(loaded, eqx.is_array)[0])
    assert len(leaves_before) == len(leaves_after)
    for a, b in zip(leaves_before, leaves_after):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_deterministic_on_same_key():
    """Same PRNGKey → identical weights."""
    key = jax.random.PRNGKey(42)
    a = EtaNet(theta_dim=1, key=key)
    b = EtaNet(theta_dim=1, key=key)
    leaves_a = jtu.tree_leaves(eqx.partition(a, eqx.is_array)[0])
    leaves_b = jtu.tree_leaves(eqx.partition(b, eqx.is_array)[0])
    for la, lb in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))
