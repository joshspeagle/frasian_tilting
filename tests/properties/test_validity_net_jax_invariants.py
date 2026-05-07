"""JAX/Equinox ``ValidityNet`` invariants (Phase F port commit 2).

Pinned invariants:

- Forward output shape matches input batch (leading axis).
- Gradient through inputs is finite (autodiff-clean across the network).
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

from frasian.learned.training.architecture import ValidityNet


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_forward_shape_matches_input():
    key = jax.random.PRNGKey(0)
    net = ValidityNet(theta_dim=1, key=key)
    for n in (1, 5, 13):
        inputs = jax.random.normal(jax.random.PRNGKey(n), (n, 2))
        out = net(inputs)
        assert out.shape == (n,)


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_forward_vector_theta():
    key = jax.random.PRNGKey(0)
    net = ValidityNet(theta_dim=4, key=key)
    inputs = jax.random.normal(jax.random.PRNGKey(1), (8, 5))  # theta_dim + 1
    out = net(inputs)
    assert out.shape == (8,)


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_grad_through_inputs_is_finite():
    """∂(sum logits)/∂inputs is finite for a smooth-MLP forward."""
    key = jax.random.PRNGKey(0)
    net = ValidityNet(theta_dim=1, key=key)
    inputs = jax.random.normal(jax.random.PRNGKey(1), (7, 2))

    def scalar(x):
        return net(x).sum()

    g = jax.grad(scalar)(inputs)
    assert g.shape == inputs.shape
    assert bool(jnp.isfinite(g).all())


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_grad_through_params_is_finite():
    """∂(sum logits)/∂params yields finite values."""
    key = jax.random.PRNGKey(0)
    net = ValidityNet(theta_dim=1, key=key)
    inputs = jax.random.normal(jax.random.PRNGKey(1), (7, 2))

    params, static = eqx.partition(net, eqx.is_array)

    def loss(p):
        return eqx.combine(p, static)(inputs).sum()

    g = jax.grad(loss)(params)
    leaves = jtu.tree_leaves(g)
    assert leaves, "expected at least one parameter leaf"
    for leaf in leaves:
        assert bool(jnp.isfinite(leaf).all())


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_serialise_round_trip_byte_equal():
    """``eqx.tree_serialise_leaves`` round-trips weights byte-equal."""
    key = jax.random.PRNGKey(7)
    net = ValidityNet(theta_dim=1, key=key)
    inputs = jax.random.normal(jax.random.PRNGKey(11), (5, 2))
    out_before = net(inputs)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "valnet.eqx"
        eqx.tree_serialise_leaves(str(path), net)
        # Reconstruct an equally-shaped skeleton and load.
        skeleton = ValidityNet(theta_dim=1, key=jax.random.PRNGKey(999))
        loaded = eqx.tree_deserialise_leaves(str(path), skeleton)

    out_after = loaded(inputs)
    np.testing.assert_array_equal(np.asarray(out_before), np.asarray(out_after))

    leaves_before = jtu.tree_leaves(eqx.partition(net, eqx.is_array)[0])
    leaves_after = jtu.tree_leaves(eqx.partition(loaded, eqx.is_array)[0])
    assert len(leaves_before) == len(leaves_after)
    for a, b in zip(leaves_before, leaves_after):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_deterministic_on_same_key():
    """Same PRNGKey → identical weights."""
    key = jax.random.PRNGKey(42)
    a = ValidityNet(theta_dim=1, key=key)
    b = ValidityNet(theta_dim=1, key=key)
    leaves_a = jtu.tree_leaves(eqx.partition(a, eqx.is_array)[0])
    leaves_b = jtu.tree_leaves(eqx.partition(b, eqx.is_array)[0])
    for la, lb in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))
