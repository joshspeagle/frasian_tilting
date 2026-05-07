"""Regression: ``fit_eta_artifact`` is bit-deterministic at fixed seed (JAX port).

Replaces the legacy ``test_torch_determinism.py``. JAX is bit-
deterministic on CPU at a fixed ``jax.random.PRNGKey`` — no
``torch.use_deterministic_algorithms`` ceremony required. The
orchestrator derives the root key from ``config.seed`` and seeds
numpy globally for any side-channel randomness.

We pin two properties:

1. ``_enable_determinism(seed)`` returns the same ``jax.Array`` PRNG
   key on repeated calls and reseeds numpy.
2. Two ``fit_eta_artifact`` runs at the same seed produce byte-equal
   ``equinox.tree_serialise_leaves`` output for both nets.
"""

from __future__ import annotations

import io
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from frasian.learned.training._setup import enable_determinism


def test_enable_determinism_returns_stable_key_and_seeds_numpy():
    """``enable_determinism(seed)`` returns a deterministic JAX key and
    reseeds numpy.

    JAX is bit-deterministic on CPU at a fixed ``PRNGKey``; the
    orchestrator routes all training-side randomness through that
    key (split per consumer). Numpy is seeded for any side-channel
    randomness that isn't routed through the per-consumer
    ``np.random.default_rng(seed)``.
    """
    k1 = enable_determinism(seed=42)
    a_n = np.random.randn(8)
    k2 = enable_determinism(seed=42)
    b_n = np.random.randn(8)
    np.testing.assert_array_equal(np.asarray(k1), np.asarray(k2))
    np.testing.assert_array_equal(a_n, b_n)
    # Sanity: drawing through the JAX key gives identical samples too.
    s1 = jax.random.normal(k1, (8,))
    s2 = jax.random.normal(k2, (8,))
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))


@pytest.mark.L4
@pytest.mark.slow
def test_fit_eta_artifact_byte_identical_serialised_leaves(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """Two runs of ``fit_eta_artifact`` at the same seed must produce
    byte-identical ``eqx.tree_serialise_leaves`` output for both nets.

    On a CPU build with a fixed ``jax.random.PRNGKey(seed)`` this is
    achievable — JAX's bit-deterministic kernels + the deterministic
    init in ``architecture._build_mlp`` yield identical weights. If a
    future JAX or Equinox version introduces a non-deterministic
    kernel that this test catches, retrain or pin.
    """
    from frasian.learned.training.sampling import (
        ExperimentConfig,
        UniformThetaDistribution,
    )
    from frasian.learned.training.train import fit_eta_artifact
    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel

    config = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(loc=0.0, scale=1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-3.0, high=3.0),
        n_grid=33,
        n_lhs=64,
        eta_explore_box=(-5.0, 5.0),
        seed=12345,
    )

    out_a = tmp_path / "a.eqx"
    out_b = tmp_path / "b.eqx"
    fit_eta_artifact(
        config=config,
        out_path=out_a,
        loss_kind="integrated_p",
        n_epochs=2,
        batch_size=16,
        n_aux=16,
        patience=2,
        antithetic=False,
        verbose=False,
    )
    fit_eta_artifact(
        config=config,
        out_path=out_b,
        loss_kind="integrated_p",
        n_epochs=2,
        batch_size=16,
        n_aux=16,
        patience=2,
        antithetic=False,
        verbose=False,
    )

    # Compare the on-disk bytes directly (header metadata includes a
    # timestamp so the full files differ; the leaves region must match).
    bytes_a = out_a.read_bytes()
    bytes_b = out_b.read_bytes()
    # Strip the 4-byte length prefix + the metadata blob; compare the
    # serialised-leaves region.
    import struct

    (meta_len_a,) = struct.unpack(">I", bytes_a[:4])
    (meta_len_b,) = struct.unpack(">I", bytes_b[:4])
    leaves_a = bytes_a[4 + meta_len_a:]
    leaves_b = bytes_b[4 + meta_len_b:]
    assert leaves_a == leaves_b, (
        "Serialised leaves must be byte-identical at fixed seed; "
        "JAX/Equinox bit-determinism regressed."
    )


@pytest.mark.L4
@pytest.mark.slow
def test_fit_eta_artifact_serialised_in_memory_matches(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """Round-trip: load both checkpoints, re-serialise the EtaNet, compare bytes.

    Belt-and-braces: even if the on-disk bytes drift due to a future
    serialiser change, the loaded PyTrees + a fresh in-memory
    ``eqx.tree_serialise_leaves`` should be byte-equal at fixed seed.
    """
    from frasian.learned.eta_artifact import EtaArtifact
    from frasian.learned.training.sampling import (
        ExperimentConfig,
        UniformThetaDistribution,
    )
    from frasian.learned.training.train import fit_eta_artifact
    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel

    config = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(loc=0.0, scale=1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-3.0, high=3.0),
        n_grid=33,
        n_lhs=64,
        eta_explore_box=(-5.0, 5.0),
        seed=12345,
    )

    out_a = tmp_path / "a.eqx"
    out_b = tmp_path / "b.eqx"
    fit_eta_artifact(
        config=config,
        out_path=out_a,
        loss_kind="integrated_p",
        n_epochs=2,
        batch_size=16,
        n_aux=16,
        patience=2,
        antithetic=False,
        verbose=False,
    )
    fit_eta_artifact(
        config=config,
        out_path=out_b,
        loss_kind="integrated_p",
        n_epochs=2,
        batch_size=16,
        n_aux=16,
        patience=2,
        antithetic=False,
        verbose=False,
    )

    art_a = EtaArtifact(artifact_path=out_a)
    art_a.load()
    art_b = EtaArtifact(artifact_path=out_b)
    art_b.load()

    buf_a, buf_b = io.BytesIO(), io.BytesIO()
    eqx.tree_serialise_leaves(buf_a, art_a._eta_net)
    eqx.tree_serialise_leaves(buf_b, art_b._eta_net)
    assert buf_a.getvalue() == buf_b.getvalue()

    buf_a, buf_b = io.BytesIO(), io.BytesIO()
    eqx.tree_serialise_leaves(buf_a, art_a._validity_net)
    eqx.tree_serialise_leaves(buf_b, art_b._validity_net)
    assert buf_a.getvalue() == buf_b.getvalue()
