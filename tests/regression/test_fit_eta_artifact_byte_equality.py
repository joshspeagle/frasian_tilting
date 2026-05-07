"""Regression: ``fit_eta_artifact`` produces byte-identical leaves at the same seed.

Closes audit finding 1.2-NN4 (post-port version). The orchestrator
is bit-deterministic on CPU at a fixed ``jax.random.PRNGKey(seed)``
since JAX has no nondeterministic kernels there. We pin the property
by running ``fit_eta_artifact`` twice at the same seed and asserting
byte-equality of the serialised leaves region of the saved
checkpoints.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import equinox as eqx
import pytest

from frasian.learned.eta_artifact import EtaArtifact
from frasian.learned.training.sampling import (
    ExperimentConfig,
    UniformThetaDistribution,
)
from frasian.learned.training.train import fit_eta_artifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel


def _read_leaves_region(path: Path) -> bytes:
    """Strip the (length-prefixed) JSON metadata header; return the
    serialised-leaves region of the checkpoint."""
    bytes_ = path.read_bytes()
    (meta_len,) = struct.unpack(">I", bytes_[:4])
    return bytes_[4 + meta_len:]


@pytest.mark.L4
@pytest.mark.slow
def test_fit_eta_artifact_byte_identical_on_same_seed(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """Two runs of ``fit_eta_artifact`` at the same seed must produce
    byte-identical serialised leaves for both EtaNet + ValidityNet.

    On a CPU build with a fixed ``jax.random.PRNGKey(seed)`` this is
    achievable; if a future JAX/Equinox version introduces a non-
    deterministic kernel that this test catches, retrain or pin.
    """
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

    # Direct on-disk leaves region must match.
    leaves_a = _read_leaves_region(out_a)
    leaves_b = _read_leaves_region(out_b)
    assert leaves_a == leaves_b, (
        "Serialised leaves must be byte-identical at fixed seed."
    )

    # Belt-and-braces: load + re-serialise in memory, compare.
    art_a = EtaArtifact(artifact_path=out_a)
    art_a.load()
    art_b = EtaArtifact(artifact_path=out_b)
    art_b.load()
    for label, mod_a, mod_b in (
        ("EtaNet", art_a._eta_net, art_b._eta_net),
        ("ValidityNet", art_a._validity_net, art_b._validity_net),
    ):
        buf_a, buf_b = io.BytesIO(), io.BytesIO()
        eqx.tree_serialise_leaves(buf_a, mod_a)
        eqx.tree_serialise_leaves(buf_b, mod_b)
        assert buf_a.getvalue() == buf_b.getvalue(), (
            f"{label} re-serialised bytes must be byte-identical at fixed seed."
        )
