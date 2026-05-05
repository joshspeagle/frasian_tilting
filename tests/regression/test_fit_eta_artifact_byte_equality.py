"""Regression: ``fit_eta_artifact`` produces byte-identical state dicts at the same seed.

Phase 4 skeptic §3: ``_enable_determinism`` uses
``torch.use_deterministic_algorithms(True, warn_only=True)`` (the safer
default — non-deterministic ops warn rather than raise). The audit's
1.2-NN4 acceptance criterion is "byte-identical checkpoints across two
GPU runs"; ``warn_only=True`` does not enforce that on its own. We pin
the property by running ``fit_eta_artifact`` twice at the same seed
and asserting byte-equality of the saved state dicts.

Torch-gated; in audit envs without torch the test is skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from frasian.learned.training.sampling import (  # noqa: E402
    ExperimentConfig,
    UniformThetaDistribution,
)
from frasian.learned.training.train import fit_eta_artifact  # noqa: E402
from frasian.models.distributions import NormalDistribution  # noqa: E402
from frasian.models.normal_normal import NormalNormalModel  # noqa: E402


def _state_dict_bytes(state_dict: dict) -> bytes:
    """Concatenate all parameter tensor bytes in sorted-key order."""
    parts: list[bytes] = []
    for k in sorted(state_dict):
        v = state_dict[k]
        parts.append(k.encode("utf-8"))
        parts.append(v.detach().cpu().contiguous().numpy().tobytes())
    return b"".join(parts)


@pytest.mark.L4
@pytest.mark.slow
def test_fit_eta_artifact_byte_identical_on_same_seed(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """Two runs of ``fit_eta_artifact`` at the same seed must produce
    byte-identical EtaNet + ValidityNet state dicts.

    On a CPU build with deterministic ops + ``torch.manual_seed(seed)``
    + ``np.random.seed(seed)`` this is achievable; if a future torch
    version introduces a non-deterministic kernel that
    ``warn_only=True`` would silently allow, this test fails and we
    promote ``warn_only=False``.
    """
    config = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(loc=0.0, scale=1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-3.0, high=3.0),
        n_grid=33,
        n_lhs=64,
        eta_explore_box=(-2.0, 2.0),
        seed=12345,
    )

    out_a = tmp_path / "a.pt"
    out_b = tmp_path / "b.pt"
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

    state_a = torch.load(str(out_a), map_location="cpu", weights_only=False)
    state_b = torch.load(str(out_b), map_location="cpu", weights_only=False)
    eta_a = _state_dict_bytes(state_a["eta_state_dict"])
    eta_b = _state_dict_bytes(state_b["eta_state_dict"])
    assert eta_a == eta_b, "EtaNet state dicts must be byte-identical at fixed seed."
    val_a = _state_dict_bytes(state_a["validity_state_dict"])
    val_b = _state_dict_bytes(state_b["validity_state_dict"])
    assert val_a == val_b, "ValidityNet state dicts must be byte-identical at fixed seed."
