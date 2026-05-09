"""L2 regression: v3 checkpoint refusal.

Phase G bumps CHECKPOINT_FORMAT_VERSION from 3 to 4. v3 checkpoints
must refuse to load with a clear migration message pointing at
scripts.train_learned_eta and the schema doc.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frasian._errors import MissingArtifactError
from frasian.learned.eta_artifact import EtaArtifact, CHECKPOINT_FORMAT_VERSION


def test_format_version_is_4():
    assert CHECKPOINT_FORMAT_VERSION == 4


@pytest.mark.L2
def test_v3_checkpoint_refused_with_migration_message(tmp_path: Path):
    """Build a fake v3 metadata header; loader refuses with a migration message."""
    import json
    import struct

    fake_v3_metadata = {
        "checkpoint_format_version": 3,
        "architecture": "EtaNet+ValidityNet",
        "eta_architecture_kwargs": {"theta_dim": 1, "hidden_sizes": [64, 64]},
        "validity_architecture_kwargs": {"theta_dim": 1, "hidden_sizes": [64, 64]},
        "experiment_config": {},
    }
    payload = json.dumps(fake_v3_metadata).encode("utf-8")
    fake_path = tmp_path / "fake_v3.eqx"
    with open(fake_path, "wb") as f:
        f.write(struct.pack(">I", len(payload)))
        f.write(payload)

    art = EtaArtifact(artifact_path=fake_path)
    with pytest.raises(MissingArtifactError, match=r"format version 3.*v4"):
        art.load()
