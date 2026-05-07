"""L1 invariant: trained Phase E EtaNet stays inside admissible η-range.

Phase E removed the architectural sigmoid clamp on η (the legacy
``MonotonicEtaNet`` had ``0.01 + 0.98·sigmoid(...)`` which forced η
into ``(0.01, 0.99)``); validity is now enforced via the boundary
penalty during training, audited by ``ValidityNet`` on (θ, η) pairs.
This test pins the empirical invariant on the shipped checkpoints:
on a dense θ-grid covering the trained ``theta_distribution``
support, the predicted η is inside the scheme's admissible range
(by ≥99%; the runtime selector clamps the residual <1% with a
warning).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frasian.learned.eta_artifact import EtaArtifact
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting

_CHECKPOINTS = [
    (
        "power_law",
        Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v0_smoke.eqx"),
        PowerLawTilting,
    ),
    (
        "ot",
        Path("artifacts/learned_eta_canonical_normal_normal_ot_v0_smoke.eqx"),
        OTTilting,
    ),
]


def _available_checkpoints():
    return [(name, p, cls) for (name, p, cls) in _CHECKPOINTS if p.exists()]


@pytest.mark.L1
@pytest.mark.properties
@pytest.mark.parametrize("scheme_name,path,scheme_cls", _available_checkpoints())
def test_eta_predominantly_in_admissible_range(scheme_name, path, scheme_cls):
    """≥99% of predicted η on the trained θ-range is admissible.

    Phase E doesn't clamp η architecturally; the boundary penalty
    pushes it into the valid region during training, but a smoke
    checkpoint can leave a small fraction past the boundary at
    extreme conflict (the runtime selector then clamps with a
    RuntimeWarning). This test pins the soft floor at 99%.
    """
    artifact = EtaArtifact(artifact_path=path)
    artifact.load()

    cfg = artifact.metadata["experiment_config"]
    theta_lo = float(cfg["theta_distribution_fingerprint"][1])
    theta_hi = float(cfg["theta_distribution_fingerprint"][2])
    sigma = float(cfg["model_fingerprint"][1])
    sigma0 = float(cfg["prior_fingerprint"][2])
    w = sigma0**2 / (sigma**2 + sigma0**2)

    theta_grid = np.linspace(theta_lo, theta_hi, 401)
    eta = artifact.predict_eta(theta_grid)

    if scheme_name == "power_law":
        eta_min = -w / (1.0 - w)
        eta_max = 1.0 / (1.0 - w)
    elif scheme_name == "ot":
        eta_min, eta_max = 0.0, 1.0
    else:
        pytest.skip(f"no admissible-range invariant for scheme {scheme_name}")

    inside = (eta > eta_min) & (eta < eta_max)
    fraction_inside = float(inside.mean())
    assert fraction_inside >= 0.99, (
        f"Only {100*fraction_inside:.2f}% of predicted η on the trained "
        f"θ-range fall in admissible ({eta_min:.3g}, {eta_max:.3g}); "
        f"<99% suggests the checkpoint is undertrained."
    )
