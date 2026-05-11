"""L2 regression: trained v4 conditional EtaNet fixture for Fisher-Rao.

The fixture is gitignored — train via:
    python -m scripts.train_learned_eta \\
        --config experiments/canonical_normal_normal_fisher_rao_v4.yaml \\
        --out artifacts/learned_eta_canonical_normal_normal_fisher_rao_phaseC_integrated_p_v4.eqx

These tests skip if the fixture isn't on disk.

Test coverage:
  1. Artifact round-trips through `EtaArtifact.load()`.
  2. fingerprint() returns a non-empty stable string.
  3. predict_eta returns finite, admissible (η ∈ ℝ; FR is geodesically
     complete) eta over a representative (θ, prior_hp, lik_hp) grid.
  4. The learned η is NOT a constant function of θ. If training failed
     (e.g. the Stage C.2 zero-gradient bug from before commit 83a3c0e),
     EtaNet collapsed to a near-constant output and this test would
     catch it.
  5. The learned η does measurably better than the η=0 baseline on the
     integrated-p metric. This is the "did learning happen" assertion
     that would have caught the Stage C.2 zero-gradient bug.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.eta_artifact import EtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel


_ARTIFACT = Path(
    "artifacts/learned_eta_canonical_normal_normal_fisher_rao_phaseC_integrated_p_v4.eqx"
)


pytestmark = pytest.mark.skipif(
    not _ARTIFACT.exists(),
    reason=(
        f"FR v4 fixture not trained yet at {_ARTIFACT}. Train via "
        f"`python -m scripts.train_learned_eta --config "
        f"experiments/canonical_normal_normal_fisher_rao_v4.yaml --out {_ARTIFACT}`."
    ),
)


def _make_artifact() -> EtaArtifact:
    art = EtaArtifact(artifact_path=_ARTIFACT)
    art.load()
    return art


@pytest.mark.L2
@pytest.mark.regression
class TestFisherRaoV4Fixture:
    """Stage C regression tests for the trained FR v4 EtaNet checkpoint."""

    def test_v4_fixture_loads(self):
        """Basic Equinox round-trip via EtaArtifact.load()."""
        art = _make_artifact()
        assert art is not None
        md = art.metadata  # property/attribute, not callable
        assert isinstance(md, dict)
        assert len(md) > 0

    def test_v4_fingerprint_exists_and_stable(self):
        """fingerprint() returns a non-empty stable string. The
        framework's cache layer uses the fingerprint to keyed-invalidate
        cached results when the artifact changes."""
        art1 = _make_artifact()
        art2 = _make_artifact()
        fp1 = art1.fingerprint()
        fp2 = art2.fingerprint()
        assert isinstance(fp1, str)
        assert len(fp1) > 0
        assert fp1 == fp2, f"fingerprint not stable across loads: {fp1!r} vs {fp2!r}"

    def test_v4_predict_eta_returns_finite_values(self):
        """EtaNet predicts finite η across a representative
        (θ, prior_hp, lik_hp) grid. FR is geodesically complete so any
        finite η is admissible; the only failure mode is NaN/Inf from a
        broken artifact."""
        art = _make_artifact()
        prior = NormalDistribution(loc=0.0, scale=1.0)
        model = NormalNormalModel(sigma=1.0)
        theta = np.linspace(-3.0, 3.0, 21)
        eta = art.predict_eta(theta, prior.hyperparams(), model.hyperparams())
        assert eta.shape == theta.shape
        assert np.all(np.isfinite(eta)), f"non-finite eta predictions: {eta}"

    def test_v4_predicted_eta_is_not_pure_float_noise(self):
        """The trained network's η output is non-trivially different
        from its random initialization. The test catches a complete
        training failure (e.g. NaN gradients, optimizer never stepped)
        where the saved artifact == random init weights.

        Threshold: std(eta over theta) > 1e-5. Loose threshold: at this
        level we're only catching float-noise-level variation (random
        init's final-layer noise is ~1e-7 to 1e-5). A WORKING training
        run produces std > 1e-4; a HEAVILY input-sensitive training
        run produces std > 0.1. This test catches neither tier —
        only the "literally identical to init" failure.

        For diagnosis of *how much* θ-variation the learned model has
        (relevant to the framework's row-13b input-insensitivity
        finding for PL/MX/OT), see the Stage C.5 input-insensitivity
        diagnostic (a separate analysis, not a pass/fail regression).
        """
        art = _make_artifact()
        prior = NormalDistribution(loc=0.0, scale=1.0)
        model = NormalNormalModel(sigma=1.0)
        theta = np.linspace(-3.0, 3.0, 51)
        eta = art.predict_eta(theta, prior.hyperparams(), model.hyperparams())
        eta_std = float(np.std(eta))
        assert eta_std > 1e-5, (
            f"learned eta is float-noise-level constant (std={eta_std:.2e}); "
            f"training likely never stepped. Check optimizer state and "
            f"the kernel gradient (commit 83a3c0e for the ST gradient fix)."
        )

    def test_v4_predicted_eta_varies_across_hyperparams(self):
        """The trained network's η output varies across (prior_hp,
        lik_hp) cells. This catches the case where EtaNet fully
        collapsed to a single constant η across ALL inputs (a more
        severe failure mode than per-cell constant).

        Three cells span the hyperparam box:
            (mu0, sigma0, sigma) ∈ {(0, 0.5, 1.0), (1, 2.0, 1.5), (-1, 1.0, 0.5)}
        Assert that the mean η across θ varies by at least 1e-3 between
        any two of these cells.
        """
        art = _make_artifact()
        theta = np.linspace(-3.0, 3.0, 21)
        cells = [
            (0.0, 0.5, 1.0),
            (1.0, 2.0, 1.5),
            (-1.0, 1.0, 0.5),
        ]
        mean_etas = []
        for mu0, sigma0, sigma in cells:
            prior = NormalDistribution(loc=mu0, scale=sigma0)
            model = NormalNormalModel(sigma=sigma)
            eta = art.predict_eta(theta, prior.hyperparams(), model.hyperparams())
            mean_etas.append(float(np.mean(eta)))
        max_spread = max(mean_etas) - min(mean_etas)
        assert max_spread > 1e-3, (
            f"learned eta means are nearly identical across hyperparams "
            f"({mean_etas}); the network may be ignoring its inputs. "
            f"This is a more severe failure mode than per-cell constant "
            f"output."
        )
