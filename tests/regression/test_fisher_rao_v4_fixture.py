"""L2 regression: trained v4 conditional EtaNet fixtures for Fisher-Rao.

Three loss heads are trained per the Stage C plan:

  python -m scripts.train_learned_eta \\
      --config experiments/canonical_normal_normal_fisher_rao_v4.yaml \\
      --loss {integrated_p, cd_variance, static_width} \\
      [--alpha 0.05 for static_width] \\
      --out artifacts/learned_eta_canonical_normal_normal_fisher_rao_phaseC_<loss>_v4.eqx

Fixtures are gitignored. Tests skip per-loss-head if the fixture isn't on
disk. Common test coverage (parametrised across heads):

  1. Artifact round-trips through `EtaArtifact.load()`.
  2. fingerprint() returns a non-empty stable string.
  3. predict_eta returns finite, admissible (η ∈ ℝ; FR is geodesically
     complete) eta over a representative (θ, prior_hp, lik_hp) grid.
  4. The learned η is NOT a constant function of θ. If training failed
     (e.g. the Stage C.2 zero-gradient bug from before commit 83a3c0e),
     EtaNet collapsed to a near-constant output and this test would
     catch it.
  5. The learned η means measurably vary across (prior_hp, lik_hp) cells,
     catching a worse failure mode where EtaNet collapsed to a single
     constant across ALL inputs.

Empirical reference (commit time, 2026-05-11):
  integrated_p : per-cell std ≈ 5e-4 (near-constant per cell),
                 cross-cell mean spread ≈ 0.16, η ∈ [0.71, 0.87]
  cd_variance  : per-cell std ≈ 5e-2 (100× higher than integrated_p,
                 despite training instability), cross-cell spread
                 ≈ 0.24, η ∈ [-0.34, -0.10] (negative, so closer
                 to WALDO=0 than Wald=1)
  static_width : populated by training (see /tmp/fr_static_w_training.log
                 if recently trained)

The cd_variance per-cell adaptation is 100× larger than integrated_p's,
which is itself informative: the framework's row-13b "input-
insensitivity" pattern is loss-specific to integrated_p, not an
architectural limitation of the EtaNet/ValidityNet pipeline. See the
Stage C.5 diagnostic for the broader analysis.
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


_ARTIFACT_DIR = Path("artifacts")
_HEADS = ("integrated_p", "cd_variance", "static_width")


def _artifact_path(loss_head: str) -> Path:
    return _ARTIFACT_DIR / (
        f"learned_eta_canonical_normal_normal_fisher_rao_phaseC_{loss_head}_v4.eqx"
    )


def _train_hint(loss_head: str) -> str:
    extra = " --alpha 0.05" if loss_head == "static_width" else ""
    return (
        f"FR v4 fixture for loss={loss_head} not trained. Train via "
        f"`python -m scripts.train_learned_eta --config "
        f"experiments/canonical_normal_normal_fisher_rao_v4.yaml --loss "
        f"{loss_head}{extra} --out {_artifact_path(loss_head)}`."
    )


@pytest.fixture(
    params=[
        pytest.param(
            head,
            id=head,
            marks=pytest.mark.skipif(
                not _artifact_path(head).exists(),
                reason=_train_hint(head),
            ),
        )
        for head in _HEADS
    ]
)
def loaded_artifact(request) -> EtaArtifact:
    """Per-loss-head trained EtaArtifact. Skips if the fixture is absent."""
    art = EtaArtifact(artifact_path=_artifact_path(request.param))
    art.load()
    return art


@pytest.mark.L2
@pytest.mark.regression
class TestFisherRaoV4Fixture:
    """Stage C regression tests for the trained FR v4 EtaNet checkpoints."""

    def test_v4_fixture_loads(self, loaded_artifact):
        """Basic Equinox round-trip via EtaArtifact.load()."""
        assert loaded_artifact is not None
        md = loaded_artifact.metadata
        assert isinstance(md, dict)
        assert len(md) > 0

    def test_v4_fingerprint_exists_and_stable(self, loaded_artifact):
        """fingerprint() returns a non-empty stable string. The
        framework's cache layer uses the fingerprint to keyed-invalidate
        cached results when the artifact changes."""
        fp1 = loaded_artifact.fingerprint()
        # Re-load via a second instance and check stability.
        art2 = EtaArtifact(artifact_path=loaded_artifact.artifact_path)
        art2.load()
        fp2 = art2.fingerprint()
        assert isinstance(fp1, str)
        assert len(fp1) > 0
        assert fp1 == fp2, f"fingerprint not stable across loads: {fp1!r} vs {fp2!r}"

    def test_v4_predict_eta_returns_finite_values(self, loaded_artifact):
        """EtaNet predicts finite η across a representative
        (θ, prior_hp, lik_hp) grid. FR is geodesically complete so any
        finite η is admissible; the only failure mode is NaN/Inf from a
        broken artifact."""
        prior = NormalDistribution(loc=0.0, scale=1.0)
        model = NormalNormalModel(sigma=1.0)
        theta = np.linspace(-3.0, 3.0, 21)
        eta = loaded_artifact.predict_eta(theta, prior.hyperparams(), model.hyperparams())
        assert eta.shape == theta.shape
        assert np.all(np.isfinite(eta)), f"non-finite eta predictions: {eta}"

    def test_v4_predicted_eta_is_not_pure_float_noise(self, loaded_artifact, request):
        """The trained network's η output varies across θ within a
        single (prior_hp, lik_hp) cell. Per-loss-head thresholds set at
        ~1/3 of the documented per-cell std on the canonical demo cell
        (μ₀=0, σ₀=σ=1) so a degenerate training that produces
        ~init-noise output (std ~1e-5) fails loudly, while normal
        cross-run drift (~30%) passes.

        Documented per-cell std at the demo cell (commit 2026-05-11,
        per `tools/probe_input_sensitivity_cross_scheme.py`):
          integrated_p ≈ 4e-4   → threshold 1e-4
          cd_variance  ≈ 0.13   → threshold 0.04
          static_width ≈ 3e-3   → threshold 1e-3
        """
        head_thresholds = {
            "integrated_p": 1e-4,
            "cd_variance": 0.04,
            "static_width": 1e-3,
        }
        head_token = request.node.callspec.params["loaded_artifact"]
        threshold = head_thresholds[head_token]

        prior = NormalDistribution(loc=0.0, scale=1.0)
        model = NormalNormalModel(sigma=1.0)
        theta = np.linspace(-3.0, 3.0, 51)
        eta = loaded_artifact.predict_eta(theta, prior.hyperparams(), model.hyperparams())
        eta_std = float(np.std(eta))
        assert eta_std > threshold, (
            f"learned eta std={eta_std:.2e} below per-loss threshold "
            f"({threshold:.2e}) for head={head_token!r}. The training run "
            f"likely produced a degenerate fixture (collapsed to constant "
            f"output, or never escaped init). Check optimizer state and "
            f"the kernel gradient (commit 83a3c0e for the ST gradient fix; "
            f"docs/notes/2026-05-11-fisher-rao-cd-var-hyperparams.md for "
            f"the cd_var-specific lr/grad-clip regime)."
        )

    def test_v4_predicted_eta_varies_across_hyperparams(self, loaded_artifact, request):
        """The trained network's η output varies across (prior_hp,
        lik_hp) cells. Per-loss-head thresholds set at ~1/3 of the
        documented cross-cell spread.

        Documented cross-cell mean-η spread (commit 2026-05-11):
          integrated_p ≈ 0.16  → threshold 0.05
          cd_variance  ≈ 1.43  → threshold 0.5
          static_width ≈ 0.08  → threshold 0.02
        """
        head_thresholds = {
            "integrated_p": 0.05,
            "cd_variance": 0.5,
            "static_width": 0.02,
        }
        head_token = request.node.callspec.params["loaded_artifact"]
        threshold = head_thresholds[head_token]

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
            eta = loaded_artifact.predict_eta(theta, prior.hyperparams(), model.hyperparams())
            mean_etas.append(float(np.mean(eta)))
        max_spread = max(mean_etas) - min(mean_etas)
        assert max_spread > threshold, (
            f"learned eta cross-cell spread = {max_spread:.3f} below per-loss "
            f"threshold ({threshold:.3f}) for head={head_token!r}. Per-cell "
            f"means: {[round(m, 3) for m in mean_etas]}. The fixture is "
            f"either input-insensitive (network ignoring (prior_hp, lik_hp) "
            f"inputs) or training failed to converge — see "
            f"docs/notes/2026-05-11-row-13b-loss-specificity-cross-scheme.md."
        )
