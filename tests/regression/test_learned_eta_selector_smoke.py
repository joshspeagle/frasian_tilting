"""Smoke test: Phase E LearnedDynamicEtaSelector wires through end-to-end.

Uses a stub artifact (no torch required) that mimics the Phase E
``EtaArtifact`` predict contract: takes a (N,) θ array and returns a
(N,) η array. The stub returns a constant η so the selector's η
lookup is deterministic.

Verifies:
  1. ``LearnedDynamicEtaSelector + StubEtaArtifact`` is constructible.
  2. ``select_grid`` returns a ``(N,)`` array of η values.
  3. ``(power_law[learned], waldo).confidence_regions`` produces a
     non-empty CI without raising.
  4. Scheme-mismatch raises ``MissingArtifactError``.
  5. Cross-experiment fingerprint mismatch raises.
  6. α-marginalised checkpoint accepts any α.

Phase 3a-1.5: tests use the θ-keyed selector signature throughout
(legacy ``|Δ|``-keyed signature with ``w=`` is gone).

Does NOT require ``torch`` — the stub stays in numpy-land.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from frasian._errors import MissingArtifactError
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.power_law import PowerLawTilting


@dataclass
class _StubEtaArtifact:
    """Minimal Phase E artifact: ``predict_eta(theta)`` returns a constant.

    Implements the same surface as ``EtaArtifact`` without importing
    torch. Used to test the selector's wiring + dispatch.
    """

    eta_value: float = 0.0
    scheme_name: str = "power_law"
    statistic_name: str = "waldo"
    alpha: float | None = None
    sigma: float = 1.0
    sigma0: float = 1.0
    mu0: float = 0.0
    name: str = "stub_phase_e"
    version: str = "v0"
    artifact_path: Path = Path("/dev/null")
    format_version: int = 3
    _loaded: bool = field(default=False, init=False, repr=False)

    def load(self) -> None:
        self._loaded = True

    def predict_eta(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded")
        theta_arr = np.asarray(theta, dtype=np.float64)
        return np.full(theta_arr.shape, self.eta_value)

    def fingerprint(self) -> str:
        h = hashlib.sha256(f"{self.name}:{self.version}:{self.eta_value}".encode())
        return h.hexdigest()[:16]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_format_version": self.format_version,
            "experiment_config": {
                "scheme_name": self.scheme_name,
                "statistic_name": self.statistic_name,
                "model_fingerprint": ["normal_normal", self.sigma],
                "prior_fingerprint": ["normal", self.mu0, self.sigma0],
            },
            "alpha": self.alpha,
        }


def _matched_pair(stub: _StubEtaArtifact) -> tuple[NormalNormalModel, NormalDistribution]:
    """Build (model, prior) whose fingerprints match the stub artifact."""
    return (
        NormalNormalModel(sigma=stub.sigma),
        NormalDistribution(loc=stub.mu0, scale=stub.sigma0),
    )


@pytest.mark.L2
class TestLearnedDynamicEtaSelectorSmoke:
    def test_select_grid_returns_constant_eta(self):
        """Stub returning η=0 ⇒ select_grid returns 0 everywhere."""
        artifact = _StubEtaArtifact(eta_value=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()
        model, prior = _matched_pair(artifact)

        theta_grid = np.linspace(-5.0, 5.0, 11)
        eta = selector.select_grid(
            theta_grid,
            scheme,
            statistic=WaldoStatistic(),
            model=model,
            prior=prior,
            alpha=0.05,
        )
        assert eta.shape == (11,)
        np.testing.assert_allclose(eta, 0.0, atol=1e-12)

    def test_end_to_end_confidence_interval(self):
        """At η=0 (constant), the dynamic CI matches bare WALDO."""
        artifact = _StubEtaArtifact(eta_value=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting(selector=selector)

        D = 1.5
        model, prior = _matched_pair(artifact)

        regions = scheme.confidence_regions(
            0.05,
            np.asarray([D]),
            model,
            prior,
            WaldoStatistic(),
        )
        assert len(regions) >= 1
        for lo, hi in regions:
            assert hi > lo, f"empty region ({lo}, {hi})"
        bare_lo, bare_hi = WaldoStatistic().confidence_interval(
            0.05,
            np.asarray([D]),
            model,
            prior,
        )
        union_lo = min(r[0] for r in regions)
        union_hi = max(r[1] for r in regions)
        np.testing.assert_allclose(union_lo, bare_lo, atol=1e-3)
        np.testing.assert_allclose(union_hi, bare_hi, atol=1e-3)

    def test_legacy_v2_checkpoint_rejected(self):
        """Audit P2 (Cluster A): selector must reject pre-Equinox v2
        checkpoints. Pre-fix the gate accepted ``v in (2, 3)`` while
        ``EtaArtifact.load`` only loads v3 — a v2 file would leak past
        ``_ensure_loaded`` and fail at the artifact level. Now both
        gates demand v3.
        """
        artifact = _StubEtaArtifact(format_version=2)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()
        model, prior = _matched_pair(artifact)

        with pytest.raises(MissingArtifactError, match="v3"):
            selector.select_grid(
                np.asarray([0.5, 1.0]),
                scheme,
                statistic=WaldoStatistic(),
                model=model,
                prior=prior,
                alpha=0.05,
            )

    def test_scheme_mismatch_raises(self):
        """Artifact trained for scheme X must not be used with scheme Y."""
        artifact = _StubEtaArtifact(scheme_name="ot")
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()
        model, prior = _matched_pair(artifact)

        with pytest.raises(MissingArtifactError, match="scheme"):
            selector.select_grid(
                np.asarray([0.5, 1.0]),
                scheme,
                statistic=WaldoStatistic(),
                model=model,
                prior=prior,
                alpha=0.05,
            )

    def test_cross_experiment_fingerprint_mismatch_raises(self):
        """Strict tuple-equal compare on prior + model fingerprints."""
        artifact = _StubEtaArtifact(sigma=1.0, sigma0=1.0, mu0=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()
        # Different prior loc — same w, but different fingerprint.
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=1.0, scale=1.0)

        with pytest.raises(MissingArtifactError, match="trained with prior"):
            selector.select_grid(
                np.asarray([0.5, 1.0]),
                scheme,
                statistic=WaldoStatistic(),
                model=model,
                prior=prior,
                alpha=0.05,
            )

    def test_alpha_marginalised_works_at_any_alpha(self):
        """Marginalised checkpoint (alpha=None) accepts any α."""
        artifact = _StubEtaArtifact(alpha=None, eta_value=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()
        model, prior = _matched_pair(artifact)

        for alpha in (0.01, 0.05, 0.10, 0.20):
            eta = selector.select_grid(
                np.asarray([0.5, 1.0]),
                scheme,
                statistic=WaldoStatistic(),
                model=model,
                prior=prior,
                alpha=alpha,
            )
            assert eta.shape == (2,)

    def test_select_uses_data_keyword(self):
        """The convenience `select` entry takes `data=`, `model=`, `prior=`,
        `alpha=`, `statistic=` kwargs (Phase 3a-1.5 — no `TiltingContext`).
        """
        artifact = _StubEtaArtifact(sigma=1.0, sigma0=1.0, mu0=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()
        model, prior = _matched_pair(artifact)

        eta = selector.select(
            scheme,
            data=np.asarray([1.0]),
            model=model,
            prior=prior,
            alpha=0.05,
            statistic=WaldoStatistic(),
        )
        assert isinstance(eta, float)

    def test_select_grid_accepts_bernoulli_checkpoint(self):
        """Phase 4c-3: a checkpoint trained on Bernoulli + Beta loads
        and serves inference without the legacy NN-only ``trained on
        model 'normal_normal'`` reject. Strict tuple-equal fingerprint
        compare still enforces cross-experiment safety, but model
        kind itself is no longer gated."""
        from frasian.models.bernoulli import BernoulliModel
        from frasian.models.distributions import BetaDistribution

        @dataclass
        class _BernoulliStubArtifact:
            eta_value: float = 0.0
            scheme_name: str = "power_law"
            statistic_name: str = "waldo"
            alpha: float | None = None
            alpha_pri: float = 2.0
            beta_pri: float = 2.0
            name: str = "stub_bernoulli"
            version: str = "v0"
            artifact_path: Path = Path("/dev/null")
            _loaded: bool = field(default=False, init=False, repr=False)

            def load(self) -> None:
                self._loaded = True

            def predict_eta(self, theta):
                if not self._loaded:
                    raise MissingArtifactError(f"{self.name} not loaded")
                arr = np.asarray(theta, dtype=np.float64)
                return np.full(arr.shape, self.eta_value)

            def fingerprint(self) -> str:
                return "bern_stub_v0"

            @property
            def metadata(self):
                return {
                    "checkpoint_format_version": 3,
                    "experiment_config": {
                        "scheme_name": self.scheme_name,
                        "statistic_name": self.statistic_name,
                        "model_fingerprint": ["bernoulli"],
                        "prior_fingerprint": ["beta", self.alpha_pri, self.beta_pri],
                        "eta_explore_box": [-2.0, 2.0],
                    },
                    "alpha": self.alpha,
                }

        artifact = _BernoulliStubArtifact(eta_value=0.3)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()
        model = BernoulliModel()
        prior = BetaDistribution(alpha=2.0, beta=2.0)

        eta = selector.select_grid(
            np.linspace(0.05, 0.95, 11),
            scheme,
            statistic=WaldoStatistic(),
            model=model,
            prior=prior,
            alpha=0.05,
        )
        assert eta.shape == (11,)
        np.testing.assert_allclose(eta, 0.3, atol=1e-12)

    def test_select_mismatched_model_fingerprint_raises(self):
        """Artifact trained on σ=1 cannot serve inference at σ=2."""
        artifact = _StubEtaArtifact(sigma=1.0, sigma0=1.0, mu0=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()
        model = NormalNormalModel(sigma=2.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        with pytest.raises(MissingArtifactError):
            selector.select(
                scheme,
                data=np.asarray([1.0]),
                model=model,
                prior=prior,
                alpha=0.05,
                statistic=WaldoStatistic(),
            )
