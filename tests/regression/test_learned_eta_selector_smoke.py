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
            "checkpoint_format_version": 2,
            "experiment_config": {
                "scheme_name": self.scheme_name,
                "statistic_name": self.statistic_name,
                "model_fingerprint": ["normal_normal", self.sigma],
                "prior_fingerprint": ["normal", self.mu0, self.sigma0],
            },
            "alpha": self.alpha,
        }


@pytest.mark.L2
class TestLearnedDynamicEtaSelectorSmoke:
    def test_select_grid_returns_constant_eta(self):
        """Stub returning η=0 ⇒ select_grid returns 0 everywhere."""
        artifact = _StubEtaArtifact(eta_value=0.0)
        selector = LearnedDynamicEtaSelector(
            artifact=artifact,
            sigma=1.0,
            mu0=0.0,
        )
        scheme = PowerLawTilting()

        ad_grid = np.linspace(0.0, 5.0, 11)
        eta = selector.select_grid(
            ad_grid,
            scheme,
            statistic=WaldoStatistic(),
            w=0.5,
            alpha=0.05,
        )
        assert eta.shape == (11,)
        np.testing.assert_allclose(eta, 0.0, atol=1e-12)

    def test_end_to_end_confidence_interval(self):
        """At η=0 (constant), the dynamic CI matches bare WALDO."""
        artifact = _StubEtaArtifact(eta_value=0.0)
        selector = LearnedDynamicEtaSelector(
            artifact=artifact,
            sigma=1.0,
            mu0=0.0,
        )
        scheme = PowerLawTilting(selector=selector)

        D = 1.5
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)

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

    def test_scheme_mismatch_raises(self):
        """Artifact trained for scheme X must not be used with scheme Y."""
        from frasian.tilting.ot import OTTilting

        artifact = _StubEtaArtifact(scheme_name="ot")
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()

        with pytest.raises(MissingArtifactError, match="scheme"):
            selector.select_grid(
                np.asarray([0.5, 1.0]),
                scheme,
                statistic=WaldoStatistic(),
                w=0.5,
                alpha=0.05,
            )

    def test_cross_experiment_fingerprint_mismatch_raises(self):
        """Strict tuple-equal compare on prior + model fingerprints."""
        artifact = _StubEtaArtifact(sigma=1.0, sigma0=1.0, mu0=0.0)
        selector = LearnedDynamicEtaSelector(
            artifact=artifact,
            sigma=1.0,
            mu0=0.0,
        )
        scheme = PowerLawTilting()

        # Different prior loc — same w, but different fingerprint.
        with pytest.raises(MissingArtifactError, match="trained with prior"):
            selector.select_grid(
                np.asarray([0.5, 1.0]),
                scheme,
                statistic=WaldoStatistic(),
                w=0.5,
                alpha=0.05,
                model_fingerprint=("normal_normal", 1.0),
                prior_fingerprint=("normal", 1.0, 1.0),
            )

    def test_alpha_marginalised_works_at_any_alpha(self):
        """Marginalised checkpoint (alpha=None) accepts any α."""
        artifact = _StubEtaArtifact(alpha=None, eta_value=0.0)
        selector = LearnedDynamicEtaSelector(
            artifact=artifact,
            sigma=1.0,
            mu0=0.0,
        )
        scheme = PowerLawTilting()

        for alpha in (0.01, 0.05, 0.10, 0.20):
            eta = selector.select_grid(
                np.asarray([0.5, 1.0]),
                scheme,
                statistic=WaldoStatistic(),
                w=0.5,
                alpha=alpha,
            )
            assert eta.shape == (2,)

    def test_select_requires_fingerprints(self):
        """`select` convenience method must reject calls missing fingerprints.

        Closes the bypass where the convenience entry point falls back to
        the w-only derived check (which cannot distinguish two ``(σ, σ₀)``
        pairs giving the same ``w``).
        """
        from frasian.tilting.base import TiltingContext

        artifact = _StubEtaArtifact(sigma=1.0, sigma0=1.0, mu0=0.0)
        selector = LearnedDynamicEtaSelector(
            artifact=artifact,
            sigma=1.0,
            mu0=0.0,
        )
        scheme = PowerLawTilting()
        ctx = TiltingContext(w=0.5, abs_delta=1.0, alpha=0.05)

        # No fingerprints → raise.
        with pytest.raises(ValueError, match="fingerprint"):
            selector.select(ctx, scheme, statistic=WaldoStatistic())

        # Mismatched model fingerprint → raise (different sigma).
        with pytest.raises(MissingArtifactError):
            selector.select(
                ctx,
                scheme,
                statistic=WaldoStatistic(),
                model_fingerprint=("normal_normal", 2.0),
                prior_fingerprint=("normal", 0.0, 1.0),
            )

        # Matching fingerprints → succeed.
        eta = selector.select(
            ctx,
            scheme,
            statistic=WaldoStatistic(),
            model_fingerprint=("normal_normal", 1.0),
            prior_fingerprint=("normal", 0.0, 1.0),
        )
        assert isinstance(eta, float)
