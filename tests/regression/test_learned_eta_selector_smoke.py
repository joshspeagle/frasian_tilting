"""Smoke test: LearnedDynamicEtaSelector wires through end-to-end.

Uses a stub artifact (no torch required) that mimics the
`MonotonicEtaArtifact` predict contract: takes a `(N, 2)` `[w, |Δ'|]`
matrix and returns `(N,)` `η'` values. The stub returns a constant
η' so the selector's η lookup is deterministic.

The test verifies:
  1. `LearnedDynamicEtaSelector + StubArtifact` is constructible.
  2. `select_grid` returns a `(N,)` array of η values in the
     scheme's admissible range.
  3. Wiring through `(power_law[learned_dynamic], waldo).confidence_interval`
     produces a non-empty CI without raising.
  4. Scheme-mismatch raises `MissingArtifactError`.
  5. Alpha-mode-mismatch raises `MissingArtifactError`.

Does NOT require `torch` — the stub artifact stays in numpy-land.
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
from frasian.learned.transforms import eta_transform_powerlaw
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.power_law import PowerLawTilting


@dataclass
class _StubArtifact:
    """Minimal artifact returning constant η' for testing.

    Implements the same predict contract as `MonotonicEtaArtifact`
    without importing torch.
    """

    eta_value: float = 0.0  # in original η space
    scheme: str = "power_law"
    alpha_mode: str = "marginalised"
    name: str = "stub"
    version: str = "v0"
    artifact_path: Path = Path("/dev/null")
    _loaded: bool = field(default=False, init=False, repr=False)

    def load(self) -> None:
        self._loaded = True

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._loaded:
            raise MissingArtifactError(f"{self.name} not loaded")
        x_arr = np.asarray(x, dtype=np.float64)
        assert x_arr.ndim == 2 and x_arr.shape[1] == 2, x_arr.shape
        # Return η' corresponding to the requested η in original space.
        # For power_law: η' = η(1-w) + w; w is the first column of x.
        w = x_arr[:, 0]
        return eta_transform_powerlaw(np.full(len(x_arr), self.eta_value), w)

    def fingerprint(self) -> str:
        h = hashlib.sha256(
            f"{self.name}:{self.version}:{self.eta_value}".encode()
        )
        return h.hexdigest()[:16]

    @property
    def metadata(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "alpha_mode": self.alpha_mode}


@pytest.mark.L2
class TestLearnedDynamicEtaSelectorSmoke:
    def test_select_grid_returns_constant_eta(self):
        """Stub artifact returning η=0 ⇒ select_grid returns 0 everywhere."""
        artifact = _StubArtifact(eta_value=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()

        ad_grid = np.linspace(0.0, 5.0, 11)
        eta = selector.select_grid(
            ad_grid, scheme, statistic=WaldoStatistic(), w=0.5, alpha=0.05,
        )
        assert eta.shape == (11,)
        np.testing.assert_allclose(eta, 0.0, atol=1e-12)

    def test_end_to_end_confidence_interval(self):
        """Wire through (power_law[learned], waldo).confidence_interval."""
        artifact = _StubArtifact(eta_value=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting(selector=selector)

        sigma, mu0, sigma0 = 1.0, 0.0, 1.0
        D = 1.5
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)

        regions = scheme.confidence_regions(
            0.05, np.asarray([D]), model, prior, WaldoStatistic(),
        )
        assert len(regions) >= 1
        for lo, hi in regions:
            assert hi > lo, f"empty region ({lo}, {hi})"
        # At eta=0 (constant), the dynamic CI should match bare WALDO.
        bare_lo, bare_hi = WaldoStatistic().confidence_interval(
            0.05, np.asarray([D]), model, prior,
        )
        union_lo = min(r[0] for r in regions)
        union_hi = max(r[1] for r in regions)
        np.testing.assert_allclose(union_lo, bare_lo, atol=1e-3)
        np.testing.assert_allclose(union_hi, bare_hi, atol=1e-3)

    def test_scheme_mismatch_raises(self):
        """Artifact trained for scheme X must not be used with scheme Y."""
        from frasian.tilting.ot import OTTilting
        artifact = _StubArtifact(scheme="ot")
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()

        with pytest.raises(MissingArtifactError, match="scheme"):
            selector.select_grid(
                np.asarray([0.5, 1.0]), scheme,
                statistic=WaldoStatistic(), w=0.5, alpha=0.05,
            )

    def test_alpha_fixed_mismatch_raises(self):
        """fixed_α artifact rejects a different inference α."""
        artifact = _StubArtifact(alpha_mode="fixed_0.05")
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()

        with pytest.raises(MissingArtifactError, match="alpha"):
            selector.select_grid(
                np.asarray([0.5, 1.0]), scheme,
                statistic=WaldoStatistic(), w=0.5, alpha=0.10,
            )

    def test_alpha_marginalised_works_at_any_alpha(self):
        """Marginalised artifact accepts any α at inference."""
        artifact = _StubArtifact(alpha_mode="marginalised", eta_value=0.0)
        selector = LearnedDynamicEtaSelector(artifact=artifact)
        scheme = PowerLawTilting()

        for alpha in (0.01, 0.05, 0.10, 0.20):
            eta = selector.select_grid(
                np.asarray([0.5, 1.0]), scheme,
                statistic=WaldoStatistic(), w=0.5, alpha=alpha,
            )
            assert eta.shape == (2,)
