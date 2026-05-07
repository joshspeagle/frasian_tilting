"""Audit P0-16: `_check_alpha` gates on `alpha_mode`, not on `alpha is None`.

Pre-fix: a checkpoint trained with loss_kind=static_width but with the
alpha field accidentally stripped to None (e.g. via metadata sanitiser)
would pass `_check_alpha` silently — the gate was `if stored is None:
return`. The new explicit `alpha_mode` field carries the contract.

These tests construct fake EtaArtifacts with controlled metadata and
verify `_check_alpha` accepts/rejects the right combinations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pytest

from frasian._errors import MissingArtifactError
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector


@dataclass
class _FakeArtifact:
    """Minimal LearnedArtifact-like fixture for `_check_alpha` testing."""

    metadata: dict[str, Any]
    name: str = "_FakeArtifact"

    def load(self) -> None:
        pass


def _selector_with_meta(meta: dict[str, Any]) -> LearnedDynamicEtaSelector:
    sel = LearnedDynamicEtaSelector(artifact=_FakeArtifact(metadata=meta))
    sel._loaded = True
    return sel


@pytest.mark.L1
class TestAlphaModeMarginalised:
    """alpha_mode='marginalised' → any inference alpha is accepted."""

    def test_any_alpha_accepted(self):
        sel = _selector_with_meta(
            {"alpha_mode": "marginalised", "alpha": None, "experiment_config": {}}
        )
        # All of these must pass without raising.
        sel._check_alpha(0.05)
        sel._check_alpha(0.10)
        sel._check_alpha(0.20)
        sel._check_alpha(0.95)


@pytest.mark.L1
class TestAlphaModeFixed:
    """alpha_mode='fixed' → only the trained alpha is accepted."""

    def test_matching_alpha_passes(self):
        sel = _selector_with_meta(
            {"alpha_mode": "fixed", "alpha": 0.05, "experiment_config": {}}
        )
        sel._check_alpha(0.05)

    def test_mismatched_alpha_raises(self):
        sel = _selector_with_meta(
            {"alpha_mode": "fixed", "alpha": 0.05, "experiment_config": {}}
        )
        with pytest.raises(MissingArtifactError, match=r"alpha=0.05"):
            sel._check_alpha(0.10)

    def test_internally_inconsistent_metadata_raises(self):
        """alpha_mode='fixed' + alpha=None ⇒ metadata is broken; refuse."""
        sel = _selector_with_meta(
            {"alpha_mode": "fixed", "alpha": None, "experiment_config": {}}
        )
        with pytest.raises(MissingArtifactError, match=r"internally inconsistent"):
            sel._check_alpha(0.05)


@pytest.mark.L1
class TestLegacyCheckpointFallback:
    """Pre-Cluster-E checkpoints lack alpha_mode; fall back to the
    `alpha is None → marginalised` heuristic with a UserWarning so the
    user knows to retrain.
    """

    def test_legacy_marginalised_passes_with_warning(self):
        sel = _selector_with_meta({"alpha": None, "experiment_config": {}})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sel._check_alpha(0.05)
        assert any(
            "legacy checkpoint" in str(w.message).lower() for w in caught
        ), [str(w.message) for w in caught]

    def test_legacy_fixed_alpha_matching_passes(self):
        sel = _selector_with_meta({"alpha": 0.05, "experiment_config": {}})
        sel._check_alpha(0.05)

    def test_legacy_fixed_alpha_mismatch_raises(self):
        sel = _selector_with_meta({"alpha": 0.05, "experiment_config": {}})
        with pytest.raises(MissingArtifactError, match=r"alpha=0.05"):
            sel._check_alpha(0.10)


@pytest.mark.L1
class TestUnknownAlphaModeRaises:
    """Future-proofing: an unknown alpha_mode value is rejected."""

    def test_unknown_alpha_mode_raises(self):
        sel = _selector_with_meta(
            {"alpha_mode": "interpolated", "alpha": 0.05, "experiment_config": {}}
        )
        with pytest.raises(MissingArtifactError, match=r"unknown alpha_mode"):
            sel._check_alpha(0.05)
