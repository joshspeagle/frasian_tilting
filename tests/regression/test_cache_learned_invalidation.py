"""Regression: learned-η checkpoint swap invalidates the cache.

Pins the contract that ``CacheKey.extra["selector_artifact_fingerprint"]``
participates in the digest. Two cache keys identical in every documented
slot except the artifact fingerprint must produce different digests —
otherwise a v1 checkpoint replacing a v0 checkpoint at the same git sha
would silently hit the stale v0 result.

Does NOT require torch — we use plain string fingerprints (the same
shape ``EtaArtifact.fingerprint()`` returns: a 16-hex-char digest).
"""

from __future__ import annotations

import pytest

from frasian.simulation.cache import CacheKey


@pytest.mark.L2
class TestLearnedCheckpointInvalidatesCache:
    def test_different_artifact_fingerprint_changes_digest(self):
        """Same everything else; only the artifact fingerprint changes."""
        key_a = CacheKey(
            experiment="coverage",
            tilting="power_law[learned_dynamic]",
            statistic="waldo",
            config_fingerprint="cfg",
            git_sha="abc123",
            raw_fingerprint="raw",
            extra={"selector_artifact_fingerprint": "abc"},
        )
        key_b = CacheKey(
            experiment="coverage",
            tilting="power_law[learned_dynamic]",
            statistic="waldo",
            config_fingerprint="cfg",
            git_sha="abc123",
            raw_fingerprint="raw",
            extra={"selector_artifact_fingerprint": "xyz"},
        )
        assert key_a.digest() != key_b.digest()

    def test_same_fingerprint_yields_same_digest(self):
        """Belt-and-braces: identical keys digest identically."""
        key_a = CacheKey(
            experiment="coverage",
            tilting="power_law[learned_dynamic]",
            statistic="waldo",
            config_fingerprint="cfg",
            git_sha="abc123",
            extra={"selector_artifact_fingerprint": "abc"},
        )
        key_b = CacheKey(
            experiment="coverage",
            tilting="power_law[learned_dynamic]",
            statistic="waldo",
            config_fingerprint="cfg",
            git_sha="abc123",
            extra={"selector_artifact_fingerprint": "abc"},
        )
        assert key_a.digest() == key_b.digest()

    def test_persist_cell_threads_artifact_fingerprint(self, tmp_path):
        """End-to-end: persist_cell with a tilting carrying a fingerprintable
        artifact records the fingerprint in the cache key.
        """
        from dataclasses import dataclass, field
        from typing import Any

        import numpy as np

        from frasian.config import Config
        from frasian.experiments.base import RawResult
        from frasian.simulation.cache import CacheKey, git_sha
        from frasian.simulation.runner import persist_cell

        @dataclass
        class _StubArtifact:
            fp: str = "stub_fp_aaa"

            def fingerprint(self) -> str:
                return self.fp

        @dataclass
        class _StubSelector:
            artifact: Any = field(default_factory=_StubArtifact)

        @dataclass
        class _StubTilting:
            selector: Any = field(default_factory=_StubSelector)

        cfg = Config.fast()
        raw = RawResult(
            experiment="coverage",
            tilting="power_law[learned_dynamic]",
            statistic="waldo",
            arrays={"x": np.zeros(1)},
            metadata={"alpha": 0.05},
        )

        tilt_a = _StubTilting(selector=_StubSelector(artifact=_StubArtifact("aaa")))
        tilt_b = _StubTilting(selector=_StubSelector(artifact=_StubArtifact("bbb")))

        path_a = persist_cell(
            raw_result=raw, config=cfg, cache_root=tmp_path / "ca",
            tilting=tilt_a,
        )
        path_b = persist_cell(
            raw_result=raw, config=cfg, cache_root=tmp_path / "cb",
            tilting=tilt_b,
        )

        # Cache directory names are the digest; different fingerprints
        # ⇒ different digests ⇒ different leaf directory names.
        assert path_a.name != path_b.name, (
            f"Expected different cache digests, got both = {path_a.name!r}; "
            f"the artifact fingerprint is not being plumbed into the key."
        )

    def test_non_learned_selector_does_not_break(self, tmp_path):
        """Tiltings without a fingerprintable selector still persist cleanly."""
        import numpy as np

        from frasian.config import Config
        from frasian.experiments.base import RawResult
        from frasian.simulation.runner import persist_cell

        cfg = Config.fast()
        raw = RawResult(
            experiment="coverage",
            tilting="power_law",
            statistic="waldo",
            arrays={"x": np.zeros(1)},
            metadata={"alpha": 0.05},
        )

        # No tilting at all.
        path_none = persist_cell(
            raw_result=raw, config=cfg, cache_root=tmp_path / "c1",
        )
        assert path_none.exists() or path_none.parent.exists()

        # Tilting whose selector has no artifact attribute — should not
        # raise, just skip the fingerprint plumbing.
        class _BareSelector:
            pass

        class _BareTilting:
            selector = _BareSelector()

        path_bare = persist_cell(
            raw_result=raw, config=cfg, cache_root=tmp_path / "c2",
            tilting=_BareTilting(),
        )
        assert path_bare.exists() or path_bare.parent.exists()
