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

    def test_persist_cell_swallows_attribute_or_type_error_on_fingerprint(
        self, tmp_path,
    ):
        """Pin the narrow except: an artifact whose `.fingerprint()` raises
        AttributeError or TypeError must be skipped silently (treated as
        "no fingerprint method available"). Real failures (OSError,
        MissingArtifactError) propagate — covered by
        ``test_persist_cell_propagates_oserror_on_fingerprint``.
        """
        from dataclasses import dataclass, field
        from typing import Any

        import numpy as np

        from frasian.config import Config
        from frasian.experiments.base import RawResult
        from frasian.simulation.runner import persist_cell

        class _AttrErrArtifact:
            def fingerprint(self):
                raise AttributeError("simulated missing-attr in fingerprint")

        class _TypeErrArtifact:
            def fingerprint(self):
                raise TypeError("simulated bad-arg in fingerprint")

        @dataclass
        class _StubSelector:
            artifact: Any = None

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

        # AttributeError path.
        tilt_attr = _StubTilting(selector=_StubSelector(artifact=_AttrErrArtifact()))
        path_attr = persist_cell(
            raw_result=raw, config=cfg, cache_root=tmp_path / "ca",
            tilting=tilt_attr,
        )
        assert path_attr.exists() or path_attr.parent.exists()

        # TypeError path.
        tilt_type = _StubTilting(selector=_StubSelector(artifact=_TypeErrArtifact()))
        path_type = persist_cell(
            raw_result=raw, config=cfg, cache_root=tmp_path / "ct",
            tilting=tilt_type,
        )
        assert path_type.exists() or path_type.parent.exists()

    def test_persist_cell_propagates_oserror_on_fingerprint(self, tmp_path):
        """Pin the narrow except: an artifact whose `.fingerprint()` raises
        OSError must propagate (it's a real I/O failure, not a "no
        fingerprint method" case). A bare ``except Exception`` would
        silently collapse the cache key onto the no-fingerprint fallback,
        which is exactly the failure mode 1.7-C1 was meant to prevent.
        """
        from dataclasses import dataclass, field
        from typing import Any

        import numpy as np
        import pytest as _pt

        from frasian.config import Config
        from frasian.experiments.base import RawResult
        from frasian.simulation.runner import persist_cell

        class _OSErrArtifact:
            def fingerprint(self):
                raise OSError("simulated checkpoint missing on disk")

        @dataclass
        class _StubSelector:
            artifact: Any = None

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
        tilt = _StubTilting(selector=_StubSelector(artifact=_OSErrArtifact()))
        with _pt.raises(OSError, match=r"simulated checkpoint missing"):
            persist_cell(
                raw_result=raw, config=cfg, cache_root=tmp_path / "co",
                tilting=tilt,
            )

    def test_runner_plumbs_artifact_fingerprint_end_to_end(
        self, tmp_path, monkeypatch, bootstrapped_registry,
    ):
        """End-to-end: ``run_experiment`` must pass ``tilting=tilting`` to
        ``persist_cell`` so the runner-level cache key picks up the
        learned-η artifact fingerprint. This catches a regression where
        someone drops ``tilting=tilting`` from the
        ``frasian._runner.run_experiment`` call site at line 163.

        Strategy: monkeypatch ``persist_cell`` in the runner to record
        the kwargs each call received, then confirm the runner forwarded
        a tilting whose selector carries a fingerprintable artifact.
        """
        from dataclasses import dataclass, field
        from pathlib import Path as _Path
        from typing import Any

        import numpy as np

        from frasian import Config, registry, run_experiment
        from frasian._registry_bootstrap import bootstrap
        from frasian.config import GridSpec
        from frasian.statistics.waldo import WaldoStatistic
        from frasian.tilting.identity import IdentityTilting

        bootstrap()

        @dataclass
        class _RecorderArtifact:
            fp: str = "stub_fingerprint_xyz"

            def fingerprint(self) -> str:
                return self.fp

        @dataclass
        class _RecorderSelector:
            artifact: Any = field(default_factory=_RecorderArtifact)
            name: str = "recorder"
            is_dynamic: bool = False

        # Wrap IdentityTilting so the runner's `_accepts_tilting` passes
        # through (waldo accepts any tilting; identity is the simplest).
        @dataclass(frozen=True)
        class _RecorderTilting(IdentityTilting):
            name: str = "identity_recorder"
            selector: Any = field(default_factory=_RecorderSelector)

            @property
            def cell_name(self) -> str:
                return self.name

        captured: list[dict] = []
        from frasian import _runner as runner_mod
        real_persist = runner_mod.persist_cell

        def _spy_persist(**kwargs):
            captured.append(kwargs)
            return real_persist(**kwargs)

        monkeypatch.setattr(runner_mod, "persist_cell", _spy_persist)

        cfg = Config.fast().from_overrides(
            n_reps=4,
            theta_grid=GridSpec("theta", -1.0, 1.0, 2),
            w_grid=GridSpec("w", 0.5, 0.5, 1),
        )
        run_experiment(
            experiment=registry.experiments["width"](),
            tiltings=[_RecorderTilting()],
            statistics=[WaldoStatistic()],
            config=cfg,
            out_dir=tmp_path,
        )

        # The runner must have called persist_cell at least once with a
        # non-None `tilting` kwarg matching our recorder. A regression
        # that drops `tilting=tilting` would fail this assertion: the
        # spy would still be called, but `kwargs.get("tilting")` would
        # be None.
        assert captured, "persist_cell was never called by run_experiment"
        forwarded = [c.get("tilting") for c in captured]
        assert all(t is not None for t in forwarded), (
            "run_experiment dropped `tilting=tilting`; "
            f"forwarded={forwarded!r}"
        )
        # And the forwarded tilting must carry our fingerprintable artifact.
        for t in forwarded:
            sel = getattr(t, "selector", None)
            art = getattr(sel, "artifact", None) if sel else None
            assert art is not None, (
                f"runner forwarded a tilting without selector.artifact: {t!r}"
            )
            assert art.fingerprint() == "stub_fingerprint_xyz"
