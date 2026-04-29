"""Tests for the plugin Registry and its decorators.

These exercise registration, conflict detection, slice operations, and the
`from_overrides` flow on Config.
"""

from __future__ import annotations

import pytest

from frasian import Config, RegistryConflictError, registry
from frasian._registry import register_statistic, register_tilting


class _DummyTilting:
    name = "dummy_tilting"


class _DummyStatistic:
    name = "dummy_stat"


class TestRegistryDecorators:
    def test_register_tilting_inserts_into_slice(self):
        decorated = register_tilting(
            name="dummy_tilting",
            brief="docs/methods/dummy.md",
            status="stub",
        )(_DummyTilting)
        assert decorated is _DummyTilting
        assert "dummy_tilting" in registry.tiltings
        entry = registry.tiltings.entries()[0]
        assert entry.kind == "tilting"
        assert entry.status == "stub"
        assert entry.brief == "docs/methods/dummy.md"

    def test_register_statistic_inserts_into_slice(self):
        register_statistic(
            name="dummy_stat", brief="docs/methods/dummy_stat.md",
        )(_DummyStatistic)
        assert "dummy_stat" in registry.statistics

    def test_double_registration_raises(self):
        register_tilting(name="dup", brief="docs/methods/dup.md")(
            type("T1", (), {"name": "dup"})
        )
        with pytest.raises(RegistryConflictError):
            register_tilting(name="dup", brief="docs/methods/dup.md")(
                type("T2", (), {"name": "dup"})
            )

    def test_slice_filtering_by_status(self):
        register_tilting(name="impl", brief="b1.md", status="implemented")(
            type("A", (), {})
        )
        register_tilting(name="stb", brief="b2.md", status="stub")(
            type("B", (), {})
        )
        impl = registry.tiltings.where(status="implemented")
        stub = registry.tiltings.where(status="stub")
        assert len(impl) == 1
        assert len(stub) == 1

    def test_slice_filtering_by_name(self):
        register_tilting(name="x", brief="x.md")(type("X", (), {}))
        register_tilting(name="y", brief="y.md")(type("Y", (), {}))
        only_x = registry.tiltings.where(name__in=["x"])
        assert len(only_x) == 1

    def test_decorator_attaches_metadata(self):
        cls = register_tilting(name="t", brief="t.md")(type("T", (), {}))
        assert cls.__frasian_registry_name__ == "t"
        assert cls.__frasian_registry_kind__ == "tilting"


class TestConfig:
    def test_default_alpha_is_005(self):
        assert Config.default().alpha == 0.05

    def test_fast_has_smaller_grids(self):
        fast = Config.fast()
        default = Config.default()
        assert fast.delta_grid.n_points < default.delta_grid.n_points
        assert fast.n_reps < default.n_reps

    def test_from_overrides_returns_new_instance(self):
        c = Config.default()
        override = c.from_overrides(alpha=0.10)
        assert c.alpha == 0.05
        assert override.alpha == 0.10

    def test_fingerprint_is_deterministic(self):
        a = Config.default()
        b = Config.default()
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_with_config(self):
        a = Config.default()
        b = Config.default().from_overrides(alpha=0.10)
        assert a.fingerprint() != b.fingerprint()
