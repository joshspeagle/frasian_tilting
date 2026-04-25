"""Shared pytest fixtures for the Frasian framework.

Test isolation: each test starts with an empty registry; the
`bootstrapped_registry` fixture restores the concrete implementations
captured at conftest-import time, so tests that need real methods can
opt in without hitting the bootstrap-singleton no-op problem.
"""

from __future__ import annotations

import itertools

import pytest

from frasian import Config, registry
from frasian._registry_bootstrap import bootstrap

# Trigger registration once at collection time and capture the entries.
bootstrap()
_BOOTSTRAPPED_ENTRIES = tuple(itertools.chain(
    registry.models.entries(),
    registry.tiltings.entries(),
    registry.statistics.entries(),
    registry.experiments.entries(),
    registry.diagnostics.entries(),
))


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Each test starts with an empty registry; restore on teardown.

    Concrete implementations remain importable directly from their
    modules — clearing the registry only affects discovery via
    `frasian.registry`. Tests that depend on registry-driven
    discovery can use the `bootstrapped_registry` fixture below.
    """
    snapshot = tuple(itertools.chain(
        registry.models.entries(),
        registry.tiltings.entries(),
        registry.statistics.entries(),
        registry.experiments.entries(),
        registry.diagnostics.entries(),
    ))
    registry.clear()
    yield
    registry.clear()
    for entry in snapshot:
        registry.register(entry)


@pytest.fixture
def bootstrapped_registry():
    """Restore the framework's concrete methods for this test.

    Use this whenever a test calls `registry.experiments[...]` or
    `run_experiment(...)` and expects the real Wald / WALDO / power_law
    implementations to be present. Cleanup is handled by the autouse
    `_isolated_registry` fixture above.
    """
    for entry in _BOOTSTRAPPED_ENTRIES:
        if entry.name not in getattr(registry, entry.kind + "s",
                                       registry.models)._entries:
            registry.register(entry)
    yield


@pytest.fixture
def fast_config() -> Config:
    """Smaller grids and fewer reps for fast tests."""
    return Config.fast()
