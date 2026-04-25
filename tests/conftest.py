"""Shared pytest fixtures for the new Frasian framework.

Step-1 fixtures cover only registry isolation and Config overrides. Property-
test strategies for Model / TiltingScheme / TestStatistic land in Step 2,
alongside the first concrete implementations.
"""

from __future__ import annotations

import pytest

from frasian import Config, registry


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot and restore the registry around each test.

    Registry mutation during one test must not leak into another. This is the
    discipline that lets new methods register at module import time without
    breaking test isolation.
    """
    snapshot = (
        list(registry.models.entries()),
        list(registry.tiltings.entries()),
        list(registry.statistics.entries()),
        list(registry.experiments.entries()),
        list(registry.diagnostics.entries()),
    )
    yield
    registry.clear()
    for entry in (*snapshot[0], *snapshot[1], *snapshot[2],
                  *snapshot[3], *snapshot[4]):
        registry.register(entry)


@pytest.fixture
def fast_config() -> Config:
    """Smaller grids and fewer reps for fast tests."""
    return Config.fast()
