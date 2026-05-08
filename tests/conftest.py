"""Shared pytest fixtures for the Frasian framework.

Test isolation: each test starts with an empty registry; the
`bootstrapped_registry` fixture restores the concrete implementations
captured at conftest-import time, so tests that need real methods can
opt in without hitting the bootstrap-singleton no-op problem.

Audit P1 M.2: Hypothesis tests get a project-wide default
`deadline=2000` (ms) so a runaway shrink can't hang CI. Per-test
overrides via `@settings(deadline=None)` still win.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import HealthCheck, settings

from frasian import Config, registry
from frasian import _registry_bootstrap as _bootstrap_mod
from frasian._registry_bootstrap import bootstrap

# Audit P1 M.2: register a project-wide Hypothesis profile with a
# 2-second deadline (the legacy default of 200 ms is too tight for the
# JAX-jit warm path on first call; 2000 ms absorbs that overhead while
# still flagging genuine runaway-shrink pathologies). Tests can still
# opt out via `@settings(deadline=None)`.
settings.register_profile(
    "frasian_default",
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("frasian_default")

# Trigger registration once at collection time and capture the entries.
bootstrap()
_BOOTSTRAPPED_ENTRIES = tuple(
    itertools.chain(
        registry.models.entries(),
        registry.tiltings.entries(),
        registry.statistics.entries(),
        registry.experiments.entries(),
        registry.diagnostics.entries(),
    )
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Each test starts with an empty registry; restore on teardown.

    Concrete implementations remain importable directly from their
    modules — clearing the registry only affects discovery via
    `frasian.registry`. Tests that depend on registry-driven
    discovery can use the `bootstrapped_registry` fixture below.

    Audit P1 M.4: also reset the `_BOOTSTRAPPED` flag in
    `_registry_bootstrap` on teardown, so a test that calls
    `bootstrap()` mid-test (or relies on bootstrap being idempotent
    across tests) gets a working flag back. Pre-fix the singleton's
    `_BOOTSTRAPPED = True` lingered across the registry-clear, and
    a follow-up `bootstrap()` call became a no-op (its early
    `if _BOOTSTRAPPED: return` short-circuit triggered).
    """
    snapshot = tuple(
        itertools.chain(
            registry.models.entries(),
            registry.tiltings.entries(),
            registry.statistics.entries(),
            registry.experiments.entries(),
            registry.diagnostics.entries(),
        )
    )
    registry.clear()
    _bootstrap_mod._BOOTSTRAPPED = False
    yield
    registry.clear()
    _bootstrap_mod._BOOTSTRAPPED = False
    for entry in snapshot:
        registry.register(entry)
    # The autouse re-registration above bypasses the bootstrap()
    # singleton, but downstream code that calls `bootstrap()` should
    # find it in the "already bootstrapped" state since the entries
    # are now present.
    _bootstrap_mod._BOOTSTRAPPED = True


@pytest.fixture
def bootstrapped_registry():
    """Restore the framework's concrete methods for this test.

    Use this whenever a test calls `registry.experiments[...]` or
    `run_experiment(...)` and expects the real Wald / WALDO / power_law
    implementations to be present. Cleanup is handled by the autouse
    `_isolated_registry` fixture above.
    """
    for entry in _BOOTSTRAPPED_ENTRIES:
        if entry.name not in getattr(registry, entry.kind + "s", registry.models)._entries:
            registry.register(entry)
    yield


@pytest.fixture
def fast_config() -> Config:
    """Smaller grids and fewer reps for fast tests."""
    return Config.fast()
