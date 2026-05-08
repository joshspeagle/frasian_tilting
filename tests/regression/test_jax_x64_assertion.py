"""Regression: `frasian._jax_setup.assert_x64()` fails fast when JAX
64-bit precision has been disabled.

Audit P2 (Cluster F) flagged that `enable_x64()` only fires if some
``frasian`` module imports it first; if a downstream import path or
third-party lib has flipped `jax_enable_x64` off, the framework
silently downgrades to float32 with no diagnostic. `assert_x64()`
provides a defensive read-back for loss / pvalue kernels.

The test toggles the JAX config off, asserts that `assert_x64()`
raises, then re-enables and asserts no raise. We restore via a
try/finally so a failing assertion does NOT leak float32 into
sibling tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.L2
class TestAssertX64:
    def test_raises_when_x64_disabled(self):
        import jax

        from frasian._jax_setup import assert_x64, enable_x64

        # Sanity: x64 should be on in this test run (frasian/__init__.py
        # imports _jax_setup which calls enable_x64()).
        assert jax.config.read("jax_enable_x64")
        try:
            jax.config.update("jax_enable_x64", False)
            with pytest.raises(RuntimeError, match="jax_enable_x64"):
                assert_x64()
        finally:
            enable_x64()
        # Restored state — `assert_x64()` is silent again.
        assert_x64()

    def test_silent_when_x64_enabled(self):
        from frasian._jax_setup import assert_x64

        # No raise when x64 is on (the framework's normal state).
        assert_x64()
