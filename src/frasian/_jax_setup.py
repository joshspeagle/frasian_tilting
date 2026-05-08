"""Framework-wide JAX configuration.

Imported once from `frasian/__init__.py` so that anyone using the public
package has 64-bit precision enabled before the first `jnp.array(...)`
call. Direct importers of submodules (e.g. test modules that do
`from frasian.tilting.power_law import ...`) get the same guarantee
because every JAX-using submodule re-imports this helper at the top.

Calling `enable_x64()` more than once is a no-op (`jax.config.update`
is idempotent for the same value).

The `assert_x64()` helper provides a defensive read-back: if some
external import path has flipped `jax_enable_x64` off after our
module-level call (rare, but possible if a third-party JAX library
updates the global config), loss / pvalue kernels can call
`assert_x64()` before evaluating gradients to fail fast with a
clear error rather than silently producing float32 numerics.
"""

from __future__ import annotations

import jax


def enable_x64() -> None:
    jax.config.update("jax_enable_x64", True)


def assert_x64() -> None:
    """Defensive read-back: raise if `jax_enable_x64` was flipped off.

    Audit P2 (Cluster F): `enable_x64()` only fires if some `frasian`
    module has been imported. If a downstream consumer imports
    `frasian.tilting.power_law` and later a third-party library
    (or test setup) calls `jax.config.update("jax_enable_x64",
    False)`, the rest of the framework's loss / pvalue kernels
    silently downgrade to float32. This helper makes that drift a
    fast `RuntimeError` instead of a numerical-mystery report.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "frasian requires JAX 64-bit precision (`jax_enable_x64=True`). "
            "Some import path has disabled it. Re-enable via "
            "`frasian._jax_setup.enable_x64()` before evaluating loss / "
            "pvalue kernels."
        )


enable_x64()
