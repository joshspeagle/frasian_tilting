"""Framework-wide JAX configuration.

Imported once from `frasian/__init__.py` so that anyone using the public
package has 64-bit precision enabled before the first `jnp.array(...)`
call. Direct importers of submodules (e.g. test modules that do
`from frasian.tilting.power_law import ...`) get the same guarantee
because every JAX-using submodule re-imports this helper at the top.

Calling `enable_x64()` more than once is a no-op (`jax.config.update`
is idempotent for the same value).
"""

from __future__ import annotations

import jax


def enable_x64() -> None:
    jax.config.update("jax_enable_x64", True)


enable_x64()
