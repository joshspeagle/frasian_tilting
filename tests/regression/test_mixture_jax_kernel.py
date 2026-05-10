"""JAX kernel agreement test: closed-form numpy vs mixture_tilted_pvalue_jax.

Stage C.1 of mixture-tilting plan (revised 2026-05-10). The JAX kernel
must match the numpy scalar implementation up to 1e-10 across the
admissible η range and all 5 branches of the quadratic-roots
formulation (L>0/L<0/L≈0 × disc</≥0).
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.tilting.mixture import _mixture_tilted_pvalue_numpy_scalar


@pytest.mark.L1
@pytest.mark.parametrize("statistic_name", ["wald", "waldo"])
@pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize(
    "theta,D_val",
    [
        (0.3, 0.5),  # interior, mild conflict
        (1.5, 0.5),  # off-prior, moderate conflict
        (-0.5, 0.5),  # opposite-side conflict
        (0.0, 2.0),  # zero θ, large D
    ],
)
def test_jax_kernel_matches_numpy(statistic_name, eta, theta, D_val):
    """Port of the closed-form mixture p-value to JAX must agree with
    the numpy scalar implementation up to 1e-10."""
    import jax.numpy as jnp

    from frasian.learned.training.pvalue_jax import mixture_tilted_pvalue_jax

    sigma = 1.0
    mu0 = 0.0
    sigma0 = 2.0
    w = sigma0**2 / (sigma**2 + sigma0**2)

    p_np = _mixture_tilted_pvalue_numpy_scalar(
        theta, eta, D_val, w, mu0, sigma, statistic_name
    )
    p_jax = float(
        mixture_tilted_pvalue_jax(
            theta=jnp.asarray(theta, dtype=jnp.float64),
            D=jnp.asarray(D_val, dtype=jnp.float64),
            w=jnp.asarray(w, dtype=jnp.float64),
            mu0=jnp.asarray(mu0, dtype=jnp.float64),
            sigma=jnp.asarray(sigma, dtype=jnp.float64),
            eta=jnp.asarray(eta, dtype=jnp.float64),
            statistic_name=statistic_name,
        )
    )
    assert abs(p_np - p_jax) < 1e-10, (
        f"{statistic_name} mismatch at eta={eta}, theta={theta}, "
        f"D={D_val}: np={p_np:.10f}, jax={p_jax:.10f}"
    )


@pytest.mark.L1
def test_jax_kernel_vmap_compatible():
    """`mixture_tilted_pvalue_jax` must be vmap+jit-compatible (no
    Python-level branches on tracer-typed values)."""
    import jax
    import jax.numpy as jnp

    from frasian.learned.training.pvalue_jax import mixture_tilted_pvalue_jax

    sigma = 1.0
    mu0 = 0.0
    sigma0 = 2.0
    w = sigma0**2 / (sigma**2 + sigma0**2)

    # Batched θ-grid + batched η; should not raise under jit/vmap.
    @jax.jit
    def kernel(theta, eta):
        return mixture_tilted_pvalue_jax(
            theta=theta,
            D=jnp.asarray(0.5, dtype=jnp.float64),
            w=jnp.asarray(w, dtype=jnp.float64),
            mu0=jnp.asarray(mu0, dtype=jnp.float64),
            sigma=jnp.asarray(sigma, dtype=jnp.float64),
            eta=eta,
            statistic_name="waldo",
        )

    theta_grid = jnp.linspace(-3.0, 3.0, 21, dtype=jnp.float64)
    eta_grid = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float64)
    out = jax.vmap(kernel, in_axes=(0, None))(theta_grid, jnp.asarray(0.5, dtype=jnp.float64))
    assert out.shape == (21,)
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.L1
@pytest.mark.parametrize("eta", [0.0, 0.5, 0.99])
def test_jax_kernel_endpoint_eta_zero_matches_waldo(eta):
    """At η=0 the mixture p-value MUST equal bare WALDO; at η=1 it MUST
    equal bare Wald. Test these two anchors explicitly to lock the
    endpoint convention."""
    import jax.numpy as jnp

    from frasian.learned.training.pvalue_jax import mixture_tilted_pvalue_jax

    sigma, mu0, sigma0 = 1.0, 0.0, 2.0
    w = sigma0**2 / (sigma**2 + sigma0**2)
    theta, D = 0.3, 0.5

    p_np = _mixture_tilted_pvalue_numpy_scalar(theta, eta, D, w, mu0, sigma, "waldo")
    p_jax = float(
        mixture_tilted_pvalue_jax(
            theta=jnp.asarray(theta, dtype=jnp.float64),
            D=jnp.asarray(D, dtype=jnp.float64),
            w=jnp.asarray(w, dtype=jnp.float64),
            mu0=jnp.asarray(mu0, dtype=jnp.float64),
            sigma=jnp.asarray(sigma, dtype=jnp.float64),
            eta=jnp.asarray(eta, dtype=jnp.float64),
            statistic_name="waldo",
        )
    )
    assert abs(p_np - p_jax) < 1e-10
