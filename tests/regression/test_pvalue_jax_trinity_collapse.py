"""Trinity collapse on the JAX training kernels for PL/OT/FR.

The learned-η training path uses JAX-traceable kernels in
`learned/training/pvalue_jax.py`. For PL/OT/FR on NN+Normal, q_η is a
single Gaussian → τ_LRTO,η = τ_SCOREO,η = τ_WALDO,η identically. The
kernels should route lrto/scoreo through the waldo arithmetic, giving
bitwise-identical p-values across the three statistic names.

Mixture has no JAX-traceable closed form for LRTO/SCOREO (requires
autodiff through scipy.optimize for the mode), so `mixture_tilted_pvalue_jax`
explicitly raises NotImplementedError for lrto/scoreo.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from frasian.learned.training.pvalue_jax import (
    fisher_rao_tilted_pvalue_jax,
    mixture_tilted_pvalue_jax,
    ot_tilted_pvalue_jax,
    power_law_tilted_pvalue_jax,
)


_PL_OT_FR = [
    (power_law_tilted_pvalue_jax, "power_law"),
    (ot_tilted_pvalue_jax, "ot"),
    (fisher_rao_tilted_pvalue_jax, "fisher_rao"),
]


@pytest.mark.L2
@pytest.mark.parametrize("kernel,name", _PL_OT_FR)
@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("theta", [-0.2, 0.0, 0.5])
def test_jax_kernel_trinity_collapse(kernel, name, eta, theta):
    """PL/OT/FR JAX kernels: lrto and scoreo give same p-value as waldo."""
    theta_j = jnp.float64(theta)
    D_j = jnp.float64(0.7)
    w_j = jnp.float64(0.6)
    mu0_j = jnp.float64(0.0)
    sigma_j = jnp.float64(1.0)
    eta_j = jnp.float64(eta)

    p_waldo = float(kernel(theta_j, D_j, w_j, mu0_j, sigma_j, eta_j, "waldo"))
    p_lrto = float(kernel(theta_j, D_j, w_j, mu0_j, sigma_j, eta_j, "lrto"))
    p_scoreo = float(kernel(theta_j, D_j, w_j, mu0_j, sigma_j, eta_j, "scoreo"))

    assert p_lrto == pytest.approx(p_waldo, rel=1e-12, abs=1e-12), (
        f"{name} eta={eta} theta={theta}: p_lrto={p_lrto} vs p_waldo={p_waldo}"
    )
    assert p_scoreo == pytest.approx(p_waldo, rel=1e-12, abs=1e-12), (
        f"{name} eta={eta} theta={theta}: p_scoreo={p_scoreo} vs p_waldo={p_waldo}"
    )


@pytest.mark.L2
@pytest.mark.parametrize("stat", ["lrto", "scoreo"])
def test_mixture_kernel_raises_for_lrto_scoreo(stat):
    """Mixture has no JAX-traceable closed form for LRTO/SCOREO."""
    with pytest.raises(NotImplementedError, match="mixture"):
        mixture_tilted_pvalue_jax(
            jnp.float64(0.0), jnp.float64(0.7), jnp.float64(0.6),
            jnp.float64(0.0), jnp.float64(1.0), jnp.float64(0.3), stat,
        )
