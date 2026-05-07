"""Regression: JAX tilted_pvalue functions match the numpy versions exactly.

Both `power_law_tilted_pvalue_jax` and `ot_tilted_pvalue_jax` are direct
ports of their numpy counterparts (see `tilting/power_law.py` and
`tilting/ot.py`). We verify the port is faithful to atol 1e-10 across
a representative grid of inputs.

This is the JAX-side analogue of `test_torch_pvalue_matches_numpy.py`.
The two tests will coexist during the learned/ migration (commit 1
adds the JAX modules alongside the torch ones); the torch test will
be retired once `learned/` no longer depends on torch (commit 4).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from frasian.learned.training.pvalue_jax import (
    ot_tilted_pvalue_jax,
    power_law_tilted_pvalue_jax,
)
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L0
@pytest.mark.parametrize("statistic_name", ["waldo", "wald"])
@pytest.mark.parametrize("D", [-1.5, 0.0, 1.0, 3.0])
@pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7, 0.95])
def test_power_law_jax_matches_numpy(statistic_name, D, sigma0, eta):
    sigma, mu0 = 1.0, 0.0
    w = sigma0**2 / (sigma**2 + sigma0**2)
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    scheme = PowerLawTilting()

    theta_np = np.linspace(D - 4 * sigma, D + 4 * sigma, 21)
    np_p = np.asarray(
        scheme.tilted_pvalue(theta_np, D, model, prior, eta, statistic_name),
        dtype=np.float64,
    )

    theta_j = jnp.asarray(theta_np)
    D_j = jnp.asarray(D)
    w_j = jnp.asarray(w)
    mu0_j = jnp.asarray(mu0)
    sigma_j = jnp.asarray(sigma)
    eta_j = jnp.asarray(eta)
    jax_p = np.asarray(
        power_law_tilted_pvalue_jax(
            theta_j, D_j, w_j, mu0_j, sigma_j, eta_j, statistic_name
        )
    )
    np.testing.assert_allclose(jax_p, np_p, atol=1e-10, rtol=1e-10)


@pytest.mark.L0
@pytest.mark.parametrize("statistic_name", ["waldo", "wald"])
@pytest.mark.parametrize("D", [-1.5, 0.0, 1.0, 3.0])
@pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7, 1.0])
def test_ot_jax_matches_numpy(statistic_name, D, sigma0, eta):
    sigma, mu0 = 1.0, 0.0
    w = sigma0**2 / (sigma**2 + sigma0**2)
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    scheme = OTTilting()

    theta_np = np.linspace(D - 4 * sigma, D + 4 * sigma, 21)
    np_p = np.asarray(
        scheme.tilted_pvalue(theta_np, D, model, prior, eta, statistic_name),
        dtype=np.float64,
    )

    theta_j = jnp.asarray(theta_np)
    D_j = jnp.asarray(D)
    w_j = jnp.asarray(w)
    mu0_j = jnp.asarray(mu0)
    sigma_j = jnp.asarray(sigma)
    eta_j = jnp.asarray(eta)
    jax_p = np.asarray(
        ot_tilted_pvalue_jax(
            theta_j, D_j, w_j, mu0_j, sigma_j, eta_j, statistic_name
        )
    )
    np.testing.assert_allclose(jax_p, np_p, atol=1e-10, rtol=1e-10)


@pytest.mark.L0
def test_jax_pvalue_grad_through_eta_is_finite():
    """JAX p-value must be jax.grad-clean through eta.

    Phase 4's learned-eta loss closes over these kernels with
    `jax.grad(loss, eta_params)`; if the kernel produced NaN gradients
    at common eta values (boundary clipping pathology), training would
    silently fail. This is the load-bearing property the JAX port has
    to preserve from the torch port.
    """
    import jax

    def loss(eta_scalar):
        theta = jnp.linspace(-3.0, 3.0, 11)
        D = jnp.asarray(0.5)
        w = jnp.asarray(0.5)
        mu0 = jnp.asarray(0.0)
        sigma = jnp.asarray(1.0)
        eta = jnp.broadcast_to(eta_scalar, theta.shape)
        p = power_law_tilted_pvalue_jax(theta, D, w, mu0, sigma, eta, "waldo")
        return jnp.sum(p)

    for eta_val in [0.0, 0.3, 0.7, 0.95]:
        g = float(jax.grad(loss)(jnp.asarray(eta_val)))
        assert np.isfinite(g), (
            f"jax.grad(power_law_tilted_pvalue_jax) NaN/Inf at eta={eta_val}: {g}"
        )
