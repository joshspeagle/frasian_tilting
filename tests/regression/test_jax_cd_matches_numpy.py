"""Regression: JAX CD density matches the numpy SH density.

Compares `cd_density_jax` (which skips signed_confidence) against
`build_cd_from_pvalue.pdf_values` on smooth p-value curves. Mirrors the
torch-side test (`test_torch_cd_matches_numpy.py`) which will be
retired once `learned/` no longer depends on torch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
import pytest

from frasian.cd.from_pvalue import build_cd_from_pvalue
from frasian.learned.training.cd_jax import cd_density_jax
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.identity import IdentityTilting


@pytest.mark.L2
@pytest.mark.parametrize("D", [-1.0, 0.5, 2.0])
@pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
def test_cd_density_jax_matches_numpy(D, sigma0):
    """`cd_density_jax(p, theta)` == `build_cd_from_pvalue(...).pdf_values`.

    Skip `signed_confidence` (uses argmax, non-diff). The pdf path is
    identical (averaged-one-sided-diff, Z-normalised).
    """
    sigma, mu0 = 1.0, 0.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)

    cd = build_cd_from_pvalue(
        IdentityTilting(),
        WaldoStatistic(),
        float(D),
        model,
        prior,
        n_grid=201,
        half_width_sigma=6.0,
    )
    theta_grid_np = np.asarray(cd.theta_grid)
    p_theta_np = np.asarray(
        WaldoStatistic().pvalue(theta_grid_np, np.asarray([float(D)]), model, prior)
    )
    p_theta_np = np.clip(p_theta_np, 0.0, 1.0)

    p_j = jnp.asarray(p_theta_np[None, :])
    theta_j = jnp.asarray(theta_grid_np)
    pdf_jax = np.asarray(cd_density_jax(p_j, theta_j)).reshape(-1)

    np.testing.assert_allclose(pdf_jax, np.asarray(cd.pdf_values), atol=5e-4, rtol=5e-4)


@pytest.mark.L2
def test_cd_density_jax_integrates_to_one():
    """Normalised CD density integrates to 1 (per batch row)."""
    theta = jnp.linspace(-5.0, 5.0, 401, dtype=jnp.float64)
    # Synthetic well-behaved p-value: Φ(b-a) + Φ(-a-b) shape.
    a = jnp.abs(theta - 1.0) / 0.7
    b = -0.3 * (theta - 0.0)
    p = jsp_stats.norm.cdf(b - a) + jsp_stats.norm.cdf(-a - b)
    p = jnp.broadcast_to(p[None, :], (3, p.shape[0]))  # batch of 3 identical rows

    pdf = cd_density_jax(p, theta)
    # Broadcast theta to match pdf for jnp.trapezoid.
    theta_b = jnp.broadcast_to(theta, pdf.shape)
    integrals = jnp.trapezoid(pdf, theta_b, axis=-1)
    np.testing.assert_allclose(np.asarray(integrals), 1.0, atol=1e-6)


@pytest.mark.L2
def test_cd_density_jax_shape_check():
    """1D p-value raises; mismatched-shape 2D theta_grid raises."""
    theta = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float64)
    with pytest.raises(ValueError, match="\\(B, N\\)"):
        cd_density_jax(jnp.zeros(5, dtype=jnp.float64), theta)
    p = jnp.zeros((2, 5), dtype=jnp.float64)
    with pytest.raises(ValueError, match="must match"):
        cd_density_jax(p, jnp.zeros((3, 5), dtype=jnp.float64))


@pytest.mark.L2
def test_cd_density_jax_per_sample_grid():
    """Per-sample 2D theta_grid produces the same density as a shared 1D grid
    when all rows of the 2D grid are identical."""
    theta_1d = jnp.linspace(-3.0, 3.0, 51, dtype=jnp.float64)
    theta_2d = jnp.broadcast_to(theta_1d[None, :], (4, theta_1d.shape[0]))
    p = jax.nn.sigmoid(-((theta_1d - 0.5) ** 2) + 0.5)
    p_b = jnp.broadcast_to(p[None, :], (4, p.shape[0]))
    pdf_shared = cd_density_jax(p_b, theta_1d)
    pdf_per = cd_density_jax(p_b, theta_2d)
    np.testing.assert_allclose(np.asarray(pdf_per), np.asarray(pdf_shared), atol=1e-12)


@pytest.mark.L0
def test_cd_density_jax_grad_through_pvalue_finite():
    """`cd_density_jax` must be jax.grad-clean through the p-value tensor.

    Phase 4's learned-eta `cd_variance_loss` closes over this with
    `jax.grad(loss, eta_params)` chained through `tilted_pvalue`. NaN
    gradients here would silently break training.
    """
    def loss(p_flat):
        p = p_flat.reshape(1, -1)
        theta = jnp.linspace(-3.0, 3.0, p.shape[1])
        return jnp.sum(cd_density_jax(p, theta))

    p_flat = jnp.linspace(0.05, 0.95, 21)
    g = jax.grad(loss)(p_flat)
    assert np.all(np.isfinite(np.asarray(g))), (
        "jax.grad(cd_density_jax) produced non-finite gradient"
    )
