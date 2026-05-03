"""Regression: torch CD density matches the numpy SH density.

Compares `cd_density_torch` (which skips signed_confidence) against
`build_cd_from_pvalue.pdf_values` on smooth p-value curves.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.cd.from_pvalue import build_cd_from_pvalue
from frasian.learned.training.cd_torch import cd_density_torch
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.identity import IdentityTilting


@pytest.mark.L2
@pytest.mark.parametrize("D", [-1.0, 0.5, 2.0])
@pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
def test_cd_density_torch_matches_numpy(D, sigma0):
    """`cd_density_torch(p, θ)` ≡ `build_cd_from_pvalue(...).pdf_values`.

    Skip `signed_confidence` (uses argmax, non-diff). The pdf path is
    identical (averaged-one-sided-diff, Z-normalised).
    """
    sigma, mu0 = 1.0, 0.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)

    cd = build_cd_from_pvalue(
        IdentityTilting(), WaldoStatistic(), float(D), model, prior,
        n_grid=201, half_width_sigma=6.0,
    )
    theta_grid_np = cd.theta_grid
    p_theta_np = WaldoStatistic().pvalue(
        theta_grid_np, np.asarray([float(D)]), model, prior,
    )
    p_theta_np = np.clip(p_theta_np, 0.0, 1.0)

    p_t = torch.tensor(p_theta_np[None, :], dtype=torch.float64)
    theta_t = torch.tensor(theta_grid_np, dtype=torch.float64)
    pdf_torch = cd_density_torch(p_t, theta_t).numpy().reshape(-1)

    np.testing.assert_allclose(pdf_torch, cd.pdf_values, atol=5e-4, rtol=5e-4)


@pytest.mark.L2
def test_cd_density_torch_integrates_to_one():
    """Normalised CD density integrates to 1 (per batch row)."""
    theta = torch.linspace(-5.0, 5.0, 401, dtype=torch.float64)
    # Synthetic well-behaved p-value: Φ(b-a) + Φ(-a-b) shape.
    a = (theta - 1.0).abs() / 0.7
    b = -0.3 * (theta - 0.0)
    from torch.distributions import Normal
    n = Normal(0.0, 1.0)
    p = n.cdf(b - a) + n.cdf(-a - b)
    p = p.unsqueeze(0).expand(3, -1)  # batch of 3 identical rows

    pdf = cd_density_torch(p, theta)
    integrals = torch.trapezoid(pdf, theta, dim=-1)
    np.testing.assert_allclose(integrals.numpy(), 1.0, atol=1e-6)


@pytest.mark.L2
def test_cd_density_torch_shape_check():
    """1D p-value raises; mismatched-shape 2D theta_grid raises."""
    theta = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="\\(B, N\\)"):
        cd_density_torch(torch.zeros(5, dtype=torch.float64), theta)
    # 2D theta_grid with the wrong shape should also fail.
    p = torch.zeros(2, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="must match"):
        cd_density_torch(p, torch.zeros(3, 5, dtype=torch.float64))


@pytest.mark.L2
def test_cd_density_torch_per_sample_grid():
    """Per-sample 2D theta_grid produces the same density as a shared 1D grid
    when all rows of the 2D grid are identical."""
    theta_1d = torch.linspace(-3.0, 3.0, 51, dtype=torch.float64)
    theta_2d = theta_1d.unsqueeze(0).expand(4, -1).contiguous()
    # Synthetic well-behaved p-value.
    p = torch.sigmoid(-(theta_1d - 0.5) ** 2 + 0.5).unsqueeze(0).expand(4, -1)
    pdf_shared = cd_density_torch(p, theta_1d)
    pdf_per = cd_density_torch(p, theta_2d)
    np.testing.assert_allclose(pdf_per.numpy(), pdf_shared.numpy(),
                                 atol=1e-12)
