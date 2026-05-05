"""Regression: torch tilted_pvalue functions match the numpy versions exactly.

Both `power_law_tilted_pvalue_torch` and `ot_tilted_pvalue_torch` are
direct ports of their numpy counterparts. We verify the port is
faithful to atol 1e-10 across a representative grid of inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.learned.training.pvalue_torch import (
    ot_tilted_pvalue_torch,
    power_law_tilted_pvalue_torch,
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
def test_power_law_torch_matches_numpy(statistic_name, D, sigma0, eta):
    sigma, mu0 = 1.0, 0.0
    w = sigma0**2 / (sigma**2 + sigma0**2)
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    scheme = PowerLawTilting()

    theta_np = np.linspace(D - 4 * sigma, D + 4 * sigma, 21)
    np_p = scheme.tilted_pvalue(theta_np, D, model, prior, eta, statistic_name)

    theta_t = torch.tensor(theta_np, dtype=torch.float64)
    torch_p = power_law_tilted_pvalue_torch(
        theta_t,
        torch.tensor(D, dtype=torch.float64),
        torch.tensor(w, dtype=torch.float64),
        torch.tensor(mu0, dtype=torch.float64),
        torch.tensor(sigma, dtype=torch.float64),
        torch.tensor(eta, dtype=torch.float64),
        statistic_name,
    ).numpy()

    np.testing.assert_allclose(torch_p, np_p, atol=1e-10)


@pytest.mark.L0
@pytest.mark.parametrize("statistic_name", ["waldo", "wald"])
@pytest.mark.parametrize("D", [-1.5, 0.0, 1.0, 3.0])
@pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7, 1.0])
def test_ot_torch_matches_numpy(statistic_name, D, sigma0, eta):
    sigma, mu0 = 1.0, 0.0
    w = sigma0**2 / (sigma**2 + sigma0**2)
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    scheme = OTTilting()

    theta_np = np.linspace(D - 4 * sigma, D + 4 * sigma, 21)
    np_p = scheme.tilted_pvalue(theta_np, D, model, prior, eta, statistic_name)

    theta_t = torch.tensor(theta_np, dtype=torch.float64)
    torch_p = ot_tilted_pvalue_torch(
        theta_t,
        torch.tensor(D, dtype=torch.float64),
        torch.tensor(w, dtype=torch.float64),
        torch.tensor(mu0, dtype=torch.float64),
        torch.tensor(sigma, dtype=torch.float64),
        torch.tensor(eta, dtype=torch.float64),
        statistic_name,
    ).numpy()

    np.testing.assert_allclose(torch_p, np_p, atol=1e-10)


@pytest.mark.L0
def test_unknown_statistic_raises():
    """Both torch p-value functions raise NotImplementedError on unknown stats."""
    theta = torch.zeros(3, dtype=torch.float64)
    args = (
        theta,
        torch.tensor(0.0),
        torch.tensor(0.5),
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(0.5),
    )
    with pytest.raises(NotImplementedError):
        power_law_tilted_pvalue_torch(*args, "lrt")
    with pytest.raises(NotImplementedError):
        ot_tilted_pvalue_torch(*args, "lrt")


@pytest.mark.L0
def test_get_torch_tilted_pvalue_unknown_scheme_raises():
    from frasian.learned.training.pvalue_torch import get_torch_tilted_pvalue

    with pytest.raises(NotImplementedError):
        get_torch_tilted_pvalue("fisher_rao")
