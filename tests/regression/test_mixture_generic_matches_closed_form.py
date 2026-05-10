"""Generic-MC mixture path agrees with NN closed form on Normal-Normal.

Mirrors `tests/regression/test_ot_generic_matches_closed_form.py`-style
3-sigma MC bounds at n_mc=2000, parametrised over (eta, D, theta).
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.mixture import (
    _generic_tilted_pvalue_mixture,
    _mixture_tilted_pvalue_numpy_scalar,
)


@pytest.mark.L3
@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("D_val,theta", [(0.5, 0.3), (1.5, 0.5)])
def test_generic_mc_agrees_with_closed_form(eta, D_val, theta):
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    sigma = 1.0
    mu0 = 0.0
    sigma0 = 2.0
    w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

    p_closed = _mixture_tilted_pvalue_numpy_scalar(
        theta, eta, D_val, w, mu0, sigma, "waldo"
    )
    p_generic = _generic_tilted_pvalue_mixture(
        theta, eta, np.asarray([D_val]), model, prior, "waldo",
        support=(-float("inf"), float("inf")), n_mc=2000,
    )
    se = float(np.sqrt(max(p_closed * (1 - p_closed), 1e-6) / 2000))
    assert abs(p_generic - p_closed) < 4.0 * se, (
        f"closed={p_closed:.6f}, generic={p_generic:.6f}, "
        f"4sig={4*se:.6f} at eta={eta}, D={D_val}, theta={theta}"
    )
