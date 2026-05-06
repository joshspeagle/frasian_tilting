"""L2 cross-check: WALDO generic (MC) matches the closed form on Normal-Normal.

The closed-form WALDO p-value is `Phi(b - a) + Phi(-a - b)`. The generic
path runs `n_mc` Monte-Carlo reference draws under H_0 ~ N(theta, sigma^2)
and reports an empirical tail probability with +1 smoothing. Agreement
within ~3/sqrt(n_mc) is the calibration test we expect; the CI inversion
likewise agrees within ~1.5*sigma_post/sqrt(n_mc).

These bounds catch a class-of-bug regression: if the generic path mis-
identifies the reference distribution, the disagreement scales like O(1)
not 1/sqrt(n_mc), and these tolerances flag it. They do NOT pin point-
estimates to MC-noise floor — that would require a much larger n_mc and
slow the suite down.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic


@pytest.mark.L3
class TestWaldoGenericMatchesClosedForm:
    @pytest.mark.parametrize(
        "D, theta",
        [
            (0.0, -0.5),
            (0.0, 0.0),
            (0.0, 0.5),
            (1.0, 0.5),
            (1.0, 1.0),
            (-1.5, -0.5),
        ],
    )
    def test_pvalue_within_mc_tolerance(self, D, theta):
        sigma = 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([D])
        n_mc = 2000
        stat = WaldoStatistic(n_mc=n_mc, seed=0)
        cf = float(stat._closed_form_pvalue(float(theta), data, model, prior))
        gn = float(stat._generic_pvalue(float(theta), data, model, prior))
        # MC standard error ≈ sqrt(p*(1-p)/n_mc) ≤ 0.5/sqrt(n_mc); 3-sigma bound
        # for a one-sided MC reference. Use a slightly looser bound to absorb
        # the +1 smoothing's O(1/n_mc) bias.
        atol = 3.0 / np.sqrt(n_mc) + 1.0 / n_mc
        assert abs(cf - gn) < atol, (
            f"WALDO p-value disagreement at D={D}, theta={theta}: "
            f"closed-form={cf:.4f}, generic={gn:.4f}, atol={atol:.4f}"
        )

    @pytest.mark.parametrize("D", [-1.0, 0.0, 1.5])
    @pytest.mark.parametrize("alpha", [0.1, 0.05])
    def test_ci_within_mc_tolerance(self, D, alpha):
        sigma = 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([D])
        n_mc = 1500
        stat = WaldoStatistic(n_mc=n_mc, seed=0)
        cf_lo, cf_hi = stat._closed_form_confidence_interval(alpha, data, model, prior)
        gn_lo, gn_hi = stat._generic_confidence_interval(alpha, data, model, prior)
        # Posterior sigma_n ~ sqrt(0.5)*sigma ≈ 0.707; 3-sigma MC tolerance.
        post = model.posterior(data, prior)
        sigma_post = float(np.sqrt(post.var()))
        tol = 1.5 * sigma_post / np.sqrt(n_mc) + 0.05
        assert abs(cf_lo - gn_lo) < tol, (
            f"CI lo disagreement at D={D}, alpha={alpha}: "
            f"closed-form={cf_lo:.4f}, generic={gn_lo:.4f}, tol={tol:.4f}"
        )
        assert abs(cf_hi - gn_hi) < tol, (
            f"CI hi disagreement at D={D}, alpha={alpha}: "
            f"closed-form={cf_hi:.4f}, generic={gn_hi:.4f}, tol={tol:.4f}"
        )
