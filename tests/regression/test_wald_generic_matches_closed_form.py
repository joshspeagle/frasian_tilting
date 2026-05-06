"""L2 cross-check: Wald generic numerical path matches closed form on Normal-Normal.

The closed-form Normal-Normal Wald statistic uses ``2*(1 - Phi(|D - theta|/sigma))``
for the p-value and ``D ± z * sigma`` for the CI. The generic numerical path
uses ``tau = (mle - theta)^2 * I(theta)`` with chi^2_1 calibration. Both must
agree to within numerical tolerance on the canonical Normal-Normal sandbox —
otherwise the generic path has a bug.

Pinning this on every commit guards against future regressions in either
branch (e.g., a JAX-port edit that subtly shifts the chi^2 calibration).
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic


_THETAS = np.array([-3.0, -1.0, -0.25, 0.0, 0.5, 1.5, 4.0])


@pytest.mark.L2
class TestWaldGenericMatchesClosedForm:
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("D", [-2.0, 0.0, 1.5])
    def test_pvalue_agrees(self, D, sigma):
        model = NormalNormalModel(sigma=sigma)
        data = np.asarray([D])
        stat = WaldStatistic()
        for theta in _THETAS:
            cf = float(stat._closed_form_pvalue(float(theta), data, model))
            gn = float(stat._generic_pvalue(float(theta), data, model))
            assert np.isfinite(cf) and np.isfinite(gn)
            assert abs(cf - gn) < 1e-10, (
                f"Wald p-value disagreement at theta={theta}, D={D}, sigma={sigma}: "
                f"closed-form={cf}, generic={gn}"
            )

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("D", [-2.0, 0.0, 1.5])
    @pytest.mark.parametrize("alpha", [0.05, 0.1])
    def test_ci_agrees(self, D, sigma, alpha):
        model = NormalNormalModel(sigma=sigma)
        data = np.asarray([D])
        stat = WaldStatistic()
        cf_lo, cf_hi = stat._closed_form_confidence_interval(alpha, data, model)
        gn_lo, gn_hi = stat._generic_confidence_interval(alpha, data, model)
        # brentq tolerance ~1e-9; closed-form is exact. Allow 1e-6 slack.
        assert abs(cf_lo - gn_lo) < 1e-6, f"CI lo: closed-form={cf_lo}, generic={gn_lo}"
        assert abs(cf_hi - gn_hi) < 1e-6, f"CI hi: closed-form={cf_hi}, generic={gn_hi}"
