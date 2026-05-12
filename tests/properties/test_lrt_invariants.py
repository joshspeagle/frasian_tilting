"""Property tests for the LRTStatistic.

Invariants (from `docs/methods/lrt.md` Derivation):
  1. p in [0, 1] for all inputs.
  2. tau_LRT >= 0 (MLE maximises the loglikelihood).
  3. p(theta_hat) == 1 (mode property).
  4. NN equivalence: lrt.pvalue == wald.pvalue elementwise (atol 1e-12).
  5. NN equivalence: lrt.CI == wald.CI elementwise (atol 1e-8).
  6. Exact chi^2_1 calibration under H0 on NN (statistical L3).
  7. Monotonicity: p(theta) strictly decreasing in |theta - D|.
  8. accepts_tilting: only `identity` returns True.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.lrt import LRTStatistic
from frasian.statistics.wald import WaldStatistic
from frasian.tilting.identity import IdentityTilting

_THETA = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_D = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_ALPHA = st.floats(min_value=0.01, max_value=0.5, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestLRTInvariants:
    @given(theta=_THETA, D=_D, sigma=_SIGMA)
    @settings(max_examples=100, deadline=None)
    def test_pvalue_in_unit_interval(self, theta, D, sigma):
        model = NormalNormalModel(sigma=sigma)
        p = LRTStatistic().pvalue(theta, np.asarray([D]), model)
        assert 0.0 <= float(p) <= 1.0

    @given(theta=_THETA, D=_D, sigma=_SIGMA)
    @settings(max_examples=100, deadline=None)
    def test_tau_nonnegative(self, theta, D, sigma):
        model = NormalNormalModel(sigma=sigma)
        tau = LRTStatistic().evaluate(theta, np.asarray([D]), model)
        assert float(tau) >= 0.0

    @given(D=_D, sigma=_SIGMA)
    @settings(max_examples=50, deadline=None)
    def test_pvalue_at_mle_equals_one(self, D, sigma):
        """For NN the MLE equals D, so p(D) == 1."""
        model = NormalNormalModel(sigma=sigma)
        p = float(LRTStatistic().pvalue(D, np.asarray([D]), model))
        assert p == pytest.approx(1.0, abs=1e-12)

    @given(theta=_THETA, D=_D, sigma=_SIGMA)
    @settings(max_examples=100, deadline=None)
    def test_matches_wald_pvalue_on_normal_normal(self, theta, D, sigma):
        """Invariant 4: NN exact equivalence with Wald p-value (atol 1e-12)."""
        model = NormalNormalModel(sigma=sigma)
        data = np.asarray([D])
        p_lrt = float(LRTStatistic().pvalue(theta, data, model))
        p_wald = float(WaldStatistic().pvalue(theta, data, model))
        assert p_lrt == pytest.approx(p_wald, abs=1e-12)

    @given(alpha=_ALPHA, D=_D, sigma=_SIGMA)
    @settings(max_examples=50, deadline=None)
    def test_matches_wald_ci_on_normal_normal(self, alpha, D, sigma):
        """Invariant 5: NN exact equivalence with Wald CI (atol 1e-8)."""
        model = NormalNormalModel(sigma=sigma)
        data = np.asarray([D])
        lo_lrt, hi_lrt = LRTStatistic().confidence_interval(alpha, data, model)
        lo_wald, hi_wald = WaldStatistic().confidence_interval(alpha, data, model)
        assert lo_lrt == pytest.approx(lo_wald, abs=1e-8)
        assert hi_lrt == pytest.approx(hi_wald, abs=1e-8)

    @given(D=_D, sigma=_SIGMA)
    @settings(max_examples=50, deadline=None)
    def test_pvalue_monotone_in_distance(self, D, sigma):
        """Invariant 7: p(theta) strictly decreasing in |theta - D| on NN."""
        model = NormalNormalModel(sigma=sigma)
        data = np.asarray([D])
        # Three points: theta1 closer to D than theta2.
        theta1 = D + 0.3 * sigma
        theta2 = D + 1.5 * sigma
        p1 = float(LRTStatistic().pvalue(theta1, data, model))
        p2 = float(LRTStatistic().pvalue(theta2, data, model))
        assert p1 > p2

    def test_accepts_only_identity_tilting(self):
        """Invariant 8: only `identity` tilting accepted (LRT ignores prior).

        Mirrors `wald`'s contract. We hand-pick a representative set rather
        than iterating the registry, since stub tiltings may not be
        instantiable with default args.
        """
        from frasian.tilting.power_law import PowerLawTilting

        stat = LRTStatistic()
        assert stat.accepts_tilting(IdentityTilting()) is True
        assert stat.accepts_tilting(PowerLawTilting()) is False
        # Anything without a `name` attribute is also rejected.
        class _Dummy:
            pass
        assert stat.accepts_tilting(_Dummy()) is False


@pytest.mark.L3
class TestLRTUniformPvalueUnderH0:
    """Statistical-tier: LRT p-values are Uniform[0,1] under H0 on NN.

    By Derivation Step 4 (`docs/methods/lrt.md`) this is exact on NN, not
    asymptotic. KS test threshold: 0.01 (matches Wald's analogous test).
    """

    def test_ks_uniform(self):
        rng = np.random.default_rng(42)
        sigma = 1.0
        theta_true = 0.7
        n = 5000
        Ds = rng.normal(loc=theta_true, scale=sigma, size=n)
        model = NormalNormalModel(sigma=sigma)
        ps = np.array(
            [float(LRTStatistic().pvalue(theta_true, np.asarray([D]), model)) for D in Ds]
        )
        ks_stat, ks_p = stats.kstest(ps, "uniform")
        assert ks_p > 0.01, f"KS p-value too low: {ks_p}, ks_stat={ks_stat}"

    def test_tau_chi2_one(self):
        """tau_LRT ~ chi^2_1 exactly under H0 on NN (Derivation Step 4)."""
        rng = np.random.default_rng(7)
        sigma = 1.0
        theta_true = -0.3
        n = 5000
        Ds = rng.normal(loc=theta_true, scale=sigma, size=n)
        model = NormalNormalModel(sigma=sigma)
        taus = np.array(
            [float(LRTStatistic().evaluate(theta_true, np.asarray([D]), model)) for D in Ds]
        )
        ks_stat, ks_p = stats.kstest(taus, "chi2", args=(1,))
        assert ks_p > 0.01, f"chi^2_1 KS p-value too low: {ks_p}, ks_stat={ks_stat}"
