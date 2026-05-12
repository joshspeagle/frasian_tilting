"""Property tests for the ScoreoStatistic.

Invariants (from `docs/methods/scoreo.md` Derivation):
  1. p in [0, 1].
  2. tau >= 0 (with equality at U_post = 0).
  3. NN+Normal Bayesian-trinity equivalence:
     scoreo.pvalue == waldo.pvalue == lrto.pvalue (atol 1e-12).
  4. CI equivalence with waldo/lrto on NN (atol 1e-8).
  5. Mode property: p(mu_n) == 1 exactly.
  6. accepts_tilting True for every TiltingScheme.
  7. Flat-prior limit: scoreo -> score as sigma_0 -> infty
     with O(1/sigma_0^2) convergence.
  8. H_0 uniformity on NN closed form (KS, L3).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel, posterior_params
from frasian.statistics.lrto import LRTOStatistic
from frasian.statistics.score import ScoreStatistic
from frasian.statistics.scoreo import ScoreoStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.identity import IdentityTilting

_THETA = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_D = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_SIGMA0 = st.floats(min_value=0.3, max_value=5.0, allow_nan=False)
_MU0 = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False)
_ALPHA = st.floats(min_value=0.01, max_value=0.5, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestScoreoInvariants:
    @given(theta=_THETA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=80, deadline=None)
    def test_pvalue_in_unit_interval(self, theta, D, sigma, sigma0, mu0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        p = ScoreoStatistic().pvalue(theta, np.asarray([D]), model, prior)
        assert 0.0 <= float(p) <= 1.0

    @given(theta=_THETA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=80, deadline=None)
    def test_tau_nonnegative(self, theta, D, sigma, sigma0, mu0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        tau = ScoreoStatistic().evaluate(theta, np.asarray([D]), model, prior)
        assert float(tau) >= 0.0

    @given(theta=_THETA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=80, deadline=None)
    def test_bayesian_trinity_pvalue(self, theta, D, sigma, sigma0, mu0):
        """Invariant 3: scoreo == waldo == lrto on NN closed form."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        data = np.asarray([D])
        p_scoreo = float(ScoreoStatistic().pvalue(theta, data, model, prior))
        p_waldo = float(WaldoStatistic().pvalue(theta, data, model, prior))
        p_lrto = float(LRTOStatistic().pvalue(theta, data, model, prior))
        assert p_scoreo == pytest.approx(p_waldo, abs=1e-12)
        assert p_scoreo == pytest.approx(p_lrto, abs=1e-12)

    @given(alpha=_ALPHA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=40, deadline=None)
    def test_bayesian_trinity_ci(self, alpha, D, sigma, sigma0, mu0):
        """Invariant 5: CI agreement across the Bayesian trinity."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        data = np.asarray([D])
        ci_scoreo = ScoreoStatistic().confidence_interval(alpha, data, model, prior)
        ci_waldo = WaldoStatistic().confidence_interval(alpha, data, model, prior)
        ci_lrto = LRTOStatistic().confidence_interval(alpha, data, model, prior)
        for s, w, l in zip(ci_scoreo, ci_waldo, ci_lrto):
            assert s == pytest.approx(w, abs=1e-8)
            assert s == pytest.approx(l, abs=1e-8)

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=40, deadline=None)
    def test_pvalue_at_map_equals_one(self, D, sigma, sigma0, mu0):
        """Invariant 6: p(mu_n) == 1 on NN closed form, because
        U_post(mu_n) = 0 ⇒ tau = 0 ⇒ p = 1."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        p = float(ScoreoStatistic().pvalue(float(mu_n), np.asarray([D]), model, prior))
        assert p == pytest.approx(1.0, abs=1e-12)

    def test_accepts_all_tiltings(self):
        """Invariant 7: scoreo is prior-aware; accepts every TiltingScheme."""
        from frasian.tilting.ot import OTTilting
        from frasian.tilting.power_law import PowerLawTilting

        stat = ScoreoStatistic()
        assert stat.accepts_tilting(IdentityTilting()) is True
        assert stat.accepts_tilting(PowerLawTilting()) is True
        assert stat.accepts_tilting(OTTilting()) is True

    @pytest.mark.parametrize("sigma0", [100.0, 1000.0, 10000.0])
    def test_flat_prior_limit_recovers_score(self, sigma0):
        """Invariant 8: as sigma_0 -> infty, scoreo -> score with
        O(1/sigma_0^2) convergence (Derivation Step 6).
        """
        sigma = 1.0
        D = 0.4
        theta_0 = 0.7  # matches Derivation Step 6 reference inputs.
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        data = np.asarray([D])
        p_scoreo = float(ScoreoStatistic().pvalue(theta_0, data, model, prior))
        p_score = float(ScoreStatistic().pvalue(theta_0, data, model))
        # Derivation Step 6 reference: gap * sigma_0^2 -> 0.33.
        # Allow 30x slack on the constant.
        expected_tol = 30.0 * 0.33 / (sigma0**2) + 1e-12
        assert abs(p_scoreo - p_score) < expected_tol, (
            f"sigma_0={sigma0}: |p_scoreo - p_score|={abs(p_scoreo - p_score):.3e} "
            f"exceeds expected tol {expected_tol:.3e}"
        )


@pytest.mark.L3
class TestScoreoUniformPvalueUnderH0:
    """Statistical-tier: NN closed-form p-values are exactly
    Uniform[0,1] under H_0 (Derivation Step 7)."""

    def test_ks_uniform(self):
        rng = np.random.default_rng(11)
        sigma = 1.5
        sigma0 = 0.7
        mu0 = 0.5
        theta_true = 1.2
        n = 2000
        Ds = rng.normal(loc=theta_true, scale=sigma, size=n)
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        ps = np.array(
            [
                float(ScoreoStatistic().pvalue(theta_true, np.asarray([D]), model, prior))
                for D in Ds
            ]
        )
        ks_stat, ks_p = stats.kstest(ps, "uniform")
        # Threshold 1e-3 — same convention as score's H_0 uniformity test.
        assert ks_p > 1e-3, f"KS p-value too low: {ks_p}, ks_stat={ks_stat}"
