"""Property tests for the ScoreStatistic.

Invariants (from `docs/methods/score.md` Derivation):
  1. p in [0, 1] for all inputs.
  2. tau >= 0 (equality iff U = 0 i.e. theta at the score's zero).
  3. NN n=1 equivalence: score.pvalue == wald.pvalue == lrt.pvalue
     (atol 1e-12 closed form).
  4. NN n=1 CI equivalence with wald/lrt (atol 1e-8).
  5. Mode property: p(Dbar) == 1 exactly (since U(Dbar) = 0).
  6. H_0 calibration: KS uniformity on NN under H_0 (L3).
  7. accepts_tilting: True only for IdentityTilting.
  8. Reparameterisation invariance at theta_0.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.lrt import LRTStatistic
from frasian.statistics.score import ScoreStatistic
from frasian.statistics.wald import WaldStatistic
from frasian.tilting.identity import IdentityTilting

_THETA = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_D = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
_SIGMA = st.floats(min_value=0.2, max_value=3.0, allow_nan=False)
_ALPHA = st.floats(min_value=0.01, max_value=0.5, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestScoreInvariants:
    @given(theta=_THETA, D=_D, sigma=_SIGMA)
    @settings(max_examples=80, deadline=None)
    def test_pvalue_in_unit_interval(self, theta, D, sigma):
        model = NormalNormalModel(sigma=sigma)
        p = ScoreStatistic().pvalue(theta, np.asarray([D]), model)
        assert 0.0 <= float(p) <= 1.0

    @given(theta=_THETA, D=_D, sigma=_SIGMA)
    @settings(max_examples=80, deadline=None)
    def test_tau_nonnegative(self, theta, D, sigma):
        model = NormalNormalModel(sigma=sigma)
        tau = ScoreStatistic().evaluate(theta, np.asarray([D]), model)
        assert float(tau) >= 0.0

    @given(theta=_THETA, D=_D, sigma=_SIGMA)
    @settings(max_examples=80, deadline=None)
    def test_matches_wald_pvalue_on_nn(self, theta, D, sigma):
        """Invariant 3: trinity collapse on NN n=1 closed form."""
        model = NormalNormalModel(sigma=sigma)
        data = np.asarray([D])
        p_score = float(ScoreStatistic().pvalue(theta, data, model))
        p_wald = float(WaldStatistic().pvalue(theta, data, model))
        p_lrt = float(LRTStatistic().pvalue(theta, data, model))
        assert p_score == pytest.approx(p_wald, abs=1e-12)
        assert p_score == pytest.approx(p_lrt, abs=1e-12)

    @given(alpha=_ALPHA, D=_D, sigma=_SIGMA)
    @settings(max_examples=40, deadline=None)
    def test_matches_wald_ci_on_nn(self, alpha, D, sigma):
        """Invariant 4: CI collapse on NN n=1 closed form."""
        model = NormalNormalModel(sigma=sigma)
        data = np.asarray([D])
        ci_score = ScoreStatistic().confidence_interval(alpha, data, model)
        ci_wald = WaldStatistic().confidence_interval(alpha, data, model)
        ci_lrt = LRTStatistic().confidence_interval(alpha, data, model)
        for s, w, l in zip(ci_score, ci_wald, ci_lrt):
            assert s == pytest.approx(w, abs=1e-8)
            assert s == pytest.approx(l, abs=1e-8)

    @given(D=_D, sigma=_SIGMA)
    @settings(max_examples=40, deadline=None)
    def test_pvalue_at_mle_equals_one(self, D, sigma):
        """Invariant 5: U(Dbar) = 0 => tau = 0 => p = 1."""
        model = NormalNormalModel(sigma=sigma)
        # On NN n=1, MLE = D.
        p = float(ScoreStatistic().pvalue(D, np.asarray([D]), model))
        assert p == pytest.approx(1.0, abs=1e-12)

    def test_accepts_only_identity(self):
        """Invariant 7: score is prior-IGNORING by construction;
        only IdentityTilting is admissible."""
        from frasian.tilting.ot import OTTilting
        from frasian.tilting.power_law import PowerLawTilting

        stat = ScoreStatistic()
        assert stat.accepts_tilting(IdentityTilting()) is True
        assert stat.accepts_tilting(PowerLawTilting()) is False
        assert stat.accepts_tilting(OTTilting()) is False

    def test_reparameterisation_invariance(self):
        """Invariant 8 (Derivation Step 5): tau_Score is exactly
        reparam-invariant at theta_0 under phi = g(theta) with
        g'(theta_0) != 0.

        Synthetic check: phi = 2*theta corresponds to a Gaussian
        likelihood `D' = 2*D ~ N(phi, (2 sigma)^2)`. The score at
        `phi_0 = 2 theta_0` under this reparameterised model should
        equal the score at `theta_0` under the original model.

        Equivalently, the p-value should match (since both have
        chi^2_1 calibration and the same tau).
        """
        sigma = 1.0
        D = 0.7
        theta_0 = 0.3
        model = NormalNormalModel(sigma=sigma)
        tau_theta = float(
            ScoreStatistic().evaluate(theta_0, np.asarray([D]), model)
        )
        # Reparam: phi = 2*theta. Then D' = 2*D, sigma' = 2*sigma, theta'_0 = 2*theta_0.
        model_phi = NormalNormalModel(sigma=2.0 * sigma)
        tau_phi = float(
            ScoreStatistic().evaluate(2.0 * theta_0, np.asarray([2.0 * D]), model_phi)
        )
        assert tau_theta == pytest.approx(tau_phi, abs=1e-12)


@pytest.mark.L3
class TestScoreUniformPvalueUnderH0:
    """Statistical-tier: NN closed-form p-values are exactly
    Uniform[0,1] under H_0 (Derivation Step 4)."""

    def test_ks_uniform(self):
        rng = np.random.default_rng(2026)
        sigma = 1.7
        theta_true = 0.4
        n = 2000
        Ds = rng.normal(loc=theta_true, scale=sigma, size=n)
        model = NormalNormalModel(sigma=sigma)
        ps = np.array(
            [
                float(ScoreStatistic().pvalue(theta_true, np.asarray([D]), model))
                for D in Ds
            ]
        )
        ks_stat, ks_p = stats.kstest(ps, "uniform")
        assert ks_p > 0.01, f"KS p-value too low: {ks_p}, ks_stat={ks_stat}"
