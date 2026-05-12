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

    def test_reparameterisation_invariance_chain_rule(self):
        """Invariant 8 (Derivation Step 5): tau_Score is exactly
        reparam-invariant at theta_0 under phi = g(theta) with
        g'(theta_0) != 0. Exercises the chain rule via the
        generic JAX-grad path so the test is NOT tautological
        (skeptic finding #1).

        Reparam phi = 2*theta keeps the data fixed; the likelihood
        in phi-coordinates is `log L_phi(phi) = log L_theta(phi/2)`,
        which we compose explicitly. Then:
          U_phi(phi_0)        = (1/2) U_theta(theta_0)
          I_phi(phi_0)        = (1/4) I_theta(theta_0)
          tau_phi             = [U_phi]^2 / I_phi
                              = [(1/2) U_theta]^2 / [(1/4) I_theta]
                              = U_theta^2 / I_theta = tau_theta

        We build a `_PhiReparamModel` that wraps NormalNormalModel
        but maps theta -> phi/2 inside `likelihood` and
        `fisher_information`. Routing through
        `force_generic=True` so the JAX-grad chain rule is
        actually exercised.
        """
        sigma = 1.0
        D = 0.7
        theta_0 = 0.3

        # Original model: D ~ N(theta, sigma^2).
        model = NormalNormalModel(sigma=sigma)
        tau_theta = float(
            ScoreStatistic(force_generic=True).evaluate(
                theta_0, np.asarray([D]), model
            )
        )

        # Phi-reparameterised model: phi = 2*theta, so theta = phi/2
        # and log L_phi(phi; D) = log L_theta(phi/2; D).
        class _PhiLik:
            def __init__(self, inner_lik):
                self._inner = inner_lik

            def __call__(self, phi):
                return self._inner(phi / 2.0)

            def loglik(self, phi):
                # JAX-traceable: divide phi by 2 before passing to the
                # inner Normal loglik. jax.grad(_PhiLik.loglik) =
                # 0.5 * inner.loglik'(phi/2) by the chain rule.
                return self._inner.loglik(phi / 2.0)

        class _PhiModel:
            name = "nn_phi_reparam"
            param_dim = 1
            # The inner NN provides sigma; we use the same sigma in phi.
            _inner: NormalNormalModel

            def __init__(self, inner):
                self._inner = inner

            def fingerprint(self):
                return ("nn_phi_reparam", self._inner.sigma)

            def likelihood(self, data):
                return _PhiLik(self._inner.likelihood(data))

            def fisher_information(self, phi):
                # I_phi(phi) = I_theta(phi/2) * (1/2)^2 = I_theta / 4.
                return self._inner.fisher_information(phi / 2.0) / 4.0

            def support(self):
                return self._inner.support()

            def mle(self, data):
                # MLE in phi-coords: 2 * theta_MLE.
                return 2.0 * self._inner.mle(data)

            def sample_data(self, phi, rng, n):
                return self._inner.sample_data(phi / 2.0, rng, n)

        model_phi = _PhiModel(model)
        # _PhiModel doesn't have NN fingerprint, so this routes through
        # the generic path automatically (no need for force_generic).
        tau_phi = float(
            ScoreStatistic().evaluate(2.0 * theta_0, np.asarray([D]), model_phi)
        )
        assert tau_theta == pytest.approx(tau_phi, abs=1e-12), (
            f"tau_theta={tau_theta}, tau_phi={tau_phi}"
        )

    def test_evaluate_array_input(self):
        """Skeptic finding #3: the array-input vmap branch of
        `_generic_evaluate` had no coverage. Verify it gives the
        same answer elementwise as scalar evaluation across a
        small theta grid.
        """
        model = NormalNormalModel(sigma=1.0)
        data = np.asarray([0.5])
        thetas = np.linspace(-1.0, 1.5, 11)
        # Closed-form path (vectorised via numpy).
        tau_arr_cf = np.asarray(
            ScoreStatistic().evaluate(thetas, data, model), dtype=np.float64
        )
        # Generic path (vmap of jax.grad).
        tau_arr_gn = np.asarray(
            ScoreStatistic(force_generic=True).evaluate(thetas, data, model),
            dtype=np.float64,
        )
        # Element-wise scalar reference: closed form at each theta.
        tau_scalar = np.array(
            [
                float(ScoreStatistic().evaluate(float(t), data, model))
                for t in thetas
            ]
        )
        np.testing.assert_allclose(tau_arr_cf, tau_scalar, atol=1e-12)
        np.testing.assert_allclose(tau_arr_gn, tau_scalar, atol=1e-12)


@pytest.mark.L3
class TestScoreUniformPvalueUnderH0:
    """Statistical-tier: NN closed-form p-values are exactly
    Uniform[0,1] under H_0 (Derivation Step 4)."""

    def test_ks_uniform(self):
        # n=10000 matches the brief's Derivation Step 4 / Invariant 6
        # numerical-check reference (KS = 0.0093, p = 0.35); skeptic
        # finding #2 (harmonised with brief).
        rng = np.random.default_rng(2026)
        sigma = 1.7
        theta_true = 0.4
        n = 10000
        Ds = rng.normal(loc=theta_true, scale=sigma, size=n)
        model = NormalNormalModel(sigma=sigma)
        ps = np.array(
            [
                float(ScoreStatistic().pvalue(theta_true, np.asarray([D]), model))
                for D in Ds
            ]
        )
        ks_stat, ks_p = stats.kstest(ps, "uniform")
        # Threshold 1e-3: flag only true miscalibration, not natural
        # finite-sample fluctuations. At n=10000 the per-test FPR is
        # 0.1% — adequate for a property test that runs once per CI.
        assert ks_p > 1e-3, f"KS p-value too low: {ks_p}, ks_stat={ks_stat}"
