"""Property tests for the LRTOStatistic.

Invariants (from `docs/methods/lrto.md` Derivation):
  1. p in [0, 1] for all inputs.
  2. tau >= 0 (with equality at theta_MAP).
  3. p(theta_MAP) == 1 on closed-form NN+Normal.
  4. NN+Normal equivalence: lrto.pvalue == waldo.pvalue (atol 1e-12).
  5. NN+Normal equivalence: lrto.CI == waldo.CI (atol 1e-8).
  6. accepts_tilting True for every concrete TiltingScheme.
  7. Flat-prior limit: as sigma_0 -> infty, lrto.pvalue -> lrt.pvalue
     with O(1/sigma_0^2) convergence.
  8. H_0 uniformity on NN+Normal closed form (KS uniform, L3).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.lrt import LRTStatistic
from frasian.statistics.lrto import LRTOStatistic
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
class TestLRTOInvariants:
    @given(theta=_THETA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=80, deadline=None)
    def test_pvalue_in_unit_interval(self, theta, D, sigma, sigma0, mu0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        p = LRTOStatistic().pvalue(theta, np.asarray([D]), model, prior)
        assert 0.0 <= float(p) <= 1.0

    @given(theta=_THETA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=80, deadline=None)
    def test_tau_nonnegative(self, theta, D, sigma, sigma0, mu0):
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        tau = LRTOStatistic().evaluate(theta, np.asarray([D]), model, prior)
        assert float(tau) >= 0.0

    @given(D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=40, deadline=None)
    def test_pvalue_at_mode_equals_one(self, D, sigma, sigma0, mu0):
        """Invariant 3: p(theta_MAP) == 1 on NN+Normal closed form.

        theta_MAP = mu_n = w*D + (1-w)*mu0; at this point a=0 and
        Phi(b) + Phi(-b) = 1 identically.
        """
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        from frasian.models.normal_normal import posterior_params

        mu_n, _, _ = posterior_params(D, mu0, sigma, sigma0)
        p = float(LRTOStatistic().pvalue(float(mu_n), np.asarray([D]), model, prior))
        assert p == pytest.approx(1.0, abs=1e-12)

    @given(theta=_THETA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=80, deadline=None)
    def test_matches_waldo_pvalue_on_nn(self, theta, D, sigma, sigma0, mu0):
        """Invariant 4: NN+Normal closed-form equivalence with WALDO."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        data = np.asarray([D])
        p_lrto = float(LRTOStatistic().pvalue(theta, data, model, prior))
        p_waldo = float(WaldoStatistic().pvalue(theta, data, model, prior))
        assert p_lrto == pytest.approx(p_waldo, abs=1e-12)

    @given(alpha=_ALPHA, D=_D, sigma=_SIGMA, sigma0=_SIGMA0, mu0=_MU0)
    @settings(max_examples=40, deadline=None)
    def test_matches_waldo_ci_on_nn(self, alpha, D, sigma, sigma0, mu0):
        """Invariant 5: NN+Normal closed-form CI equivalence with WALDO."""
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        data = np.asarray([D])
        lo_lrto, hi_lrto = LRTOStatistic().confidence_interval(
            alpha, data, model, prior
        )
        lo_waldo, hi_waldo = WaldoStatistic().confidence_interval(
            alpha, data, model, prior
        )
        assert lo_lrto == pytest.approx(lo_waldo, abs=1e-8)
        assert hi_lrto == pytest.approx(hi_waldo, abs=1e-8)

    def test_accepts_all_tiltings(self):
        """Invariant 6: lrto is prior-aware; accepts every concrete tilting."""
        from frasian.tilting.ot import OTTilting
        from frasian.tilting.power_law import PowerLawTilting

        stat = LRTOStatistic()
        assert stat.accepts_tilting(IdentityTilting()) is True
        assert stat.accepts_tilting(PowerLawTilting()) is True
        assert stat.accepts_tilting(OTTilting()) is True

    @pytest.mark.parametrize("sigma0", [100.0, 1000.0, 10000.0])
    def test_flat_prior_limit_recovers_lrt(self, sigma0):
        """Invariant 7: as sigma_0 -> infty, p_lrto -> p_lrt with
        O(1/sigma_0^2) convergence (Derivation Step 6)."""
        sigma = 1.0
        D = 0.7
        theta0 = -0.2
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.5, scale=sigma0)
        data = np.asarray([D])
        p_lrto = float(LRTOStatistic().pvalue(theta0, data, model, prior))
        p_lrt = float(LRTStatistic().pvalue(theta0, data, model))
        # Derivation Step 6 numerical table: diff ~ 3.7 * (1/sigma_0^2).
        # Add a small floor (1e-12) for sigma_0 -> infty where float64
        # rounding kicks in. Allow 10x slack on the predicted constant.
        expected_tol = 10.0 * 3.7 / (sigma0**2) + 1e-12
        assert abs(p_lrto - p_lrt) < expected_tol, (
            f"sigma_0={sigma0}: |p_lrto - p_lrt|={abs(p_lrto - p_lrt):.3e} "
            f"exceeds expected tol {expected_tol:.3e}"
        )

    def test_wrong_map_triggers_warning(self):
        """Generic-path mode-finder protection: if the posterior's
        argmax disagrees with what `_find_theta_map` returns,
        `_generic_evaluate` warns and clamps. We force this by
        wrapping NormalDistribution and overriding `logpdf` so the
        true mode is OUTSIDE the support's bracket (the optimiser
        won't find it within `support()`).
        """
        import warnings

        class _PathologicalPosterior:
            """Mock unimodal posterior whose mode is at +1e6.

            `logpdf(theta)` is monotone increasing in theta, so the
            true argmax over R is at +infinity. `_find_theta_map`
            with unbounded NN support uses a 10*sigma bracket around
            the mean, which will return a non-maximiser inside that
            bracket.
            """

            def logpdf(self, theta):
                arr = np.asarray(theta, dtype=np.float64)
                # Monotone increasing — no interior maximum.
                return arr  # pdf ∝ exp(theta), unbounded.

            def mean(self):
                return 0.0

            def var(self):
                return 1.0

        class _MockModel:
            name = "mock"
            param_dim = 1

            def fingerprint(self):
                return ("mock",)

            def support(self):
                return (-np.inf, np.inf)

            def posterior(self, data, prior):
                return _PathologicalPosterior()

        model = _MockModel()
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.0])
        stat = LRTOStatistic(force_generic=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            # Evaluate at a theta far ABOVE the mode-finder's bracket
            # midpoint; logpdf there is larger than at the returned MAP.
            tau = float(stat.evaluate(50.0, data, model, prior))
        # tau should be clamped to 0 (because clamp triggered).
        assert tau == 0.0
        runtime_warnings = [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) >= 1
        assert "did not converge" in str(runtime_warnings[0].message)


@pytest.mark.L3
class TestLRTOUniformPvalueUnderH0:
    """Statistical-tier: closed-form NN+Normal p-values are exactly
    Uniform[0,1] under H_0 (Derivation Step 8).
    """

    def test_ks_uniform(self):
        rng = np.random.default_rng(42)
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
                float(LRTOStatistic().pvalue(theta_true, np.asarray([D]), model, prior))
                for D in Ds
            ]
        )
        ks_stat, ks_p = stats.kstest(ps, "uniform")
        assert ks_p > 0.01, f"KS p-value too low: {ks_p}, ks_stat={ks_stat}"
