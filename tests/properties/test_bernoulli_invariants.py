"""Property tests for BernoulliModel — `Model` protocol invariants.

These exercise the same generic invariants that
`tests/properties/test_normal_distribution.py` exercises for the Normal
model, demonstrating that the Model protocol accommodates a non-Normal
sampling distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import BetaDistribution
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.power_law import PowerLawTilting

_ALPHA = st.floats(min_value=0.5, max_value=10.0, allow_nan=False)
_BETA = st.floats(min_value=0.5, max_value=10.0, allow_nan=False)
_THETA = st.floats(min_value=0.05, max_value=0.95, allow_nan=False)


@pytest.mark.L1
@pytest.mark.properties
class TestBernoulliModelInvariants:
    @given(theta=_THETA)
    @settings(max_examples=20, deadline=None)
    def test_mle_is_consistent(self, theta):
        """MLE concentrates at theta_true under increasing n."""
        model = BernoulliModel()
        rng = np.random.default_rng(0)
        samples = model.sample_data(theta, rng, 5000)
        np.testing.assert_allclose(model.mle(samples), theta, atol=0.03)

    @given(alpha=_ALPHA, beta=_BETA, n_success=st.integers(0, 20))
    @settings(max_examples=30, deadline=None)
    def test_posterior_mean_between_prior_and_mle(self, alpha, beta, n_success):
        """The Beta posterior mean lies between prior mean and MLE."""
        n_total = 20
        model = BernoulliModel()
        prior = BetaDistribution(alpha=alpha, beta=beta)
        data = np.concatenate([np.ones(n_success), np.zeros(n_total - n_success)])
        post = model.posterior(data, prior)
        prior_mean = prior.mean()
        mle = float(model.mle(data))
        post_mean = post.mean()
        lo, hi = (prior_mean, mle) if prior_mean <= mle else (mle, prior_mean)
        assert (
            lo - 1e-9 <= post_mean <= hi + 1e-9
        ), f"posterior mean {post_mean} outside [{lo}, {hi}]"

    @given(alpha=_ALPHA, beta=_BETA, theta=_THETA)
    @settings(max_examples=10, deadline=None)
    def test_posterior_var_decays_with_n(self, alpha, beta, theta):
        """As n grows at fixed theta_true, posterior var -> 0.

        (Note: posterior var is NOT generally <= prior var when prior
        and MLE disagree — the brief was wrong on that. The correct
        asymptotic statement is the one tested here.)
        """
        model = BernoulliModel()
        prior = BetaDistribution(alpha=alpha, beta=beta)
        rng = np.random.default_rng(0)
        d_small = model.sample_data(theta, rng, 50)
        d_big = model.sample_data(theta, rng, 500)
        v_small = model.posterior(d_small, prior).var()
        v_big = model.posterior(d_big, prior).var()
        assert v_big < v_small, (
            f"variance did not decrease: small({len(d_small)})={v_small}, "
            f"big({len(d_big)})={v_big}"
        )


@pytest.mark.L1
@pytest.mark.properties
class TestBernoulliPairingsGenericVsRaise:
    """Phase-2 generic numerical paths now light up Wald (and WALDO) on
    BernoulliModel; PowerLawTilting still raises (Phase-3 work).

    Wald: tau = (mle - theta)^2 * I(theta), chi^2_1 calibration. Works
    against any Model implementing `mle` and `fisher_information`.
    WALDO: t = (mu_post - theta)^2 / sigma_post^2 with MC reference
    distribution under H0 sampled from `model.sample_data(theta_0, ...)`.
    """

    def test_wald_pvalue_runs_generic_on_bernoulli(self):
        p = WaldStatistic().pvalue(0.5, np.array([1.0, 0.0, 1.0, 1.0]), BernoulliModel())
        p_f = float(np.asarray(p))
        assert 0.0 <= p_f <= 1.0
        assert np.isfinite(p_f)

    def test_wald_ci_runs_generic_on_bernoulli(self):
        lo, hi = WaldStatistic().confidence_interval(
            0.05,
            np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
            BernoulliModel(),
        )
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < hi
        # MLE is 4/6 ≈ 0.667; CI must contain it.
        assert lo <= 4.0 / 6.0 <= hi

    def test_waldo_pvalue_runs_generic_on_bernoulli(self):
        """WALDO generic path: t = (mu_post - theta)^2 / sigma_post^2,
        MC reference under H_0 sampled via model.sample_data."""
        prior = BetaDistribution(alpha=2.0, beta=2.0)
        stat = WaldoStatistic(n_mc=200, seed=0)
        p = stat.pvalue(0.5, np.array([1.0, 0.0, 1.0, 1.0]), BernoulliModel(), prior)
        p_f = float(np.asarray(p))
        assert 0.0 < p_f <= 1.0  # +1 smoothing makes p strictly > 0
        assert np.isfinite(p_f)

    def test_waldo_ci_runs_generic_on_bernoulli(self):
        prior = BetaDistribution(alpha=2.0, beta=2.0)
        stat = WaldoStatistic(n_mc=200, seed=0)
        lo, hi = stat.confidence_interval(
            0.05,
            np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
            BernoulliModel(),
            prior,
        )
        assert np.isfinite(lo) and np.isfinite(hi)
        assert 0.0 <= lo < hi <= 1.0  # CI lives on the parameter support [0, 1]

    def test_power_law_tilt_runs_generic_on_bernoulli(self):
        """Phase 3b: PowerLawTilting.tilt() now runs generically on
        non-Normal models via the numerical `_generic_tilt` path
        (`log L + (1-eta) * log pi`, normalised on a theta-grid).
        The closed-form Theorem 6 path stays the fast path on
        (NormalDistribution, NormalDistribution, GaussianLikelihood)
        triples; everything else routes through the GridDistribution
        wrapper.

        Previously raised `NotImplementedError`. Now returns a
        `GridDistribution` with finite moments. End-to-end CI inversion
        through `tilted_pvalue` is Phase 3c (not yet wired).
        """
        from frasian.models.distributions import BernoulliLikelihood
        from frasian.tilting._grid_distribution import GridDistribution

        prior = BetaDistribution(alpha=2.0, beta=2.0)
        model = BernoulliModel()
        post = model.posterior(np.array([1.0, 0.0]), prior)
        lik = BernoulliLikelihood(n_success=1, n_total=2)
        tilted = PowerLawTilting().tilt(post, prior, lik, 0.0)
        assert isinstance(tilted, GridDistribution)
        m = tilted.mean()
        v = tilted.var()
        assert 0.0 <= m <= 1.0
        assert v > 0.0
        # Cross-check: at eta=0 the tilted distribution recovers the posterior.
        assert abs(m - post.mean()) < 5e-3
        assert abs(v - post.var()) < 5e-3
