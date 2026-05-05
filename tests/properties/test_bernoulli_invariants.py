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
class TestBernoulliPairingsRaise:
    """The Normal-only methods must raise clear NotImplementedError when
    paired with BernoulliModel, not silently produce nonsense."""

    def test_wald_pvalue_raises(self):
        with pytest.raises(NotImplementedError, match="NormalNormalModel"):
            WaldStatistic().pvalue(0.5, np.array([1.0, 0.0]), BernoulliModel())

    def test_wald_ci_raises(self):
        with pytest.raises(NotImplementedError, match="NormalNormalModel"):
            WaldStatistic().confidence_interval(
                0.05,
                np.array([1.0, 0.0]),
                BernoulliModel(),
            )

    def test_waldo_pvalue_raises(self):
        prior = BetaDistribution(alpha=2.0, beta=2.0)
        with pytest.raises(NotImplementedError, match="NormalNormalModel"):
            WaldoStatistic().pvalue(0.5, np.array([1.0, 0.0]), BernoulliModel(), prior)

    def test_power_law_tilt_raises(self):
        prior = BetaDistribution(alpha=2.0, beta=2.0)
        model = BernoulliModel()
        post = model.posterior(np.array([1.0, 0.0]), prior)
        from frasian.models.distributions import BernoulliLikelihood

        lik = BernoulliLikelihood(n_success=1, n_total=2)
        with pytest.raises(NotImplementedError):
            PowerLawTilting().tilt(post, prior, lik, 0.0)
