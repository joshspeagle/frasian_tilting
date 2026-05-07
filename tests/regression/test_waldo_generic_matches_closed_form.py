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


@pytest.mark.L2
class TestWaldoNGreaterThanOneRoutesGeneric:
    """Public WALDO dispatch must NOT take the closed-form NN path for n>1.

    The closed-form posterior in NormalNormalModel collapses ``data`` to its
    mean and uses sigma^2 (NOT sigma^2/n); the generic MC reference uses
    n_obs = data.size correctly. Routing n>1 to closed form silently
    produces a too-wide CI by a factor of sqrt(n). We pin both behaviours:
    the public dispatcher routes n>1 to the generic path, and the closed
    form on n=1 still hits the analytic codepath.
    """

    def _fixtures(self):
        sigma = 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        return model, prior

    def test_pvalue_n1_takes_closed_form_dispatch(self):
        from frasian.statistics.waldo import _is_normal_normal_n1

        model, prior = self._fixtures()
        assert _is_normal_normal_n1(model, prior, np.asarray([0.5])) is True

    def test_pvalue_ngt1_routes_to_generic(self):
        from frasian.statistics.waldo import _is_normal_normal_n1

        model, prior = self._fixtures()
        # n=8 NN data: must NOT take the closed form path.
        assert _is_normal_normal_n1(model, prior, np.asarray([0.5] * 8)) is False

    def test_non_normal_pair_routes_to_generic(self):
        from frasian.models.bernoulli import BernoulliModel
        from frasian.models.distributions import BetaDistribution
        from frasian.statistics.waldo import _is_normal_normal_n1

        model = BernoulliModel()
        prior = BetaDistribution(alpha=2.0, beta=2.0)
        assert _is_normal_normal_n1(model, prior, np.asarray([1.0])) is False


@pytest.mark.L2
class TestWaldoDegenerateVarianceFloorAgreement:
    """The variance-floor for `_generic_evaluate` and `_generic_mc_reference`
    must agree — both produce NaN on degenerate posterior variance, and
    `_generic_pvalue` raises rather than returning a fake CI.

    Pre-fix: `_generic_evaluate` floored at 1e-300 (huge t_obs) while
    `_generic_mc_reference` set t=0 (always-not-extreme), collapsing the
    empirical p to ~1/(n_mc+1) regardless of θ.
    """

    def test_evaluate_returns_nan_on_zero_variance(self):
        """A degenerate posterior (var=0) must produce NaN, not a 1e-300-clamped huge value."""
        # Construct a synthetic distribution-like posterior whose var() == 0.
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _DegenerateModel:
            def fingerprint(self):  # pragma: no cover - test fixture
                return ("_degenerate",)

            def support(self):  # pragma: no cover
                return (-np.inf, np.inf)

            def sample_data(self, theta, rng, n):  # pragma: no cover
                return np.full(n, float(theta))

            def posterior(self, data, prior):
                return _PointMass(loc=0.0)

        @dataclass(frozen=True)
        class _PointMass:
            loc: float

            def mean(self):
                return self.loc

            def var(self):
                return 0.0

        @dataclass(frozen=True)
        class _NoOpPrior:
            def fingerprint(self):  # pragma: no cover
                return ("_noop",)

        stat = WaldoStatistic(n_mc=10, seed=0)
        t = float(np.asarray(stat._generic_evaluate(0.5, np.asarray([0.0]), _DegenerateModel(), _NoOpPrior())))
        assert np.isnan(t), f"expected NaN on var=0 posterior, got {t!r}"

    def test_pvalue_raises_when_observed_variance_degenerate(self):
        """If the observed-data posterior is degenerate, _generic_pvalue must raise."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _DegenerateModel:
            def fingerprint(self):  # pragma: no cover
                return ("_degenerate",)

            def support(self):  # pragma: no cover
                return (-np.inf, np.inf)

            def sample_data(self, theta, rng, n):  # pragma: no cover
                # MC draws give a non-degenerate posterior so we isolate the
                # `t_obs is NaN` failure path.
                return np.asarray(rng.normal(loc=float(theta), scale=1.0, size=n), dtype=np.float64)

            def posterior(self, data, prior):
                # Observed-data posterior is degenerate; MC posteriors below
                # would (in principle) be non-degenerate — but we never get
                # there because `_generic_pvalue` short-circuits on the
                # degenerate observed t_obs.
                return _PointMass(loc=0.0)

        @dataclass(frozen=True)
        class _PointMass:
            loc: float

            def mean(self):
                return self.loc

            def var(self):
                return 0.0

        @dataclass(frozen=True)
        class _NoOpPrior:
            def fingerprint(self):  # pragma: no cover
                return ("_noop",)

        stat = WaldoStatistic(n_mc=10, seed=0)
        with pytest.raises(ValueError, match=r"degenerate"):
            stat._generic_pvalue(0.5, np.asarray([0.0]), _DegenerateModel(), _NoOpPrior())
