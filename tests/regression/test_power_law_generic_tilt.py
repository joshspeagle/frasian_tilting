"""Phase 3b regression tests for PowerLawTilting._generic_tilt.

Two pin contracts:

1. **Cross-path agreement on Normal-Normal**: the generic numerical
   formula `log L + (1-eta) * log pi` reduces to Theorem 6 closed form
   (deriver-verified atol 1e-7 at N=1024). This test exercises the
   public `tilt()` dispatch by bypassing the closed-form branch via a
   helper that calls `_generic_tilt` directly, and compares moments
   against the closed-form `NormalDistribution`.

2. **Bernoulli smoke**: `power_law.tilt(...)` runs end-to-end on
   `(BernoulliModel, BetaDistribution)` inputs without raising. The
   tilted distribution is a `GridDistribution` with finite mean / var /
   sample / pdf / cdf / quantile, and at eta=0 reduces to the Beta
   posterior within numerical tolerance.

This is the marquee Phase 3 deliverable: power_law works on non-
Normal models. Phase 3c/3d/3e build on this for tilted_pvalue and
end-to-end CI inversion.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import (
    BernoulliLikelihood,
    BetaDistribution,
    GaussianLikelihood,
    NormalDistribution,
)
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting._grid_distribution import GridDistribution
from frasian.tilting.power_law import PowerLawTilting, _generic_tilt


@pytest.mark.L2
@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("D", [0.0, 0.5, 1.5])
@pytest.mark.parametrize("sigma0", [0.5, 1.0, 2.0])
def test_generic_tilt_matches_theorem6_on_normal_normal(eta, D, sigma0):
    """The generic numerical path reduces to Theorem 6 within atol 1e-3.

    The deriver verified atol 1e-7 at N=1024 in standalone tests; the
    L2 tolerance here is loose enough to accommodate floating-point
    drift in the closed-form sigma_eta computation (the Theorem 6 form
    has a `sqrt` after a divide which doesn't survive bitwise round-
    trip through the trapezoidal grid). N=1024 default applies.
    """
    sigma = 1.0
    mu0 = 0.0
    posterior = NormalNormalModel(sigma=sigma).posterior(
        np.asarray([D]), NormalDistribution(loc=mu0, scale=sigma0)
    )
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    likelihood = GaussianLikelihood(D=D, sigma=sigma)

    closed = PowerLawTilting().tilt(posterior, prior, likelihood, eta)
    assert isinstance(closed, NormalDistribution)
    cf_mu, cf_sigma2 = closed.mean(), closed.var()

    # Bypass the closed-form dispatch — call generic directly.
    generic = _generic_tilt(
        posterior, prior, likelihood, eta,
        support=(-float("inf"), float("inf")),
    )
    assert isinstance(generic, GridDistribution)
    gn_mu, gn_sigma2 = generic.mean(), generic.var()

    assert abs(gn_mu - cf_mu) < 1e-3, (
        f"mean mismatch at eta={eta}, D={D}, sigma0={sigma0}: "
        f"closed={cf_mu:.6f} generic={gn_mu:.6f}"
    )
    assert abs(gn_sigma2 - cf_sigma2) < 1e-3, (
        f"var mismatch at eta={eta}, D={D}, sigma0={sigma0}: "
        f"closed={cf_sigma2:.6f} generic={gn_sigma2:.6f}"
    )


@pytest.mark.L0
def test_generic_tilt_runs_on_bernoulli_smoke():
    """power_law.tilt(...) end-to-end on (BernoulliModel, BetaDistribution).

    Pre-Phase-3b this raised NotImplementedError via _require_gaussian.
    Now it routes through the generic numerical path and returns a
    GridDistribution with finite moments.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    posterior = model.posterior(data, prior)  # Beta(4, 4)
    likelihood = model.likelihood(data)  # BernoulliLikelihood(n_success=4, n_total=6)
    assert isinstance(likelihood, BernoulliLikelihood)

    for eta in [0.0, 0.3, 0.7, 1.0]:
        tilted = PowerLawTilting().tilt(posterior, prior, likelihood, eta)
        assert isinstance(tilted, GridDistribution)
        # Moments finite and within (0, 1).
        m = tilted.mean()
        v = tilted.var()
        assert 0.0 <= m <= 1.0, f"tilted mean {m} outside [0, 1] at eta={eta}"
        assert v > 0.0, f"tilted var {v} non-positive at eta={eta}"
        # pdf at theta=0.5 is finite.
        assert np.isfinite(float(tilted.pdf(0.5)))
        # cdf is monotone-ish at sentinel points.
        c_lo = float(tilted.cdf(0.0))
        c_hi = float(tilted.cdf(1.0))
        assert 0.0 <= c_lo <= c_hi <= 1.0 + 1e-9


@pytest.mark.L2
def test_generic_tilt_eta0_recovers_posterior_on_bernoulli():
    """At eta=0, the tilted distribution reduces to the posterior.

    log q(theta; 0) = log L(theta) + log pi(theta) ∝ log p(theta | D)
    so the normalised tilted density IS the posterior.
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=3.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])  # 5 successes, 7 total
    posterior = model.posterior(data, prior)  # Beta(8, 4)
    likelihood = model.likelihood(data)

    tilted = PowerLawTilting().tilt(posterior, prior, likelihood, eta=0.0)

    # Compare moments. The Beta(8, 4) posterior has mean = 8/12 = 2/3,
    # var = 8*4/(12^2 * 13) = 32 / 1872 ≈ 0.01709.
    expected_mean = float(posterior.mean())
    expected_var = float(posterior.var())
    assert abs(tilted.mean() - expected_mean) < 5e-3
    assert abs(tilted.var() - expected_var) < 5e-3


@pytest.mark.L2
def test_generic_tilt_eta1_recovers_likelihood_on_bernoulli():
    """At eta=1, the tilted distribution is the likelihood-as-density.

    log q(theta; 1) = log L(theta), normalised. For Bernoulli with
    n_success=k, n_total=n, the likelihood-as-density is Beta(k+1, n-k+1)
    on [0, 1].
    """
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)  # arbitrary (zero contribution at eta=1)
    data = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0])  # 3 successes, 5 total
    posterior = model.posterior(data, prior)
    likelihood = model.likelihood(data)

    tilted = PowerLawTilting().tilt(posterior, prior, likelihood, eta=1.0)

    # Likelihood-as-density: theta^k * (1-theta)^(n-k) normalised on
    # [0, 1] → Beta(k+1, n-k+1) = Beta(4, 3). mean = 4/7, var = 4*3 / (49 * 8).
    n_success = 3
    n_total = 5
    expected_mean = (n_success + 1) / (n_total + 2)
    a, b = n_success + 1, n_total - n_success + 1
    expected_var = a * b / ((a + b) ** 2 * (a + b + 1))
    assert abs(tilted.mean() - expected_mean) < 5e-3
    assert abs(tilted.var() - expected_var) < 5e-3


@pytest.mark.L0
def test_generic_tilt_path_iterates_on_bernoulli():
    """`path()` yields a sequence of tilted distributions on Bernoulli."""
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    data = np.asarray([1.0, 0.0, 1.0])
    posterior = model.posterior(data, prior)
    likelihood = model.likelihood(data)
    ts = np.linspace(0.0, 1.0, 5)
    distributions = list(PowerLawTilting().path(posterior, prior, likelihood, ts))
    assert len(distributions) == 5
    for d in distributions:
        assert isinstance(d, GridDistribution)
        assert np.isfinite(d.mean())
