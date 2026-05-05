"""Property tests for `MixtureTilting` (the m-geodesic).

Mirrors the invariants block in `audit/tier2/mixture_derivation.md`.
Each test pins one closed-form claim verified by the deriver agent
(MC-checked at N=2e6 and quadrature-checked at atol < 1e-15).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, stats

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.base import TiltingContext
from frasian.tilting.mixture import MixtureTilting

pytestmark = [pytest.mark.L1, pytest.mark.properties]


def _setup(mu0: float = 0.0, sigma0: float = 1.0, sigma: float = 1.0, D: float = 0.0):
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    lik = GaussianLikelihood(D=D, sigma=sigma)
    post = model.posterior(np.array([D]), prior)
    return model, prior, lik, post


@pytest.mark.parametrize("D", [-1.0, 0.0, 3.0])
def test_eta_zero_returns_posterior(D: float) -> None:
    """At eta=0 the m-geodesic identity element is the posterior."""
    _model, prior, lik, post = _setup(D=D)
    out = MixtureTilting().tilt(post, prior, lik, 0.0)
    grid = np.linspace(-6.0, 6.0, 401)
    np.testing.assert_allclose(out.pdf(grid), post.pdf(grid), atol=1e-12)


@pytest.mark.parametrize("D", [-1.0, 0.0, 3.0])
def test_eta_one_returns_likelihood_gaussian(D: float) -> None:
    """At eta=1 the m-geodesic recovers the likelihood-induced N(D, sigma^2)."""
    _model, prior, lik, post = _setup(D=D)
    out = MixtureTilting().tilt(post, prior, lik, 1.0)
    grid = np.linspace(-6.0, 6.0, 401)
    expected = stats.norm.pdf(grid, loc=lik.D, scale=lik.sigma)
    np.testing.assert_allclose(out.pdf(grid), expected, atol=1e-12)


@pytest.mark.parametrize(
    "mu0, sigma0, sigma, D, eta",
    [
        (0.0, 1.0, 1.0, 0.0, 0.5),
        (0.0, 1.0, 1.0, 3.0, 0.7),
        (0.0, 0.5, 2.0, -2.0, 0.2),
    ],
)
def test_mixture_mean_and_var_closed_form(
    mu0: float, sigma0: float, sigma: float, D: float, eta: float
) -> None:
    """Mean and variance match the Step-3 closed forms (sympy-verified)."""
    _model, prior, lik, post = _setup(mu0=mu0, sigma0=sigma0, sigma=sigma, D=D)
    out = MixtureTilting().tilt(post, prior, lik, eta)
    mu_n = post.loc
    sigma_n = post.scale
    expected_mean = (1.0 - eta) * mu_n + eta * D
    expected_var = (
        (1.0 - eta) * sigma_n**2
        + eta * sigma**2
        + eta * (1.0 - eta) * (mu_n - D) ** 2
    )
    assert abs(out.mean() - expected_mean) < 1e-12
    assert abs(out.var() - expected_var) < 1e-12


@pytest.mark.parametrize(
    "mu0, sigma0, sigma, D, eta",
    [
        (0.0, 1.0, 1.0, 0.0, 0.5),
        (0.0, 1.0, 1.0, 3.0, 0.7),
        (0.0, 0.5, 2.0, -2.0, 0.2),
    ],
)
def test_mixture_pdf_integrates_to_one(
    mu0: float, sigma0: float, sigma: float, D: float, eta: float
) -> None:
    """Convex combination of two normalised Gaussians integrates to 1."""
    _model, prior, lik, post = _setup(mu0=mu0, sigma0=sigma0, sigma=sigma, D=D)
    out = MixtureTilting().tilt(post, prior, lik, eta)
    norm, _ = integrate.quad(lambda t: float(out.pdf(t)), -50.0, 50.0, limit=200)
    assert abs(norm - 1.0) < 1e-9


def test_mixture_is_non_gaussian_at_intermediate_eta() -> None:
    """At strong conflict + eta=0.5 the mixture has negative excess kurtosis.

    Behboodian threshold for w=0.5 is `|Delta| > 2*sqrt(0.5) ~ 1.414`. The
    setting D=4 (|Delta|=2) is empirically bimodal at eta=0.5 (Step 2 of
    the derivation), so excess kurtosis < 0 — distinct from Gaussian's 0.
    """
    _model, prior, lik, post = _setup(D=4.0)
    out = MixtureTilting().tilt(post, prior, lik, 0.5)
    rng = np.random.default_rng(0)
    samples = out.sample(rng, 200_000)
    excess_kurt = float(stats.kurtosis(samples, fisher=True))
    assert excess_kurt < -0.2, f"expected bimodal, got excess kurtosis {excess_kurt}"


@pytest.mark.parametrize("eta", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_wald_pvalue_eta_invariant(eta: float) -> None:
    """Wald is eta-independent: the m-geodesic does not change the MLE."""
    model, prior, _, _ = _setup(D=0.0)
    scheme = MixtureTilting()
    p_eta = float(
        scheme.tilted_pvalue(np.asarray(1.0), 0.0, model, prior, eta, "wald")
    )
    p_bare = float(2.0 * stats.norm.sf(1.0 / model.sigma))
    assert abs(p_eta - p_bare) < 1e-12


@pytest.mark.parametrize(
    "eta1, eta2",
    [(0.0, 1e-6), (0.5, 0.5 + 1e-6), (1.0, 1.0 - 1e-6)],
)
def test_tilt_continuous_in_eta(eta1: float, eta2: float) -> None:
    """tilt(eta) is continuous in eta on [0, 1] (linear in density space)."""
    _model, prior, lik, post = _setup(D=2.0)
    scheme = MixtureTilting()
    p1 = scheme.tilt(post, prior, lik, eta1)
    p2 = scheme.tilt(post, prior, lik, eta2)
    grid = np.linspace(-6.0, 6.0, 201)
    np.testing.assert_allclose(p1.pdf(grid), p2.pdf(grid), atol=1e-5)


@given(
    mu0=st.floats(-3, 3),
    sigma0=st.floats(0.3, 3.0),
    sigma=st.floats(0.3, 3.0),
    D=st.floats(-3, 3),
    eta=st.floats(0.0, 1.0),
)
@settings(max_examples=50, deadline=None)
def test_variance_bound(
    mu0: float, sigma0: float, sigma: float, D: float, eta: float
) -> None:
    """`var(p_eta) <= max(sigma_n^2, sigma^2) + (mu_n - D)^2 / 4` (Step 3)."""
    _model, prior, lik, post = _setup(mu0=mu0, sigma0=sigma0, sigma=sigma, D=D)
    out = MixtureTilting().tilt(post, prior, lik, eta)
    mu_n, sigma_n = post.loc, post.scale
    bound = max(sigma_n**2, sigma**2) + (mu_n - D) ** 2 / 4.0
    assert out.var() <= bound + 1e-12


def test_admissible_range_is_unit_interval() -> None:
    """Admissible eta is [0, 1] regardless of context."""
    scheme = MixtureTilting()
    ctx = TiltingContext(w=0.5, abs_delta=0.0, alpha=0.05)
    lo, hi = scheme.admissible_range(ctx)
    assert lo == 0.0 and hi == 1.0
    # Independent of (w, abs_delta).
    ctx2 = TiltingContext(w=0.9, abs_delta=2.0, alpha=0.05)
    assert scheme.admissible_range(ctx2) == (lo, hi)


def test_waldo_pvalue_closed_form_matches_canonical_substitution() -> None:
    """Mixture's tilted WALDO equals the canonical bare-WALDO formula
    with the mixture's first two moments substituted in.

    The canonical "tilted WALDO" structure (shared with `power_law`
    and `fisher_rao`) is `Phi(b - a) + Phi(-a - b)` with
        a = sigma * |mu_eta - theta| / sigma_eta^2
        b = (1-w) * (mu0 - theta) / (w * sigma)
    and `(mu_eta, sigma_eta^2)` taken from the mixture's first/second
    moments. This test verifies that `tilted_pvalue` evaluates that
    exact closed form (independent recomputation, atol 1e-15).
    """
    model, prior, _lik, _post = _setup(mu0=0.0, sigma0=1.0, sigma=1.0, D=0.0)
    scheme = MixtureTilting()
    sigma = 1.0
    sigma0 = 1.0
    mu0 = 0.0
    w = sigma0**2 / (sigma**2 + sigma0**2)
    for D, eta, theta in [
        (0.0, 0.5, -1.0),
        (0.0, 0.5, 0.5),
        (2.0, 0.3, 1.0),
        (-1.5, 0.7, 0.0),
    ]:
        mu_n = w * D + (1.0 - w) * mu0
        sigma_n_sq = w * sigma**2
        mu_eta = (1.0 - eta) * mu_n + eta * D
        sigma_eta_sq = (
            (1.0 - eta) * sigma_n_sq
            + eta * sigma**2
            + eta * (1.0 - eta) * (mu_n - D) ** 2
        )
        a = sigma * abs(mu_eta - theta) / sigma_eta_sq
        b = (1.0 - w) * (mu0 - theta) / (w * sigma)
        expected = float(stats.norm.cdf(b - a) + stats.norm.cdf(-a - b))
        got = float(
            scheme.tilted_pvalue(np.asarray(theta), D, model, prior, eta, "waldo")
        )
        assert abs(got - expected) < 1e-15, (
            f"closed-form mismatch at D={D}, eta={eta}, theta={theta}: "
            f"got {got!r} vs expected {expected!r}"
        )


@pytest.mark.parametrize(
    "sigma,sigma0",
    [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)],
)
def test_tilted_waldo_at_eta_zero_equals_bare_waldo(
    sigma: float, sigma0: float
) -> None:
    """At eta=0 the mixture-tilted WALDO must collapse to bare WALDO.

    Phase 6 skeptic vector #1: the tilting protocol's identity
    invariant requires `tilted_pvalue('waldo', eta=eta_identity)` to
    equal `WaldoStatistic.pvalue`. The canonical formula uses
    `a = sigma * |mu_eta - theta| / sigma_eta^2` and bare WALDO's
    `b`, which collapses correctly at eta=0 (where the mixture's
    weight-1 component is the posterior, so mu_eta=mu_n,
    sigma_eta^2=sigma_n^2).
    """
    from frasian.statistics.waldo import WaldoStatistic

    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=sigma0)
    D = np.array([2.0])
    theta = 1.5
    bare = float(np.asarray(WaldoStatistic().pvalue(theta, D, model, prior)).item())
    tilted = float(
        np.asarray(
            MixtureTilting().tilted_pvalue(
                theta, D, model, prior, eta=0.0, statistic_name="waldo"
            )
        ).item()
    )
    assert abs(tilted - bare) < 1e-9, (
        f"mixture tilted WALDO at eta=0 ({tilted!r}) does not match "
        f"bare WALDO ({bare!r}) for (sigma, sigma0)=({sigma!r}, {sigma0!r})."
    )
