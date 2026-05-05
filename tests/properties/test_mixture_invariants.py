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


def test_waldo_pvalue_matches_monte_carlo() -> None:
    """Closed-form WALDO p-value matches MC integration to MC SE.

    The deriver verified this at N=2e6 with errors <= 6e-4. We use a
    lighter N here for test speed; tolerance is set to `5/sqrt(N)`.
    """
    model, prior, _lik, _post = _setup(mu0=0.0, sigma0=1.0, sigma=1.0, D=0.0)
    scheme = MixtureTilting()
    eta = 0.5
    rng = np.random.default_rng(42)
    n = 100_000
    # Sample from the mixture posterior at eta=0.5.
    sigma = 1.0
    mu_n = 0.0  # since D=0, mu0=0
    sigma_n = np.sqrt(0.5)
    D = 0.0
    mu_eta = (1.0 - eta) * mu_n + eta * D
    # Component indicators: half from each Gaussian.
    which = rng.uniform(0.0, 1.0, size=n) >= (1.0 - eta)
    z = rng.standard_normal(n)
    samples = np.where(which, D + sigma * z, mu_n + sigma_n * z)
    for theta in [-1.0, 0.5, 2.0]:
        p_cf = float(
            scheme.tilted_pvalue(np.asarray(theta), D, model, prior, eta, "waldo")
        )
        z_thresh = abs(theta - mu_eta)
        p_mc = float(np.mean(np.abs(samples - mu_eta) >= z_thresh))
        # MC SE ~ sqrt(p(1-p)/n) <= 0.5/sqrt(n) ~ 1.6e-3 at n=1e5.
        assert abs(p_cf - p_mc) < 5.0 / np.sqrt(n), (
            f"theta={theta}: closed-form {p_cf} vs MC {p_mc} differ by {p_cf - p_mc}"
        )
