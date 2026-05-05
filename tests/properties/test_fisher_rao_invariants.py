"""Property tests for `FisherRaoTilting`.

Mirrors the invariants block in `audit/tier2/fisher_rao_derivation.md`.
Each test pins one closed-form claim verified by the deriver agent
(ODE-cross-checked at atol 7.6e-14).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.base import TiltingContext
from frasian.tilting.fisher_rao import (
    _VERTICAL_THRESHOLD,
    FisherRaoTilting,
    _fisher_rao_path,
    fisher_rao_distance,
)
from frasian.tilting.ot import OTTilting

pytestmark = [pytest.mark.L1, pytest.mark.properties]


def _setup(mu0: float = 0.0, sigma0: float = 1.0, sigma: float = 1.0, D: float = 0.0):
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    lik = GaussianLikelihood(D=D, sigma=sigma)
    post = model.posterior(np.array([D]), prior)
    return model, prior, lik, post


@pytest.mark.parametrize("D", [-1.0, 0.0, 3.0])
def test_eta_zero_returns_posterior(D: float) -> None:
    """At eta=0 the FR geodesic identity element is the posterior."""
    _model, prior, lik, post = _setup(D=D)
    out = FisherRaoTilting().tilt(post, prior, lik, 0.0)
    assert abs(out.loc - post.loc) < 1e-12
    assert abs(out.scale - post.scale) < 1e-12


@pytest.mark.parametrize("D", [-1.0, 0.0, 3.0])
def test_eta_one_returns_likelihood_gaussian(D: float) -> None:
    """At eta=1 the FR geodesic recovers `N(D, sigma^2)`."""
    _model, prior, lik, post = _setup(D=D)
    out = FisherRaoTilting().tilt(post, prior, lik, 1.0)
    assert abs(out.loc - lik.D) < 1e-12
    assert abs(out.scale - lik.sigma) < 1e-12


def test_vertical_geometric_mean_signature() -> None:
    """Vertical case midpoint: sigma = sqrt(sigma_p * sigma_q) (geometric mean).

    This is the FR fingerprint that distinguishes it from `ot` (arithmetic
    mean) and `power_law` (no clean closed form). Per Step 3a.
    """
    mu_p, sigma_p, mu_q, sigma_q = 0.5, 0.7, 0.5, 3.0
    mu, sigma = _fisher_rao_path(mu_p, sigma_p, mu_q, sigma_q, 0.5)
    assert abs(mu - mu_p) < 1e-12  # mu unchanged
    assert abs(sigma - math.sqrt(sigma_p * sigma_q)) < 1e-12


def test_vertical_mu_unchanged_at_all_eta() -> None:
    """Vertical case: mu(eta) = mu_p for every eta in [0, 1]."""
    mu_p = 1.5
    sigma_p, sigma_q = 0.5, 2.0
    for eta in np.linspace(0.0, 1.0, 11):
        mu, _ = _fisher_rao_path(mu_p, sigma_p, mu_p, sigma_q, float(eta))
        assert abs(mu - mu_p) < 1e-12


def test_semicircle_matches_ode_integration() -> None:
    """Closed-form (mu(eta), sigma(eta)) matches ODE integration to atol 1e-13.

    Integrate the geodesic equations on the Poincaré half-plane via DOP853
    starting from `(u_p, sigma_p)` with the analytic unit-speed initial
    velocity tangent to the semicircle, integrate for `eta * L_P`, and
    compare to the closed form. The deriver verified this at 7.6e-14;
    we use a tolerance of 1e-11 here to absorb minor solver variation
    (we're using SciPy DOP853 which can vary slightly across versions).
    """
    from scipy.integrate import solve_ivp

    mu_p, sigma_p, mu_q, sigma_q = -1.0, 0.5, 3.0, 1.5
    eta = 0.3

    # Closed form.
    mu_closed, sigma_closed = _fisher_rao_path(mu_p, sigma_p, mu_q, sigma_q, eta)

    # ODE integration on the Poincaré half-plane (u, sigma) with unit
    # Poincaré speed. Geodesic equations:
    #     u''     = (2 / sigma) * u' * sigma'
    #     sigma'' = (-(u')^2 + (sigma')^2) / sigma
    # We need the initial unit-speed velocity tangent to the semicircle at
    # (u_p, sigma_p). On the Poincaré speed parameterisation, the tangent
    # to the semicircle pointing toward (u_q, sigma_q) is:
    #   (du/ds, dsigma/ds) = sigma_p * (cos(t_p), sin(t_p)) ... but we
    # need it normalised such that ds_P/dt = 1.
    SQRT2 = math.sqrt(2.0)
    u_p = mu_p / SQRT2
    u_q = mu_q / SQRT2
    u_c = (u_q**2 - u_p**2 + sigma_q**2 - sigma_p**2) / (2.0 * (u_q - u_p))
    R = math.sqrt((u_p - u_c) ** 2 + sigma_p**2)
    t_p = math.atan2(sigma_p, u_p - u_c)
    t_q = math.atan2(sigma_q, u_q - u_c)
    L_P = abs(math.log(math.tan(t_q / 2.0)) - math.log(math.tan(t_p / 2.0)))
    # Direction (sign of dt/ds): t increases iff t_q > t_p in the chosen
    # parametrisation. ds_P/dt = 1/sin(t), so dt/ds = sin(t).
    sgn = 1.0 if t_q > t_p else -1.0
    # Initial velocity in (u, sigma) at unit Poincaré speed:
    # u' = du/dt * dt/ds = -R sin(t) * sgn * sin(t_p) = -R sin(t_p)^2 * sgn
    # sigma' = R cos(t) * sgn * sin(t_p) = R cos(t_p) sin(t_p) * sgn
    du0 = -R * math.sin(t_p) ** 2 * sgn
    dsigma0 = R * math.cos(t_p) * math.sin(t_p) * sgn

    def rhs(s: float, y: np.ndarray) -> np.ndarray:
        u_, sigma_, du, dsigma = y
        d_du = (2.0 / sigma_) * du * dsigma
        d_dsigma = (-du * du + dsigma * dsigma) / sigma_
        return np.asarray([du, dsigma, d_du, d_dsigma])

    s_target = eta * L_P
    y0 = np.asarray([u_p, sigma_p, du0, dsigma0])
    sol = solve_ivp(
        rhs,
        (0.0, s_target),
        y0,
        method="DOP853",
        rtol=1e-13,
        atol=1e-13,
    )
    assert sol.success
    u_ode, sigma_ode = sol.y[0, -1], sol.y[1, -1]
    mu_ode = SQRT2 * u_ode

    assert abs(mu_ode - mu_closed) < 1e-11, f"mu err {mu_ode - mu_closed}"
    assert abs(sigma_ode - sigma_closed) < 1e-11, f"sigma err {sigma_ode - sigma_closed}"


def test_output_is_gaussian_with_positive_sigma() -> None:
    """tilt(eta) returns a NormalDistribution with sigma > 0 for every eta in [0, 1]."""
    _model, prior, lik, post = _setup(D=2.0)
    scheme = FisherRaoTilting()
    for eta in np.linspace(0.0, 1.0, 21):
        out = scheme.tilt(post, prior, lik, float(eta))
        assert isinstance(out, NormalDistribution)
        assert out.scale > 0


@pytest.mark.parametrize("eta", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_wald_pvalue_eta_invariant(eta: float) -> None:
    """Wald is eta-independent (matches power_law / ot / mixture)."""
    model, prior, _, _ = _setup(D=0.0)
    scheme = FisherRaoTilting()
    p_eta = float(
        scheme.tilted_pvalue(np.asarray(1.0), 0.0, model, prior, eta, "wald")
    )
    p_bare = float(2.0 * stats.norm.sf(1.0 / model.sigma))
    assert abs(p_eta - p_bare) < 1e-12


def test_continuity_in_eta() -> None:
    """tilt(eta) is C^1 in eta with bounded speed; finite differences track."""
    _model, prior, lik, post = _setup(D=2.0)
    scheme = FisherRaoTilting()
    # Pick a few interior points; finite-difference shift bounded by C * h.
    for eta in [0.1, 0.5, 0.9]:
        out_a = scheme.tilt(post, prior, lik, eta)
        out_b = scheme.tilt(post, prior, lik, eta + 1e-7)
        # |loc_b - loc_a| and |scale_b - scale_a| both bounded by O(1e-7).
        assert abs(out_b.loc - out_a.loc) < 1e-5
        assert abs(out_b.scale - out_a.scale) < 1e-5


def test_distance_matches_costa_closed_form() -> None:
    """`fisher_rao_distance` matches Costa et al. 2015 Eq. 12."""
    settings = [
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 1.0, 2.0, 1.0),
        (-1.0, 0.5, 3.0, 1.5),
    ]
    for mu_p, sigma_p, mu_q, sigma_q in settings:
        # The function itself implements Costa Eq. 12; verify it via an
        # independent recomputation in a different algebraic form.
        # Eq. 12: d_FR = sqrt(2) * arccosh(1 + ((mu_p-mu_q)^2/2 + (sigma_p-sigma_q)^2)
        #                                   / (2 sigma_p sigma_q))
        d_a = fisher_rao_distance(mu_p, sigma_p, mu_q, sigma_q)
        # Independent: integrate ds_P/dt dt = dt/sin(t) over the arc, then
        # multiply by sqrt(2). For the vertical case, |log(sigma_q/sigma_p)|.
        if abs(mu_p - mu_q) < 1e-12:
            d_b = math.sqrt(2.0) * abs(math.log(sigma_q / sigma_p))
        else:
            SQRT2 = math.sqrt(2.0)
            u_p = mu_p / SQRT2
            u_q = mu_q / SQRT2
            u_c = (u_q**2 - u_p**2 + sigma_q**2 - sigma_p**2) / (
                2.0 * (u_q - u_p)
            )
            t_p = math.atan2(sigma_p, u_p - u_c)
            t_q = math.atan2(sigma_q, u_q - u_c)
            d_b = SQRT2 * abs(
                math.log(math.tan(t_q / 2.0)) - math.log(math.tan(t_p / 2.0))
            )
        assert abs(d_a - d_b) < 1e-12, f"setting {(mu_p, sigma_p, mu_q, sigma_q)}: {d_a} vs {d_b}"


def test_distance_is_symmetric() -> None:
    """d(P, Q) = d(Q, P) for the Fisher-Rao distance."""
    settings = [
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 1.0, 2.0, 1.0),
        (-1.0, 0.5, 3.0, 1.5),
    ]
    for mu_p, sigma_p, mu_q, sigma_q in settings:
        d_pq = fisher_rao_distance(mu_p, sigma_p, mu_q, sigma_q)
        d_qp = fisher_rao_distance(mu_q, sigma_q, mu_p, sigma_p)
        assert abs(d_pq - d_qp) < 1e-15


def test_differs_from_ot_when_sigmas_differ() -> None:
    """FR != OT whenever sigma_p != sigma_q.

    Setting `(mu_p, sigma_p, mu_q, sigma_q) = (0, 1, 0, 2)` at eta=0.5:
      Fisher-Rao gives sigma = sqrt(2) ~ 1.41421 (geometric mean)
      OT gives sigma = 1.5 (arithmetic mean)
    Difference > 0.08, well above any tolerance.
    """
    # Set up a Normal-Normal context whose posterior + likelihood pair
    # gives the desired (mu_p, sigma_p) = (0, 1) and (mu_q, sigma_q) = (0, 2).
    # Posterior is N(mu_n, sigma_n^2) with mu_n = w*D + (1-w)*mu0, sigma_n = sqrt(w)*sigma.
    # We want sigma_n = 1, sigma = 2: so w = 0.25, sigma0^2 / (sigma^2 + sigma0^2) = 0.25
    # => sigma0^2 = sigma^2 / 3 = 4/3 => sigma0 = 2/sqrt(3).
    # Want mu_n = 0 with mu0 = 0, D = 0 => mu_n = 0 trivially.
    # That gives posterior N(0, 1) and likelihood-as-Gaussian N(0, 2).
    sigma = 2.0
    sigma0 = 2.0 / math.sqrt(3.0)
    mu0 = 0.0
    D = 0.0
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    lik = GaussianLikelihood(D=D, sigma=sigma)
    post = model.posterior(np.array([D]), prior)
    # post should be N(0, 1) approximately:
    assert abs(post.loc - 0.0) < 1e-12
    assert abs(post.scale - 1.0) < 1e-12

    fr = FisherRaoTilting().tilt(post, prior, lik, 0.5)
    ot = OTTilting().tilt(post, prior, lik, 0.5)
    # FR midpoint scale: sqrt(1 * 2) = sqrt(2) ~ 1.41421
    assert abs(fr.scale - math.sqrt(2.0)) < 1e-12
    # OT midpoint scale: (1 + 2) / 2 = 1.5
    assert abs(ot.scale - 1.5) < 1e-12
    # And they differ by ~0.085.
    assert abs(fr.scale - ot.scale) > 0.08


def test_no_branch_discontinuity_at_threshold() -> None:
    """Vertical / semicircle branch boundary must be continuous in `delta_u`.

    Sweeps `|u_p - u_q|` across the `_VERTICAL_THRESHOLD` boundary and
    asserts that the FR-tilted sigma matches the vertical-branch closed
    form (geometric mean) to atol 1e-7. Phase 6 skeptic vector #3:
    at the previous threshold of 1e-12, the semicircle formula's
    `u_c = (u_q^2 - u_p^2 + sigma_q^2 - sigma_p^2) / (2*(u_q - u_p))`
    poisons just outside the vertical branch and produces a 1.7e-4
    sigma jump. Raising the threshold to 1e-8 drops the worst-case
    mismatch below 3e-8 (well below the 1e-7 tolerance here, but
    well above 1e-10 — float-cancellation in `u_q^2 - u_p^2` near
    the boundary can amplify on the order of `delta_u`).
    """
    sigma_p, sigma_q = 1.0, 2.0
    eta = 0.5
    sigma_vertical = sigma_p * (sigma_q / sigma_p) ** eta  # exact geometric mean
    SQRT2 = math.sqrt(2.0)
    # Sweep across the branch boundary, including just-outside-vertical
    # where the semicircle formula was previously poisoned.
    for frac in (0.5, 0.9, 1.0, 1.1, 2.0, 10.0, 100.0):
        delta_u = frac * _VERTICAL_THRESHOLD
        # Place mu_p, mu_q so |u_p - u_q| = delta_u.
        mu_p = 0.0
        mu_q = SQRT2 * delta_u  # u_q - u_p = delta_u
        _mu_eta, sigma_eta = _fisher_rao_path(mu_p, sigma_p, mu_q, sigma_q, eta)
        diff = abs(sigma_eta - sigma_vertical)
        assert diff < 1e-7, (
            f"branch discontinuity at delta_u={delta_u!r}: "
            f"sigma={sigma_eta!r} vs vertical={sigma_vertical!r} "
            f"(diff {diff!r})"
        )


@pytest.mark.parametrize(
    "sigma,sigma0",
    [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)],
)
def test_tilted_waldo_at_eta_zero_equals_bare_waldo(
    sigma: float, sigma0: float
) -> None:
    """At eta=0 the FR-tilted WALDO must collapse to bare WALDO.

    Phase 6 skeptic vector #2: the tilting protocol's identity
    invariant requires `tilted_pvalue('waldo', eta=eta_identity)` to
    equal `WaldoStatistic.pvalue`. The canonical formula uses
    `a = sigma * |mu_eta - theta| / sigma_eta^2` and bare WALDO's
    `b`, which collapses correctly at eta=0 (where mu_eta=mu_n,
    sigma_eta=sigma_n).
    """
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=sigma0)
    D = np.array([2.0])
    theta = 1.5
    bare = float(np.asarray(WaldoStatistic().pvalue(theta, D, model, prior)).item())
    tilted = float(
        np.asarray(
            FisherRaoTilting().tilted_pvalue(
                theta, D, model, prior, eta=0.0, statistic_name="waldo"
            )
        ).item()
    )
    assert abs(tilted - bare) < 1e-9, (
        f"FR tilted WALDO at eta=0 ({tilted!r}) does not match bare "
        f"WALDO ({bare!r}) for (sigma, sigma0)=({sigma!r}, {sigma0!r})."
    )


def test_admissible_range_is_unit_interval() -> None:
    """eta in [0, 1] always; no clamp."""
    scheme = FisherRaoTilting()
    ctx = TiltingContext(w=0.5, abs_delta=0.0, alpha=0.05)
    lo, hi = scheme.admissible_range(ctx)
    assert lo == 0.0 and hi == 1.0
    # Independent of context.
    ctx2 = TiltingContext(w=0.9, abs_delta=2.0, alpha=0.05)
    assert scheme.admissible_range(ctx2) == (lo, hi)
