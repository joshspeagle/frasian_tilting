"""STUB: Fisher-Rao geodesic on a parametric distribution family.

Fisher-Rao is the Riemannian (Levi-Civita) geodesic of a parametric
manifold equipped with the Fisher information metric. It is the
*third* affine connection compatible with the Fisher metric, distinct
from the e-connection (`power_law`'s log-linear path) and the
m-connection (`mixture`'s linear-in-density path) of Amari's dually
flat structure.

On the univariate Gaussian family the Fisher metric makes the manifold
the upper half-plane in `(mu, sigma)` (after rescaling `mu` by sqrt(2)
so the Gaussian curvature is -1), and geodesics between two Gaussians
have a closed form (Costa et al. 2015 Eq. 12). Compared to the W2
geodesic (`ot`), Fisher-Rao respects the *information-geometric*
structure rather than the displacement of mass; the two coincide only
when sigma_a = sigma_b.

Implementation scope (this stub): Gaussian-only closed form first, with
NotImplementedError on non-Gaussian endpoints. A general
ParametricFamily interface (which would let us run Fisher-Rao on
other families) is a follow-up refactor.

Endpoints follow the framework's posterior <-> likelihood convention
(matching `power_law` and `ot`): eta=0 -> posterior, eta=1 ->
likelihood-induced Gaussian N(D, sigma^2).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import diffrax
import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as _scalar_scipy_stats

from .. import _jax_setup as _x64  # noqa: F401  -- ensure float64 active
from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..models._dispatch import is_normal_normal
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..statistics.base import TestStatistic
from ._dynamic import dynamic_ci_scan
from ._solvers import brentq_with_doubling
from .base import EtaSelector, ParamSpec
from .eta_selectors import FixedEtaSelector

if TYPE_CHECKING:
    from ..config import Config

_FORCE_X64 = _x64

# Numerical guards
_VERTICAL_CASE_EPS = 1e-12       # numpy scalar path: threshold for |mu_a - mu_b| -> vertical-line geodesic
_VERTICAL_CASE_EPS_JAX = 1e-6    # JAX path: wider so gradients through the arc denominator don't blow up
                                  # (autograd reverses through 1/safe_denom; the arc formula has a coordinate
                                  # singularity at denom=0, so wraps must trigger BEFORE the gradient corrupts.
                                  # At |denom|<1e-7 the autograd-vs-FD gap explodes; 1e-6 leaves a safety margin.
                                  # See test_jax_geodesic_gradient_through_vertical for the calibration grid.)
_SIGMA_FLOOR = 1e-300            # absolute floor on sigma along the path


# ----------------------------------------------------------------------
# Closed-form half-plane geodesic helpers
# ----------------------------------------------------------------------


def _fr_geodesic_gaussian_scalar(
    mu_a: float, sigma_a: float, mu_b: float, sigma_b: float, t: float
) -> tuple[float, float]:
    """Constant-speed Fisher-Rao geodesic between two Gaussians at parameter t.

    Returns ``(mu_t, sigma_t)`` along the half-plane geodesic from
    ``(mu_a, sigma_a)`` to ``(mu_b, sigma_b)``. Vertical-line case when
    ``mu_a == mu_b``; circular-arc case otherwise.

    Constant-speed parametrisation (rev 1): on the generic arc,
    ``ds/dphi = 1/sin(phi)`` (NOT constant), so linear-in-phi is NOT
    constant-speed. The correct constant-speed (= arc-length) param
    uses the antiderivative ``s(phi) = ln tan(phi/2)``:

        s(t)  = (1 - t) * ln tan(phi_a/2) + t * ln tan(phi_b/2)
        phi(t) = 2 * arctan(exp(s(t)))

    Derivation: docs/methods/fisher_rao.md "Definition" (rev 1) +
    deriver output Step 5 at docs/superpowers/specs/2026-05-11-fisher-
    rao-deriver-output.md (sympy-verified).
    """
    if not (sigma_a > 0.0 and sigma_b > 0.0):
        raise TiltingDomainError(
            f"FisherRaoTilting requires positive sigmas; got sigma_a={sigma_a!r}, sigma_b={sigma_b!r}."
        )
    sqrt2 = math.sqrt(2.0)
    mu_a_t = mu_a / sqrt2
    mu_b_t = mu_b / sqrt2
    if abs(mu_a_t - mu_b_t) < _VERTICAL_CASE_EPS:
        s_t = sigma_a ** (1.0 - t) * sigma_b ** t
        return float(mu_a), float(s_t)
    c_tilde = ((mu_a_t * mu_a_t - mu_b_t * mu_b_t) + (sigma_a * sigma_a - sigma_b * sigma_b)) \
              / (2.0 * (mu_a_t - mu_b_t))
    r = math.sqrt((mu_a_t - c_tilde) ** 2 + sigma_a * sigma_a)
    phi_a = math.atan2(sigma_a, mu_a_t - c_tilde)
    phi_b = math.atan2(sigma_b, mu_b_t - c_tilde)
    s_a = math.log(math.tan(phi_a / 2.0))
    s_b = math.log(math.tan(phi_b / 2.0))
    s_t = (1.0 - t) * s_a + t * s_b
    phi_t = 2.0 * math.atan(math.exp(s_t))
    mu_t_tilde = c_tilde + r * math.cos(phi_t)
    s_sigma = r * math.sin(phi_t)
    if s_sigma <= _SIGMA_FLOOR:
        raise TiltingDomainError(
            f"FisherRaoTilting: geodesic crossed sigma=0 boundary at t={t!r} "
            f"(numerical instability or out-of-admissible parameter)."
        )
    return float(sqrt2 * mu_t_tilde), float(s_sigma)


def _fr_arc_length_costa(
    mu_a: float, sigma_a: float, mu_b: float, sigma_b: float
) -> float:
    """Closed-form Fisher-Rao arc-length per Costa et al. 2015 Eqs. 5-6.

    d_FR = sqrt(2) * arccosh(1 + ((mu_a-mu_b)^2/2 + (sigma_a-sigma_b)^2) / (2*sigma_a*sigma_b))

    (Note: cited in earlier drafts as "Eq. 12" — that was wrong; Eq. 12
    is the symmetrised KL distance. Correct citation is Eqs. 5-6.)
    """
    arg = 1.0 + ((mu_a - mu_b) ** 2 / 2.0 + (sigma_a - sigma_b) ** 2) / (2.0 * sigma_a * sigma_b)
    # Stage A audit finding #14: arg is structurally >= 1 (sum of
    # squares / positive denominator + 1), but floating-point roundoff
    # at identical endpoints can produce arg = 1 - epsilon, sending
    # math.acosh into a domain error. Clamp at construction.
    arg = max(1.0, arg)
    return float(math.sqrt(2.0) * math.acosh(arg))


def _fr_geodesic_arc_length_numerical(
    mu_a: float, sigma_a: float, mu_b: float, sigma_b: float, n_steps: int = 10000
) -> float:
    """Trapezoidal arc-length integration along the closed-form geodesic.

    Used by tests to verify that the constant-speed parametrisation
    integrates to the Costa et al. 2015 closed form. The half-plane
    metric ds = sqrt((dmu_tilde^2 + dsigma^2)/sigma^2) where
    mu_tilde = mu/sqrt(2). Multiply by sqrt(2) to get the Fisher arc
    length d_F = sqrt(2) * d_H.
    """
    import numpy as np
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    pts = np.empty((n_steps + 1, 2))
    for i, t in enumerate(ts):
        pts[i] = _fr_geodesic_gaussian_scalar(mu_a, sigma_a, mu_b, sigma_b, float(t))
    sqrt2 = math.sqrt(2.0)
    diffs = np.diff(pts, axis=0)
    sigma_mid = 0.5 * (pts[:-1, 1] + pts[1:, 1])
    diffs[:, 0] /= sqrt2  # mu -> mu_tilde for half-plane metric
    ds = np.sqrt((diffs ** 2).sum(axis=1)) / sigma_mid
    return float(sqrt2 * ds.sum())  # multiply by sqrt(2) for Fisher arc length


# ----------------------------------------------------------------------
# Stage B: general-purpose JAX autodiff machinery (NN-validation only)
#
# Closed-form Gaussian Fisher metric, used as the known-correct reference
# that the autodiff/diffrax shooting in subsequent helpers is validated
# against. Per docs/superpowers/specs/2026-05-11-fisher-rao-tilting-design.md
# (rev 2), Stage B is a general-purpose machinery that operates on a
# `g_fn: theta -> matrix` callable; the Gaussian metric is the only
# concrete metric this PR exercises. No non-Gaussian endpoint paths.
# ----------------------------------------------------------------------


def _gaussian_fisher_metric(theta: jax.Array) -> jax.Array:
    """Closed-form Fisher information metric on the univariate Gaussian family.

    For ``theta = (mu, sigma)``, returns the 2x2 matrix
    ``g(mu, sigma) = diag(1/sigma^2, 2/sigma^2)``.

    Verified by the deriver agent (sympy: negative expected Hessian of
    log N(x; mu, sigma^2) returns exactly (1/sigma^2, 2/sigma^2, 0) for
    the (mu, mu), (sigma, sigma), (mu, sigma) entries). See
    `docs/methods/fisher_rao.md` Derivation Step 1.

    This is the metric callable that `_christoffel_from_metric` and
    `_fr_geodesic_numerical` (Stage B helpers) consume. They operate
    on the abstract `g_fn: (theta,) -> (D, D)` signature, so plugging
    in a different family's metric in a future PR would extend FR
    to that family with no other code changes.
    """
    sigma = theta[1]
    return jnp.diag(jnp.array([1.0 / sigma ** 2, 2.0 / sigma ** 2]))


def _christoffel_from_metric(g_fn, theta: jax.Array) -> jax.Array:
    """Christoffel symbols of the Levi-Civita connection from a metric tensor field.

    Args:
        g_fn: callable ``theta -> (D, D) matrix`` returning the metric
              tensor at the given parameter point. Must be JAX-traceable
              (we apply ``jax.jacrev`` to it).
        theta: ``(D,)`` parameter vector.

    Returns:
        ``(D, D, D)`` tensor with ``gamma[k, i, j] = Γ^k_{ij}``, the
        Christoffel symbols of the second kind. Symmetric in ``(i, j)``.

    Formula::

        Γ^k_{ij} = 0.5 · g^{kl} · ( ∂_i g_{lj} + ∂_j g_{li} − ∂_l g_{ij} )

    where ``g^{kl}`` is the inverse metric and Einstein summation is over
    ``l``. Verified against the Gaussian closed form in
    ``TestFisherRaoChristoffel`` for the metric ``diag(1/σ², 2/σ²)`` on
    the half-plane (the only metric this PR exercises; rev 2 spec).

    This is the second piece of the general-purpose autodiff machinery
    (after ``_gaussian_fisher_metric`` in B.2). The geodesic ODE rhs
    (B.4) and the shooting BVP (B.5) consume it.
    """
    g = g_fn(theta)                                # (D, D)
    g_inv = jnp.linalg.inv(g)                      # (D, D)
    # jax.jacrev convention: jacrev(f)(theta)[output_axes, theta_axis]
    # so dg[i, j, k] = ∂_k g_{ij}.
    dg = jax.jacrev(g_fn)(theta)                   # (D, D, D)
    # Build the bracket: ∂_i g_{lj} + ∂_j g_{li} − ∂_l g_{ij}.
    # Using axis names, dg has shape (i_out, j_out, k_diff) ≡ (a, b, c)
    # interpreted as `∂_c g_{ab}`.
    # We need terms in (i, j, l) space (i.e. the bracket for Γ^k_{ij}):
    #   ∂_i g_{lj}  →  dg[l, j, i]  → transpose perm (2, 1, 0):
    #       new[i,j,l] = dg[old] where old[2]=i, old[1]=j, old[0]=l → dg[l,j,i] ✓
    #   ∂_j g_{li}  →  dg[l, i, j]  → transpose perm (1, 2, 0):
    #       new[i,j,l] = dg[old] where old[1]=i, old[2]=j, old[0]=l → dg[l,i,j] ✓
    #   ∂_l g_{ij}  →  dg[i, j, l]  → identity perm (0, 1, 2):
    #       new[i,j,l] = dg[old] where old[0]=i, old[1]=j, old[2]=l → dg[i,j,l] ✓
    term_a = jnp.transpose(dg, (2, 1, 0))          # [i, j, l] = ∂_i g_{lj}
    term_b = jnp.transpose(dg, (1, 2, 0))          # [i, j, l] = ∂_j g_{li}
    term_c = jnp.transpose(dg, (0, 1, 2))          # [i, j, l] = ∂_l g_{ij}
    bracket = 0.5 * (term_a + term_b - term_c)     # (i, j, l)
    # Γ^k_{ij} = g^{kl} · bracket[i, j, l] — contract over l.
    gamma = jnp.einsum("kl,ijl->kij", g_inv, bracket)
    return gamma


def _geodesic_ode_rhs(t: jax.Array, state: jax.Array, args) -> jax.Array:
    """First-order ODE rhs for the Levi-Civita geodesic equation.

    Args:
        t: scalar parameter (unused — the ODE is autonomous, but
           diffrax passes it for API symmetry).
        state: ``(2*D,)`` concatenation of ``(theta, v)`` where
            ``v = dtheta/dt``.
        args: ``(g_fn,)`` 1-tuple containing the metric-tensor callable
            ``g_fn: theta -> (D, D) matrix``. Passed via diffrax's
            ``ODETerm(..., args=...)`` plumbing.

    Returns:
        ``(2*D,)`` concatenation of ``(v, accel)`` where
        ``accel^k = -Γ^k_{ij}(theta) v^i v^j``.

    Computes Christoffel symbols by autodiff (``_christoffel_from_metric``)
    at every rhs call; the Christoffel call is the dominant cost
    inside the diffrax solve. ``_christoffel_from_metric`` is
    JIT-compatible (verified in B.3), so the full rhs JITs through
    diffrax cleanly.

    The geodesic ODE is *autonomous* (no explicit t-dependence), but
    diffrax's ODETerm signature requires accepting t as the first
    positional arg.
    """
    g_fn, = args
    D = state.shape[0] // 2
    theta = state[:D]
    v = state[D:]
    gamma = _christoffel_from_metric(g_fn, theta)              # (D, D, D)
    # accel^k = -Γ^k_{ij} v^i v^j (Einstein sum over i, j)
    accel = -jnp.einsum("kij,i,j->k", gamma, v, v)             # (D,)
    return jnp.concatenate([v, accel])


# ----------------------------------------------------------------------
# Stage B.5: diffrax shooting BVP for the geodesic
# ----------------------------------------------------------------------
#
# Forward-integrate the geodesic IVP with diffrax (Tsit5 + PID controller)
# and shoot via Newton on the initial velocity to satisfy the BVP
# theta(0) = theta_a, theta(1) = theta_b. Validated against the closed-form
# Gaussian half-plane geodesic in TestFisherRaoGeodesicNumerical.


# Solver settings — tight tolerances to hit atol ~1e-7 vs closed-form
_DIFFRAX_RTOL: float = 1e-8
_DIFFRAX_ATOL: float = 1e-10
_DIFFRAX_MAX_STEPS: int = 10000
_BVP_NEWTON_MAX_ITERS: int = 30
_BVP_NEWTON_TOL: float = 1e-9


def _solve_geodesic_forward(g_fn, theta_a: jax.Array, v0: jax.Array) -> jax.Array:
    """Forward-integrate the geodesic ODE from theta_a with initial velocity v0.

    Args:
        g_fn: metric-tensor callable ``theta -> (D, D)``.
        theta_a: ``(D,)`` initial parameter point.
        v0: ``(D,)`` initial velocity.

    Returns:
        ``theta(1)`` as a ``(D,)`` JAX array (the parameter point at t=1).

    Uses diffrax.Tsit5 with PIDController(rtol=1e-8, atol=1e-10), max
    10k steps. Saves only at t=1 (we don't need the full trajectory
    for the BVP residual). JIT-traceable; gradient-through-solve works
    via diffrax's implicit-function adjoint.
    """
    D = theta_a.shape[0]
    state0 = jnp.concatenate([theta_a, v0])
    term = diffrax.ODETerm(_geodesic_ode_rhs)
    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=1.0, dt0=0.01,
        y0=state0,
        args=(g_fn,),
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(
            rtol=_DIFFRAX_RTOL, atol=_DIFFRAX_ATOL,
        ),
        max_steps=_DIFFRAX_MAX_STEPS,
    )
    # sol.ys has shape (1, 2*D) since SaveAt(t1=True) returns 1 timepoint.
    return sol.ys[0, :D]


def _shoot_bvp(
    g_fn, theta_a: jax.Array, theta_b: jax.Array,
    max_iters: int = _BVP_NEWTON_MAX_ITERS, tol: float = _BVP_NEWTON_TOL,
) -> jax.Array:
    """Newton iteration on initial velocity to satisfy the BVP θ(0)=θ_a, θ(1)=θ_b.

    Args:
        g_fn: metric-tensor callable.
        theta_a, theta_b: ``(D,)`` boundary points.
        max_iters: Newton iteration budget (default 30).
        tol: convergence tolerance on ``||θ(1) - θ_b||`` (default 1e-9).

    Returns:
        ``v0`` = ``(D,)`` initial velocity such that forward-integrating
        with ``_solve_geodesic_forward(g_fn, theta_a, v0)`` lands at
        ``theta_b`` within ``tol``.

    Raises:
        RuntimeError if Newton fails to converge in ``max_iters``.

    Initial guess: ``v_0 = θ_b - θ_a`` (parameter-space linear interp
    rate). Refined via **damped Newton with backtracking line search**
    on the boundary residual. The Jacobian is computed via
    ``jax.jacrev``; the linear solve uses ``jnp.linalg.solve`` and falls
    back to ``lstsq`` if singular.

    Damping rationale: the curved half-plane metric makes the
    parameter-space linear guess ``v_0 = θ_b - θ_a`` overshoot the true
    geodesic velocity by ~2-4× in typical Gaussian endpoint settings.
    Undamped Newton from this guess can diverge (huge step, near-singular
    Jacobian, overflow). Backtracking line search (halve step until
    residual decreases or until 20 backtracks) globalizes convergence;
    we also reject any step that produces non-finite forward integration.
    """
    v0 = theta_b - theta_a

    def residual(v):
        return _solve_geodesic_forward(g_fn, theta_a, v) - theta_b

    r = residual(v0)
    r_norm = float(jnp.linalg.norm(r))
    for _ in range(max_iters):
        if r_norm < tol:
            return v0
        J = jax.jacrev(residual)(v0)                    # (D, D)
        try:
            dv = jnp.linalg.solve(J, -r)
        except Exception:
            dv = -jnp.linalg.lstsq(J, r)[0]
        # Backtracking line search: halve the step until residual strictly
        # decreases. Cap at 20 backtracks (alpha=2^-20 ≈ 1e-6).
        alpha = 1.0
        accepted = False
        for _ in range(20):
            v_trial = v0 + alpha * dv
            try:
                r_trial = residual(v_trial)
                r_trial_norm = float(jnp.linalg.norm(r_trial))
                if jnp.all(jnp.isfinite(r_trial)) and r_trial_norm < r_norm:
                    v0 = v_trial
                    r = r_trial
                    r_norm = r_trial_norm
                    accepted = True
                    break
            except Exception:
                # Forward solve exploded; halve alpha and try again.
                pass
            alpha *= 0.5
        if not accepted:
            raise RuntimeError(
                f"FR geodesic shooting BVP: line search failed to find a "
                f"residual-reducing step (theta_a={theta_a}, theta_b={theta_b}, "
                f"|r|={r_norm:.3e})."
            )
    raise RuntimeError(
        f"FR geodesic shooting BVP did not converge in {max_iters} iters "
        f"(theta_a={theta_a}, theta_b={theta_b}, last_residual_norm={r_norm:.3e})."
    )


def _fr_geodesic_numerical(
    theta_a, theta_b, t: float, g_fn=None,
):
    """Generic Fisher-Rao geodesic at parameter t in [0, 1] (or extrapolation).

    Args:
        theta_a, theta_b: ``(D,)`` boundary parameter points (numpy or JAX).
        t: scalar in ``[0, 1]`` (or extrapolation).
        g_fn: metric-tensor callable; defaults to ``_gaussian_fisher_metric``
            (the only metric this PR exercises; rev 2 spec).

    Returns:
        ``theta_t`` as a numpy ``(D,)`` array — the parameter point along
        the FR geodesic at the given t.

    Pipeline:
    1. Solve the BVP for the initial velocity ``v_0`` such that
       ``_solve_geodesic_forward(theta_a, v_0)`` lands at ``theta_b``.
    2. Forward-integrate from ``theta_a`` with ``v_0`` to time t, return
       the parameter point.

    Validated against the closed-form Gaussian half-plane geodesic in
    ``TestFisherRaoGeodesicNumerical``: atol ~1e-7 on the parameter
    point at three representative endpoint pairs.

    The BVP solve is the dominant cost (~10-20 diffrax solves per Newton
    iteration × ~5-10 iterations = 50-200 solves total). Per-call cost
    on CPU: ~100-500 ms depending on the geodesic length. Used only by
    the `fr_dyn_numerical_generic` audit flavor (B.7); the production
    NN path uses the closed-form ``_fr_geodesic_gaussian_scalar``.
    """
    if g_fn is None:
        g_fn = _gaussian_fisher_metric
    theta_a_j = jnp.asarray(theta_a, dtype=jnp.float64)
    theta_b_j = jnp.asarray(theta_b, dtype=jnp.float64)
    v0 = _shoot_bvp(g_fn, theta_a_j, theta_b_j)
    if t == 0.0:
        return np.asarray(theta_a_j)
    if t == 1.0:
        return np.asarray(theta_b_j)
    # Forward-integrate to time t.
    state0 = jnp.concatenate([theta_a_j, v0])
    term = diffrax.ODETerm(_geodesic_ode_rhs)
    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=float(t), dt0=0.01 if t > 0.0 else None,
        y0=state0,
        args=(g_fn,),
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(
            rtol=_DIFFRAX_RTOL, atol=_DIFFRAX_ATOL,
        ),
        max_steps=_DIFFRAX_MAX_STEPS,
    )
    D = theta_a_j.shape[0]
    return np.asarray(sol.ys[0, :D])


# ----------------------------------------------------------------------
# Tilted-WALDO p-value (adaptive quadrature with brentq boundary finding)
# ----------------------------------------------------------------------
#
# Rev 1 finding: FR's tilted-WALDO p-value has NO closed form at interior
# eta because mu_FR(eta; X) is non-linear in X along the curved geodesic.
# Trapezoidal quadrature on the discontinuous indicator 1{tau_rep >= tau_obs}
# converges only O(1/n) -- ~1e-2 at n=256. The proper fix uses brentq to
# find the discontinuity boundaries X* where tau_rep(X*) = tau_obs, then
# integrates pdf(X) analytically as sums of Gaussian CDF differences over
# the accept intervals. Machine precision; one brentq per boundary.


_FR_COARSE_GRID_N: int = 256          # coarse grid for sign-change discovery
_FR_X_RANGE: float = 12.0             # X grid spans theta +/- X_RANGE * sigma
_FR_BRENTQ_TOL: float = 1e-12         # brentq xtol


def _fr_tau_rep_minus_obs(
    X: float, theta_f: float, eta_f: float, mu0: float,
    w: float, sigma: float, sigma_n: float, tau_obs: float,
) -> float:
    """g(X) = tau_rep(X) - tau_obs, with tau_rep(X) computed via the
    closed-form Fisher-Rao geodesic at replicate X.
    """
    mu_n_X = w * X + (1.0 - w) * mu0
    mu_t_X, s_t_X = _fr_geodesic_gaussian_scalar(mu_n_X, sigma_n, X, sigma, eta_f)
    tau_rep = (mu_t_X - theta_f) ** 2 / (s_t_X ** 2)
    return tau_rep - tau_obs


def _fr_tilted_pvalue_numpy_scalar(
    theta_f: float,
    eta_f: float,
    D_f: float,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
    *,
    n_grid: int = _FR_COARSE_GRID_N,
) -> float:
    """Scalar Fisher-Rao tilted-WALDO / tilted-Wald p-value on Normal-Normal.

    Wald: closed-form ``2 * Phi(-|D - theta| / sigma)`` (Wald ignores the prior).

    WALDO at endpoints (eta == 0 or eta == 1): closed-form bare-WALDO /
    bare-Wald fast paths.

    WALDO at interior eta: adaptive quadrature via brentq boundary finding
    + analytical Gaussian-CDF integration over the accept intervals.
    Machine precision; per-call cost is bounded by the brentq iteration
    count times the per-call geodesic cost (typically 0-2 brentq calls
    each ~20 evaluations, so ~10x scalar geodesic + a coarse-grid sweep).

    n_grid: number of points in the coarse sign-change search grid.
    **The coarse-grid default n_grid=256 is sufficient to catch sign
    changes across typical (theta, eta, D) regimes; at extreme eta
    (close to 0 or 1) the rapid tau_rep variation near the boundary
    may require denser grids — pass n_grid=1024 for safety in
    adversarial regimes.**

    Derivation: docs/methods/fisher_rao.md Derivation (rev 1) Steps 7-9.
    """
    from scipy.optimize import brentq

    if statistic_name == "wald":
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))
    if statistic_name != "waldo":
        raise NotImplementedError(
            f"FisherRaoTilting tilted p-value: unknown statistic_name={statistic_name!r}."
        )
    sigma_n = math.sqrt(w) * sigma
    mu_n = w * D_f + (1.0 - w) * mu0
    # Endpoint fast paths (deriver Step 9)
    if np.isclose(eta_f, 0.0, atol=1e-15):
        a = abs(mu_n - theta_f) / (w * sigma)
        b = (1.0 - w) * (mu0 - theta_f) / (w * sigma)
        p = _scalar_scipy_stats.norm.cdf(b - a) + _scalar_scipy_stats.norm.cdf(-a - b)
        return float(p)
    if np.isclose(eta_f, 1.0, atol=1e-15):
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))
    # Adaptive quadrature: find brentq roots of g(X) = tau_rep(X) - tau_obs
    mu_t_obs, s_t_obs = _fr_geodesic_gaussian_scalar(mu_n, sigma_n, D_f, sigma, eta_f)
    tau_obs = (mu_t_obs - theta_f) ** 2 / (s_t_obs ** 2)

    def g(X: float) -> float:
        return _fr_tau_rep_minus_obs(X, theta_f, eta_f, mu0, w, sigma, sigma_n, tau_obs)

    # Coarse-grid sign-change discovery
    X_lo, X_hi = theta_f - _FR_X_RANGE * sigma, theta_f + _FR_X_RANGE * sigma
    coarse_X = np.linspace(X_lo, X_hi, n_grid)
    coarse_g = np.array([g(float(x)) for x in coarse_X])
    # Find sign-change indices
    sign_changes = np.where(np.diff(np.sign(coarse_g)) != 0)[0]
    if len(sign_changes) == 0:
        # No sign change: integrand is constant on the grid.
        # Determine constant value by checking sign at any X.
        if coarse_g[0] >= 0.0:
            return 1.0  # accept everywhere
        return 0.0  # reject everywhere
    # Refine each sign change with brentq
    roots = []
    for idx in sign_changes:
        a, b = float(coarse_X[idx]), float(coarse_X[idx + 1])
        try:
            root = brentq(g, a, b, xtol=_FR_BRENTQ_TOL, maxiter=100)
            roots.append(root)
        except ValueError:
            # Brentq failed (e.g. sign of g(a), g(b) is the same; numerical
            # artifact of the coarse grid). Skip this candidate.
            continue
    if not roots:
        # All sign changes were numerical noise; fall back to constant check
        if coarse_g[0] >= 0.0:
            return 1.0
        return 0.0
    roots = sorted(roots)
    # Build accept intervals. The line is partitioned by roots into intervals;
    # determine which intervals are "accept" (g >= 0) by checking g at midpoints.
    intervals = []  # list of (a, b) where g >= 0
    boundaries = [-np.inf] + list(roots) + [np.inf]
    for i in range(len(boundaries) - 1):
        a_i, b_i = boundaries[i], boundaries[i + 1]
        # Probe midpoint (or use a finite midpoint for ±inf endpoints)
        if a_i == -np.inf:
            mid = b_i - 1.0
        elif b_i == np.inf:
            mid = a_i + 1.0
        else:
            mid = 0.5 * (a_i + b_i)
        if g(float(mid)) >= 0.0:
            intervals.append((a_i, b_i))
    # Sum P(X in accept interval) under X ~ N(theta_f, sigma^2)
    p = 0.0
    for a_i, b_i in intervals:
        if a_i == -np.inf:
            cdf_a = 0.0
        else:
            cdf_a = float(_scalar_scipy_stats.norm.cdf((a_i - theta_f) / sigma))
        if b_i == np.inf:
            cdf_b = 1.0
        else:
            cdf_b = float(_scalar_scipy_stats.norm.cdf((b_i - theta_f) / sigma))
        p += cdf_b - cdf_a
    # Clamp to [0, 1] (numerical safety)
    return float(max(0.0, min(1.0, p)))


# ----------------------------------------------------------------------
# JAX kernel — batched / jit-friendly trap-rule tilted p-value
# ----------------------------------------------------------------------
#
# Used by the learned-eta training loop (loss gradients through the
# p-value need a differentiable, jit-traceable kernel). Brentq inside
# JIT is hard, so we use fine-grain trapezoidal (n_grid=8000) rather
# than the numpy-scalar's adaptive boundary finding. Precision ~4e-4
# at n=8000 is sufficient for gradient signal. The numpy scalar path
# (adaptive brentq) is the production-precision route for audit CIs.


_FR_JAX_N_GRID_DEFAULT: int = 8000


def _fr_geodesic_gaussian_jax(
    mu_a: jax.Array, sigma_a: jax.Array, mu_b: jax.Array, sigma_b: jax.Array, t: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """JAX-traceable Fisher-Rao geodesic on Gaussian endpoints. Vmap-friendly.

    Uses constant-speed parametrisation s(t) -> phi(t) (rev 1 correction).
    Branches on |mu_a - mu_b| < eps via jnp.where (no Python control flow).

    **Gradient correctness (Stage A audit finding #1):** the naive
    ``safe_denom = where(is_vertical, 1.0, denom)`` pattern catches
    forward NaNs at denom=0 but does NOT prevent reverse-mode gradient
    corruption when ``denom`` is small-but-above-threshold — autograd
    flows through ``c_tilde ~ 1/safe_denom`` whose Jacobian blows up
    as ``1/denom**2`` even though the eventual ``mu_t_tilde - c_tilde``
    and ``sigma`` outputs are bounded. The symbolic double-where trick
    used here computes BOTH branches with ``safe_denom`` (which equals
    1.0 in the vertical regime), then selects via ``where`` at the
    output level. The JAX-specific threshold ``_VERTICAL_CASE_EPS_JAX
    = 1e-6`` is wider than the numpy scalar's ``1e-12`` because the
    arc-branch derivative magnitudes don't stabilise until ~1e-7;
    ``1e-6`` adds a one-decade safety margin. Validated against
    central-difference FD in
    ``test_jax_geodesic_gradient_through_vertical``.
    """
    sqrt2 = jnp.sqrt(jnp.array(2.0))
    mu_a_t = mu_a / sqrt2
    mu_b_t = mu_b / sqrt2
    denom = mu_a_t - mu_b_t
    is_vertical = jnp.abs(denom) < _VERTICAL_CASE_EPS_JAX
    # Symbolic double-where: arc-branch computation uses safe_denom (= 1.0 in
    # vertical regime), so its forward value is finite and its reverse-mode
    # gradient is bounded irrespective of the true denom.
    safe_denom = jnp.where(is_vertical, jnp.array(1.0), denom)
    c_tilde_arc = ((mu_a_t * mu_a_t - mu_b_t * mu_b_t) + (sigma_a * sigma_a - sigma_b * sigma_b)) \
                  / (2.0 * safe_denom)
    r_arc = jnp.sqrt((mu_a_t - c_tilde_arc) ** 2 + sigma_a * sigma_a)
    phi_a = jnp.arctan2(sigma_a, mu_a_t - c_tilde_arc)
    phi_b = jnp.arctan2(sigma_b, mu_b_t - c_tilde_arc)
    # Constant-speed param: s(t) = (1-t)*ln(tan(phi_a/2)) + t*ln(tan(phi_b/2))
    s_a = jnp.log(jnp.tan(phi_a / 2.0))
    s_b = jnp.log(jnp.tan(phi_b / 2.0))
    s_t = (1.0 - t) * s_a + t * s_b
    phi_t = 2.0 * jnp.arctan(jnp.exp(s_t))
    mu_t_tilde_arc = c_tilde_arc + r_arc * jnp.cos(phi_t)
    s_sigma_arc = r_arc * jnp.sin(phi_t)
    # Vertical branch (bit-equal to closed-form geometric mean): sigma(t) =
    # sigma_a^(1-t) * sigma_b^t, mu unchanged.
    mu_t_tilde_vert = mu_a_t
    s_sigma_vert = sigma_a ** (1.0 - t) * sigma_b ** t
    mu_t_tilde = jnp.where(is_vertical, mu_t_tilde_vert, mu_t_tilde_arc)
    s_sigma = jnp.where(is_vertical, s_sigma_vert, s_sigma_arc)
    return sqrt2 * mu_t_tilde, s_sigma


@partial(jax.jit, static_argnames=("statistic_name", "n_grid"))
def _fr_tilted_pvalue_kernel(
    theta: jax.Array,
    eta: jax.Array,
    D: jax.Array,
    w: jax.Array,
    mu0: jax.Array,
    sigma: jax.Array,
    statistic_name: str,
    n_grid: int = _FR_JAX_N_GRID_DEFAULT,
) -> jax.Array:
    """JIT'd Fisher-Rao tilted p-value on Normal-Normal.

    Wald: closed-form ``2 * jsp_stats.norm.sf(|D - theta|/sigma)``.

    WALDO: trapezoidal quadrature over X under H_0 (no closed form at
    interior eta; rev 1 finding). **~4e-4 precision at n_grid=8000**
    (O(1/n) trap-rule on discontinuous indicator). Endpoints (eta=0,
    eta=1) go through quadrature too for jit-stability — they agree
    with closed-form bare-WALDO / bare-Wald to quadrature truncation.
    """
    if statistic_name == "wald":
        z = jnp.abs(D - theta) / sigma
        return 2.0 * jsp_stats.norm.sf(z)
    if statistic_name != "waldo":
        raise NotImplementedError(
            f"FisherRaoTilting JAX kernel: unknown statistic_name={statistic_name!r}."
        )
    sigma_n = jnp.sqrt(w) * sigma
    mu_n = w * D + (1.0 - w) * mu0
    mu_t_obs, s_t_obs = _fr_geodesic_gaussian_jax(mu_n, sigma_n, D, sigma, eta)
    tau_obs = (mu_t_obs - theta) ** 2 / (s_t_obs ** 2)
    X_grid = jnp.linspace(theta - _FR_X_RANGE * sigma, theta + _FR_X_RANGE * sigma, n_grid)
    # Vmap per-X geodesic + tau computation
    def _tau_at_X(X):
        mu_n_X = w * X + (1.0 - w) * mu0
        mu_t_X, s_t_X = _fr_geodesic_gaussian_jax(mu_n_X, sigma_n, X, sigma, eta)
        return (mu_t_X - theta) ** 2 / (s_t_X ** 2)
    tau_rep = jax.vmap(_tau_at_X)(X_grid)
    pdf = jsp_stats.norm.pdf(X_grid, loc=theta, scale=sigma)
    indicator = jnp.where(tau_rep >= tau_obs, 1.0, 0.0)
    p = jnp.trapezoid(pdf * indicator, X_grid)
    return p


# ----------------------------------------------------------------------
# FisherRaoTilting class
# ----------------------------------------------------------------------


@register_tilting(name="fisher_rao", brief="docs/methods/fisher_rao.md", status="implemented")
@dataclass(frozen=True)
class FisherRaoTilting:
    """Levi-Civita (Riemannian) geodesic of the Fisher information metric
    on the Gaussian half-plane manifold. eta=0 -> posterior, eta=1 ->
    likelihood-induced N(D, sigma^2).

    See docs/methods/fisher_rao.md for the derivation and predicted
    behavior.
    """

    name: ClassVar[str] = "fisher_rao"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        eta_likelihood_only=1.0,
        description=(
            "t in [0, 1] along the Fisher-Rao geodesic between posterior (t=0) "
            "and the likelihood-induced Gaussian N(D, sigma^2) (t=1)."
        ),
        # None matches PL/OT — let learned-η explore freely; mixture is
        # the only scheme with a hard structural bound. FR's Riemannian
        # geodesic is geometrically smooth for eta in R (except at the
        # sigma=0 boundary, which `_fr_geodesic_gaussian_scalar` checks
        # internally); there is no density-positivity constraint that
        # restricts eta to [0, 1].
        training_output_bounds=None,
    )
    selector: EtaSelector = field(default_factory=lambda: FixedEtaSelector(eta=0.0))

    def __str__(self) -> str:
        if isinstance(self.selector, FixedEtaSelector) and self.selector.eta == 0.0:
            return "fisher_rao"
        return f"fisher_rao[{self.selector.name}]"

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> Posterior:
        """Tilt along the Fisher-Rao Gaussian-half-plane geodesic.

        Stage A's `tilt()` admissibility is `σ(η) > 0` along the
        Gaussian-family geodesic (enforced inside
        `_fr_geodesic_gaussian_scalar`); Stage B's generic ParametricFamily
        path will add family-specific admissibility (e.g. α, β > 0 for
        Beta). The Riemannian geodesic itself is smooth for η ∈ ℝ — the
        [0, 1] segment is the "posterior ↔ likelihood interpolant"
        interpretation but η outside [0, 1] is a well-defined
        extrapolation along the same geodesic.
        """
        eta_f = float(np.asarray(eta).item())
        if not np.isfinite(eta_f):
            raise TiltingDomainError(
                f"FisherRaoTilting requires finite eta, got {eta_f!r}."
            )
        if not (
            isinstance(posterior, NormalDistribution)
            and isinstance(prior, NormalDistribution)
            and isinstance(likelihood, GaussianLikelihood)
        ):
            raise NotImplementedError(
                "FisherRaoTilting.tilt currently requires Normal posterior + "
                "Normal prior + Gaussian likelihood. Generic numerical path is "
                "implemented in Stage B (see _generic_tilt_fr); a future PR will "
                "extend tilt() dispatch to non-Gaussian endpoints."
            )
        mu_t, sigma_t = _fr_geodesic_gaussian_scalar(
            posterior.loc, posterior.scale, likelihood.D, likelihood.sigma, eta_f
        )
        if sigma_t <= 0.0:
            raise TiltingDomainError(
                f"FisherRaoTilting: sigma_t={sigma_t!r} non-positive at eta={eta_f!r} "
                f"(geodesic crossed the sigma=0 boundary)."
            )
        return NormalDistribution(loc=mu_t, scale=sigma_t)

    def path(
        self,
        posterior: Posterior,
        prior: Prior,
        likelihood: Likelihood,
        ts: NDArray[np.float64],
    ) -> Iterable[NormalDistribution]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, float(t))

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    def tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float | NDArray[np.float64],
        model: Model,
        prior: NormalDistribution,
        eta: ArrayLike,
        statistic_name: str,
    ) -> float | NDArray[np.float64]:
        """Selector-free Fisher-Rao tilted p-value on Normal-Normal.

        Scalar fast path for scalar (theta, eta, D) — uses the adaptive
        brentq quadrature. JAX kernel for batched inputs — uses fine-grain
        trap rule. See docstrings on the helpers for the precision
        characteristics of each path.
        """
        if not is_normal_normal(model):
            raise NotImplementedError(
                "FisherRaoTilting.tilted_pvalue currently requires NormalNormalModel; "
                f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "FisherRaoTilting.tilted_pvalue currently requires a NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        # Detect scalar vs batched
        theta_arr = np.asarray(theta)
        eta_arr = np.asarray(eta)
        D_arr = np.asarray(D)
        is_scalar = theta_arr.ndim == 0 and eta_arr.ndim == 0 and D_arr.ndim == 0
        if is_scalar:
            return _fr_tilted_pvalue_numpy_scalar(
                theta_f=float(theta_arr), eta_f=float(eta_arr), D_f=float(D_arr),
                w=w, mu0=mu0, sigma=sigma, statistic_name=statistic_name,
            )
        # Batched JAX path
        return np.asarray(_fr_tilted_pvalue_kernel(
            jnp.asarray(theta), jnp.asarray(eta), jnp.asarray(D),
            jnp.asarray(w), jnp.asarray(mu0), jnp.asarray(sigma), statistic_name,
        ))

    # ------------------------------------------------------------------
    # Selector-aware CI / regions / pvalue (Stage A.5)
    # ------------------------------------------------------------------

    def confidence_regions(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: "Config | None" = None,
    ) -> list[tuple[float, float]]:
        from .eta_selectors import (
            DynamicNumericalEtaSelector,
            LearnedDynamicEtaSelector,
            NumericalEtaSelector,
        )
        if not is_normal_normal(model):
            raise NotImplementedError(
                "FisherRaoTilting.confidence_regions currently requires NormalNormalModel; "
                "Stage B's generic numerical path will route here via WaldoStatistic(force_generic=True)."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "FisherRaoTilting.confidence_regions currently requires NormalDistribution prior."
            )
        D = float(np.asarray(data).mean())
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        statistic_name = getattr(statistic, "name", type(statistic).__name__.lower())
        if getattr(statistic, "force_generic", False):
            raise NotImplementedError(
                "FisherRaoTilting.confidence_regions: force_generic=True is "
                "the Stage B generic numerical path; not implemented in Stage A."
            )
        if isinstance(self.selector, FixedEtaSelector):
            eta_resolved = float(self.selector.eta)
            return self._static_confidence_regions(
                alpha=alpha, eta=eta_resolved, D=D, w=w, mu0=mu0, sigma=sigma,
                statistic_name=statistic_name,
            )
        if isinstance(self.selector, NumericalEtaSelector):
            eta_resolved = float(self.selector.select(
                self, data=data, model=model, prior=prior, alpha=alpha, statistic=statistic
            ))
            return self._static_confidence_regions(
                alpha=alpha, eta=eta_resolved, D=D, w=w, mu0=mu0, sigma=sigma,
                statistic_name=statistic_name,
            )
        if isinstance(self.selector, (DynamicNumericalEtaSelector, LearnedDynamicEtaSelector)):
            n_grid = int(getattr(self.selector, "n_grid", 401))
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            search_mult = float(getattr(self.selector, "search_mult", 8.0))

            def _tilted_pvalue_fn(theta_v: float, eta_v: float) -> float:
                return _fr_tilted_pvalue_numpy_scalar(
                    theta_f=float(theta_v), eta_f=float(eta_v), D_f=D,
                    w=w, mu0=mu0, sigma=sigma, statistic_name=statistic_name,
                )

            # Stage A audit finding #6: this vec helper is a Python loop
            # over scalar `_fr_tilted_pvalue_numpy_scalar` calls (vs OT's
            # true vectorisation), so dynamic-CI inversion on FR is slower
            # per cell than on OT. Stage A limitation; a proper batched
            # path is deferred to Stage B alongside the generic numerical
            # machinery.
            def _tilted_pvalue_vec_fn(
                theta_arr: np.ndarray, eta_arr: np.ndarray
            ) -> np.ndarray:
                out = np.empty(theta_arr.shape, dtype=np.float64)
                for i in range(theta_arr.shape[0]):
                    out[i] = _fr_tilted_pvalue_numpy_scalar(
                        theta_f=float(theta_arr[i]), eta_f=float(eta_arr[i]),
                        D_f=D, w=w, mu0=mu0, sigma=sigma,
                        statistic_name=statistic_name,
                    )
                return out

            regions, _, _ = dynamic_ci_scan(
                tilted_pvalue_fn=_tilted_pvalue_fn,
                tilted_pvalue_vec_fn=_tilted_pvalue_vec_fn,
                alpha=alpha,
                D=D,
                w=w,
                mu0=mu0,
                sigma=sigma,
                eta_selector=self.selector,
                scheme=self,
                statistic_name=statistic_name,
                n_grid=n_grid,
                coarse_n=coarse_n,
                search_mult=search_mult,
                model_fingerprint=model.fingerprint(),
                prior_fingerprint=prior.fingerprint(),
                model=model,
                prior=prior,
            )
            if not regions:
                raise RuntimeError(
                    f"FisherRaoTilting dynamic CI inversion produced no regions at D={D!r}"
                )
            return regions
        raise NotImplementedError(
            f"FisherRaoTilting: selector {type(self.selector).__name__} not yet wired."
        )

    def _static_confidence_regions(
        self, *, alpha: float, eta: float, D: float, w: float, mu0: float,
        sigma: float, statistic_name: str,
    ) -> list[tuple[float, float]]:
        """Brent-invert the closed-form scalar p-value at fixed eta.

        Uses brentq_with_doubling on f(theta) = p_FR(theta) - alpha,
        bracketing around (mu_t, s_t) at eta.
        """
        sigma_n = math.sqrt(w) * sigma
        mu_n = w * D + (1.0 - w) * mu0
        mu_t, s_t = _fr_geodesic_gaussian_scalar(mu_n, sigma_n, D, sigma, eta)

        def f(th: float) -> float:
            return _fr_tilted_pvalue_numpy_scalar(
                theta_f=th, eta_f=eta, D_f=D, w=w, mu0=mu0, sigma=sigma,
                statistic_name=statistic_name,
            ) - alpha

        lo = brentq_with_doubling(
            f, midpoint=mu_t, initial_half_width=s_t, direction=-1, max_doublings=20,
        )
        hi = brentq_with_doubling(
            f, midpoint=mu_t, initial_half_width=s_t, direction=+1, max_doublings=20,
        )
        return [(float(lo), float(hi))]

    def confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: "Config | None" = None,
    ) -> tuple[float, float]:
        regions = self.confidence_regions(alpha, data, model, prior, statistic, config=config)
        los = [lo for lo, _ in regions]
        his = [hi for _, hi in regions]
        return (float(min(los)), float(max(his)))

    def pvalue(
        self,
        theta: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> NDArray[np.float64]:
        from .eta_selectors import (
            DynamicNumericalEtaSelector,
            LearnedDynamicEtaSelector,
        )
        if not is_normal_normal(model):
            raise NotImplementedError(
                "FisherRaoTilting.pvalue currently requires NormalNormalModel."
            )
        if getattr(statistic, "force_generic", False):
            raise NotImplementedError(
                "FisherRaoTilting.pvalue: force_generic=True is the Stage B "
                "generic numerical path; not implemented in Stage A."
            )
        D = float(np.asarray(data).mean())
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        statistic_name = getattr(statistic, "name", type(statistic).__name__.lower())
        thetas = np.atleast_1d(np.asarray(theta))
        if isinstance(self.selector, (DynamicNumericalEtaSelector, LearnedDynamicEtaSelector)):
            etas = self.selector.select_grid(
                thetas, scheme=self, model=model, prior=prior, alpha=0.05,
                statistic=statistic,
            )
        else:
            eta_single = self.selector.select(
                self, data=data, model=model, prior=prior, alpha=0.05, statistic=statistic,
            )
            etas = np.full_like(thetas, float(eta_single))
        etas_arr = np.asarray(etas, dtype=np.float64)
        pvals = np.empty_like(thetas, dtype=np.float64)
        for i, (th, et) in enumerate(zip(thetas, etas_arr)):
            pvals[i] = _fr_tilted_pvalue_numpy_scalar(
                theta_f=float(th), eta_f=float(et), D_f=D, w=w, mu0=mu0, sigma=sigma,
                statistic_name=statistic_name,
            )
        return pvals
