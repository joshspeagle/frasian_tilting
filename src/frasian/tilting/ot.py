"""Optimal-transport (Wasserstein-2) geodesic tilting.

The W2 geodesic between two 1D distributions p and q at parameter
`t in [0, 1]` is the **quantile-mixture**: the distribution whose
quantile function is the linear interpolation of the endpoint
quantiles,

    F_t^{-1}(u) = (1 - t) * F_p^{-1}(u) + t * F_q^{-1}(u),  u in [0, 1].

This is a general result for any two endpoints exposing `quantile`,
which is in the framework's `Distribution` protocol — so OT tilting
applies to *any* (posterior, likelihood-as-distribution) pair, not
just Gaussians. On the Normal-Normal sandbox, the geodesic stays in
the Gaussian family with the closed form

    mu_t = (1 - t) * mu_a + t * mu_b
    sigma_t = (1 - t) * sigma_a + t * sigma_b

(linear in `(mu, sigma)`, *not* in `(mu, sigma^2)`). The tilt method
recognises this Gaussian fast path and returns a `NormalDistribution`
directly; non-Gaussian endpoints fall back to a `QuantileMixturePath`
wrapper that derives `pdf` / `cdf` numerically.

Endpoints follow the framework's posterior <-> likelihood convention
(matching `power_law`): eta=0 -> posterior, eta=1 -> likelihood-induced
Gaussian N(D, sigma^2). At eta=0 the tilted-WALDO p-value reduces to
bare WALDO; at eta=1 it reduces to bare two-sided Wald — so OT is a
*different* (smoother) path between the same WALDO/Wald endpoints
that `power_law` already interpolates between.

For the (ot, waldo) cell on Normal-Normal, the tilted p-value has the
closed form

    s_t        = (w + eta * (1 - w)) * sigma
    mu_t       = (1 - eta) * mu_n + eta * D
    a(theta)   = |mu_t - theta| / s_t
    b(theta)   = (1 - eta) * (1 - w) * (mu0 - theta) / s_t
    p(theta)   = Phi(b - a) + Phi(-a - b)

derived by substituting the OT-tilted (mu_t, s_t) for (mu_n, w*sigma)
in WALDO's two-Gaussian-CDF formula. See the brief at
docs/methods/ot.md for the full derivation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar, cast

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray

# scipy: used by the numpy-eager scalar fast path inside tilted_pvalue.
# Same boundary discipline as power_law.py — see that file's comments
# for the rationale (~10 us scalar vs ~200 us under JAX dispatch).
from scipy import stats as _scalar_scipy_stats

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..models._dispatch import is_normal_normal
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..statistics.base import TestStatistic
from .base import EtaSelector, ParamSpec
from .eta_selectors import FixedEtaSelector
from .quantile_mixture import QuantileMixturePath

if TYPE_CHECKING:
    from ..config import Config

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _ot_tilted_pvalue_numpy_scalar(
    theta_f: float,
    eta_f: float,
    D_f: float,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> float:
    """Numpy-eager scalar fast path. Mirrors `_ot_tilted_pvalue_kernel`
    but runs on Python floats + scipy.stats.norm. Used when caller
    passes scalar (theta, eta, D) — the brentq inner-loop pattern.
    ~10 us/call.
    """
    if statistic_name == "wald":
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))
    if statistic_name == "waldo":
        mu_n = w * D_f + (1.0 - w) * mu0
        mu_t = (1.0 - eta_f) * mu_n + eta_f * D_f
        s_t = (w + eta_f * (1.0 - w)) * sigma
        a = abs(mu_t - theta_f) / s_t
        b = (1.0 - eta_f) * (1.0 - w) * (mu0 - theta_f) / s_t
        return float(
            _scalar_scipy_stats.norm.cdf(b - a) + _scalar_scipy_stats.norm.cdf(-a - b)
        )
    raise NotImplementedError(
        f"_ot_tilted_pvalue_numpy_scalar not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo'."
    )


@partial(jax.jit, static_argnames=("statistic_name",))
def _ot_tilted_pvalue_kernel(
    theta: jax.Array,
    eta: jax.Array,
    D: jax.Array,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> jax.Array:
    """Pure JAX arithmetic kernel for the OT-tilted p-value.

    Same shape / discipline / jit rationale as
    `power_law._tilted_pvalue_kernel`. Scalar inputs do NOT reach
    this kernel — the public method's shape dispatch routes them to
    the numpy fast path.
    """
    if statistic_name == "wald":
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - jsp_stats.norm.cdf(z))
    if statistic_name == "waldo":
        mu_n = w * D + (1.0 - w) * mu0
        mu_t = (1.0 - eta) * mu_n + eta * D
        s_t = (w + eta * (1.0 - w)) * sigma
        a = jnp.abs(mu_t - theta) / s_t
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / s_t
        return jsp_stats.norm.cdf(b - a) + jsp_stats.norm.cdf(-a - b)
    raise NotImplementedError(
        f"_ot_tilted_pvalue_kernel not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo'."
    )


def _data_to_scalar_D(data: NDArray[np.float64]) -> float:
    """Coerce ``data`` to a single scalar D for the n=1 sandbox.

    The framework's Normal-Normal contract is single-observation
    (CLAUDE.md "1D conjugate Normal-Normal sandbox"). Earlier code
    silently used ``data.mean()`` which produced wrong CI widths for
    n>1 (the effective σ would shrink as σ/√n, not σ). Make the
    assumption explicit: refuse n>1 with a clear message rather than
    silently mis-scaling. Tier 1.5-O8 in the audit.
    """
    arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if arr.size != 1:
        raise NotImplementedError(
            f"Normal-Normal sandbox is single-observation (n=1); got "
            f"data.size={arr.size}. For n>1, use sigma_eff=sigma/sqrt(n) "
            f"and pass data.mean() with a model whose sigma is sigma_eff."
        )
    # Use ``arr.item()`` so shape-(1,1) and other non-flat single-element
    # inputs reduce cleanly. ``float(arr[0])`` would crash with
    # ``TypeError: only 0-dimensional arrays can be converted to Python
    # scalars`` on a shape-(1,1) input where ``arr[0]`` is a shape-(1,)
    # array. Phase 5 skeptic vector #3.
    return float(arr.item())


# ----- Generic numerical path (any Distribution-conforming inputs) -----

# Default knobs for the generic MC tilted-pvalue path. Mirrors PowerLaw
# (see `power_law.py::_GENERIC_TILTED_PVALUE_*`). `_GENERIC_TILTED_PVALUE_BASE_SEED`
# is sourced from `_generic_pvalue` so PowerLaw and OT share the same
# CRN seed at fixed (data, prior, eta, alpha) — enables direct cross-
# scheme MC comparison in the smoothness experiment.
from ._generic_pvalue import _GENERIC_TILTED_PVALUE_BASE_SEED  # noqa: F401

_GENERIC_TILT_N_GRID: int = 1024
_GENERIC_TILTED_PVALUE_N_MC: int = 200
_GENERIC_TILTED_PVALUE_N_GRID_MC: int = 256


def _generic_tilt_ot(
    posterior: Posterior,
    likelihood: Likelihood,
    eta: float,
    *,
    model: object,
    data: NDArray[np.float64],
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
):
    """Numerical W2 tilt: QuantileMixturePath between posterior and
    likelihood-as-distribution at parameter ``t = eta``.

    The geodesic *segment* is ``t in [0, 1]``; extrapolation along the
    W2 displacement line is admissible whenever the resulting law is
    still a valid distribution (audit P0-4). The monotonicity check
    that gates extrapolation is enforced by ``QuantileMixturePath``'s
    own ``__post_init__`` validator — this helper only refuses
    non-finite ``eta`` upfront so callers don't pay the
    likelihood-as-distribution construction cost on a doomed call.

    Endpoints:
      - p = posterior (passed in, already a Distribution).
      - q = likelihood_as_distribution(model, data, support, n_grid).

    Both expose `.quantile(u)` so the W2 geodesic is the linear
    interpolation of their quantile functions — that's exactly what
    `QuantileMixturePath` implements. Returns a QuantileMixturePath
    (also a Distribution).
    """
    from ._generic_pvalue import likelihood_as_distribution

    if not np.isfinite(float(eta)):
        raise TiltingDomainError(
            f"OTTilting requires finite eta, got {float(eta)!r}."
        )
    q = likelihood_as_distribution(model, data, support, n_grid=n_grid)
    # Monotonicity probe at construction time (QuantileMixturePath
    # __post_init__) raises ValueError for non-monotone extrapolation;
    # callers in the validity helper / CI inversion already catch
    # ValueError → NaN, so non-Gaussian extrapolation that violates
    # the inverse-density slope balance fails loudly at the right gate.
    return QuantileMixturePath(p=posterior, q=q, t=float(eta))


def _generic_tilted_moments_ot(
    posterior: Posterior,
    likelihood: Likelihood,
    eta: float,
    *,
    model: object,
    data: NDArray[np.float64],
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
) -> tuple[float, float]:
    """(mean, var) of the OT-tilted distribution at parameter t = eta.

    Builds the QuantileMixturePath via `_generic_tilt_ot`; reads off
    `mean()` (closed-form linear-in-t) and `var()` (Gauss-Legendre
    on the quantile, see `quantile_mixture.py::QuantileMixturePath.var`).
    Used as the hot inner kernel of `_generic_tilted_pvalue_ot`.
    """
    qmp = _generic_tilt_ot(
        posterior, likelihood, eta,
        model=model, data=data, support=support, n_grid=n_grid,
    )
    return float(qmp.mean()), float(qmp.var())


def _generic_tilted_t_statistic_ot(
    theta_f: float,
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta: float,
    *,
    support: tuple[float, float],
    n_grid: int = _GENERIC_TILT_N_GRID,
) -> float:
    """t = (mu_tilted - theta)^2 / sigma_tilted^2 at observed data.

    OT analogue of `power_law._generic_tilted_t_statistic`. Constructs
    the posterior + likelihood at observed data, then the tilted
    moments via `_generic_tilted_moments_ot`.
    """
    posterior = model.posterior(data, prior)
    likelihood = model.likelihood(data)
    mu, var = _generic_tilted_moments_ot(
        posterior, likelihood, eta,
        model=model, data=data, support=support, n_grid=n_grid,
    )
    var_safe = max(var, 1e-300)
    diff = mu - theta_f
    return diff * diff / var_safe


def _generic_tilted_pvalue_ot(
    theta: float,
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta: float,
    statistic_name: str,
    *,
    n_mc: int = _GENERIC_TILTED_PVALUE_N_MC,
    derived_seed: int | None = None,
    alpha: float = 0.0,
    base_seed: int = _GENERIC_TILTED_PVALUE_BASE_SEED,
    obs_moments: tuple[float, float] | None = None,
) -> float:
    """Generic MC tilted p-value for OTTilting on any (model, prior).

    Mirrors `power_law._generic_tilted_pvalue` structurally:
    - statistic_name="wald": eta-independent, delegates to WaldStatistic.
    - statistic_name="waldo": MC reference under H_0 via
      `model.sample_data(theta, ...)`, recompute t per draw using
      `_generic_tilted_t_statistic_ot`. Conservative `(k+1)/(n+1)`
      smoothing. CRN-seeded via blake2b stable hash.
    """
    from ..statistics.wald import WaldStatistic
    from ._generic_pvalue import _resolve_support, _stable_tilted_pvalue_seed

    if statistic_name == "wald":
        return float(np.asarray(WaldStatistic()._generic_pvalue(theta, data, model)))
    if statistic_name != "waldo":
        raise NotImplementedError(
            f"OTTilting generic tilted_pvalue not implemented for "
            f"statistic={statistic_name!r}; supported: 'wald', 'waldo'."
        )

    data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if data_arr.ndim != 1:
        raise NotImplementedError(
            f"OTTilting generic tilted_pvalue expects 1-D data; got data.ndim={data_arr.ndim}."
        )

    support = _resolve_support(model, data_arr)
    theta_f = float(theta)
    eta_f = float(eta)
    # Audit P0-4: OT is well-defined on the full W2 displacement line.
    # Per-eta admissibility (monotonicity of the linear quantile combo)
    # is enforced inside `_generic_tilt_ot` → `QuantileMixturePath`;
    # only refuse non-finite eta here, before the seed-derivation cost.
    if not np.isfinite(eta_f):
        raise TiltingDomainError(
            f"OTTilting requires finite eta, got {eta_f!r}."
        )

    if derived_seed is None:
        derived_seed = _stable_tilted_pvalue_seed(
            data_arr, model, prior, eta_f, alpha, base_seed
        )

    # Hoist observed moments (skeptic finding from PowerLaw 3c-fix1).
    if obs_moments is not None:
        mu_obs, var_obs = obs_moments
    else:
        posterior_obs = model.posterior(data_arr, prior)
        likelihood_obs = model.likelihood(data_arr)
        mu_obs, var_obs = _generic_tilted_moments_ot(
            posterior_obs, likelihood_obs, eta_f,
            model=model, data=data_arr, support=support,
            n_grid=_GENERIC_TILT_N_GRID,
        )
    var_obs_safe = max(var_obs, 1e-300)
    diff_obs = mu_obs - theta_f
    t_obs = diff_obs * diff_obs / var_obs_safe

    rng = np.random.default_rng(derived_seed)
    n_obs = int(data_arr.size)
    t_samples = np.empty(n_mc, dtype=np.float64)
    n_collapsed = 0
    for i in range(n_mc):
        D_prime = model.sample_data(theta_f, rng, n_obs)
        try:
            t_samples[i] = _generic_tilted_t_statistic_ot(
                theta_f, D_prime, model, prior, eta_f, support=support,
                n_grid=_GENERIC_TILTED_PVALUE_N_GRID_MC,
            )
        except (ValueError, RuntimeError, ArithmeticError):
            t_samples[i] = 0.0
            n_collapsed += 1
    if n_collapsed > n_mc // 2:
        import warnings
        warnings.warn(
            f"OTTilting._generic_tilted_pvalue: {n_collapsed}/{n_mc} MC samples "
            f"collapsed (theta={theta_f}, eta={eta_f}); empirical p is "
            f"strongly biased upward. Increase data size or reduce eta.",
            RuntimeWarning,
            stacklevel=2,
        )
    p = (float(np.sum(t_samples >= t_obs)) + 1.0) / (float(n_mc) + 1.0)
    return float(p)


def _generic_tilted_confidence_interval_ot(
    alpha: float,
    data: NDArray[np.float64],
    model: object,
    prior: Prior,
    eta: float,
    statistic_name: str,
    *,
    n_mc: int = _GENERIC_TILTED_PVALUE_N_MC,
    base_seed: int = _GENERIC_TILTED_PVALUE_BASE_SEED,
) -> tuple[float, float]:
    """Generic CI inversion for OTTilting via brentq + CRN.

    Mirrors `power_law._generic_tilted_confidence_interval` structurally,
    including the explicit boundary detection from Phase 3c-fix1.
    """
    from .._errors import BracketingFailed
    from ._generic_pvalue import _resolve_support, _stable_tilted_pvalue_seed
    from ._solvers import brentq_with_doubling

    data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    eta_f = float(eta)
    # Skeptic LOW #8: validate eta at function entry rather than letting
    # _generic_tilt_ot raise mid-inversion. Saves the seed-derivation +
    # posterior-construction cost on a doomed call. Audit P0-4: only
    # finite-eta is required upfront — admissibility along the W2
    # displacement line is enforced by `QuantileMixturePath`'s runtime
    # monotonicity probe, not a [0, 1] interval clamp.
    if not np.isfinite(eta_f):
        raise TiltingDomainError(
            f"OTTilting requires finite eta, got {eta_f!r}."
        )
    derived_seed = _stable_tilted_pvalue_seed(
        data_arr, model, prior, eta_f, alpha, base_seed
    )

    support = _resolve_support(model, data_arr)
    support_lo, support_hi = support

    # Hoist observed moments.
    posterior_at_obs = model.posterior(data_arr, prior)
    likelihood_at_obs = model.likelihood(data_arr)
    mu_obs, var_obs = _generic_tilted_moments_ot(
        posterior_at_obs, likelihood_at_obs, eta_f,
        model=model, data=data_arr, support=support,
        n_grid=_GENERIC_TILT_N_GRID,
    )
    var_obs_safe = max(var_obs, 1e-300)
    sigma_tilted = float(np.sqrt(var_obs_safe))

    def f(theta_val: float) -> float:
        theta_safe = max(support_lo, min(support_hi, float(theta_val)))
        return _generic_tilted_pvalue_ot(
            theta_safe,
            data_arr,
            model,
            prior,
            eta_f,
            statistic_name,
            n_mc=n_mc,
            derived_seed=derived_seed,
            alpha=alpha,
            base_seed=base_seed,
            obs_moments=(mu_obs, var_obs),
        ) - alpha

    half = max(4.0 * sigma_tilted, 1e-3)

    # Explicit boundary detection (Phase 3c-fix1).
    if np.isfinite(support_lo):
        ci_extends_below = (f(support_lo) >= 0.0)
    else:
        ci_extends_below = False
    if np.isfinite(support_hi):
        ci_extends_above = (f(support_hi) >= 0.0)
    else:
        ci_extends_above = False

    if ci_extends_below:
        lower = support_lo
    else:
        try:
            lower = brentq_with_doubling(
                f, midpoint=mu_obs, initial_half_width=half, direction=-1
            )
        except BracketingFailed:
            lower = support_lo
    if ci_extends_above:
        upper = support_hi
    else:
        try:
            upper = brentq_with_doubling(
                f, midpoint=mu_obs, initial_half_width=half, direction=+1
            )
        except BracketingFailed:
            upper = support_hi
    return (max(lower, support_lo), min(upper, support_hi))


@register_tilting(name="ot", brief="docs/methods/ot.md")
@dataclass(frozen=True)
class OTTilting:
    """Wasserstein-2 geodesic tilting (general 1D, Gaussian fast path)."""

    name: ClassVar[str] = "ot"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description=(
            "Position along the W2 displacement line through posterior "
            "(eta=0) and likelihood-induced Gaussian N(D, sigma^2) "
            "(eta=1). The geodesic *segment* is [0, 1]; eta outside that "
            "extrapolates along the same line and is admissible whenever "
            "the resulting distribution is well-defined (Gaussian path: "
            "sigma_t > 0; closed-form pvalue: s_t > 0 ⇔ eta > -w/(1-w))."
        ),
    )
    selector: EtaSelector = field(default_factory=lambda: FixedEtaSelector(eta=0.0))

    @property
    def cell_name(self) -> str:
        sel_name = getattr(self.selector, "name", "")
        if isinstance(self.selector, FixedEtaSelector) and self.selector.eta == 0.0:
            return self.name
        return f"{self.name}[{sel_name}]"

    # ----- TiltingScheme protocol -----

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> Posterior:
        """W2-geodesic tilt between posterior and likelihood-as-distribution.

        Three dispatch paths:
        - **Gaussian fast path**: closed-form linear interpolation in
          (mu, sigma) when both posterior and likelihood are Gaussian.
          Returns `NormalDistribution`.
        - **Gaussian-likelihood quantile-mixture**: when likelihood is
          `GaussianLikelihood` but posterior isn't (rare on the
          Normal-Normal sandbox). Constructs `q = N(D, sigma)`
          directly and wraps in `QuantileMixturePath`.
        - **Generic numerical (Phase 3d)**: when neither path applies.
          Constructs `q` as a `GridDistribution` from `log L(theta)`
          on the model support; returns `QuantileMixturePath(p=posterior,
          q=q, t=eta)`. Works for `(BernoulliModel, BetaDistribution)`
          and other (model, prior) pairs.

        ``data`` is required for the generic path (to evaluate
        `log L(theta)`). Callers from `confidence_regions` etc.
        plumb it through; direct callers must use the
        `data`-aware OT generic helpers.
        """
        eta_arr = np.asarray(eta, dtype=np.float64)
        if eta_arr.ndim != 0:
            raise NotImplementedError(
                "tilt() expects scalar eta; vectorised eta is consumed via "
                "repeated scalar calls (see `path`)."
            )
        t = float(eta_arr)
        if not np.isfinite(t):
            raise TiltingDomainError(f"OTTilting requires finite eta, got {t!r}.")

        # Gaussian fast path: linear interpolation in (mu, sigma). The W2
        # displacement line is well-defined for any finite t, but the
        # output is only a valid Gaussian when sigma_t > 0.
        # Audit P1 H.6: `prior` is intentionally unused on this path —
        # OT interpolates between posterior and the likelihood-induced
        # Gaussian; the prior is already absorbed in `posterior`. The
        # parameter stays in the protocol signature for uniformity
        # with `power_law`/`mixture` etc. that DO consume the prior.
        if isinstance(posterior, NormalDistribution) and isinstance(likelihood, GaussianLikelihood):
            mu_a, sigma_a = posterior.loc, posterior.scale
            mu_b, sigma_b = float(likelihood.D), float(likelihood.sigma)
            sigma_t = (1.0 - t) * sigma_a + t * sigma_b
            if sigma_t <= 0.0:
                raise TiltingDomainError(
                    f"OTTilting Gaussian fast path requires sigma_t > 0, got "
                    f"sigma_t={sigma_t!r} at eta={t!r} "
                    f"(sigma_post={sigma_a!r}, sigma_lik={sigma_b!r})."
                )
            return NormalDistribution(
                loc=(1.0 - t) * mu_a + t * mu_b,
                scale=sigma_t,
            )

        # Gaussian-likelihood quantile-mixture path (likelihood admits a
        # Distribution view directly).
        if isinstance(likelihood, GaussianLikelihood):
            q = NormalDistribution(loc=likelihood.D, scale=likelihood.sigma)
            return QuantileMixturePath(p=posterior, q=q, t=t)

        # Phase 3d generic numerical path. We need access to the model
        # and data to construct the likelihood-as-distribution; the
        # `tilt()` signature doesn't carry them, so we cannot compute
        # this path directly here. Surface a clear error pointing
        # callers to the data-aware entry point in `confidence_regions`.
        raise NotImplementedError(
            f"OTTilting.tilt() can't construct the generic numerical path "
            f"from a {type(likelihood).__name__!r} without knowing the model "
            f"and data — use `confidence_regions(alpha, data, model, prior, "
            f"statistic)` (which routes through the generic path internally) "
            f"or call `_generic_tilt_ot(posterior, likelihood, eta, model=, "
            f"data=, support=)` directly."
        )

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[Posterior]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, float(t))

    def is_identity(self, eta: float) -> bool:
        return float(eta) == self.param_space.eta_identity

    # ----- (TiltingScheme, TestStatistic) cross-product specialisations -----

    def tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float | NDArray[np.float64],
        model: Model,
        prior: NormalDistribution,
        eta: ArrayLike,
        statistic_name: str,
    ) -> float | jax.Array:
        """Tilted p-value evaluated against the W2-tilted Gaussian.

        Specialized for (ot, waldo) and (ot, wald) on Normal-Normal:

          (ot, wald): eta-independent two-sided Wald, 2 * Phi(-|D-theta|/sigma)
          (ot, waldo): closed form derived in docs/methods/ot.md, with the
            standard error s_t = (w + eta*(1-w))*sigma.

        Endpoint sanity: at eta=0 reduces to bare WALDO; at eta=1 reduces
        to bare Wald (s_t -> sigma, mu_t -> D, b -> 0, a -> |D-theta|/sigma).

        Returns ``jax.Array`` for bulk inputs (length >= 2 in any of
        theta/eta/D); returns Python float for the all-scalar fast
        path. Same shape-dispatch + return-type discipline as
        `PowerLawTilting.tilted_pvalue` — see that method's docstring
        for the rationale.
        """
        from ..models.normal_normal import NormalNormalModel

        if not is_normal_normal(model):
            raise NotImplementedError(
                "OTTilting.tilted_pvalue currently requires NormalNormalModel; "
                f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "OTTilting.tilted_pvalue currently requires a NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        # Validation runs in numpy (JAX can't raise mid-trace). The
        # closed-form WALDO p-value is parameterised by s_t = (w +
        # eta*(1-w))*sigma; it is well-defined whenever s_t > 0, i.e.
        # eta > -w/(1-w). Above that there is no upper bound from s_t.
        eta_np = np.asarray(eta, dtype=np.float64)
        eta_lower = -w / (1.0 - w)  # exclusive boundary
        invalid = ~(np.isfinite(eta_np) & (eta_np > eta_lower))
        if np.any(invalid):
            if eta_np.ndim == 0:
                raise TiltingDomainError(
                    f"OTTilting.tilted_pvalue requires eta > {eta_lower!r} "
                    f"(s_t > 0) and finite, got {float(eta_np)!r}."
                )
            bad = int(np.argmax(invalid))
            raise TiltingDomainError(
                f"OTTilting.tilted_pvalue requires eta > {eta_lower!r} and "
                f"finite; offending index {bad} eta={float(eta_np.flat[bad])!r}."
            )

        # Shape dispatch: scalar -> numpy fast path; bulk -> jit'd kernel.
        theta_np = np.asarray(theta, dtype=np.float64)
        D_np = np.asarray(D, dtype=np.float64)
        if theta_np.size == 1 and eta_np.size == 1 and D_np.size == 1:
            # Returns Python float, NOT jnp.asarray(float) — see the
            # corresponding note in PowerLawTilting.tilted_pvalue. The
            # signature's `float | jax.Array` union makes the relaxed
            # return type honest at the type level.
            return _ot_tilted_pvalue_numpy_scalar(
                float(theta_np.item()),
                float(eta_np.item()),
                float(D_np.item()),
                w,
                mu0,
                sigma,
                statistic_name,
            )
        # Bulk JAX kernel — autodiff-clean, no Python control flow.
        theta_arr = jnp.asarray(theta, dtype=jnp.float64)
        eta_arr = jnp.asarray(eta, dtype=jnp.float64)
        D_arr = jnp.asarray(D, dtype=jnp.float64)
        return _ot_tilted_pvalue_kernel(theta_arr, eta_arr, D_arr, w, mu0, sigma, statistic_name)

    def tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: Model,
        prior: NormalDistribution,
        eta: float,
        statistic_name: str,
    ) -> tuple[float, float]:
        """Numerical CI inversion of `tilted_pvalue` via brentq_with_doubling."""
        from ..models.normal_normal import NormalNormalModel
        from ._solvers import brentq_with_doubling

        if not is_normal_normal(model):
            raise NotImplementedError(
                "OTTilting.tilted_confidence_interval currently requires " "NormalNormalModel."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "OTTilting.tilted_confidence_interval currently requires a "
                "NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)
        mu_n = w * D + (1.0 - w) * mu0

        mid = float(D) if statistic_name == "wald" else float((1.0 - eta) * mu_n + eta * D)  # waldo

        def f(theta_val: float) -> float:
            return (
                float(self.tilted_pvalue(float(theta_val), D, model, prior, eta, statistic_name))
                - alpha
            )

        half = 4.0 * sigma
        lo = brentq_with_doubling(f, midpoint=mid, initial_half_width=half, direction=-1)
        hi = brentq_with_doubling(f, midpoint=mid, initial_half_width=half, direction=+1)
        return (lo, hi)

    def dynamic_tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float,
        model: Model,
        prior: NormalDistribution,
        statistic_name: str,
        eta_at_theta: ArrayLike,
    ) -> NDArray[np.float64]:
        """p(theta) with eta varying per theta via a precomputed lookup."""
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        eta_arr = np.atleast_1d(np.asarray(eta_at_theta, dtype=np.float64))
        if theta_arr.shape != eta_arr.shape:
            raise ValueError(
                f"theta and eta_at_theta must have the same shape; got "
                f"{theta_arr.shape!r} and {eta_arr.shape!r}."
            )
        # eta validation now lives in `tilted_pvalue` (single source of
        # truth; was previously duplicated here for the Phase 1 skeptic
        # vector #5 contract). Removing the upfront check eliminates
        # ~5 µs of redundant work per scan and keeps error messages
        # consistent regardless of whether a caller hits this dynamic
        # entry-point or the inner `tilted_pvalue` directly (e.g. through
        # the vec_fn callback at `_dynamic.py:151`).
        # Vectorised path: `tilted_pvalue` now broadcasts over array eta,
        # so one bulk call replaces the scalar Python loop (Tier 1.3 N3).
        out = np.asarray(
            self.tilted_pvalue(theta_arr, D, model, prior, eta_arr, statistic_name),
            dtype=np.float64,
        )
        return out if out.size > 1 else np.asarray(float(out.item()))

    def dynamic_tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: Model,
        prior: NormalDistribution,
        statistic_name: str,
        eta_selector,
        n_grid: int = 401,
        coarse_n: int = 25,
        search_mult: float = 8.0,
    ) -> tuple[list[tuple[float, float]], float, int]:
        """Dynamic-eta CI: eta = eta*(|Delta(theta)|) per theta.

        Delegates to
        `frasian.tilting._dynamic.dynamic_ci_scan`; the scheme-specific
        bit is the `tilted_pvalue` closure.
        """
        from ..models.normal_normal import NormalNormalModel
        from ._dynamic import dynamic_ci_scan

        if not is_normal_normal(model):
            raise NotImplementedError(
                "OTTilting.dynamic_tilted_confidence_interval currently "
                "requires NormalNormalModel."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        def _tilted_pvalue_fn(theta: float, eta: float) -> float:
            return float(
                self.tilted_pvalue(
                    theta,
                    D,
                    model,
                    prior,
                    eta,
                    statistic_name,
                )
            )

        def _tilted_pvalue_vec_fn(
            theta_arr: np.ndarray, eta_arr: np.ndarray
        ) -> np.ndarray:
            # Bulk path: tilted_pvalue broadcasts over array eta when
            # eta has the same shape as theta. Single numpy/scipy call
            # replaces the scalar Python loop in dynamic_ci_scan
            # (Tier 1.3 N1).
            return np.asarray(
                self.tilted_pvalue(
                    theta_arr, D, model, prior, eta_arr, statistic_name
                ),
                dtype=np.float64,
            )

        return dynamic_ci_scan(
            tilted_pvalue_fn=_tilted_pvalue_fn,
            tilted_pvalue_vec_fn=_tilted_pvalue_vec_fn,
            alpha=alpha,
            D=D,
            w=w,
            mu0=mu0,
            sigma=sigma,
            eta_selector=eta_selector,
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

    # ----- Uniform CI / regions / pvalue interface -----

    def _require_normal_sandbox(self, model: Model, prior: Prior) -> None:
        """Enforce Normal-Normal sandbox. Used by paths that have no
        generic fallback (currently the dynamic-η selector path).
        """
        from ..models.normal_normal import NormalNormalModel

        if not is_normal_normal(model):
            raise NotImplementedError(
                f"OTTilting requires NormalNormalModel for the uniform CI "
                f"interface; got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                f"OTTilting requires a NormalDistribution prior; " f"got {type(prior).__name__!r}."
            )

    @staticmethod
    def _is_normal_normal_pair(model: Model, prior: Prior) -> bool:
        """Predicate-only counterpart to `_require_normal_sandbox`."""
        from ..models.normal_normal import NormalNormalModel

        return is_normal_normal(model) and isinstance(prior, NormalDistribution)

    def confidence_regions(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: Config | None = None,
    ) -> list[tuple[float, float]]:
        """Selector-aware region list. Single-element for static selectors;
        multi-element for dynamic selectors at conflict-band D where the
        dynamic p-value is multimodal.

        ``config`` (optional, kw-only): when supplied, the dynamic-CI
        scan reads ``n_grid`` / ``coarse_n`` / ``search_mult`` from
        ``Config.dynamic_*``. When ``None`` (default), falls back to
        the selector's own attributes for backward compatibility.
        Skeptic Phase 5 vector #2.

        Phase 3d: non-Normal-Normal pairings supported via the generic
        numerical path with a STATIC selector. Dynamic selectors still
        require Normal-Normal (the dynamic scanner builds its θ-window
        from `D ± search_mult * sigma`, which is Normal-Normal-flavoured).
        """
        if not self._is_normal_normal_pair(model, prior):
            if getattr(self.selector, "is_dynamic", False):
                raise NotImplementedError(
                    "OTTilting dynamic-η selector currently requires "
                    "NormalNormalModel + NormalDistribution prior. Use a "
                    "static selector (FixedEtaSelector) for non-Normal-Normal "
                    "pairings."
                )
            eta = float(
                self.selector.select(
                    self,
                    data=data,
                    model=model,
                    prior=prior,
                    alpha=alpha,
                    statistic=statistic,
                )
            )
            lo, hi = _generic_tilted_confidence_interval_ot(
                alpha, data, model, prior, eta, statistic.name
            )
            return [(lo, hi)]

        self._require_normal_sandbox(model, prior)
        # Narrow types after the dispatch check (mypy can't infer through it).
        # `cast` is `-O`-safe; the runtime gate is `_require_normal_sandbox` above.
        from ..models.normal_normal import NormalNormalModel

        model = cast(NormalNormalModel, model)
        prior = cast(NormalDistribution, prior)
        D = _data_to_scalar_D(data)
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
            if config is not None:
                n_grid = int(config.dynamic_n_grid)
                coarse_n = int(config.dynamic_coarse_n)
                search_mult = float(config.dynamic_search_mult)
            else:
                n_grid = int(getattr(self.selector, "n_grid", 401))
                coarse_n = int(getattr(self.selector, "coarse_n", 25))
                search_mult = float(getattr(self.selector, "search_mult", 8.0))
            regions, _, _ = self.dynamic_tilted_confidence_interval(
                alpha,
                D,
                model,
                prior,
                statistic.name,
                self.selector,
                n_grid=n_grid,
                coarse_n=coarse_n,
                search_mult=search_mult,
            )
            if not regions:
                raise RuntimeError(f"dynamic CI inversion produced no regions at D={D!r}")
            return regions

        eta = float(
            self.selector.select(
                self,
                data=data,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=statistic,
            )
        )
        return [
            self.tilted_confidence_interval(
                alpha,
                D,
                model,
                prior,
                eta,
                statistic.name,
            )
        ]

    def confidence_interval(
        self,
        alpha: float,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
        *,
        config: Config | None = None,
    ) -> tuple[float, float]:
        """Convex hull of `confidence_regions`; ``config`` forwarded
        to that method for the dynamic-CI scan resolution.
        """
        regions = self.confidence_regions(
            alpha, data, model, prior, statistic, config=config
        )
        lo = float(min(r[0] for r in regions))
        hi = float(max(r[1] for r in regions))
        return (lo, hi)

    def pvalue(
        self,
        theta: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> NDArray[np.float64]:
        # Phase 3d: dispatch generic for non-Normal-Normal pairings.
        if not self._is_normal_normal_pair(model, prior):
            if getattr(self.selector, "is_dynamic", False):
                raise NotImplementedError(
                    "OTTilting.pvalue dynamic-η selector currently requires "
                    "NormalNormalModel + NormalDistribution prior."
                )
            from ..config import Config as _Config
            from ._generic_pvalue import _stable_tilted_pvalue_seed
            alpha = float(_Config.default().alpha)
            eta = float(
                self.selector.select(
                    self,
                    data=data,
                    model=model,
                    prior=prior,
                    alpha=alpha,
                    statistic=statistic,
                )
            )
            theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
            data_arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
            derived_seed = _stable_tilted_pvalue_seed(
                data_arr, model, prior, eta, alpha,
                _GENERIC_TILTED_PVALUE_BASE_SEED,
            )
            out = np.empty(theta_arr.shape, dtype=np.float64)
            for i, th in enumerate(theta_arr):
                out[i] = _generic_tilted_pvalue_ot(
                    float(th), data_arr, model, prior, eta, statistic.name,
                    derived_seed=derived_seed, alpha=alpha,
                )
            return out

        self._require_normal_sandbox(model, prior)
        # Narrow types after the dispatch check (mypy can't infer through it).
        # `cast` is `-O`-safe; the runtime gate is `_require_normal_sandbox` above.
        from ..models.normal_normal import NormalNormalModel

        model = cast(NormalNormalModel, model)
        prior = cast(NormalDistribution, prior)
        from ..config import Config
        from .eta_selectors import _NamedStatistic

        alpha = float(Config.default().alpha)
        D = _data_to_scalar_D(data)
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
            # Phase 3a-1: θ-space directly.
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            theta_lo = float(theta_arr.min())
            theta_hi = float(theta_arr.max())
            half_pad = 1e-6 * max(1.0, abs(theta_hi - theta_lo))
            coarse_grid = np.linspace(theta_lo - half_pad, theta_hi + half_pad, coarse_n)
            coarse_eta = self.selector.select_grid(  # type: ignore[attr-defined]
                coarse_grid,
                self,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=_NamedStatistic(statistic.name),
            )
            eta_at_theta = np.interp(theta_arr, coarse_grid, coarse_eta)
            return self.dynamic_tilted_pvalue(
                theta_arr,
                D,
                model,
                prior,
                statistic.name,
                eta_at_theta,
            )

        eta = float(
            self.selector.select(
                self,
                data=data,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=statistic,
            )
        )
        return self.tilted_pvalue(theta_arr, D, model, prior, eta, statistic.name)
