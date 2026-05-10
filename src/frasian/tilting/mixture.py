"""m-geodesic — convex mixture between posterior and likelihood.

The m-geodesic of information geometry: linear interpolation in
density space between the posterior and the likelihood-as-Gaussian,

    q(theta; eta) = (1 - eta) * post(theta) + eta * L(theta) / Z_L,

with the framework's posterior <-> likelihood endpoint convention:
`eta=0` recovers the posterior, `eta=1` recovers the likelihood-as-
Gaussian. See `docs/methods/mixture.md` for the full derivation.

For Gaussian endpoints the mixture is a 2-component Gaussian mixture
(NOT Gaussian itself) -- bimodal when prior and likelihood disagree
strongly (Behboodian 1970: bimodal iff |mu_n - D| > 2 * min(sigma_n, sigma)).
The framework's `GaussianMixtureDistribution` wraps this case.

**Admissibility on Normal-Normal:**
    R_max  = (1/sqrt(w)) * exp(Delta^2 / (2*(1-w)))
    eta_max = R_max / (R_max - 1)        if R_max > 1, else +inf

eta in [0, eta_max] is admissible. eta < 0 is never admissible on
Normal-Normal (prior tails go to 0). Per-call validity check refuses
inadmissible eta.

**Closed-form tilted-WALDO p-value (Normal-Normal sandbox).**
The tilted moments depend on X (the replicate):
    mu_til(X)    = alpha*X + (1-alpha)*mu_0,   alpha = w + eta*(1-w)
    sigma2_til(X) = C_0 + C_1*(mu_0 - X)^2
    C_0 = (1-eta)*sigma_n^2 + eta*sigma^2
    C_1 = (1-eta)*eta*(1-w)^2

The accept set {X : t(theta;X) >= t_obs(theta)} is the solution of a
quadratic-in-X inequality Q(X) = L*X^2 + 2*M*X + N >= 0 (5-case
branching on sign(L) and sign(disc)). Closed form gives a sum of
Gaussian-CDF terms at the quadratic roots. See docs/methods/mixture.md
for the full derivation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as _scalar_scipy_stats

from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..models._dispatch import is_normal_normal
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import (
    GaussianLikelihood,
    GaussianMixtureDistribution,
    NormalDistribution,
)
from ..statistics.base import TestStatistic
from ._dynamic import dynamic_ci_scan
from ._solvers import brentq_with_doubling
from .base import EtaSelector, ParamSpec
from .eta_selectors import FixedEtaSelector

if TYPE_CHECKING:
    from ..config import Config


# Coefficient guards -- same scale as PL/OT denom guards.
_QUADRATIC_LEADING_EPS = 1e-12
_DISCRIMINANT_EPS = 1e-12


def _admissibility_normal_normal(
    eta: float, w: float, mu0: float, D: float, sigma: float
) -> tuple[bool, float]:
    """Closed-form admissibility check for the m-geodesic on NN.

    Returns ``(is_admissible, eta_max)``. eta_max = +inf when R_max <= 1.

    Derivation: see docs/methods/mixture.md "Admissibility on Normal-Normal".
    """
    eta_f = float(eta)
    if eta_f < 0.0:
        return False, float("inf")
    Delta = (1.0 - w) * (mu0 - D) / sigma
    log_R_max = -0.5 * np.log(w) + (Delta * Delta) / (2.0 * (1.0 - w))
    R_max = float(np.exp(log_R_max))
    if R_max <= 1.0:
        return (eta_f >= 0.0), float("inf")
    eta_max = R_max / (R_max - 1.0)
    return (eta_f <= eta_max), float(eta_max)


def _mixture_tilted_pvalue_numpy_scalar(
    theta_f: float,
    eta_f: float,
    D_f: float,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> float:
    """Closed-form scalar tilted-WALDO / tilted-Wald p-value on Normal-Normal.

    Derivation: docs/methods/mixture.md "Closed-form tilted-WALDO p-value".

    The accept set {X : t(theta;X) >= t_obs(theta)} is the solution of a
    quadratic-in-X inequality
        Q(X) = L*X^2 + 2*M*X + N >= 0
    where L, M, N are functions of (theta, eta, w, mu0, sigma, t_obs).
    Branches:
      L > 0, disc > 0:  X <= X- or X >= X+
      L > 0, disc <= 0: Q >= 0 everywhere -> p = 1
      L < 0, disc > 0:  X- <= X <= X+
      L < 0, disc <= 0: Q < 0 everywhere -> p = 0
      L = 0:            linear (single half-line)
    The p-value is the probability of the accept set under X ~ N(theta, sigma^2).
    """
    if statistic_name == "wald":
        # Wald ignores prior; reduces to bare 2-sided Wald regardless of eta.
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))

    if statistic_name != "waldo":
        raise NotImplementedError(
            f"_mixture_tilted_pvalue_numpy_scalar: statistic={statistic_name!r}; "
            f"supported: 'wald', 'waldo'."
        )

    # Endpoint shortcuts (mathematically redundant; numerically robust).
    sigma_n_sq = w * sigma * sigma
    if eta_f == 0.0:
        # Bare WALDO closed form.
        mu_n = w * D_f + (1.0 - w) * mu0
        a = abs(mu_n - theta_f) / (w * sigma)
        b = (1.0 - w) * (mu0 - theta_f) / (w * sigma)
        return float(
            _scalar_scipy_stats.norm.cdf(b - a)
            + _scalar_scipy_stats.norm.cdf(-a - b)
        )
    if eta_f == 1.0:
        # Bare 2-sided Wald.
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))

    # Interior: full quadratic branching.
    alpha_lin = w + eta_f * (1.0 - w)
    A = alpha_lin
    B = (1.0 - alpha_lin) * mu0 - theta_f
    C0 = (1.0 - eta_f) * sigma_n_sq + eta_f * sigma * sigma
    C1 = (1.0 - eta_f) * eta_f * (1.0 - w) * (1.0 - w)

    # t_obs at observed D.
    mu_til_D = alpha_lin * D_f + (1.0 - alpha_lin) * mu0
    var_til_D = C0 + C1 * (mu0 - D_f) ** 2
    if var_til_D <= 0.0:
        return float("nan")
    t_obs = (mu_til_D - theta_f) ** 2 / var_til_D

    # Quadratic Q(X) = L*X^2 + 2*M*X + N >= 0
    L = A * A - t_obs * C1
    M = A * B + t_obs * C1 * mu0
    N = B * B - t_obs * C0 - t_obs * C1 * mu0 * mu0
    # discriminant of L*X^2 + 2*M*X + N = 0 is 4*(M^2 - L*N).
    disc_quarter = M * M - L * N

    Phi = _scalar_scipy_stats.norm.cdf

    def F(x: float) -> float:
        return float(Phi((x - theta_f) / sigma))

    # Branch on L.
    if abs(L) <= _QUADRATIC_LEADING_EPS:
        # Linear: 2*M*X + N >= 0
        if abs(M) <= _QUADRATIC_LEADING_EPS:
            # Constant Q. p = 1 if N >= 0 else 0.
            return 1.0 if N >= 0.0 else 0.0
        x_root = -N / (2.0 * M)
        if M > 0.0:
            return 1.0 - F(x_root)
        return F(x_root)

    if disc_quarter < -_DISCRIMINANT_EPS:
        return 1.0 if L > 0.0 else 0.0

    sqrt_disc = float(np.sqrt(max(disc_quarter, 0.0)))
    x_minus = (-M - sqrt_disc) / L
    x_plus = (-M + sqrt_disc) / L
    if x_minus > x_plus:
        x_minus, x_plus = x_plus, x_minus

    if L > 0.0:
        # Outer half-lines.
        return F(x_minus) + (1.0 - F(x_plus))
    # L < 0: inner interval.
    return F(x_plus) - F(x_minus)


def _mixture_tilted_pvalue_array(
    theta: NDArray[np.float64],
    eta: NDArray[np.float64],
    D: float,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> NDArray[np.float64]:
    """Vectorised wrapper around `_mixture_tilted_pvalue_numpy_scalar`.

    Used by the dynamic-CI scan (per-theta varying eta). The closed-form
    branching has Python control flow per (theta, eta), so the array
    path is a Python loop over the scalar kernel; this matches OT's
    closed-form vec path (ot.py also Python-loops the scalar kernel
    when eta varies per row).
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    eta_arr = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    if theta_arr.shape != eta_arr.shape:
        # Broadcast scalar eta to theta shape for convenience.
        if eta_arr.size == 1:
            eta_arr = np.full(theta_arr.shape, float(eta_arr.item()), dtype=np.float64)
        else:
            raise ValueError(
                f"_mixture_tilted_pvalue_array: theta {theta_arr.shape} vs "
                f"eta {eta_arr.shape} shape mismatch."
            )
    out = np.empty_like(theta_arr)
    for i in range(theta_arr.size):
        out.flat[i] = _mixture_tilted_pvalue_numpy_scalar(
            float(theta_arr.flat[i]),
            float(eta_arr.flat[i]),
            D, w, mu0, sigma, statistic_name,
        )
    return out


@register_tilting(name="mixture", brief="docs/methods/mixture.md")
@dataclass(frozen=True)
class MixtureTilting:
    """m-geodesic: q = (1 - eta) * posterior + eta * likelihood-as-Gaussian.

    Endpoint convention matches the framework: eta=0 recovers the
    posterior (identity), eta=1 recovers the likelihood-as-Gaussian.

    The selector is owned by the tilting (matches PowerLaw/OT). The
    default `FixedEtaSelector(eta=0.0)` makes the bare MixtureTilting()
    behave as the identity at `eta_default=0.0`.
    """

    selector: EtaSelector = field(default_factory=lambda: FixedEtaSelector(eta=0.0))
    name: ClassVar[str] = "mixture"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="eta in [0, eta_max(D, prior, model)]; 0=posterior, 1=likelihood.",
    )

    # ------ Tilt: NN closed-form path (returns GaussianMixtureDistribution) ------

    def tilt(
        self,
        posterior: Posterior,
        prior: Prior,
        likelihood: Likelihood,
        eta: ArrayLike,
    ) -> GaussianMixtureDistribution:
        """Closed-form NN: returns a 2-component GaussianMixtureDistribution.

        Validates admissibility on NN. Raises ValueError on inadmissible
        eta (callers in the CI inversion catch ValueError -> NaN).
        """
        eta_f = float(np.asarray(eta).item())
        if not np.isfinite(eta_f):
            raise TiltingDomainError(
                f"MixtureTilting requires finite eta, got {eta_f!r}."
            )
        if not (
            isinstance(posterior, NormalDistribution)
            and isinstance(prior, NormalDistribution)
            and isinstance(likelihood, GaussianLikelihood)
        ):
            raise NotImplementedError(
                "MixtureTilting closed-form path requires Normal posterior + "
                "Normal prior + Gaussian likelihood. Generic numerical path "
                "is implemented in Stage B (see _generic_tilt_mixture)."
            )
        D = float(likelihood.D)
        sigma = float(likelihood.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        is_ok, eta_max = _admissibility_normal_normal(eta_f, w, mu0, D, sigma)
        if not is_ok:
            raise ValueError(
                f"MixtureTilting: eta={eta_f!r} is inadmissible "
                f"(admissible set on Normal-Normal is [0, {eta_max:.4g}]; "
                f"density would be negative somewhere). See "
                f"docs/methods/mixture.md 'Admissibility on Normal-Normal'."
            )
        mu_n = w * D + (1.0 - w) * mu0
        sigma_n = float(np.sqrt(w)) * sigma
        return GaussianMixtureDistribution(
            weights=(1.0 - eta_f, eta_f),
            means=(mu_n, D),
            scales=(sigma_n, sigma),
        )

    def path(
        self,
        posterior: Posterior,
        prior: Prior,
        likelihood: Likelihood,
        ts: NDArray[np.float64],
    ) -> Iterable[GaussianMixtureDistribution]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, t)

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    # --- Numerical-no-selector helpers (mirror of PL.tilted_pvalue family) ---

    def tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float | NDArray[np.float64],
        model: object,
        prior: NormalDistribution,
        eta: ArrayLike,
        statistic_name: str,
    ) -> float | NDArray[np.float64]:
        """p-value of `statistic_name` against the tilted reference (NN closed form).

        ``eta`` accepts a scalar or an array broadcastable to ``theta``;
        the array path is what the dynamic-eta scan uses.
        """
        if not is_normal_normal(model):
            raise NotImplementedError(
                "MixtureTilting.tilted_pvalue currently requires NormalNormalModel; "
                f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "MixtureTilting.tilted_pvalue currently requires a NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        eta_np = np.asarray(eta, dtype=np.float64)
        if not np.all(np.isfinite(eta_np)):
            bad_idx = int(np.argmax(~np.isfinite(eta_np)))
            raise TiltingDomainError(
                f"MixtureTilting.tilted_pvalue requires finite eta; "
                f"offending index {bad_idx} eta="
                f"{float(eta_np.flat[bad_idx])!r}"
            )

        theta_np = np.asarray(theta, dtype=np.float64)
        D_np = np.asarray(D, dtype=np.float64)

        # Scalar fast path.
        if theta_np.size == 1 and eta_np.size == 1 and D_np.size == 1:
            return _mixture_tilted_pvalue_numpy_scalar(
                float(theta_np.item()),
                float(eta_np.item()),
                float(D_np.item()),
                w, mu0, sigma, statistic_name,
            )
        # Array path. D must be a scalar for now (matches PL/OT contract);
        # broadcast eta to theta shape inside the helper.
        if D_np.size != 1:
            raise NotImplementedError(
                "MixtureTilting.tilted_pvalue: array D not supported; "
                "the framework's NN sandbox is single-observation."
            )
        return _mixture_tilted_pvalue_array(
            theta_np, eta_np, float(D_np.item()), w, mu0, sigma, statistic_name,
        )

    def dynamic_tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float,
        model: object,
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
        out = np.asarray(
            self.tilted_pvalue(theta_arr, D, model, prior, eta_arr, statistic_name),
            dtype=np.float64,
        )
        return out if out.size > 1 else np.asarray(float(out.item()))

    def tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: object,
        prior: NormalDistribution,
        eta: float,
        statistic_name: str,
    ) -> tuple[float, float]:
        """Numerical CI inversion of `tilted_pvalue` via brentq_with_doubling."""
        if not is_normal_normal(model):
            raise NotImplementedError(
                "tilted_confidence_interval currently requires NormalNormalModel."
            )
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        mu0 = float(prior.loc)

        # Bracket midpoint: tilted-mixture mean (alpha*D + (1-alpha)*mu0).
        alpha_lin = w + float(eta) * (1.0 - w)
        if statistic_name == "wald":
            mu_eta = float(D)
        else:
            mu_eta = alpha_lin * float(D) + (1.0 - alpha_lin) * mu0

        def f(theta_val: float) -> float:
            return (
                float(self.tilted_pvalue(
                    float(theta_val), D, model, prior, float(eta), statistic_name
                ))
                - alpha
            )

        half = 4.0 * sigma
        lo = brentq_with_doubling(
            f, midpoint=float(mu_eta), initial_half_width=half, direction=-1
        )
        hi = brentq_with_doubling(
            f, midpoint=float(mu_eta), initial_half_width=half, direction=+1
        )
        return (lo, hi)

    def dynamic_tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: object,
        prior: NormalDistribution,
        statistic_name: str,
        eta_selector,
        n_grid: int = 401,
        coarse_n: int = 25,
        search_mult: float = 8.0,
    ) -> tuple[list[tuple[float, float]], float, int]:
        """Dynamic-eta CI: eta = eta*(|Delta(theta)|) per theta."""
        if not is_normal_normal(model):
            raise NotImplementedError(
                "dynamic_tilted_confidence_interval currently requires NormalNormalModel."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        def _tilted_pvalue_fn(theta: float, eta: float) -> float:
            return float(
                self.tilted_pvalue(theta, D, model, prior, eta, statistic_name)
            )

        def _tilted_pvalue_vec_fn(
            theta_arr: np.ndarray, eta_arr: np.ndarray
        ) -> np.ndarray:
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

    # ------ Public TiltingScheme API ------

    def pvalue(
        self,
        theta: ArrayLike,
        data: NDArray[np.float64],
        model: Model,
        prior: Prior,
        statistic: TestStatistic,
    ) -> NDArray[np.float64]:
        """Selector-aware p-value at hypothesised theta values.

        Static selector: resolve a single eta via `selector.select(...)`.
        Dynamic selector: precompute coarse eta*(theta) lookup, interpolate.
        """
        if not (is_normal_normal(model) and isinstance(prior, NormalDistribution)):
            raise NotImplementedError(
                "MixtureTilting.pvalue: generic path implemented in Stage B."
            )
        from ..config import Config
        from .eta_selectors import _NamedStatistic

        alpha = float(Config.default().alpha)
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        D = float(np.asarray(data, dtype=np.float64).item())
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))

        if getattr(self.selector, "is_dynamic", False):
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            theta_lo = float(theta_arr.min())
            theta_hi = float(theta_arr.max())
            half_pad = 1e-6 * max(1.0, abs(theta_hi - theta_lo))
            coarse_grid = np.linspace(
                theta_lo - half_pad, theta_hi + half_pad, coarse_n
            )
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
                theta_arr, D, model, prior, statistic.name, eta_at_theta,
            )

        # Static selector: single eta.
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
        result = self.tilted_pvalue(
            theta_arr, D, model, prior, eta, statistic.name
        )
        return np.atleast_1d(np.asarray(result, dtype=np.float64))

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
        """Selector-aware region list."""
        if not (is_normal_normal(model) and isinstance(prior, NormalDistribution)):
            raise NotImplementedError(
                "MixtureTilting.confidence_regions: generic path in Stage B."
            )
        D = float(np.asarray(data, dtype=np.float64).item())

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
                raise RuntimeError(
                    f"dynamic CI inversion produced no regions at D={D!r}"
                )
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
                alpha, D, model, prior, eta, statistic.name,
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
        config: "Config | None" = None,
    ) -> tuple[float, float]:
        """Convex hull of `confidence_regions` (single-tuple summary).

        Multi-region cells (dynamic-eta, bimodal mixture) over-count the
        gaps with this hull -- coverage / width experiments use
        `confidence_regions` directly.
        """
        regions = self.confidence_regions(
            alpha, data, model, prior, statistic, config=config
        )
        if not regions:
            return (float("nan"), float("nan"))
        lo = float(min(r[0] for r in regions))
        hi = float(max(r[1] for r in regions))
        return (lo, hi)
