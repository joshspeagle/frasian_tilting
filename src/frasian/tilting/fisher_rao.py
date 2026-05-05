"""Fisher-Rao (Levi-Civita) geodesic tilting on the Gaussian half-plane.

Fisher-Rao is the Riemannian (Levi-Civita) geodesic of the parametric
Gaussian manifold equipped with the Fisher information metric. It is
the third affine connection compatible with the Fisher metric, distinct
from the e-connection (`power_law`) and the m-connection (`mixture`)
of Amari's dually-flat structure (Amari & Nagaoka 2000 §3).

On the univariate Gaussian family the manifold is the upper half-plane
in `(mu, sigma)` (rescaling `mu` by `sqrt(2)` makes the metric the
standard Poincaré half-plane of constant curvature `K = -1`), and
geodesics between two Gaussians have a closed form (Costa et al. 2015,
Eq. 12; Pinele et al. 2020 Eq. 22).

Two cases:
  - **Vertical** (`mu_p = mu_q`): geodesic stays at fixed mu and
    interpolates sigma as the geometric mean,
        mu(eta)    = mu_p
        sigma(eta) = sigma_p * (sigma_q / sigma_p) ** eta
  - **Semicircle** (`mu_p != mu_q`): in `u = mu/sqrt(2)` coords, the
    geodesic is a Euclidean semicircle centred on the boundary
    `sigma=0`. Parametrise the angle `t` so `log tan(t/2)` is linear
    in eta — that's the arc-length parametrisation on the half-plane.

The geodesic stays in the Gaussian family because the manifold *is*
the Gaussian family, so `tilt(...)` returns a `NormalDistribution`
(unlike `mixture`, which leaves the family).

Endpoints follow the framework's posterior <-> likelihood convention
(matching `power_law`, `ot`, `mixture`): eta=0 -> posterior, eta=1 ->
likelihood-induced Gaussian `N(D, sigma^2)`.

See `audit/tier2/fisher_rao_derivation.md` for the full derivation,
ODE-verified at atol 7.6e-14.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..statistics.base import TestStatistic
from .base import EtaSelector, ParamSpec, TiltingContext
from .eta_selectors import FixedEtaSelector

if TYPE_CHECKING:
    from ..config import Config


# Threshold for the vertical-vs-semicircle branch decision in
# `(u_p, u_q)` coordinates. The semicircle branch divides by
# `(u_q - u_p)` and so loses precision below this threshold; the
# vertical formula is exact in the limit, so the switch is safe.
_VERTICAL_THRESHOLD = 1e-12

_SQRT2 = math.sqrt(2.0)


def _require_gaussian(
    posterior: Posterior, prior: Prior, likelihood: Likelihood
) -> tuple[NormalDistribution, NormalDistribution, GaussianLikelihood]:
    """Recognize a Normal-Normal context. Raise if any input is not Gaussian."""
    if not isinstance(posterior, NormalDistribution):
        raise NotImplementedError(
            "FisherRaoTilting requires a NormalDistribution posterior; "
            f"got {type(posterior).__name__!r}. The Gaussian-only "
            f"closed form lands first; non-Gaussian families need a "
            f"general ParametricFamily interface (future extension)."
        )
    if not isinstance(prior, NormalDistribution):
        raise NotImplementedError(
            "FisherRaoTilting requires a NormalDistribution prior; "
            f"got {type(prior).__name__!r}."
        )
    if not isinstance(likelihood, GaussianLikelihood):
        raise NotImplementedError(
            "FisherRaoTilting requires a GaussianLikelihood; "
            f"got {type(likelihood).__name__!r}."
        )
    return posterior, prior, likelihood


def _data_to_scalar_D(data: NDArray[np.float64]) -> float:
    """Coerce ``data`` to a single scalar D for the n=1 sandbox."""
    arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if arr.size != 1:
        raise NotImplementedError(
            f"Normal-Normal sandbox is single-observation (n=1); got "
            f"data.size={arr.size}. For n>1, use sigma_eff=sigma/sqrt(n) "
            f"and pass data.mean() with a model whose sigma is sigma_eff."
        )
    return float(arr.item())


def _fisher_rao_path(
    mu_p: float, sigma_p: float, mu_q: float, sigma_q: float, eta: float
) -> tuple[float, float]:
    """Closed-form Fisher-Rao geodesic on the Gaussian half-plane.

    Returns ``(mu_eta, sigma_eta)`` per Step 3 of
    `audit/tier2/fisher_rao_derivation.md`.

    eta=0 -> (mu_p, sigma_p);  eta=1 -> (mu_q, sigma_q).
    """
    u_p = mu_p / _SQRT2
    u_q = mu_q / _SQRT2

    if abs(u_p - u_q) < _VERTICAL_THRESHOLD:
        # Vertical case: mu fixed, sigma is geometric-mean interpolated.
        # sigma(eta) = sigma_p * (sigma_q / sigma_p) ** eta
        return (float(mu_p), float(sigma_p * (sigma_q / sigma_p) ** eta))

    # Semicircle case. Centre `u_c` on the boundary `sigma = 0`,
    # radius R, parametrise the angle `t in (0, pi)`.
    u_c = (u_q**2 - u_p**2 + sigma_q**2 - sigma_p**2) / (2.0 * (u_q - u_p))
    R = math.sqrt((u_p - u_c) ** 2 + sigma_p**2)
    t_p = math.atan2(sigma_p, u_p - u_c)
    t_q = math.atan2(sigma_q, u_q - u_c)
    # Arc-length-normalised parametrisation: log tan(t/2) is linear in eta.
    log_tan_half = (
        (1.0 - eta) * math.log(math.tan(t_p / 2.0))
        + eta * math.log(math.tan(t_q / 2.0))
    )
    t_eta = 2.0 * math.atan(math.exp(log_tan_half))
    u_eta = u_c + R * math.cos(t_eta)
    sigma_eta = R * math.sin(t_eta)
    mu_eta = _SQRT2 * u_eta
    return (float(mu_eta), float(sigma_eta))


def fisher_rao_distance(
    mu_p: float, sigma_p: float, mu_q: float, sigma_q: float
) -> float:
    """Closed-form Fisher-Rao distance (Costa et al. 2015 Eq. 12).

    `d_FR = sqrt(2) * arccosh(1 + ((mu_p-mu_q)^2 / 2 + (sigma_p-sigma_q)^2)
                              / (2 sigma_p sigma_q))`.
    """
    num = (mu_p - mu_q) ** 2 / 2.0 + (sigma_p - sigma_q) ** 2
    arg = 1.0 + num / (2.0 * sigma_p * sigma_q)
    return float(_SQRT2 * math.acosh(arg))


@register_tilting(name="fisher_rao", brief="docs/methods/fisher_rao.md")
@dataclass(frozen=True)
class FisherRaoTilting:
    """Fisher-Rao (Levi-Civita) geodesic tilting on the Gaussian half-plane.

    Output is `NormalDistribution` (the geodesic stays in the Gaussian
    family). Wald is eta-independent (matches `power_law` / `ot` / `mixture`
    Wald branches); WALDO uses the standard Phi-pair formula on the tilted
    `(mu_eta, sigma_eta)`.
    """

    name: ClassVar[str] = "fisher_rao"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description=(
            "eta in [0, 1] arc-length on the Fisher-Rao geodesic; "
            "eta=0 -> posterior, eta=1 -> N(D, sigma^2)."
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
    ) -> NormalDistribution:
        """Closed-form Fisher-Rao geodesic on the Gaussian half-plane.

        Two-branch logic: vertical when `mu_post = mu_likelihood`
        (within `_VERTICAL_THRESHOLD` in u-coordinates), semicircle
        otherwise. See `audit/tier2/fisher_rao_derivation.md` Step 3.
        """
        post, _pri, lik = _require_gaussian(posterior, prior, likelihood)

        eta_arr = np.asarray(eta, dtype=np.float64)
        if eta_arr.ndim != 0:
            raise NotImplementedError(
                "tilt() expects scalar eta; vectorised eta is consumed via "
                "repeated scalar calls (see `path`)."
            )
        eta_f = float(eta_arr)
        if not (0.0 <= eta_f <= 1.0):
            raise TiltingDomainError(
                f"FisherRaoTilting requires eta in [0, 1], got {eta_f!r}."
            )

        mu_eta, sigma_eta = _fisher_rao_path(
            mu_p=float(post.loc),
            sigma_p=float(post.scale),
            mu_q=float(lik.D),
            sigma_q=float(lik.sigma),
            eta=eta_f,
        )
        return NormalDistribution(loc=mu_eta, scale=sigma_eta)

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
        return float(eta) == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        # eta in [0, 1] is always valid: the half-plane is geodesically
        # complete on sigma > 0 and both endpoints lie in the open
        # region. No clamp.
        return (0.0, 1.0)

    # ----- (TiltingScheme, TestStatistic) cross-product specialisations -----

    def _tilted_gaussian_params(
        self,
        D: float | NDArray[np.float64],
        mu0: float,
        sigma: float,
        sigma0: float,
        eta: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per-element (mu_eta, sigma_eta) along the FR geodesic.

        Vectorised over `eta` (and over `D` if it is array-shaped). The
        scheme's `_fisher_rao_path` is scalar; this helper applies it
        element-wise via `np.vectorize` for the array path.
        """
        D_arr = np.asarray(D, dtype=np.float64)
        eta_arr = np.asarray(eta, dtype=np.float64)
        # Posterior endpoint at the data D (broadcast over D if needed).
        w = sigma0**2 / (sigma**2 + sigma0**2)
        mu_n = w * D_arr + (1.0 - w) * mu0
        sigma_n = float(np.sqrt(w) * sigma)

        if eta_arr.ndim == 0 and D_arr.ndim == 0:
            mu_eta, sigma_eta = _fisher_rao_path(
                mu_p=float(mu_n), sigma_p=sigma_n, mu_q=float(D_arr),
                sigma_q=sigma, eta=float(eta_arr),
            )
            return (np.asarray(mu_eta, dtype=np.float64),
                    np.asarray(sigma_eta, dtype=np.float64))

        # Broadcast pair-wise. We use np.vectorize for clarity over
        # writing per-element loops — performance is not a hot path here
        # because the dynamic-CI scan vectorises by batching theta, not
        # by vectorising tilt.
        mu_b, eta_b = np.broadcast_arrays(D_arr, eta_arr)
        mu_n_b = w * mu_b + (1.0 - w) * mu0  # broadcast
        out_mu = np.empty_like(mu_b, dtype=np.float64)
        out_sigma = np.empty_like(mu_b, dtype=np.float64)
        for idx in np.ndindex(mu_b.shape):
            mu_eta_v, sigma_eta_v = _fisher_rao_path(
                mu_p=float(mu_n_b[idx]),
                sigma_p=sigma_n,
                mu_q=float(mu_b[idx]),
                sigma_q=sigma,
                eta=float(eta_b[idx]),
            )
            out_mu[idx] = mu_eta_v
            out_sigma[idx] = sigma_eta_v
        return out_mu, out_sigma

    def tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float | NDArray[np.float64],
        model: Model,
        prior: NormalDistribution,
        eta: ArrayLike,
        statistic_name: str,
    ) -> NDArray[np.float64]:
        """Tilted p-value of `statistic_name` against the FR-tilted Gaussian.

        Specialized for (fisher_rao, waldo) — closed form on Q_eta — and
        (fisher_rao, wald) — eta-independent (Wald ignores the prior).

        ``eta`` accepts scalar or array broadcastable to ``theta_arr``.
        """
        from ..models.normal_normal import NormalNormalModel

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "FisherRaoTilting.tilted_pvalue currently requires "
                f"NormalNormalModel; got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "FisherRaoTilting.tilted_pvalue currently requires a "
                "NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)

        theta_arr = np.asarray(theta, dtype=np.float64)
        eta_arr = np.asarray(eta, dtype=np.float64)

        # eta validation: in [0, 1] and finite.
        invalid = ~(np.isfinite(eta_arr) & (eta_arr >= 0.0) & (eta_arr <= 1.0))
        if np.any(invalid):
            if eta_arr.ndim == 0:
                raise TiltingDomainError(
                    f"FisherRaoTilting.tilted_pvalue requires eta in [0, 1], "
                    f"got {float(eta_arr)!r}."
                )
            bad = int(np.argmax(invalid))
            raise TiltingDomainError(
                f"FisherRaoTilting.tilted_pvalue requires eta in [0, 1] and "
                f"finite; offending index {bad} eta="
                f"{float(eta_arr.flat[bad])!r}."
            )

        if statistic_name == "wald":
            # eta-independent: matches power_law / ot / mixture Wald branches.
            z = np.abs(D - theta_arr) / sigma
            return np.asarray(2.0 * stats.norm.sf(z), dtype=np.float64)

        if statistic_name == "waldo":
            # WALDO under the FR-tilted Gaussian: standard Phi-pair on
            # `(mu_eta, sigma_eta)`. b is the prior z-score (the prior
            # is fixed; the tilting only re-parametrises the posterior
            # along the geodesic, so b retains its bare-WALDO meaning).
            mu_eta, sigma_eta = self._tilted_gaussian_params(
                D=D, mu0=mu0, sigma=sigma, sigma0=sigma0, eta=eta_arr,
            )
            # Broadcast to theta_arr's shape.
            mu_eta_b, sigma_eta_b, theta_b = np.broadcast_arrays(
                mu_eta, sigma_eta, theta_arr,
            )
            a = np.abs(mu_eta_b - theta_b) / sigma_eta_b
            b = (mu0 - theta_b) / sigma0
            return np.asarray(
                stats.norm.cdf(b - a) + stats.norm.cdf(-a - b),
                dtype=np.float64,
            )

        raise NotImplementedError(
            f"FisherRaoTilting.tilted_pvalue not implemented for "
            f"statistic={statistic_name!r}; supported: 'wald', 'waldo'."
        )

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

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "FisherRaoTilting.tilted_confidence_interval currently "
                "requires NormalNormalModel."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "FisherRaoTilting.tilted_confidence_interval currently "
                "requires a NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)

        # Bracket midpoint: the FR-tilted mean (CI mode for WALDO);
        # for Wald, centre on D. For WALDO, derive via the path.
        if statistic_name == "wald":
            mid = float(D)
        else:
            w = sigma0**2 / (sigma**2 + sigma0**2)
            mu_n = w * D + (1.0 - w) * mu0
            sigma_n = float(np.sqrt(w) * sigma)
            mu_eta, _ = _fisher_rao_path(
                mu_p=mu_n, sigma_p=sigma_n, mu_q=D, sigma_q=sigma, eta=eta,
            )
            mid = float(mu_eta)

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

        Delegates to `frasian.tilting._dynamic.dynamic_ci_scan`.
        """
        from ..models.normal_normal import NormalNormalModel
        from ._dynamic import dynamic_ci_scan

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "FisherRaoTilting.dynamic_tilted_confidence_interval currently "
                "requires NormalNormalModel."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

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
        )

    # ----- Uniform CI / regions / pvalue interface -----

    def _require_normal_sandbox(self, model: Model, prior: Prior) -> None:
        from ..models.normal_normal import NormalNormalModel

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                f"FisherRaoTilting requires NormalNormalModel for the uniform "
                f"CI interface; got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                f"FisherRaoTilting requires a NormalDistribution prior; "
                f"got {type(prior).__name__!r}."
            )

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
        """Selector-aware region list."""
        self._require_normal_sandbox(model, prior)
        from ..models.normal_normal import NormalNormalModel

        model = cast(NormalNormalModel, model)
        prior = cast(NormalDistribution, prior)
        D = _data_to_scalar_D(data)
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)
        abs_delta = abs((1.0 - w) * (prior.loc - D) / sigma)
        ctx = TiltingContext(w=w, abs_delta=abs_delta, alpha=alpha)

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

        eta = float(self.selector.select(ctx, self, statistic=statistic))
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
        """Convex hull of `confidence_regions`."""
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
        self._require_normal_sandbox(model, prior)
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
            abs_delta_theta = np.abs((1.0 - w) * (prior.loc - theta_arr) / sigma)
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            ad_max = float(abs_delta_theta.max()) + 1e-6
            coarse_grid = np.linspace(0.0, ad_max, coarse_n)
            select_kwargs = dict(
                statistic=_NamedStatistic(statistic.name),
                w=w,
                alpha=alpha,
            )
            try:
                import inspect

                sig = inspect.signature(self.selector.select_grid)  # type: ignore[attr-defined]
                if "model_fingerprint" in sig.parameters:
                    select_kwargs["model_fingerprint"] = model.fingerprint()
                if "prior_fingerprint" in sig.parameters:
                    select_kwargs["prior_fingerprint"] = prior.fingerprint()
            except (TypeError, ValueError):
                pass
            coarse_eta = self.selector.select_grid(  # type: ignore[attr-defined]
                coarse_grid,
                self,
                **select_kwargs,
            )
            eta_at_theta = np.interp(abs_delta_theta, coarse_grid, coarse_eta)
            return self.dynamic_tilted_pvalue(
                theta_arr,
                D,
                model,
                prior,
                statistic.name,
                eta_at_theta,
            )

        abs_delta = abs((1.0 - w) * (prior.loc - D) / sigma)
        ctx = TiltingContext(w=w, abs_delta=abs_delta, alpha=alpha)
        eta = float(self.selector.select(ctx, self, statistic=statistic))
        return self.tilted_pvalue(theta_arr, D, model, prior, eta, statistic.name)
