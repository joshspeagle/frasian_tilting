"""Mixture tilting (the m-geodesic) — dual partner of `power_law`.

The m-geodesic of information geometry between the WALDO posterior and the
likelihood-as-Gaussian is the linear interpolation in density space:

    p_eta(theta) = (1 - eta) * N(theta; mu_n, sigma_n^2)
                 +       eta * N(theta; D,    sigma^2),     eta in [0, 1].

This is the dually-flat partner of `power_law`'s e-geodesic under the
Fisher metric (Amari & Nagaoka 2000 §3): power-law is the log-linear /
geometric-mean path, mixture is the linear-density / arithmetic-mean
path. On the Normal-Normal sandbox the tilted output is a 2-component
Gaussian mixture — *not* in the Normal family — and is bimodal beyond
the Behboodian threshold `|mu_n - D| > 2 min(sigma_n, sigma)`.

Endpoints follow the framework's posterior <-> likelihood convention
(matching `power_law`, `ot`, `fisher_rao`): eta=0 -> posterior (identity
element), eta=1 -> likelihood-as-Gaussian. See
`audit/tier2/mixture_derivation.md` for the full derivation including
MC-verified closed forms.

Tilted-pvalue closed forms (Step 4 of the derivation):

  Wald: eta-independent — Wald ignores the prior, so the mixture
        does not change the MLE-based statistic. Returns the bare
        two-sided Wald p-value `2 * Phi(-|D - theta| / sigma)`.

  WALDO: each mixture component contributes a Phi-pair around the
         shifted mean `mu_eta := (1-eta) * mu_n + eta * D`:

           P_k = Phi((m_k - mu_eta - z) / s_k)
               + Phi((mu_eta - m_k - z) / s_k),     z = |theta - mu_eta|
           p(theta; eta) = (1 - eta) * P_1 + eta * P_2

         with `(m_1, s_1) = (mu_n, sigma_n)` and `(m_2, s_2) = (D, sigma)`.

Admissible range: `eta in [0, 1]` always; the convex combination of two
valid densities is itself a valid density without any clamp.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import (
    GaussianLikelihood,
    MixtureDistribution,
    NormalDistribution,
)
from ..statistics.base import TestStatistic
from .base import EtaSelector, ParamSpec, TiltingContext
from .eta_selectors import FixedEtaSelector

if TYPE_CHECKING:
    from ..config import Config


def _require_gaussian(
    posterior: Posterior, prior: Prior, likelihood: Likelihood
) -> tuple[NormalDistribution, NormalDistribution, GaussianLikelihood]:
    """Recognize a Normal-Normal context. Raise if any input is not Gaussian."""
    if not isinstance(posterior, NormalDistribution):
        raise NotImplementedError(
            "MixtureTilting requires a NormalDistribution posterior; "
            f"got {type(posterior).__name__!r}. Numerical fallback is a "
            f"future extension."
        )
    if not isinstance(prior, NormalDistribution):
        raise NotImplementedError(
            "MixtureTilting requires a NormalDistribution prior; "
            f"got {type(prior).__name__!r}."
        )
    if not isinstance(likelihood, GaussianLikelihood):
        raise NotImplementedError(
            "MixtureTilting requires a GaussianLikelihood; "
            f"got {type(likelihood).__name__!r}."
        )
    return posterior, prior, likelihood


def _data_to_scalar_D(data: NDArray[np.float64]) -> float:
    """Coerce ``data`` to a single scalar D for the n=1 sandbox.

    Mirrors `power_law._data_to_scalar_D` and `ot._data_to_scalar_D`. See
    those functions for the n=1 contract rationale.
    """
    arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if arr.size != 1:
        raise NotImplementedError(
            f"Normal-Normal sandbox is single-observation (n=1); got "
            f"data.size={arr.size}. For n>1, use sigma_eff=sigma/sqrt(n) "
            f"and pass data.mean() with a model whose sigma is sigma_eff."
        )
    return float(arr.item())


@register_tilting(name="mixture", brief="docs/methods/mixture.md")
@dataclass(frozen=True)
class MixtureTilting:
    """The m-geodesic tilting scheme (linear-density interpolation).

    Output is a 2-component Gaussian mixture (`MixtureDistribution`)
    rather than a Normal — so unlike `power_law` / `ot` / `fisher_rao`,
    `tilt(...)` does *not* return a `NormalDistribution`. The framework's
    CD constructor and CI inversion pieces consume the mixture through
    the `Distribution` protocol surface.
    """

    name: ClassVar[str] = "mixture"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description=(
            "eta in [0, 1] along the m-geodesic between posterior (eta=0) "
            "and likelihood-induced Gaussian N(D, sigma^2) (eta=1)."
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
    ) -> MixtureDistribution:
        """Linear-density tilt: 2-component Gaussian mixture.

        At eta=0, the mixture collapses to the posterior (weight 1 on
        the posterior component, weight 0 on the likelihood-Gaussian).
        At eta=1, it collapses to the likelihood-induced Gaussian
        N(D, sigma^2). For eta in (0, 1) the output is a genuine
        2-component mixture and is bimodal beyond
        `|mu_n - D| > 2 min(sigma_n, sigma)` (Behboodian 1970).
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
                f"MixtureTilting requires eta in [0, 1], got {eta_f!r}."
            )

        # Mixture weights (1-eta, eta), components (posterior, likelihood-as-Gaussian).
        # Even at the endpoints we return a MixtureDistribution with one
        # weight pinned to 0; this keeps the return type uniform across
        # the full eta in [0, 1] range and exercises the mixture-Distribution
        # path consistently. The sample / quantile fall-back paths handle
        # zero-weight components correctly.
        return MixtureDistribution(
            weights=(1.0 - eta_f, eta_f),
            means=(float(post.loc), float(lik.D)),
            sigmas=(float(post.scale), float(lik.sigma)),
        )

    def path(
        self,
        posterior: Posterior,
        prior: Prior,
        likelihood: Likelihood,
        ts: NDArray[np.float64],
    ) -> Iterable[MixtureDistribution]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, float(t))

    def is_identity(self, eta: float) -> bool:
        return float(eta) == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        # Trivially [0, 1]; convex combination of two valid densities is
        # always a valid density. No w-dependence, no buffer (the
        # mixture density is bounded everywhere on the open interval).
        return (0.0, 1.0)

    # ----- (TiltingScheme, TestStatistic) cross-product specialisations -----

    def tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float | NDArray[np.float64],
        model: Model,
        prior: NormalDistribution,
        eta: ArrayLike,
        statistic_name: str,
    ) -> NDArray[np.float64]:
        """Tilted p-value of `statistic_name` under the m-geodesic mixture.

        Specialized for (mixture, waldo) — Phi-pair-per-component closed
        form — and (mixture, wald) — eta-independent (Wald ignores the
        prior).

        ``eta`` accepts either a scalar or an array broadcastable to
        ``theta_arr``; the array path is what `dynamic_ci_scan` uses to
        evaluate per-θ varying η in one bulk numpy call.
        ``D`` accepts either a scalar or an array broadcastable to
        ``theta_arr``.
        """
        from ..models.normal_normal import NormalNormalModel

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "MixtureTilting.tilted_pvalue currently requires "
                f"NormalNormalModel; got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "MixtureTilting.tilted_pvalue currently requires a "
                "NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        theta_arr = np.asarray(theta, dtype=np.float64)
        eta_arr = np.asarray(eta, dtype=np.float64)

        # Vectorised admissibility: eta in [0, 1] and finite.
        invalid = ~(np.isfinite(eta_arr) & (eta_arr >= 0.0) & (eta_arr <= 1.0))
        if np.any(invalid):
            if eta_arr.ndim == 0:
                raise TiltingDomainError(
                    f"MixtureTilting.tilted_pvalue requires eta in [0, 1], "
                    f"got {float(eta_arr)!r}."
                )
            bad = int(np.argmax(invalid))
            raise TiltingDomainError(
                f"MixtureTilting.tilted_pvalue requires eta in [0, 1] and "
                f"finite; offending index {bad} eta="
                f"{float(eta_arr.flat[bad])!r}."
            )

        if statistic_name == "wald":
            # Wald is eta-independent: the m-geodesic does not change
            # the MLE-based statistic. Same closed form as `power_law`
            # and `ot` Wald branches.
            z = np.abs(D - theta_arr) / sigma
            return np.asarray(2.0 * stats.norm.sf(z), dtype=np.float64)

        if statistic_name == "waldo":
            # WALDO under the mixture posterior: per-component Phi-pair
            # around the shifted centre `mu_eta = (1-eta)*mu_n + eta*D`.
            # See Step 4b of `audit/tier2/mixture_derivation.md`.
            mu_n = w * D + (1.0 - w) * mu0
            sigma_n = np.sqrt(w) * sigma
            mu_eta = (1.0 - eta_arr) * mu_n + eta_arr * D
            z = np.abs(theta_arr - mu_eta)
            Phi = stats.norm.cdf
            P1 = Phi((mu_n - mu_eta - z) / sigma_n) + Phi((mu_eta - mu_n - z) / sigma_n)
            P2 = Phi((D - mu_eta - z) / sigma) + Phi((mu_eta - D - z) / sigma)
            return np.asarray(
                (1.0 - eta_arr) * P1 + eta_arr * P2,
                dtype=np.float64,
            )

        raise NotImplementedError(
            f"MixtureTilting.tilted_pvalue not implemented for "
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
                "MixtureTilting.tilted_confidence_interval currently requires "
                "NormalNormalModel."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "MixtureTilting.tilted_confidence_interval currently requires a "
                "NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)
        mu_n = w * D + (1.0 - w) * mu0

        # Bracket midpoint: shifted mean for WALDO, D for Wald.
        mid = float(D) if statistic_name == "wald" else float((1.0 - eta) * mu_n + eta * D)

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

        Delegates to `frasian.tilting._dynamic.dynamic_ci_scan`; the
        scheme-specific bit is the `tilted_pvalue` closure.
        """
        from ..models.normal_normal import NormalNormalModel
        from ._dynamic import dynamic_ci_scan

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "MixtureTilting.dynamic_tilted_confidence_interval currently "
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
                f"MixtureTilting requires NormalNormalModel for the uniform CI "
                f"interface; got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                f"MixtureTilting requires a NormalDistribution prior; "
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
        """Selector-aware region list. Single-element for static selectors;
        multi-element for dynamic selectors (or for static eta in (0, 1) at
        bimodal-mixture conflict where the WALDO p-curve is non-monotone).

        ``config``: see `power_law.confidence_regions` for the dynamic-CI
        scan resolution semantics.
        """
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
