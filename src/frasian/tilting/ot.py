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

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..statistics.base import TestStatistic
from .base import ParamSpec, TiltingContext
from .eta_selectors import FixedEtaSelector
from .quantile_mixture import QuantileMixturePath


@register_tilting(name="ot", brief="docs/methods/ot.md")
@dataclass(frozen=True)
class OTTilting:
    """Wasserstein-2 geodesic tilting (general 1D, Gaussian fast path)."""

    name: str = "ot"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description=(
            "t in [0, 1] along the W2 geodesic between posterior (t=0) "
            "and likelihood-induced Gaussian N(D, sigma^2) (t=1)."
        ),
    )
    selector: object = field(
        default_factory=lambda: FixedEtaSelector(eta=0.0)
    )

    @property
    def cell_name(self) -> str:
        sel_name = getattr(self.selector, "name", "")
        if isinstance(self.selector, FixedEtaSelector) and self.selector.eta == 0.0:
            return self.name
        return f"{self.name}[{sel_name}]"

    # ----- TiltingScheme protocol -----

    def tilt(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             eta: ArrayLike) -> Posterior:
        """W2-geodesic tilt between posterior and likelihood-as-distribution."""
        eta_arr = np.asarray(eta, dtype=np.float64)
        if eta_arr.ndim != 0:
            raise NotImplementedError(
                "tilt() expects scalar eta; vectorised eta is consumed via "
                "repeated scalar calls (see `path`)."
            )
        t = float(eta_arr)
        if not (0.0 <= t <= 1.0):
            raise TiltingDomainError(
                f"OTTilting requires eta in [0, 1], got {t!r}."
            )

        # Gaussian fast path: linear interpolation in (mu, sigma).
        if (isinstance(posterior, NormalDistribution)
                and isinstance(likelihood, GaussianLikelihood)):
            mu_a, sigma_a = posterior.loc, posterior.scale
            mu_b, sigma_b = float(likelihood.D), float(likelihood.sigma)
            return NormalDistribution(
                loc=(1.0 - t) * mu_a + t * mu_b,
                scale=(1.0 - t) * sigma_a + t * sigma_b,
            )

        # General 1D path: quantile-mixture. Requires the likelihood to
        # admit a Distribution view; on the Gaussian-likelihood sandbox we
        # construct N(D, sigma) directly. Other Likelihood types raise
        # NotImplementedError, matching `power_law`'s discipline.
        if isinstance(likelihood, GaussianLikelihood):
            q = NormalDistribution(loc=likelihood.D, scale=likelihood.sigma)
            return QuantileMixturePath(p=posterior, q=q, t=t)

        raise NotImplementedError(
            "OTTilting requires a GaussianLikelihood for likelihood-to-"
            "distribution conversion. General conversion is a future "
            "extension."
        )

    def path(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             ts: NDArray[np.float64]) -> Iterable[Posterior]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, float(t))

    def is_identity(self, eta: float) -> bool:
        return float(eta) == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        return (0.0, 1.0)

    # ----- (TiltingScheme, TestStatistic) cross-product specialisations -----

    def tilted_pvalue(self, theta: ArrayLike, D: float, model: Model,
                      prior: NormalDistribution, eta: float,
                      statistic_name: str) -> NDArray[np.float64]:
        """Tilted p-value evaluated against the W2-tilted Gaussian.

        Specialized for (ot, waldo) and (ot, wald) on Normal-Normal:

          (ot, wald): eta-independent two-sided Wald, 2 * Phi(-|D-theta|/sigma)
          (ot, waldo): closed form derived in docs/methods/ot.md, with the
            standard error s_t = (w + eta*(1-w))*sigma.

        Endpoint sanity: at eta=0 reduces to bare WALDO; at eta=1 reduces
        to bare Wald (s_t -> sigma, mu_t -> D, b -> 0, a -> |D-theta|/sigma).
        """
        from ..models.normal_normal import NormalNormalModel

        if not isinstance(model, NormalNormalModel):
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
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        theta_arr = np.asarray(theta, dtype=np.float64)

        if statistic_name == "wald":
            z = np.abs(D - theta_arr) / sigma
            return np.asarray(2.0 * stats.norm.sf(z), dtype=np.float64)

        if statistic_name == "waldo":
            mu_n = w * D + (1.0 - w) * mu0
            mu_t = (1.0 - eta) * mu_n + eta * D
            s_t = (w + eta * (1.0 - w)) * sigma
            a = np.abs(mu_t - theta_arr) / s_t
            b = (1.0 - eta) * (1.0 - w) * (mu0 - theta_arr) / s_t
            return np.asarray(
                stats.norm.cdf(b - a) + stats.norm.cdf(-a - b),
                dtype=np.float64,
            )

        raise NotImplementedError(
            f"OTTilting.tilted_pvalue not implemented for "
            f"statistic={statistic_name!r}; supported: 'wald', 'waldo'."
        )

    def tilted_confidence_interval(
        self, alpha: float, D: float, model: Model,
        prior: NormalDistribution, eta: float, statistic_name: str,
    ) -> tuple[float, float]:
        """Numerical CI inversion of `tilted_pvalue` via brentq_with_doubling."""
        from ..models.normal_normal import NormalNormalModel
        from ._solvers import brentq_with_doubling

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "OTTilting.tilted_confidence_interval currently requires "
                "NormalNormalModel."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "OTTilting.tilted_confidence_interval currently requires a "
                "NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        mu_n = w * D + (1.0 - w) * mu0

        if statistic_name == "wald":
            mid = float(D)
        else:  # waldo
            mid = float((1.0 - eta) * mu_n + eta * D)

        def f(theta_val: float) -> float:
            return float(self.tilted_pvalue(
                float(theta_val), D, model, prior, eta, statistic_name
            )) - alpha

        half = 4.0 * sigma
        lo = brentq_with_doubling(f, midpoint=mid,
                                    initial_half_width=half, direction=-1)
        hi = brentq_with_doubling(f, midpoint=mid,
                                    initial_half_width=half, direction=+1)
        return (lo, hi)

    def dynamic_tilted_pvalue(self, theta: ArrayLike, D: float,
                                model: Model, prior: NormalDistribution,
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
        out = np.empty_like(theta_arr)
        for i in range(theta_arr.size):
            out[i] = float(self.tilted_pvalue(
                float(theta_arr[i]), D, model, prior, float(eta_arr[i]),
                statistic_name,
            ))
        return out if out.size > 1 else np.asarray(float(out.item()))

    def dynamic_tilted_confidence_interval(
        self, alpha: float, D: float, model: Model,
        prior: NormalDistribution, statistic_name: str,
        eta_selector,
        n_grid: int = 401, coarse_n: int = 25, search_mult: float = 8.0,
    ) -> tuple[list[tuple[float, float]], float, int]:
        """Dynamic-eta CI: eta = eta*(|Delta(theta)|) per theta.

        Algorithm mirrors `PowerLawTilting.dynamic_tilted_confidence_interval`;
        the only OT-specific bit is the `tilted_pvalue` callback. See the
        power-law version for the documented step-by-step recipe.
        """
        from ..models.normal_normal import NormalNormalModel
        from scipy import optimize
        from .eta_selectors import _NamedStatistic

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "OTTilting.dynamic_tilted_confidence_interval currently "
                "requires NormalNormalModel."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        search_half = search_mult * sigma
        theta_lo = D - search_half
        theta_hi = D + search_half
        theta_grid = np.linspace(theta_lo, theta_hi, n_grid)

        abs_delta_theta = np.abs((1.0 - w) * (mu0 - theta_grid) / sigma)
        ad_max = float(abs_delta_theta.max()) + 1e-6
        coarse_grid = np.linspace(0.0, ad_max, coarse_n)
        coarse_eta = eta_selector.select_grid(
            coarse_grid, self, statistic=_NamedStatistic(statistic_name),
            w=w, alpha=alpha,
        )
        eta_at_theta = np.interp(abs_delta_theta, coarse_grid, coarse_eta)

        p_theta = np.empty_like(theta_grid)
        for i in range(theta_grid.size):
            p_theta[i] = float(self.tilted_pvalue(
                float(theta_grid[i]), D, model, prior,
                float(eta_at_theta[i]), statistic_name,
            ))

        diff = p_theta - alpha
        crossings: list[float] = []
        for i in range(theta_grid.size - 1):
            if diff[i] * diff[i + 1] < 0.0:
                def _f(theta_val: float, _i=i) -> float:
                    ad = abs((1.0 - w) * (mu0 - theta_val) / sigma)
                    eta = float(np.interp(ad, coarse_grid, coarse_eta))
                    return float(self.tilted_pvalue(
                        theta_val, D, model, prior, eta, statistic_name,
                    )) - alpha
                try:
                    cross = optimize.brentq(
                        _f, theta_grid[i], theta_grid[i + 1], xtol=1e-9,
                    )
                    crossings.append(float(cross))
                except ValueError:
                    t = diff[i] / (diff[i] - diff[i + 1])
                    crossings.append(
                        float(theta_grid[i] + t * (theta_grid[i + 1]
                                                    - theta_grid[i]))
                    )

        regions: list[tuple[float, float]] = []
        if not crossings:
            if p_theta[len(p_theta) // 2] >= alpha:
                regions = [(float(theta_lo), float(theta_hi))]
        else:
            entries = list(crossings)
            if p_theta[0] >= alpha:
                entries = [float(theta_lo)] + entries
            if p_theta[-1] >= alpha:
                entries = entries + [float(theta_hi)]
            for i in range(0, len(entries) - 1, 2):
                regions.append((entries[i], entries[i + 1]))

        total = float(sum(hi - lo for lo, hi in regions))
        return regions, total, len(regions)

    # ----- Uniform CI / regions / pvalue interface -----

    def _require_normal_sandbox(self, model: Model, prior: Prior) -> None:
        from ..models.normal_normal import NormalNormalModel
        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                f"OTTilting requires NormalNormalModel for the uniform CI "
                f"interface; got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                f"OTTilting requires a NormalDistribution prior; "
                f"got {type(prior).__name__!r}."
            )

    def confidence_regions(self, alpha: float, data: NDArray[np.float64],
                            model: Model, prior: Prior,
                            statistic: TestStatistic
                            ) -> list[tuple[float, float]]:
        self._require_normal_sandbox(model, prior)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        abs_delta = abs((1.0 - w) * (prior.loc - D) / sigma)
        ctx = TiltingContext(w=w, abs_delta=abs_delta, alpha=alpha)

        if getattr(self.selector, "is_dynamic", False):
            regions, _, _ = self.dynamic_tilted_confidence_interval(
                alpha, D, model, prior, statistic.name, self.selector,
                n_grid=getattr(self.selector, "n_grid", 401),
                coarse_n=getattr(self.selector, "coarse_n", 25),
                search_mult=getattr(self.selector, "search_mult", 8.0),
            )
            if not regions:
                raise RuntimeError(
                    f"dynamic CI inversion produced no regions at D={D!r}"
                )
            return regions

        eta = float(self.selector.select(ctx, self, statistic=statistic))
        return [self.tilted_confidence_interval(
            alpha, D, model, prior, eta, statistic.name,
        )]

    def confidence_interval(self, alpha: float, data: NDArray[np.float64],
                            model: Model, prior: Prior,
                            statistic: TestStatistic
                            ) -> tuple[float, float]:
        regions = self.confidence_regions(alpha, data, model, prior, statistic)
        lo = float(min(r[0] for r in regions))
        hi = float(max(r[1] for r in regions))
        return (lo, hi)

    def pvalue(self, theta: ArrayLike, data: NDArray[np.float64],
               model: Model, prior: Prior,
               statistic: TestStatistic) -> NDArray[np.float64]:
        self._require_normal_sandbox(model, prior)
        from ..config import Config
        from .eta_selectors import _NamedStatistic

        alpha = float(Config.default().alpha)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        if getattr(self.selector, "is_dynamic", False):
            abs_delta_theta = np.abs((1.0 - w) * (prior.loc - theta_arr) / sigma)
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            ad_max = float(abs_delta_theta.max()) + 1e-6
            coarse_grid = np.linspace(0.0, ad_max, coarse_n)
            coarse_eta = self.selector.select_grid(
                coarse_grid, self, statistic=_NamedStatistic(statistic.name),
                w=w, alpha=alpha,
            )
            eta_at_theta = np.interp(abs_delta_theta, coarse_grid, coarse_eta)
            return self.dynamic_tilted_pvalue(
                theta_arr, D, model, prior, statistic.name, eta_at_theta,
            )

        abs_delta = abs((1.0 - w) * (prior.loc - D) / sigma)
        ctx = TiltingContext(w=w, abs_delta=abs_delta, alpha=alpha)
        eta = float(self.selector.select(ctx, self, statistic=statistic))
        return self.tilted_pvalue(theta_arr, D, model, prior, eta,
                                    statistic.name)
