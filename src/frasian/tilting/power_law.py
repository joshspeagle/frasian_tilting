"""Power-law tilting (the legacy eta-tilting, ported into the new shape).

  q(theta; eta) ∝ L(theta) * pi(theta)^(1 - eta)

For the conjugate Normal-Normal model this admits the closed form (Theorem 6
in the legacy derivations):

  denom    = 1 - eta * (1 - w)
  mu_eta   = (w*D + (1-eta)*(1-w)*mu0) / denom
  sigma_eta^2 = w * sigma^2 / denom
  w_eta    = w / denom

Identity element is `eta = 0` (recovers the WALDO posterior). The motivating
research observation — the reason this whole framework exists — is that the
*selection* of eta as a function of |Delta| produces a sharp transition
between posterior-driven and likelihood-driven behavior; the smoothness
experiment makes that complaint quantitative.

Admissible range is bounded below by the non-negativity of `denom`:
  eta_min = -w/(1-w) + buffer       (variance positive)
  eta_max = +inf in principle; capped at 1 in practice (Wald limit).

This implementation specializes on `NormalNormalModel`. Calling `tilt` with a
non-conjugate-Normal posterior raises `NotImplementedError`, by design — the
generic numerical fallback is a future extension and would obscure the math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..config import Config
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..models.normal_normal import weight as _weight
from ..statistics.base import TestStatistic
from .base import ParamSpec, TiltingContext
from .eta_selectors import (DynamicNumericalEtaSelector, FixedEtaSelector,
                            NumericalEtaSelector)


_ETA_MIN_BUFFER = Config.default().eta_min_buffer


def _require_gaussian(posterior: Posterior, prior: Prior, likelihood: Likelihood
                      ) -> tuple[NormalDistribution, NormalDistribution,
                                 GaussianLikelihood]:
    """Recognize a Normal-Normal context. Raise if any input is not Gaussian."""
    if not isinstance(posterior, NormalDistribution):
        raise NotImplementedError(
            "PowerLawTilting requires a NormalDistribution posterior; "
            f"got {type(posterior).__name__!r}. Numerical fallback is a "
            f"future extension."
        )
    if not isinstance(prior, NormalDistribution):
        raise NotImplementedError(
            "PowerLawTilting requires a NormalDistribution prior; "
            f"got {type(prior).__name__!r}."
        )
    if not isinstance(likelihood, GaussianLikelihood):
        raise NotImplementedError(
            "PowerLawTilting requires a GaussianLikelihood; "
            f"got {type(likelihood).__name__!r}."
        )
    return posterior, prior, likelihood


def _denom(w: float, eta: float) -> float:
    return 1.0 - eta * (1.0 - w)


@register_tilting(name="power_law", brief="docs/methods/power_law.md")
@dataclass(frozen=True)
class PowerLawTilting:
    """The legacy eta-tilting scheme as a `TiltingScheme` implementation.

    Parameterised by an `EtaSelector` that decides the η used at CI
    inversion time. Static selectors (`FixedEtaSelector`,
    `NumericalEtaSelector`) route through `tilted_confidence_interval`;
    dynamic selectors (`DynamicNumericalEtaSelector`) route through
    `dynamic_tilted_confidence_interval` and produce per-θ varying η.
    The cell display name picks up the selector's name when non-default
    so the runner emits distinguishable cells like
    `power_law[dynamic_numerical]`.
    """

    name: str = "power_law"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="eta=0 recovers WALDO; eta=1 recovers Wald.",
    )
    selector: object = field(
        default_factory=lambda: FixedEtaSelector(eta=0.0)
    )

    @property
    def cell_name(self) -> str:
        """Display name for the runner's cell key.

        `power_law` for the default fixed-η-zero selector (matches the
        identity tilting numerically); `power_law[<selector>]` otherwise.
        """
        sel_name = getattr(self.selector, "name", "")
        if isinstance(self.selector, FixedEtaSelector) and self.selector.eta == 0.0:
            return self.name
        return f"{self.name}[{sel_name}]"

    # ----- TiltingScheme protocol -----

    def tilt(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             eta: ArrayLike) -> NormalDistribution:
        """Closed-form tilted Normal posterior (Theorem 6 in legacy docs)."""
        post, pri, lik = _require_gaussian(posterior, prior, likelihood)

        eta_arr = np.asarray(eta, dtype=np.float64)
        if eta_arr.ndim != 0:
            raise NotImplementedError(
                "tilt() expects scalar eta; vectorised eta is consumed "
                "by the smoothness experiment via repeated scalar calls."
            )
        eta_f = float(eta_arr)

        w = _weight(lik.sigma, pri.scale)
        denom = _denom(w, eta_f)
        if denom <= 0.0:
            raise TiltingDomainError(
                f"eta={eta_f!r} drives the tilted-posterior denominator to "
                f"{denom!r} <= 0 with w={w!r}; admissible range is "
                f"({-w/(1-w):+.6g}, inf)."
            )

        mu_eta = (w * lik.D + (1.0 - eta_f) * (1.0 - w) * pri.loc) / denom
        sigma_eta_sq = w * lik.sigma ** 2 / denom
        sigma_eta = float(np.sqrt(sigma_eta_sq))
        return NormalDistribution(loc=float(mu_eta), scale=sigma_eta)

    def path(self, posterior: Posterior, prior: Prior, likelihood: Likelihood,
             ts: NDArray[np.float64]) -> Iterable[NormalDistribution]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, t)

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
        w = context.w
        if not (0.0 < w < 1.0):
            raise ValueError(f"context.w must lie in (0, 1), got {w!r}")
        # denom = 1 - eta*(1 - w) > 0  ⇔  eta < 1/(1 - w); also eta > -w/(1-w)
        # for the equivalent variance-positivity condition with the buffer.
        eta_low = -w / (1.0 - w) + _ETA_MIN_BUFFER
        eta_high = 1.0 / (1.0 - w) - _ETA_MIN_BUFFER
        return (eta_low, eta_high)

    # ----- (TiltingScheme, TestStatistic) cross-product specialisations -----
    #
    # The smoothness experiment needs `tilted p-value` and `tilted CI` —
    # quantities that depend on *both* the tilting scheme and the test
    # statistic. The cleanest factoring would be multiple dispatch on
    # (scheme, statistic) types; for now we dispatch on `statistic.name`
    # inside the scheme. Documented as a temporary bridge in
    # `docs/methods/power_law.md`. The generalisation lands when a
    # second non-Wald/WALDO statistic gets implemented.

    def tilted_pvalue(self, theta: ArrayLike, D: float, model: object,
                      prior: NormalDistribution, eta: float,
                      statistic_name: str) -> NDArray[np.float64]:
        """p-value of `statistic_name` evaluated against the tilted posterior.

        Specialized for (power_law, waldo) — Theorem 8 closed form — and
        (power_law, wald) — eta-independent two-sided Wald.
        """
        from ..models.normal_normal import NormalNormalModel

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "tilted_pvalue currently requires NormalNormalModel; "
                f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "tilted_pvalue currently requires a NormalDistribution prior."
            )
        sigma = model.sigma
        mu0 = prior.loc
        sigma0 = prior.scale
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        denom = _denom(w, eta)
        if denom <= 0.0:
            raise TiltingDomainError(
                f"eta={eta!r} drives denom to {denom!r} <= 0 with w={w!r}."
            )

        theta_arr = np.asarray(theta, dtype=np.float64)
        if statistic_name == "wald":
            # Wald is eta-independent: 2 * (1 - Phi(|D - theta| / sigma)).
            from scipy import stats as _stats
            z = np.abs(D - theta_arr) / sigma
            return np.asarray(2.0 * _stats.norm.sf(z), dtype=np.float64)
        if statistic_name == "waldo":
            from scipy import stats as _stats
            mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * mu0) / denom
            norm_factor = w * sigma / denom
            a_eta = np.abs(mu_eta - theta_arr) / norm_factor
            b_eta = (1.0 - eta) * (1.0 - w) * (mu0 - theta_arr) / (denom * norm_factor)
            return np.asarray(
                _stats.norm.cdf(b_eta - a_eta) + _stats.norm.cdf(-a_eta - b_eta),
                dtype=np.float64,
            )
        raise NotImplementedError(
            f"PowerLawTilting.tilted_pvalue not implemented for "
            f"statistic={statistic_name!r}; supported: 'wald', 'waldo'."
        )

    def dynamic_tilted_pvalue(self, theta: ArrayLike, D: float,
                                model: object,
                                prior: NormalDistribution,
                                statistic_name: str,
                                eta_at_theta: ArrayLike,
                                ) -> NDArray[np.float64]:
        """p(theta) with η varying per θ via a precomputed lookup.

        Caller supplies `eta_at_theta`, the η* value chosen *for each θ*.
        This is what `dynamic_tilted_confidence_interval` does internally:
        run a coarse η-selector, then interpolate η*(|Δ(θ)|) across the
        fine θ scan.
        """
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
        self,
        alpha: float, D: float,
        model: object,
        prior: NormalDistribution,
        statistic_name: str,
        eta_selector,
        n_grid: int = 401,
        coarse_n: int = 25,
        search_mult: float = 8.0,
    ) -> tuple[list[tuple[float, float]], float, int]:
        """Dynamic-η CI: η = η*(|Δ(θ)|) per θ.

        Delegates the scan + α-crossing algorithm to
        `frasian.tilting._dynamic.dynamic_ci_scan` (see that function's
        docstring for the documented step-by-step recipe). The
        scheme-specific bit is the `tilted_pvalue` closure.
        """
        from ..models.normal_normal import NormalNormalModel
        from ._dynamic import dynamic_ci_scan

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "dynamic_tilted_confidence_interval currently requires "
                "NormalNormalModel."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        def _tilted_pvalue_fn(theta: float, eta: float) -> float:
            return float(self.tilted_pvalue(
                theta, D, model, prior, eta, statistic_name,
            ))

        return dynamic_ci_scan(
            tilted_pvalue_fn=_tilted_pvalue_fn,
            alpha=alpha, D=D, w=w, mu0=mu0, sigma=sigma,
            eta_selector=eta_selector, scheme=self,
            statistic_name=statistic_name,
            n_grid=n_grid, coarse_n=coarse_n, search_mult=search_mult,
        )

    # ----- Uniform CI / regions / pvalue interface (called by experiments) -----

    def _require_normal_sandbox(self, model: Model, prior: Prior) -> None:
        from ..models.normal_normal import NormalNormalModel
        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "PowerLawTilting requires NormalNormalModel; "
                f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "PowerLawTilting requires a NormalDistribution prior; "
                f"got {type(prior).__name__!r}."
            )

    def confidence_regions(self, alpha: float, data: NDArray[np.float64],
                            model: Model, prior: Prior,
                            statistic: TestStatistic
                            ) -> list[tuple[float, float]]:
        """Selector-aware region list. Single-element for static selectors;
        multi-element for dynamic selectors at conflict-band D where the
        dynamic p-value is multimodal."""
        self._require_normal_sandbox(model, prior)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        sigma = model.sigma
        sigma0 = prior.scale
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
        """Convex hull of `confidence_regions` — single (lo, hi) summary.

        Multi-region cells (e.g. dynamic-η Dyn-WALDO at low |Δ|) collapse
        to `(min lo, max hi)` here; consumers that need union semantics
        should call `confidence_regions` directly.
        """
        regions = self.confidence_regions(alpha, data, model, prior, statistic)
        lo = float(min(r[0] for r in regions))
        hi = float(max(r[1] for r in regions))
        return (lo, hi)

    def pvalue(self, theta: ArrayLike, data: NDArray[np.float64],
               model: Model, prior: Prior,
               statistic: TestStatistic) -> NDArray[np.float64]:
        """Selector-aware p-value at hypothesised θ values.

        Static selector: resolve a single η via `selector.select(...)`,
        evaluate `tilted_pvalue(θ, D, …, η, statistic.name)`.

        Dynamic selector: precompute the coarse η*(|Δ|) lookup once, then
        interpolate to the per-θ |Δ_θ| values in `theta`, and evaluate
        `dynamic_tilted_pvalue` with the resulting `eta_at_theta` array.

        The α used by the dynamic selector defaults to
        `Config.default().alpha` (= 0.05) — overridable via
        `metadata={'alpha': ...}` on the prior or via subclassing if a
        downstream consumer needs different per-α p-value evaluation.
        """
        self._require_normal_sandbox(model, prior)
        from ..config import Config
        from .eta_selectors import _NamedStatistic

        alpha = float(Config.default().alpha)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        sigma = model.sigma
        sigma0 = prior.scale
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        if getattr(self.selector, "is_dynamic", False):
            # Per-θ |Δ_θ| values, then interpolate η*(|Δ|) onto them.
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

        # Static selector: single η for the whole evaluation.
        abs_delta = abs((1.0 - w) * (prior.loc - D) / sigma)
        ctx = TiltingContext(w=w, abs_delta=abs_delta, alpha=alpha)
        eta = float(self.selector.select(ctx, self, statistic=statistic))
        return self.tilted_pvalue(theta_arr, D, model, prior, eta, statistic.name)

    def tilted_confidence_interval(self, alpha: float, D: float,
                                    model: object,
                                    prior: NormalDistribution,
                                    eta: float,
                                    statistic_name: str
                                    ) -> tuple[float, float]:
        """Numerical CI inversion of `tilted_pvalue` via brentq_with_doubling."""
        from ..models.normal_normal import NormalNormalModel
        from ._solvers import brentq_with_doubling

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "tilted_confidence_interval currently requires NormalNormalModel."
            )
        sigma = model.sigma
        sigma0 = prior.scale
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        # Use the tilted posterior mean as the bracket midpoint (CI mode for WALDO).
        denom = _denom(w, eta)
        if denom <= 0.0:
            raise TiltingDomainError(
                f"eta={eta!r} drives denom to {denom!r} <= 0 with w={w!r}."
            )
        mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * prior.loc) / denom
        if statistic_name == "wald":
            mu_eta = D  # Wald CI is centred on D, not mu_eta.

        def f(theta_val: float) -> float:
            return float(self.tilted_pvalue(
                float(theta_val), D, model, prior, eta, statistic_name
            )) - alpha

        half = 4.0 * sigma
        lo = brentq_with_doubling(f, midpoint=float(mu_eta),
                                    initial_half_width=half, direction=-1)
        hi = brentq_with_doubling(f, midpoint=float(mu_eta),
                                    initial_half_width=half, direction=+1)
        return (lo, hi)
