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
between posterior-driven and likelihood-driven behavior. The smoothness
diagnostic (Step 5) makes that complaint quantitative.

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
                "tilt() expects scalar eta; vectorized eta lands with "
                "the smoothness experiment in Step 5."
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

    # ----- (TiltingScheme, TestStatistic) cross-product specializations -----
    #
    # The Step-5 SmoothnessExperiment needs `tilted p-value` and `tilted CI`
    # — quantities that depend on *both* the tilting scheme and the test
    # statistic. The cleanest factoring (multiple dispatch on (scheme,
    # statistic) types) is deferred to Step 6 when more cells exist; for
    # now we dispatch on `statistic.name` inside the scheme. Documented as
    # a temporary bridge in docs/methods/power_law.md.

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

        Algorithm:
          1. Build a coarse |Δ| grid covering the search range.
          2. Compute η*(|Δ|) on the coarse grid via `eta_selector.select_grid`.
          3. Build a fine θ scan; interpolate η* to each θ from the coarse grid.
          4. Compute the dynamic p-value at each θ.
          5. Find α-crossings; refine each via brentq.
          6. Stitch crossings into intervals (region count may be > 1: the
             dynamic p-value can be multimodal at low |Δ|).

        Returns `(regions, total_width, n_regions)`.
        """
        from ..models.normal_normal import NormalNormalModel
        from scipy import optimize

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "dynamic_tilted_confidence_interval currently requires "
                "NormalNormalModel."
            )
        sigma = model.sigma
        mu0 = prior.loc
        sigma0 = prior.scale
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)

        # Search range centred on D; theta and D agree in scale on canonical sandbox.
        search_half = search_mult * sigma
        theta_lo = D - search_half
        theta_hi = D + search_half
        theta_grid = np.linspace(theta_lo, theta_hi, n_grid)

        # |Delta(theta)| over the scan.
        abs_delta_theta = np.abs((1.0 - w) * (mu0 - theta_grid) / sigma)

        # Coarse η* lookup.
        from .eta_selectors import _NamedStatistic
        ad_max = float(abs_delta_theta.max()) + 1e-6
        coarse_grid = np.linspace(0.0, ad_max, coarse_n)
        coarse_eta = eta_selector.select_grid(
            coarse_grid, self, statistic=_NamedStatistic(statistic_name),
            w=w, alpha=alpha,
        )
        # Interpolate to the fine θ scan.
        eta_at_theta = np.interp(abs_delta_theta, coarse_grid, coarse_eta)

        # Dynamic p-values at each θ on the fine grid.
        p_theta = np.empty_like(theta_grid)
        for i in range(theta_grid.size):
            p_theta[i] = float(self.tilted_pvalue(
                float(theta_grid[i]), D, model, prior,
                float(eta_at_theta[i]), statistic_name,
            ))

        # Find α-crossings.
        diff = p_theta - alpha
        crossings: list[float] = []
        for i in range(theta_grid.size - 1):
            if diff[i] * diff[i + 1] < 0.0:
                # Brentq refines on the dynamic-p function.
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
                    # Linear interpolation fallback.
                    t = diff[i] / (diff[i] - diff[i + 1])
                    crossings.append(
                        float(theta_grid[i] + t * (theta_grid[i + 1]
                                                    - theta_grid[i]))
                    )

        # Build intervals from crossings. We treat the scan endpoints as
        # closed: if p > alpha there, we include the endpoint as an
        # implicit crossing.
        regions: list[tuple[float, float]] = []
        if not crossings:
            # No crossings — either entirely inside or entirely outside.
            if p_theta[len(p_theta) // 2] >= alpha:
                regions = [(float(theta_lo), float(theta_hi))]
        else:
            entries = list(crossings)
            # Prepend / append endpoints if needed so the parity comes out right.
            if p_theta[0] >= alpha:
                entries = [float(theta_lo)] + entries
            if p_theta[-1] >= alpha:
                entries = entries + [float(theta_hi)]
            # Pair them up.
            for i in range(0, len(entries) - 1, 2):
                regions.append((entries[i], entries[i + 1]))

        total = float(sum(hi - lo for lo, hi in regions))
        return regions, total, len(regions)

    # ----- Uniform CI interface (called by experiments) -----

    def confidence_interval(self, alpha: float, data: NDArray[np.float64],
                            model: Model, prior: Prior,
                            statistic: TestStatistic
                            ) -> tuple[float, float]:
        """Uniform CI: pick η via `self.selector`, then invert.

        Static selector → one-shot `tilted_confidence_interval` at the
        selected η. Dynamic selector → `dynamic_tilted_confidence_interval`
        with η*(|Δ(θ)|); when the result is multi-region we return the
        convex hull (min lo, max hi). Region-count metadata is not
        propagated through this interface — callers that need it should
        invoke `dynamic_tilted_confidence_interval` directly.
        """
        from ..models.normal_normal import NormalNormalModel

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "PowerLawTilting.confidence_interval requires "
                f"NormalNormalModel; got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "PowerLawTilting.confidence_interval requires a "
                f"NormalDistribution prior; got {type(prior).__name__!r}."
            )

        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        sigma = model.sigma
        sigma0 = prior.scale
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        # |Δ| at the observed D, used as the single context for static selectors.
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
            lo = float(min(r[0] for r in regions))
            hi = float(max(r[1] for r in regions))
            return (lo, hi)

        eta = float(self.selector.select(ctx, self, statistic=statistic))
        return self.tilted_confidence_interval(
            alpha, D, model, prior, eta, statistic.name,
        )

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
