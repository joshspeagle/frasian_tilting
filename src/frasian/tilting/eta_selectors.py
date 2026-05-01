"""η-selectors: strategies for picking η given a tilting context.

The legacy code conflated three solvers (numerical search, closed-form
approximation, learned MLP) inside `tilting.py`. Here they live behind
the `EtaSelector` protocol and can be swapped via configuration.

Selectors come in two flavours flagged by `is_dynamic`:
  - **static**: `select(context)` returns a single η for the whole CI
    inversion. `FixedEtaSelector` (constant η) and `NumericalEtaSelector`
    (η minimising tilted CI width at a representative D) are static.
  - **dynamic**: η varies per θ. `DynamicNumericalEtaSelector` is the
    canonical example — it precomputes η*(|Δ|) on a coarse grid then
    interpolates per θ at CI-inversion time.

Static selectors compose with `TiltingScheme.tilted_confidence_interval`
(one-shot inversion); dynamic selectors compose with
`TiltingScheme.dynamic_tilted_confidence_interval` (scan + crossings).
Both paths route through `TiltingScheme.confidence_interval`, the
uniform interface the experiments call.

Coverage caveat — read before using `NumericalEtaSelector`
==========================================================

The two selector flavours have **different coverage properties**:

  - `DynamicNumericalEtaSelector` (per-θ): the η used at each θ depends
    only on θ (not on the data D). The WALDO p-value at any fixed η is
    U[0,1] under H0: θ = θ_true, so the CI achieves exact 1-α coverage
    by construction. **This is the calibrated default** used by
    `default_tiltings()` and the `coverage` / `width` experiments.

  - `NumericalEtaSelector` (static, post-selection): η = argmin_η
    |CI_η(D)|. Width is monotone non-increasing in flexibility, so this
    CI is always ≤ WALDO and asymptotes to Wald at large |Δ|. **But the
    coverage drops below nominal** (~93% empirically at α=0.05; see
    `tests/regression/test_post_selection_coverage.py`) because picking
    the narrowest CI per D is post-selection inference. Use this
    selector only as a *baseline for studying the coverage / width
    trade-off* — it is not a calibrated estimator. The framework's
    research goal is to find smoother tilting families that retrieve
    static-η-opt's narrowness *while* keeping per-θ-η's calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..statistics.base import TestStatistic
from .base import TiltingContext, TiltingDomainError, TiltingScheme


def _D_from_abs_delta(abs_delta: float, w: float, sigma: float, mu0: float
                      ) -> float:
    """Invert Delta = (1 - w)(mu0 - D)/sigma with the convention Delta >= 0."""
    return float(mu0 - abs_delta * sigma / max(1.0 - w, 1e-12))


@dataclass(frozen=True)
class _NamedStatistic:
    """Minimal `.name`-only adapter for selector calls that only need dispatch.

    `dynamic_tilted_confidence_interval` knows the statistic by name (the
    cell-evaluator dispatch in PowerLawTilting is keyed on `statistic.name`)
    and does not have a concrete `TestStatistic` instance handy. Wrapping
    the name in this adapter satisfies the selector's type contract without
    pulling in the registry.
    """
    name: str


@dataclass(frozen=True)
class FixedEtaSelector:
    """Always return the same η. Identity selector for any tilting scheme.

    `(power_law, FixedEtaSelector(0.0))` is numerically identical to the
    `IdentityTilting` (η=0 recovers the input WALDO posterior); we keep
    the class so other tiltings can declare a non-zero identity element.
    """

    eta: float = 0.0
    name: str = "fixed"
    is_dynamic: bool = False

    def select(self, context: TiltingContext, scheme: TiltingScheme,
               *, statistic: TestStatistic) -> float:
        return float(self.eta)


@dataclass(frozen=True)
class NumericalEtaSelector:
    """Pick η minimizing the (analytic) CI width for the tilted posterior.

    DOES NOT MAINTAIN NOMINAL COVERAGE. Choosing η = argmin_η |CI_η(D)|
    per data sample is post-selection inference: the resulting CI is
    biased toward narrow CIs that systematically exclude θ_true at a
    rate higher than α. Empirically the shortfall is ~2 percentage
    points at α=0.05 (see `tests/regression/test_post_selection_coverage.py`),
    and grows with |Δ|.

    Use this selector only as a baseline for illustrating the
    coverage / width trade-off. For calibrated CIs, use
    `DynamicNumericalEtaSelector` (the per-θ varying selector that the
    framework defaults to in `default_tiltings()`).

    Width-wise, this selector is the genuine optimum: ≤ WALDO at every
    D, → Wald at large |Δ|. That's exactly why coverage drops — the
    procedure is too eager to shrink.

    The width is computed by inverting the tilted p-value at a single
    representative `D` (no Monte-Carlo). For the (power_law, waldo) cell
    this matches the legacy `optimal_eta_numerical` solver up to bracket
    bookkeeping.
    """

    name: str = "numerical"
    sigma: float = 1.0
    mu0: float = 0.0
    eta_min_buffer: float = 1e-3
    is_dynamic: bool = False

    def select(self, context: TiltingContext, scheme: TiltingScheme,
               *, statistic: TestStatistic) -> float:
        eta_lo, eta_hi = scheme.admissible_range(context)
        # Cap the upper end at +1 (the Wald limit); anything beyond is
        # mathematically valid but uninteresting in practice.
        eta_hi = min(eta_hi, 1.0 - self.eta_min_buffer)

        sigma0 = float(np.sqrt(context.w / max(1.0 - context.w, 1e-12))
                        * self.sigma)
        prior = NormalDistribution(loc=self.mu0, scale=sigma0)
        model = NormalNormalModel(sigma=self.sigma)
        D = _D_from_abs_delta(context.abs_delta, context.w, self.sigma,
                                self.mu0)

        def width(eta: float) -> float:
            try:
                if hasattr(scheme, "tilted_confidence_interval"):
                    lo, hi = scheme.tilted_confidence_interval(
                        context.alpha, D, model, prior, eta,
                        statistic.name,
                    )
                else:
                    raise NotImplementedError(
                        f"{type(scheme).__name__} does not implement "
                        f"`tilted_confidence_interval` (Step 5 bridge)."
                    )
            except (NotImplementedError, TiltingDomainError, ValueError, RuntimeError):
                return np.inf
            w_ci = float(hi - lo)
            return w_ci if (w_ci > 0 and np.isfinite(w_ci)) else np.inf

        result = optimize.minimize_scalar(
            width, bounds=(eta_lo, eta_hi), method="bounded",
            options={"xatol": 1e-3},
        )
        return float(result.x)

    def select_grid(self, abs_delta_grid, scheme: TiltingScheme,
                    *, statistic: TestStatistic, w: float, alpha: float
                    ):
        """Vectorised: η* at every |Δ| in `abs_delta_grid`.

        Used by `dynamic_tilted_confidence_interval` to pre-compute a coarse
        η*(|Δ|) lookup that is then interpolated across the fine θ-scan grid.
        Repeating `optimize.minimize_scalar` for every fine-grid θ would be
        prohibitively slow.
        """
        out = np.empty(len(abs_delta_grid), dtype=np.float64)
        for i, ad in enumerate(abs_delta_grid):
            ctx = TiltingContext(w=w, abs_delta=float(ad), alpha=alpha)
            out[i] = self.select(ctx, scheme, statistic=statistic)
        return out


@dataclass
class DynamicNumericalEtaSelector:
    """Per-θ varying η* via the coarse-grid + interpolation strategy.

    The framework's **calibrated default** for `power_law` cells.
    Maintains 1-α coverage by construction: the η used at each θ depends
    only on θ (not on D), so the WALDO p-value at any fixed η is U[0,1]
    under H0, and the CI = {θ : p_dyn(θ; D) ≥ α} has the correct level.
    Contrast with `NumericalEtaSelector`, which picks η post hoc per D
    and undercovers (see that class's docstring).

    Width behaviour: at large |Δ| the dynamic procedure approaches Wald
    *eventually*, but takes a detour past Wald in the conflict band
    (around |Δ| ≈ 2-3 for w=0.5) — the η used at θ values near the
    prior is heavily prior-amplifying even when D is far from the
    prior, dragging the CI toward μ₀. This non-monotone width is the
    structural pathology that smoother tilting families (OT, geodesic,
    mixture) are intended to fix without sacrificing the calibration.

    Wraps `NumericalEtaSelector.select_grid`: the inner machinery is the
    same width-minimising solver, but `is_dynamic = True` flags to the
    tilting that it should route CI inversion through
    `dynamic_tilted_confidence_interval` (scan + α-crossings) rather than
    `tilted_confidence_interval` (one-shot bracket inversion).

    `n_grid` and `coarse_n` control resolution of the per-θ scan and the
    coarse η*(|Δ|) lookup, respectively. `search_mult` controls the half-
    width (in σ) of the θ scan window centred on D.

    `select_grid` is memoized by `(w, α, statistic.name, scheme_name, ad_max)`
    so a per-cell experiment loop (which holds w, α, statistic constant
    and only varies D) does not recompute η*(|Δ|) from scratch on every
    sample — the dominant cost without this cache.
    """

    name: str = "dynamic_numerical"
    sigma: float = 1.0
    mu0: float = 0.0
    eta_min_buffer: float = 1e-3
    n_grid: int = 401
    coarse_n: int = 25
    search_mult: float = 8.0
    is_dynamic: bool = True
    # Mutable cache; we rely on dataclass(eq=False) at class level to keep
    # the selector hashable when needed, and use `field(default_factory=...)`.
    _cache: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._cache is None:
            object.__setattr__(self, "_cache", {})

    def _inner(self) -> NumericalEtaSelector:
        return NumericalEtaSelector(sigma=self.sigma, mu0=self.mu0,
                                    eta_min_buffer=self.eta_min_buffer)

    def select(self, context: TiltingContext, scheme: TiltingScheme,
               *, statistic: TestStatistic) -> float:
        """Convenience: a single-context η* (delegates to the static inner)."""
        return self._inner().select(context, scheme, statistic=statistic)

    def select_grid(self, abs_delta_grid, scheme: TiltingScheme,
                    *, statistic: TestStatistic, w: float, alpha: float
                    ):
        ad_max = float(np.asarray(abs_delta_grid).max())
        coarse_n = len(abs_delta_grid)
        scheme_name = getattr(scheme, "name", type(scheme).__name__)
        # Cache key drops `ad_max`: η*(|Δ|) is monotone with η* → 1 (Wald
        # limit) as |Δ| → ∞, so `np.interp` clamping at the cached
        # boundary returns the correct asymptotic value for any |Δ|
        # exceeding the cached range. This makes the per-cell loop
        # (constant w, α, statistic; varying D) hit the cache once and
        # then run at the cost of a vector interp per D.
        key = (w, alpha, statistic.name, scheme_name, coarse_n)
        cached = self._cache.get(key)
        if cached is not None:
            cached_grid, cached_eta = cached
            # If a later call needs a wider ad_max than we cached, extend
            # the lookup once and cache the wider version.
            if ad_max > cached_grid[-1] * 1.05:
                wider = max(ad_max, cached_grid[-1]) * 1.5
                cached_grid = np.linspace(0.0, wider, coarse_n)
                cached_eta = self._inner().select_grid(
                    cached_grid, scheme, statistic=statistic,
                    w=w, alpha=alpha,
                )
                self._cache[key] = (cached_grid, cached_eta)
            return np.interp(abs_delta_grid, cached_grid, cached_eta)
        # Fresh compute — slightly pad ad_max so subsequent calls with a
        # marginally wider range still hit the cache.
        wider = max(ad_max, 1e-3) * 1.5
        coarse_grid_full = np.linspace(0.0, wider, coarse_n)
        eta_full = self._inner().select_grid(
            coarse_grid_full, scheme, statistic=statistic, w=w, alpha=alpha,
        )
        self._cache[key] = (coarse_grid_full, eta_full)
        return np.interp(abs_delta_grid, coarse_grid_full, eta_full)
