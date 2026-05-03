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

from dataclasses import dataclass, field

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
                        f"`tilted_confidence_interval`."
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

    `select_grid` is memoized by
    `(w, α, statistic.name, scheme_name, coarse_n, ad_max_bin)`, where
    `ad_max_bin = ceil(max(abs_delta_grid) / cache_bin_width) *
    cache_bin_width` — the smallest 0.5-wide cache bin that covers the
    requested range. This makes the cache **order-independent**: every
    call with the same `ad_max_bin` returns the same lookup (the same
    `np.linspace(0, ad_max_bin, coarse_n)` evaluated by the inner
    width-minimising solver), regardless of what other queries have hit
    the selector first.
    """

    name: str = "dynamic_numerical"
    sigma: float = 1.0
    mu0: float = 0.0
    eta_min_buffer: float = 1e-3
    n_grid: int = 401
    coarse_n: int = 25
    search_mult: float = 8.0
    cache_bin_width: float = 0.5
    is_dynamic: bool = True
    # Mutable cache; `compare=False, repr=False` keeps the dataclass's
    # auto-generated `__eq__` / `__repr__` independent of the cache state.
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

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
        coarse_n = len(abs_delta_grid)
        scheme_name = getattr(scheme, "name", type(scheme).__name__)
        ad_max = float(np.asarray(abs_delta_grid).max())
        # Bin `ad_max` to the next multiple of `cache_bin_width`. This
        # makes the cached coarse grid `linspace(0, ad_max_bin, coarse_n)`
        # deterministic given the bin — different `ad_max` values within
        # the same bin produce identical grid points and η values.
        # Multiple bins coexist in the cache.
        ad_max_bin = float(np.ceil(max(ad_max, self.cache_bin_width)
                                     / self.cache_bin_width)
                           * self.cache_bin_width)
        key = (w, alpha, statistic.name, scheme_name, coarse_n, ad_max_bin)
        cached = self._cache.get(key)
        if cached is None:
            coarse_grid_full = np.linspace(0.0, ad_max_bin, coarse_n)
            eta_full = self._inner().select_grid(
                coarse_grid_full, scheme, statistic=statistic,
                w=w, alpha=alpha,
            )
            self._cache[key] = (coarse_grid_full, eta_full)
            cached_grid, cached_eta = coarse_grid_full, eta_full
        else:
            cached_grid, cached_eta = cached
        return np.interp(abs_delta_grid, cached_grid, cached_eta)


@dataclass
class LearnedDynamicEtaSelector:
    """Per-θ varying η* via a trained `LearnedArtifact`.

    Replaces the inner `NumericalEtaSelector.select_grid` with a
    direct neural-network lookup. The artifact's MLP is trained to
    minimise the **dynamic-procedure** loss directly (e.g. integrated
    CI width `∫ p_dyn(θ; D, η) dθ`), instead of the static-per-D
    width that `NumericalEtaSelector` minimises pointwise. The
    learned η*(|Δ|; w) curve is smooth and monotone by architectural
    construction (positive-weight ReLU pathway), avoiding the
    lower-clamp kink that inflates dynamic-CI widths at conflict.

    Calibration is preserved by the same argument as
    `DynamicNumericalEtaSelector`: η depends only on θ (and `w`),
    never on D, so the WALDO p-value at fixed η is U[0,1] under H0
    and the dynamic CI hits 1-α coverage exactly.

    α handling
    ----------
    The MLP architecture is α-agnostic; α is a property of the
    *checkpoint*, not of the model:
      - `alpha_mode = "marginalised"`: trained on an α-independent
        loss (integrated-p or CD-variance). Selector ignores its
        `alpha` argument; valid at any α.
      - `alpha_mode = "fixed_<α>"`: trained on the static CI width
        at that specific α. Selector verifies the inference α
        matches and raises if it doesn't.

    Scheme compatibility
    --------------------
    The artifact records the scheme it was trained for. The selector
    raises if a different scheme is passed at inference time;
    retrain to use a different scheme.

    Caching
    -------
    `select_grid` calls `artifact.predict` directly with no coarse
    grid + interpolate step (the MLP itself is the dense lookup),
    so no per-(w, α) cache is needed beyond the artifact load.
    """

    artifact: object  # MonotonicEtaArtifact (avoid hard import here)
    name: str = "learned_dynamic"
    sigma: float = 1.0
    mu0: float = 0.0
    is_dynamic: bool = True
    _loaded: bool = field(default=False, init=False, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.artifact.load()
            self._loaded = True

    def _check_scheme(self, scheme: TiltingScheme) -> None:
        meta = self.artifact.metadata
        trained_scheme = meta.get("scheme")
        if trained_scheme is None:
            return
        if scheme.name != trained_scheme:
            from .._errors import MissingArtifactError
            raise MissingArtifactError(
                f"{self.artifact.name} trained for scheme={trained_scheme!r}, "
                f"but inference scheme={scheme.name!r}; retrain or use a "
                f"different artifact."
            )

    def _check_alpha(self, alpha: float) -> None:
        meta = self.artifact.metadata
        mode = meta.get("alpha_mode", "marginalised")
        if mode == "marginalised":
            return
        if isinstance(mode, str) and mode.startswith("fixed_"):
            trained_alpha = float(mode[len("fixed_"):])
            if abs(alpha - trained_alpha) > 1e-9:
                from .._errors import MissingArtifactError
                raise MissingArtifactError(
                    f"{self.artifact.name} trained at alpha={trained_alpha}, "
                    f"but inference alpha={alpha}; retrain or use the "
                    f"marginalised artifact."
                )

    def select(self, context: TiltingContext, scheme: TiltingScheme,
               *, statistic: TestStatistic) -> float:
        """Single-context η. Convenience for non-dynamic callers."""
        self._ensure_loaded()
        self._check_scheme(scheme)
        self._check_alpha(context.alpha)
        out = self.select_grid(
            np.asarray([context.abs_delta]), scheme,
            statistic=statistic, w=context.w, alpha=context.alpha,
        )
        return float(out[0])

    def select_grid(self, abs_delta_grid, scheme: TiltingScheme,
                    *, statistic: TestStatistic, w: float, alpha: float):
        """Per-θ η* lookup via the trained MLP.

        Builds a `(N, 2)` `[w, |Δ'|]` feature matrix, calls
        `artifact.predict` to get `η'`, then `eta_inverse` to recover η.
        """
        self._ensure_loaded()
        self._check_scheme(scheme)
        self._check_alpha(alpha)

        from ..learned.transforms import delta_transform, eta_inverse

        ad = np.asarray(abs_delta_grid, dtype=np.float64)
        delta_prime = delta_transform(ad)
        x = np.column_stack([np.full_like(delta_prime, float(w)), delta_prime])
        eta_prime = self.artifact.predict(x)
        return eta_inverse(scheme.name, eta_prime, w)
