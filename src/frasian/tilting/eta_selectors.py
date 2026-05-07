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
from typing import ClassVar

import numpy as np

# scipy: minimize_scalar / brentq used by NumericalEtaSelector;
# objective evaluations consume `scheme.tilted_pvalue` output via
# `np.asarray(...)` at the boundary so the JAX vs numpy seam is
# explicit. See `docs/jax_style.md`.
from scipy import optimize

from ..learned.base import LearnedArtifact
from ..models.base import Model, Prior
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel
from ..models.normal_normal import weight as _weight
from ..statistics.base import TestStatistic
from .base import TiltingContext, TiltingDomainError, TiltingScheme


def _D_from_abs_delta(abs_delta: float, w: float, sigma: float, mu0: float) -> float:
    """Invert Delta = (1 - w)(mu0 - D)/sigma with the convention Delta >= 0.

    Normal-Normal-specific helper retained for `experiments/smoothness.py`,
    which still owns its own |Δ|-keyed sweep until commit 3a-3 relocates
    it. New selector code paths consume θ directly via the model/prior
    instances passed at call time; do not introduce new callers of this
    function outside the smoothness experiment.
    """
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
    name: ClassVar[str] = "fixed"
    is_dynamic: bool = False

    def select(
        self,
        scheme_or_context=None,
        scheme: TiltingScheme | None = None,
        *,
        data: np.ndarray | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        statistic: TestStatistic | None = None,
    ) -> float:
        """Phase 3a-1: dual signature; both forms return the constant η.

          - **New**: `select(scheme, *, data, model, prior, alpha, statistic)`.
          - **Legacy**: `select(context, scheme, *, statistic)`.
        """
        return float(self.eta)


@dataclass(frozen=True)
class NumericalEtaSelector:
    """Pick η numerically per cell.

    Two objectives, switchable via the `objective` constructor argument:

      - **`"static_width"`** (default, backwards-compatible): minimize
        the analytic CI width `|C_α(D, η)|` at the alpha read from
        the context. **DOES NOT MAINTAIN NOMINAL COVERAGE** — picking
        η = argmin_η |CI_η(D)| per data sample is post-selection
        inference; coverage drops by ~2 percentage points at α=0.05
        (see `tests/regression/test_post_selection_coverage.py`).
        Width-wise it is the genuine optimum: ≤ WALDO at every D,
        → Wald at large |Δ|.

      - **`"integrated_p"`** (new, the apples-to-apples baseline for
        `LearnedDynamicEtaSelector`): minimize the integrated p-value
        `∫_θ p_dyn(θ; D, η) dθ` over a θ-grid (D ± `search_mult`·σ,
        `n_grid` points). This is *the same loss* the learned MLP
        minimizes — `NumericalEtaSelector(objective="integrated_p")`
        gives the per-cell scipy-optimization equivalent without the
        architectural smoothness/monotonicity constraint.
        Coverage behavior is similar to static_width (post-selection
        per cell), but the optimum η differs: integrated-p prefers
        larger η at moderate |Δ| because the loss surface averages
        over all α via the Cavalieri / layer-cake identity.

    Use this selector only as a baseline; for calibrated CIs use
    `DynamicNumericalEtaSelector` (per-θ static η) or
    `LearnedDynamicEtaSelector` (per-θ MLP η).

    Width is computed by inverting the tilted p-value at a single
    representative D (no Monte-Carlo). For (`power_law`, `waldo`),
    `objective="static_width"` matches the legacy `optimal_eta_numerical`
    solver up to bracket bookkeeping.
    """

    name: ClassVar[str] = "numerical"
    # Phase 3a-1: `sigma` and `mu0` are deprecated — they came in
    # originally only because |Δ| had no model/prior context. The
    # call-time signature now receives `model` and `prior`, so these
    # are unused. Retained as informational fields for legacy demos
    # / smoothness experiment; will be removed in commit 3a-3.
    sigma: float = 1.0
    mu0: float = 0.0
    eta_min_buffer: float = 1e-3
    is_dynamic: bool = False

    # New: objective + integrated_p hyperparameters.
    objective: str = "static_width"  # "static_width" | "integrated_p"
    n_grid: int = 401  # only used if objective="integrated_p"
    search_mult: float = 8.0  # only used if objective="integrated_p"

    def __post_init__(self) -> None:
        if self.objective not in ("static_width", "integrated_p"):
            raise ValueError(
                f"NumericalEtaSelector.objective must be "
                f"'static_width' or 'integrated_p'; got {self.objective!r}."
            )

    def _normal_normal_w(self, model: Model, prior: Prior) -> float:
        """Compute w from a NormalNormalModel + NormalDistribution prior.

        Phase 3a-1: NumericalEtaSelector remains Normal-Normal-only by
        construction; the static-width / integrated-p objectives use
        the closed-form `w = sigma0**2 / (sigma**2 + sigma0**2)` to
        derive the η bracket. Non-NormalNormal callers must use the
        generic numerical selector planned for Phase 3 follow-ups.
        """
        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                f"NumericalEtaSelector currently requires NormalNormalModel; "
                f"got {type(model).__name__!r}. Generic-model selector is a "
                f"future extension (Phase 3 follow-up)."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                f"NumericalEtaSelector currently requires a NormalDistribution "
                f"prior; got {type(prior).__name__!r}."
            )
        return float(_weight(model.sigma, prior.scale))

    def select(
        self,
        scheme_or_context,
        scheme: TiltingScheme | None = None,
        *,
        data: np.ndarray | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        statistic: TestStatistic | None = None,
    ) -> float:
        """Phase 3a-1: dual signature during the transition.

          - **New**: `select(scheme, *, data, model, prior, alpha, statistic)`.
          - **Legacy**: `select(context, scheme, *, statistic)` —
            constructs Normal-Normal model/prior from `(self.sigma,
            self.mu0, ctx.w)` and inverts `ctx.abs_delta` to D.
            Removed in commit 3a-3.
        """
        # Dispatch on first positional: TiltingContext (legacy) vs scheme (new).
        if isinstance(scheme_or_context, TiltingContext):
            ctx = scheme_or_context
            assert scheme is not None, "legacy `select(ctx, scheme, ...)` requires scheme."
            assert statistic is not None, "legacy `select(...)` requires statistic kwarg."
            sigma0 = float(np.sqrt(ctx.w / max(1.0 - ctx.w, 1e-12)) * self.sigma)
            legacy_prior = NormalDistribution(loc=self.mu0, scale=sigma0)
            legacy_model = NormalNormalModel(sigma=self.sigma)
            D = _D_from_abs_delta(ctx.abs_delta, ctx.w, self.sigma, self.mu0)
            return self._select_inner(
                scheme,
                D=D,
                model=legacy_model,
                prior=legacy_prior,
                alpha=ctx.alpha,
                statistic=statistic,
            )

        # New signature.
        scheme_obj = scheme_or_context
        assert data is not None and model is not None and prior is not None
        assert alpha is not None and statistic is not None
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        return self._select_inner(
            scheme_obj,
            D=D,
            model=model,
            prior=prior,
            alpha=alpha,
            statistic=statistic,
        )

    def _select_inner(self, scheme, *, D, model, prior, alpha, statistic):
        w = self._normal_normal_w(model, prior)
        ctx = TiltingContext(w=w, alpha=alpha)
        eta_lo, eta_hi = scheme.admissible_range(ctx)
        eta_hi = min(eta_hi, 1.0 - self.eta_min_buffer)

        if self.objective == "static_width":
            objective_fn = self._make_static_width_objective(
                scheme=scheme,
                statistic=statistic,
                D=D,
                model=model,
                prior=prior,
                alpha=alpha,
            )
        else:  # integrated_p
            objective_fn = self._make_integrated_p_objective(
                scheme=scheme,
                statistic=statistic,
                D=D,
                model=model,
                prior=prior,
            )

        result = optimize.minimize_scalar(
            objective_fn,
            bounds=(eta_lo, eta_hi),
            method="bounded",
            options={"xatol": 1e-3},
        )
        return float(result.x)

    def _make_static_width_objective(self, *, scheme, statistic, D, model, prior, alpha):
        def width(eta: float) -> float:
            try:
                if hasattr(scheme, "tilted_confidence_interval"):
                    lo, hi = scheme.tilted_confidence_interval(
                        alpha,
                        D,
                        model,
                        prior,
                        eta,
                        statistic.name,
                    )
                else:
                    raise NotImplementedError(
                        f"{type(scheme).__name__} does not implement "
                        f"`tilted_confidence_interval`."
                    )
            except (NotImplementedError, TiltingDomainError, ValueError, RuntimeError):
                return float("inf")
            w_ci = float(hi - lo)
            return w_ci if (w_ci > 0 and np.isfinite(w_ci)) else float("inf")

        return width

    def _make_integrated_p_objective(self, *, scheme, statistic, D, model, prior):
        """Build `f(eta) = ∫_θ p_dyn(θ; D, η) dθ` for scipy.optimize."""
        if not hasattr(scheme, "tilted_pvalue"):
            raise NotImplementedError(
                f"{type(scheme).__name__} does not implement "
                f"`tilted_pvalue`; integrated_p mode requires it."
            )
        # Use the model's sigma for the search window (Normal-Normal scale).
        sigma = float(model.sigma) if hasattr(model, "sigma") else 1.0
        half = self.search_mult * sigma
        theta_grid = np.linspace(D - half, D + half, self.n_grid)

        def integrated_p(eta: float) -> float:
            try:
                p = scheme.tilted_pvalue(theta_grid, D, model, prior, eta, statistic.name)
            except (NotImplementedError, TiltingDomainError, ValueError, RuntimeError):
                return float("inf")
            p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
            val = float(np.trapezoid(p, theta_grid))
            return val if np.isfinite(val) else float("inf")

        return integrated_p

    def select_grid(
        self,
        grid,
        scheme: TiltingScheme,
        *,
        statistic: TestStatistic,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        w: float | None = None,
    ):
        """Vectorised: η* at every θ in `grid` (Phase 3a-1 θ-keyed) or
        every |Δ| in `grid` (legacy, pending 3a-2 cleanup).

        Phase 3a-1: dual signature during the transition.

          - **New (preferred)**: `select_grid(theta_grid, scheme, *,
            statistic, model=, prior=, alpha=)`. Each grid point is
            treated as a "data sufficient statistic" — we call
            `self.select(scheme, data=[theta], ...)` for each θ.
            Preserves the "static-η optimum at each θ" semantics.

          - **Legacy**: `select_grid(abs_delta_grid, scheme, *,
            statistic, w=, alpha=)`. Inverts |Δ| → D via the legacy
            `_D_from_abs_delta` helper using the selector's `sigma`/`mu0`
            attributes. Will be removed in commit 3a-2 once
            `_dynamic.py`'s interior is θ-refactored.
        """
        if alpha is None:
            raise ValueError("select_grid requires `alpha` keyword.")
        # Dispatch: new-style (model/prior provided) vs legacy (w provided).
        if model is not None and prior is not None:
            theta_grid = np.asarray(grid, dtype=np.float64)
            out = np.empty(len(theta_grid), dtype=np.float64)
            for i, theta in enumerate(theta_grid):
                out[i] = self.select(
                    scheme,
                    data=np.asarray([float(theta)]),
                    model=model,
                    prior=prior,
                    alpha=alpha,
                    statistic=statistic,
                )
            return out

        # Legacy |Δ|-grid path: construct Normal-Normal model + prior
        # from `(self.sigma, self.mu0, w)` and dispatch each |Δ| via
        # `_D_from_abs_delta`. Dropped in commit 3a-2.
        if w is None:
            raise ValueError(
                "select_grid (legacy path) requires `w` keyword. "
                "Pass `model` and `prior` for the new θ-keyed path."
            )
        sigma0 = float(np.sqrt(w / max(1.0 - w, 1e-12)) * self.sigma)
        legacy_prior = NormalDistribution(loc=self.mu0, scale=sigma0)
        legacy_model = NormalNormalModel(sigma=self.sigma)
        out = np.empty(len(grid), dtype=np.float64)
        for i, ad in enumerate(grid):
            D = _D_from_abs_delta(float(ad), w, self.sigma, self.mu0)
            out[i] = self.select(
                scheme,
                data=np.asarray([D]),
                model=legacy_model,
                prior=legacy_prior,
                alpha=alpha,
                statistic=statistic,
            )
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

    name: ClassVar[str] = "dynamic_numerical"
    # `sigma` and `mu0` are retained as informational fields for the
    # legacy demos that still inspect them; new code paths never read
    # them — model/prior come in at call time. They are no longer
    # required to match the inference-time model/prior.
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
        return NumericalEtaSelector(eta_min_buffer=self.eta_min_buffer)

    def select(
        self,
        scheme_or_context,
        scheme: TiltingScheme | None = None,
        *,
        data: np.ndarray | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        statistic: TestStatistic | None = None,
    ) -> float:
        """Convenience: a single-context η* (delegates to the static inner).

        Phase 3a-1: dual signature.
          - **New**: `select(scheme, *, data, model, prior, alpha, statistic)`.
          - **Legacy**: `select(context, scheme, *, statistic)`.
        """
        if isinstance(scheme_or_context, TiltingContext):
            return self._inner().select(scheme_or_context, scheme, statistic=statistic)
        scheme_obj = scheme_or_context
        assert data is not None and model is not None and prior is not None
        assert alpha is not None and statistic is not None
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        out = self.select_grid(
            np.asarray([D]),
            scheme_obj,
            statistic=statistic,
            model=model,
            prior=prior,
            alpha=alpha,
        )
        return float(out[0])

    def select_grid(
        self,
        grid,
        scheme: TiltingScheme,
        *,
        statistic: TestStatistic,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        w: float | None = None,
    ):
        """Per-θ η* lookup with caching.

        Phase 3a-1: dual signature during the transition.
          - **New (preferred)**: `select_grid(theta_grid, scheme, *,
            statistic, model=, prior=, alpha=)`. Cache key includes the
            θ-grid hash and model/prior fingerprint.
          - **Legacy**: `select_grid(abs_delta_grid, scheme, *,
            statistic, w=, alpha=)`. Cache key includes the |Δ| bin.
            Will be removed in commit 3a-2 once `_dynamic.py`'s
            interior is θ-refactored.
        """
        if alpha is None:
            raise ValueError("select_grid requires `alpha` keyword.")
        scheme_name = getattr(scheme, "name", type(scheme).__name__)
        coarse_n = len(grid)

        # New θ-keyed path.
        if model is not None and prior is not None:
            theta_arr = np.asarray(grid, dtype=np.float64)
            # Cache key absorbs tiny numerical drift via 1e-6 binning of
            # the (theta_min, theta_max) endpoints — repeat calls with
            # identical or near-identical bounds hit the cache. Includes
            # the model + prior fingerprints so different experiments
            # don't collide.
            t_min_bin = float(np.round(theta_arr.min(), 6))
            t_max_bin = float(np.round(theta_arr.max(), 6))
            model_fp = getattr(model, "fingerprint", lambda: None)()
            prior_fp = getattr(prior, "fingerprint", lambda: None)()
            key = (
                "theta",
                alpha,
                statistic.name,
                scheme_name,
                coarse_n,
                t_min_bin,
                t_max_bin,
                tuple(model_fp) if model_fp is not None else None,
                tuple(prior_fp) if prior_fp is not None else None,
            )
            cached = self._cache.get(key)
            if cached is None:
                eta_full = self._inner().select_grid(
                    theta_arr,
                    scheme,
                    model=model,
                    prior=prior,
                    alpha=alpha,
                    statistic=statistic,
                )
                # Store (grid, eta) for symmetry with the legacy path.
                self._cache[key] = (theta_arr, eta_full)
                cached_grid, cached_eta = theta_arr, eta_full
            else:
                cached_grid, cached_eta = cached
            return np.interp(theta_arr, cached_grid, cached_eta)

        # Legacy |Δ|-keyed path.
        if w is None:
            raise ValueError(
                "select_grid (legacy path) requires `w` keyword. "
                "Pass `model` and `prior` for the new θ-keyed path."
            )
        abs_delta_grid = np.asarray(grid, dtype=np.float64)
        ad_max = float(abs_delta_grid.max())
        ad_max_bin = float(
            np.ceil(max(ad_max, self.cache_bin_width) / self.cache_bin_width) * self.cache_bin_width
        )
        key = (w, alpha, statistic.name, scheme_name, coarse_n, ad_max_bin)
        cached = self._cache.get(key)
        if cached is None:
            coarse_grid_full = np.linspace(0.0, ad_max_bin, coarse_n)
            eta_full = self._inner().select_grid(
                coarse_grid_full,
                scheme,
                statistic=statistic,
                w=w,
                alpha=alpha,
            )
            self._cache[key] = (coarse_grid_full, eta_full)
            cached_grid, cached_eta = coarse_grid_full, eta_full
        else:
            cached_grid, cached_eta = cached
        return np.interp(abs_delta_grid, cached_grid, cached_eta)


# Threshold for the runtime safety clamp in `LearnedDynamicEtaSelector`.
# If the predicted η is out of admissible range for more than this
# fraction of the batch, raise rather than silently clamp — a checkpoint
# that drifts that far is either undertrained or for the wrong
# experiment, and silently clamping would mask a calibration failure.
_CLAMP_FAIL_THRESHOLD = 0.20


@dataclass
class LearnedDynamicEtaSelector:
    """Per-θ varying η* via a trained `LearnedArtifact`.

    Replaces the inner `NumericalEtaSelector.select_grid` with a
    direct neural-network lookup. The artifact's MLP is trained to
    minimise the **dynamic-procedure** loss directly (e.g. integrated
    CI width `∫ p_dyn(θ; D, η) dθ`), instead of the static-per-D
    width that `NumericalEtaSelector` minimises pointwise. Phase E
    `EtaNet` is a smooth GELU-MLP on θ — no monotonicity prior, no
    bounded sigmoid; admissibility is enforced by the boundary
    penalty during training, not by architecture. The result is a
    smooth η(θ) curve that avoids the lower-clamp kink that inflates
    dynamic-CI widths at conflict.

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

    artifact: LearnedArtifact  # any Phase E v2 dual-head artifact (concrete EtaArtifact, NullArtifact stub, ...)
    name: ClassVar[str] = "learned_dynamic"
    sigma: float = 1.0
    mu0: float = 0.0
    is_dynamic: bool = True
    _loaded: bool = field(default=False, init=False, repr=False, compare=False)
    # Diagnostic counters tracking how often the runtime safety clamp
    # has fired (skeptic E pre-PR review #2). Cumulative across calls
    # in the lifetime of this selector instance.
    _clamped_calls: int = field(default=0, init=False, repr=False, compare=False)
    _last_clamped_fraction: float = field(default=0.0, init=False, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.artifact.load()
            self._loaded = True
            # Phase E checkpoints: legacy torch format v2 + post-port
            # Equinox format v3 are both accepted (the on-disk schema
            # changed; the in-memory metadata fields are compatible).
            from .._errors import MissingArtifactError

            meta = self.artifact.metadata
            v = meta.get("checkpoint_format_version", None)
            if v not in (2, 3) or "experiment_config" not in meta:
                raise MissingArtifactError(
                    f"{self.artifact.name}: expected Phase E (v2 or v3) "
                    f"checkpoint with `experiment_config`; got "
                    f"format_version={v!r}. Re-train via "
                    f"`python -m scripts.train_learned_eta --config "
                    f"<experiment.yaml>`."
                )

    def _check_scheme(self, scheme: TiltingScheme) -> None:
        from .._errors import MissingArtifactError

        if not self._loaded:
            raise MissingArtifactError(
                f"{self.artifact.name} not loaded; " f"call .load() before _check_scheme()."
            )
        meta = self.artifact.metadata
        trained_scheme = meta["experiment_config"]["scheme_name"]
        if scheme.name != trained_scheme:
            raise MissingArtifactError(
                f"{self.artifact.name} trained for scheme={trained_scheme!r}, "
                f"but inference scheme={scheme.name!r}; retrain or use a "
                f"different artifact."
            )

    def _check_alpha(self, alpha: float) -> None:
        from .._errors import MissingArtifactError

        if not self._loaded:
            raise MissingArtifactError(
                f"{self.artifact.name} not loaded; " f"call .load() before _check_alpha()."
            )
        meta = self.artifact.metadata
        # Phase E records `alpha` (None for marginalised losses, fixed
        # value for static_width).
        stored = meta.get("alpha")
        if stored is None:
            return
        trained_alpha = float(stored)
        if abs(alpha - trained_alpha) > 1e-9:
            raise MissingArtifactError(
                f"{self.artifact.name} trained at alpha={trained_alpha}, "
                f"but inference alpha={alpha}; retrain or use a "
                f"marginalised checkpoint."
            )

    def _check_experiment(
        self,
        w: float,
        model_fingerprint: tuple | None = None,
        prior_fingerprint: tuple | None = None,
    ) -> None:
        """Phase E: verify inference matches the trained experiment.

        Strict tuple-equal compare on (model.fingerprint(),
        prior.fingerprint()) when both are plumbed through from the
        caller. Falls back to a w-only derived check if not — this
        catches gross mismatches but cannot distinguish two
        ``(σ, σ₀)`` pairs giving the same ``w``.
        """
        from .._errors import MissingArtifactError

        meta = self.artifact.metadata["experiment_config"]
        trained_model_fp = tuple(meta["model_fingerprint"])
        trained_prior_fp = tuple(meta["prior_fingerprint"])
        # Normal-Normal-only inversion path today.
        if trained_model_fp[0] != "normal_normal":
            raise MissingArtifactError(
                f"{self.artifact.name} trained on model "
                f"{trained_model_fp[0]!r}; only Normal-Normal is "
                f"supported by the Phase E inversion path."
            )
        if trained_prior_fp[0] != "normal":
            raise MissingArtifactError(
                f"{self.artifact.name} trained with prior "
                f"{trained_prior_fp[0]!r}; only NormalDistribution is "
                f"supported by the Phase E inversion path."
            )

        # Strict per-fingerprint compare when available.
        if model_fingerprint is not None and tuple(model_fingerprint) != trained_model_fp:
            raise MissingArtifactError(
                f"{self.artifact.name} trained on model "
                f"{trained_model_fp!r}, but inference model is "
                f"{tuple(model_fingerprint)!r}. Train a new "
                f"checkpoint for this experiment."
            )
        if prior_fingerprint is not None and tuple(prior_fingerprint) != trained_prior_fp:
            raise MissingArtifactError(
                f"{self.artifact.name} trained with prior "
                f"{trained_prior_fp!r}, but inference prior is "
                f"{tuple(prior_fingerprint)!r}. Train a new "
                f"checkpoint for this experiment."
            )

        # Derived w check (catches rough mismatches even when
        # fingerprints aren't plumbed through).
        sigma_trained = float(trained_model_fp[1])
        sigma0_trained = float(trained_prior_fp[2])
        w_trained = sigma0_trained**2 / (sigma_trained**2 + sigma0_trained**2)
        if abs(w - w_trained) > 1e-6:
            raise MissingArtifactError(
                f"{self.artifact.name} trained at w={w_trained:.6f}, "
                f"but inference w={w:.6f}; this checkpoint is "
                f"per-experiment and cannot be reused across w values. "
                f"Train a new checkpoint for this prior/likelihood pair."
            )
        # Mirror the training-time degenerate-w guard so that hand-
        # edited or out-of-band checkpoints can't slip through with
        # w → 0 / w → 1 (where the torch port's denom-clamp distorts
        # silently).
        _W_EPS = 1e-3
        if not (_W_EPS < w < 1.0 - _W_EPS):
            raise MissingArtifactError(
                f"{self.artifact.name}: inference w={w:.6f} is outside "
                f"({_W_EPS}, {1.0 - _W_EPS}); the torch port's "
                f"denom-clamp distorts silently in this regime."
            )

    def select(
        self,
        scheme_or_context,
        scheme: TiltingScheme | None = None,
        *,
        data: np.ndarray | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        statistic: TestStatistic | None = None,
        model_fingerprint: tuple | None = None,
        prior_fingerprint: tuple | None = None,
    ) -> float:
        """Single-context η. Convenience for non-dynamic callers.

        Phase 3a-1: dual signature during the transition.
          - **New**: `select(scheme, *, data, model, prior, alpha, statistic)`.
          - **Legacy**: `select(context, scheme, *, statistic,
            model_fingerprint=, prior_fingerprint=)`.
        """
        # Dispatch on first positional: TiltingContext (legacy) vs scheme (new).
        if isinstance(scheme_or_context, TiltingContext):
            ctx = scheme_or_context
            assert scheme is not None, "legacy `select(ctx, scheme, ...)` requires scheme."
            if model_fingerprint is None or prior_fingerprint is None:
                raise ValueError(
                    f"{type(self).__name__}.select requires explicit "
                    f"`model_fingerprint` and `prior_fingerprint` kwargs "
                    f"to enforce strict cross-experiment refusal. Pass "
                    f"`model.fingerprint()` and `prior.fingerprint()` from "
                    f"the inference call site."
                )
            if statistic is None:
                raise ValueError("legacy select(...) requires `statistic` kwarg.")
            self._ensure_loaded()
            self._check_scheme(scheme)
            self._check_alpha(ctx.alpha)
            out = self.select_grid(
                np.asarray([0.0]),  # |Δ| = 0 == single representative point
                scheme,
                statistic=statistic,
                w=ctx.w,
                alpha=ctx.alpha,
                model_fingerprint=model_fingerprint,
                prior_fingerprint=prior_fingerprint,
            )
            return float(out[0])

        # New signature.
        scheme_obj = scheme_or_context
        assert data is not None and model is not None and prior is not None
        assert alpha is not None and statistic is not None
        self._ensure_loaded()
        self._check_scheme(scheme_obj)
        self._check_alpha(alpha)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        out = self.select_grid(
            np.asarray([D]),
            scheme_obj,
            statistic=statistic,
            model=model,
            prior=prior,
            alpha=alpha,
        )
        return float(out[0])

    def select_grid(
        self,
        grid,
        scheme: TiltingScheme,
        *,
        statistic: TestStatistic,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        w: float | None = None,
        model_fingerprint: tuple | None = None,
        prior_fingerprint: tuple | None = None,
    ):
        """Per-θ η* lookup via the trained Phase E EtaNet.

        Phase 3a-1: dual signature during the transition.
          - **New (preferred)**: `select_grid(theta_grid, scheme, *,
            statistic, model=, prior=, alpha=)`. EtaNet receives θ
            directly (its natural input); no |Δ| inversion / two-branch
            averaging.
          - **Legacy**: `select_grid(abs_delta_grid, scheme, *,
            statistic, w=, alpha=, model_fingerprint=, prior_fingerprint=)`.
            Inverts |Δ| → θ via the trained config's μ₀, σ and averages
            the two branches. Will be removed in commit 3a-2.
        """
        if alpha is None:
            raise ValueError("select_grid requires `alpha` keyword.")
        self._ensure_loaded()
        self._check_scheme(scheme)
        self._check_alpha(alpha)

        # New θ-keyed path: EtaNet input is θ directly.
        if model is not None and prior is not None:
            theta_arr = np.asarray(grid, dtype=np.float64)
            model_fp = getattr(model, "fingerprint", lambda: None)()
            prior_fp = getattr(prior, "fingerprint", lambda: None)()
            if hasattr(model, "sigma") and hasattr(prior, "scale"):
                w_eff = float(prior.scale) ** 2 / (
                    float(model.sigma) ** 2 + float(prior.scale) ** 2
                )
            else:  # pragma: no cover — non-Normal models will fail below
                w_eff = 0.5
            self._check_experiment(
                w_eff,
                model_fingerprint=tuple(model_fp) if model_fp is not None else None,
                prior_fingerprint=tuple(prior_fp) if prior_fp is not None else None,
            )
            eta = self.artifact.predict_eta(theta_arr)
            return self._maybe_clamp_eta(eta, scheme=scheme, w=w_eff, alpha=alpha)

        # Legacy |Δ|-keyed path.
        if w is None:
            raise ValueError(
                "select_grid (legacy path) requires `w` keyword. "
                "Pass `model` and `prior` for the new θ-keyed path."
            )
        ad = np.asarray(grid, dtype=np.float64)
        self._check_experiment(
            w,
            model_fingerprint=model_fingerprint,
            prior_fingerprint=prior_fingerprint,
        )
        meta = self.artifact.metadata["experiment_config"]
        mu0 = float(meta["prior_fingerprint"][1])
        sigma = float(meta["model_fingerprint"][1])
        offset = sigma * ad / max(1.0 - w, 1e-12)
        theta_lo = mu0 - offset
        theta_hi = mu0 + offset
        eta_lo = self.artifact.predict_eta(theta_lo)
        eta_hi = self.artifact.predict_eta(theta_hi)
        eta = 0.5 * (eta_lo + eta_hi)
        return self._maybe_clamp_eta(eta, scheme=scheme, w=w, alpha=alpha)

    def _maybe_clamp_eta(self, eta, *, scheme, w, alpha):
        """Runtime safety net for predicted η outside admissible range."""
        ctx = TiltingContext(w=w, alpha=alpha)
        # Catch only the documented protocol exception for invalid
        # context; any other exception (TypeError from a buggy ctx,
        # AttributeError from a stub scheme, etc.) should propagate
        # so we don't silently disable the safety net by falling
        # through to (-inf, inf).
        try:
            lo, hi = scheme.admissible_range(ctx)
        except (ValueError, TiltingDomainError):
            lo, hi = -np.inf, np.inf
        margin = 1e-6 * max(1.0, abs(hi - lo) if np.isfinite(hi - lo) else 1.0)
        lo_safe = lo + margin if np.isfinite(lo) else -np.inf
        hi_safe = hi - margin if np.isfinite(hi) else np.inf
        out_of_range = (eta < lo_safe) | (eta > hi_safe)
        if out_of_range.any():
            import warnings

            frac = float(out_of_range.mean())
            self._clamped_calls += 1
            self._last_clamped_fraction = frac
            warnings.warn(
                f"{self.artifact.name}: {100*frac:.1f}% of predicted "
                f"η values fell outside the admissible range "
                f"({lo_safe:.4g}, {hi_safe:.4g}); clamping. This "
                f"signals the checkpoint is undertrained at the "
                f"conflict band — consider a longer training run.",
                RuntimeWarning,
                stacklevel=3,
            )
            # Stricter: if more than `_CLAMP_FAIL_THRESHOLD` of the
            # batch is out of range, refuse rather than clamp. This
            # closes the "silent escape hatch" — a checkpoint that
            # routinely drifts past the boundary should not pass
            # silently as if calibration held.
            if frac > _CLAMP_FAIL_THRESHOLD:
                from .._errors import MissingArtifactError

                raise MissingArtifactError(
                    f"{self.artifact.name}: {100*frac:.1f}% of predicted "
                    f"η values fell outside the admissible range — over "
                    f"the {100*_CLAMP_FAIL_THRESHOLD:.0f}% threshold. "
                    f"This checkpoint is undertrained or for the wrong "
                    f"experiment; refuse rather than train on a clamp."
                )
            eta = np.clip(eta, lo_safe, hi_safe)
        return eta
