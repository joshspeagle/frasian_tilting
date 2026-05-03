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

    name: str = "numerical"
    sigma: float = 1.0
    mu0: float = 0.0
    eta_min_buffer: float = 1e-3
    is_dynamic: bool = False

    # New: objective + integrated_p hyperparameters.
    objective: str = "static_width"  # "static_width" | "integrated_p"
    n_grid: int = 401                # only used if objective="integrated_p"
    search_mult: float = 8.0         # only used if objective="integrated_p"

    def __post_init__(self) -> None:
        if self.objective not in ("static_width", "integrated_p"):
            raise ValueError(
                f"NumericalEtaSelector.objective must be "
                f"'static_width' or 'integrated_p'; got {self.objective!r}."
            )

    def select(self, context: TiltingContext, scheme: TiltingScheme,
               *, statistic: TestStatistic) -> float:
        eta_lo, eta_hi = scheme.admissible_range(context)
        eta_hi = min(eta_hi, 1.0 - self.eta_min_buffer)

        sigma0 = float(np.sqrt(context.w / max(1.0 - context.w, 1e-12))
                        * self.sigma)
        prior = NormalDistribution(loc=self.mu0, scale=sigma0)
        model = NormalNormalModel(sigma=self.sigma)
        D = _D_from_abs_delta(context.abs_delta, context.w, self.sigma,
                                self.mu0)

        if self.objective == "static_width":
            objective_fn = self._make_static_width_objective(
                scheme=scheme, statistic=statistic,
                D=D, model=model, prior=prior, alpha=context.alpha,
            )
        else:  # integrated_p
            objective_fn = self._make_integrated_p_objective(
                scheme=scheme, statistic=statistic,
                D=D, model=model, prior=prior,
            )

        result = optimize.minimize_scalar(
            objective_fn, bounds=(eta_lo, eta_hi), method="bounded",
            options={"xatol": 1e-3},
        )
        return float(result.x)

    def _make_static_width_objective(self, *, scheme, statistic, D, model,
                                        prior, alpha):
        def width(eta: float) -> float:
            try:
                if hasattr(scheme, "tilted_confidence_interval"):
                    lo, hi = scheme.tilted_confidence_interval(
                        alpha, D, model, prior, eta, statistic.name,
                    )
                else:
                    raise NotImplementedError(
                        f"{type(scheme).__name__} does not implement "
                        f"`tilted_confidence_interval`."
                    )
            except (NotImplementedError, TiltingDomainError,
                     ValueError, RuntimeError):
                return np.inf
            w_ci = float(hi - lo)
            return w_ci if (w_ci > 0 and np.isfinite(w_ci)) else np.inf
        return width

    def _make_integrated_p_objective(self, *, scheme, statistic, D, model,
                                       prior):
        """Build `f(eta) = ∫_θ p_dyn(θ; D, η) dθ` for scipy.optimize."""
        if not hasattr(scheme, "tilted_pvalue"):
            raise NotImplementedError(
                f"{type(scheme).__name__} does not implement "
                f"`tilted_pvalue`; integrated_p mode requires it."
            )
        half = self.search_mult * self.sigma
        theta_grid = np.linspace(D - half, D + half, self.n_grid)

        def integrated_p(eta: float) -> float:
            try:
                p = scheme.tilted_pvalue(theta_grid, D, model, prior,
                                          eta, statistic.name)
            except (NotImplementedError, TiltingDomainError,
                     ValueError, RuntimeError):
                return np.inf
            p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
            val = float(np.trapezoid(p, theta_grid))
            return val if np.isfinite(val) else np.inf
        return integrated_p

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

    artifact: object  # MonotonicEtaArtifact (legacy v1) or EtaArtifact (Phase E v2)
    name: str = "learned_dynamic"
    sigma: float = 1.0
    mu0: float = 0.0
    is_dynamic: bool = True
    _loaded: bool = field(default=False, init=False, repr=False, compare=False)
    _is_phase_e: bool = field(default=True, init=False, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.artifact.load()
            self._loaded = True
            # E.4 retired the legacy v1 checkpoint format; only Phase E
            # (format v2) is supported. Verify here so a stale v1
            # checkpoint produces a clear error rather than a confusing
            # downstream failure.
            from .._errors import MissingArtifactError
            meta = self.artifact.metadata
            v = meta.get("checkpoint_format_version", None)
            if v != 2 or "experiment_config" not in meta:
                raise MissingArtifactError(
                    f"{self.artifact.name}: expected Phase E (v2) "
                    f"checkpoint with `experiment_config`; got "
                    f"format_version={v!r}. Legacy v1 was removed in "
                    f"E.4 — re-train via "
                    f"`python -m scripts.train_learned_eta --config "
                    f"<experiment.yaml>`."
                )

    def _check_scheme(self, scheme: TiltingScheme) -> None:
        from .._errors import MissingArtifactError
        if not self._loaded:
            raise MissingArtifactError(
                f"{self.artifact.name} not loaded; "
                f"call .load() before _check_scheme()."
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
                f"{self.artifact.name} not loaded; "
                f"call .load() before _check_alpha()."
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
        if model_fingerprint is not None:
            if tuple(model_fingerprint) != trained_model_fp:
                raise MissingArtifactError(
                    f"{self.artifact.name} trained on model "
                    f"{trained_model_fp!r}, but inference model is "
                    f"{tuple(model_fingerprint)!r}. Train a new "
                    f"checkpoint for this experiment."
                )
        if prior_fingerprint is not None:
            if tuple(prior_fingerprint) != trained_prior_fp:
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
        w_trained = sigma0_trained ** 2 / (
            sigma_trained ** 2 + sigma0_trained ** 2
        )
        if abs(w - w_trained) > 1e-6:
            raise MissingArtifactError(
                f"{self.artifact.name} trained at w={w_trained:.6f}, "
                f"but inference w={w:.6f}; this checkpoint is "
                f"per-experiment and cannot be reused across w values. "
                f"Train a new checkpoint for this prior/likelihood pair."
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
                    *, statistic: TestStatistic, w: float, alpha: float,
                    model_fingerprint: tuple | None = None,
                    prior_fingerprint: tuple | None = None):
        """Per-θ η* lookup via the trained Phase E EtaNet.

        Converts ``abs_delta_grid`` back to θ using the trained
        config's μ₀ and σ (averaging the two θ branches for symmetric
        Normal-Normal training), then calls ``EtaArtifact.predict_eta``.

        ``model_fingerprint`` and ``prior_fingerprint`` are plumbed
        through ``dynamic_ci_scan`` for strict cross-experiment
        validation.
        """
        self._ensure_loaded()
        self._check_scheme(scheme)
        self._check_alpha(alpha)

        ad = np.asarray(abs_delta_grid, dtype=np.float64)

        if self._is_phase_e:
            # Phase E: invert |Δ| → θ using trained (μ₀, σ).
            self._check_experiment(
                w,
                model_fingerprint=model_fingerprint,
                prior_fingerprint=prior_fingerprint,
            )
            meta = self.artifact.metadata["experiment_config"]
            mu0 = float(meta["prior_fingerprint"][1])
            sigma = float(meta["model_fingerprint"][1])
            # |Δ| = (1-w)|μ₀ - θ|/σ has two θ-branches:
            #   θ_lo = μ₀ - σ·|Δ|/(1-w)  (θ < μ₀)
            #   θ_hi = μ₀ + σ·|Δ|/(1-w)  (θ > μ₀)
            # The downstream `dynamic_ci_scan` indexes η by |Δ|, so we
            # need η as a function of |Δ| — but EtaNet is θ-indexed.
            # For Normal-Normal training (symmetric θ-distribution
            # about μ₀), the optimal η(θ) is approximately symmetric;
            # we average the two branches to get a symmetric η(|Δ|),
            # which is what the contract demands. Bias is bounded by
            # the deviation of the trained η(θ) from symmetry.
            offset = sigma * ad / max(1.0 - w, 1e-12)
            theta_lo = mu0 - offset
            theta_hi = mu0 + offset
            eta_lo = self.artifact.predict_eta(theta_lo)
            eta_hi = self.artifact.predict_eta(theta_hi)
            eta = 0.5 * (eta_lo + eta_hi)

            # Phase E removed the architectural sigmoid clamp on η,
            # relying on the boundary penalty during training to keep
            # predictions inside the admissible range. A poorly-
            # trained checkpoint can still drift past the boundary at
            # extreme conflict; rather than crash mid-CI, clamp here
            # with a warning. The fingerprint check on _check_experiment
            # already refuses cross-experiment use, so this is a safety
            # net for "this checkpoint hasn't trained enough", not for
            # "this checkpoint is for the wrong experiment".
            ctx = TiltingContext(w=w, abs_delta=0.0, alpha=alpha)
            try:
                lo, hi = scheme.admissible_range(ctx)
            except Exception:
                lo, hi = -np.inf, np.inf
            margin = 1e-6 * max(1.0, abs(hi - lo) if np.isfinite(hi - lo) else 1.0)
            lo_safe = lo + margin if np.isfinite(lo) else -np.inf
            hi_safe = hi - margin if np.isfinite(hi) else np.inf
            out_of_range = (eta < lo_safe) | (eta > hi_safe)
            if out_of_range.any():
                import warnings
                frac = float(out_of_range.mean())
                warnings.warn(
                    f"{self.artifact.name}: {100*frac:.1f}% of predicted "
                    f"η values fell outside the admissible range "
                    f"({lo_safe:.4g}, {hi_safe:.4g}); clamping. This "
                    f"signals the checkpoint is undertrained at the "
                    f"conflict band — consider a longer training run.",
                    RuntimeWarning,
                )
                eta = np.clip(eta, lo_safe, hi_safe)
            return eta

        from .._errors import MissingArtifactError
        raise MissingArtifactError(
            f"{self.artifact.name}: only Phase E (v2) checkpoints are "
            f"supported; legacy v1 was removed in E.4."
        )
