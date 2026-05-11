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
from ..models._dispatch import is_normal_normal
from ..models.base import Model, Prior
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel  # noqa: F401  (legacy field-access typing)
from ..models.normal_normal import weight as _weight
from ..statistics.base import TestStatistic
from .base import TiltingDomainError, TiltingScheme


def _D_from_abs_delta(abs_delta: float, w: float, sigma: float, mu0: float) -> float:
    """Invert Delta = (1 - w)(mu0 - D)/sigma with the convention Delta >= 0.

    Normal-Normal-specific helper retained solely for
    `experiments/smoothness.py`, which sweeps `|Δ|` and inverts
    back to D for plotting. New selector code paths consume θ
    directly via the model/prior instances passed at call time;
    do not introduce new callers of this function outside the
    smoothness experiment.
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
    # Post-selection inference flag: True iff `select(data=...)` reads D
    # to pick eta. FixedEtaSelector returns a constant; not post-selection.
    is_post_selection: ClassVar[bool] = False

    def select(
        self,
        scheme: TiltingScheme,
        *,
        data: np.ndarray | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        statistic: TestStatistic | None = None,
    ) -> float:
        """Return the constant η. All kwargs are accepted for protocol
        parity but ignored — `FixedEtaSelector` is η-only.
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
    eta_min_buffer: float = 1e-3
    is_dynamic: bool = False
    # NumericalEtaSelector reads `D = data.mean()` inside `select(...)`
    # and minimises width / integrated-p AT that D — η = η(D), not η(θ).
    # That's post-selection inference: the resulting CI undercovers by
    # ~2 points at α=0.05 (see test_post_selection_coverage.py).
    # Calibrated callers should use DynamicNumericalEtaSelector instead.
    is_post_selection: ClassVar[bool] = True

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

        NumericalEtaSelector remains Normal-Normal-only by construction;
        the static-width / integrated-p objectives use the closed-form
        `w = sigma0**2 / (sigma**2 + sigma0**2)` to derive the η bracket.
        Non-NormalNormal callers must use the generic numerical selector
        planned for Phase 3 follow-ups.
        """
        if not is_normal_normal(model):
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

    def _eta_bounds(self, model: Model, prior: Prior) -> tuple[float, float]:
        """η search bracket for `scipy.optimize.minimize_scalar`.

        The bracket is the closed-form Normal-Normal `power_law`
        admissible range `(-w/(1-w) + buffer, 1/(1-w) - buffer)`
        (also valid for `ot` since `(0, 1) ⊂ (-w/(1-w), 1/(1-w))`
        for any `w ∈ (0, 1)`); `scheme.tilted_pvalue` will raise
        `TiltingDomainError` for any η in the bracket that the
        scheme actually rejects, and `_make_*_objective` returns
        `+inf` on that exception, so over-broad brackets are
        self-correcting at the optimizer.
        """
        w = self._normal_normal_w(model, prior)
        eta_lo = -w / (1.0 - w) + self.eta_min_buffer
        eta_hi = 1.0 / (1.0 - w) - self.eta_min_buffer
        return (eta_lo, eta_hi)

    def select(
        self,
        scheme: TiltingScheme,
        *,
        data: np.ndarray | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        statistic: TestStatistic | None = None,
    ) -> float:
        """Pick η numerically per cell.

        See class docstring for the two `objective` modes
        (`static_width` vs `integrated_p`).
        """
        assert data is not None and model is not None and prior is not None
        assert alpha is not None and statistic is not None
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        return self._select_inner(
            scheme,
            D=D,
            model=model,
            prior=prior,
            alpha=alpha,
            statistic=statistic,
        )

    def _select_inner(self, scheme, *, D, model, prior, alpha, statistic):
        # The bracket is the closed-form Normal-Normal `power_law`
        # admissible range (also valid for `ot`'s W2 displacement line);
        # the optimizer's objective returns +inf on TiltingDomainError so
        # over-broad brackets are self-correcting at the optimum.
        eta_lo, eta_hi = self._eta_bounds(model, prior)

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
        from .._errors import BracketingFailed

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
            except (NotImplementedError, TiltingDomainError, ValueError,
                    RuntimeError, BracketingFailed):
                # `BracketingFailed` is the brentq sentinel raised by
                # `tilted_confidence_interval` when the CI extends past
                # the search box (boundary-hit case). Treating it the
                # same as the other "this η is unworkable" exceptions
                # — return +inf so `minimize_scalar` skips it. Without
                # this catch the entire (D, prior) cell errored out
                # whenever a single eta probe in scipy's search hit a
                # boundary, observed in `pl_dyn_numerical` at extreme
                # |Δ| (Phase D audit, midpoint=-12.1 / half=4 → 16
                # bracket doublings exhausted).
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
        model: Model,
        prior: Prior,
        alpha: float,
    ):
        """Vectorised: η* at every θ in `grid` (θ-keyed).

        Each grid point is treated as a "data sufficient statistic" —
        we call `self.select(scheme, data=[theta], ...)` for each θ.
        This preserves the "static-η optimum at each θ" semantics
        used by `DynamicNumericalEtaSelector`.
        """
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
    eta_min_buffer: float = 1e-3
    n_grid: int = 401
    coarse_n: int = 25
    search_mult: float = 8.0
    cache_bin_width: float = 0.5
    is_dynamic: bool = True
    # `select_grid(theta_grid, ...)` is θ-only — η at each θ depends on
    # θ, not D — so the dynamic CI is calibrated. The single-context
    # `select(data=[D], ...)` shim falls back to NumericalEtaSelector at
    # θ=D and inherits its post-selection bias; prefer `select_grid`.
    is_post_selection: ClassVar[bool] = False
    # Memoization cache for `select_grid`. The selector is otherwise a
    # value object (frozen dataclass + value-equal `compare=` fields);
    # the cache is excluded from `__eq__` / `__hash__` / `__repr__`
    # via `compare=False, repr=False, hash=False` so two selectors
    # with the same configuration but different cache populations are
    # equal AND interchangeable. Read by `select_grid` only; never
    # written from outside this module.
    _cache: dict = field(
        default_factory=dict, compare=False, repr=False, hash=False
    )

    def _inner(self) -> NumericalEtaSelector:
        return NumericalEtaSelector(eta_min_buffer=self.eta_min_buffer)

    def select(
        self,
        scheme: TiltingScheme,
        *,
        data: np.ndarray | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        statistic: TestStatistic | None = None,
    ) -> float:
        """Convenience: a single-context η* (delegates to the static inner)."""
        assert data is not None and model is not None and prior is not None
        assert alpha is not None and statistic is not None
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        out = self.select_grid(
            np.asarray([D]),
            scheme,
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
        model: Model,
        prior: Prior,
        alpha: float,
    ):
        """Per-θ η* lookup with caching.

        Cache key includes the θ-grid endpoints (1e-6-binned to absorb
        tiny numerical drift) and the model/prior fingerprints so
        different experiments do not collide.
        """
        scheme_name = getattr(scheme, "name", type(scheme).__name__)
        coarse_n = len(grid)

        theta_arr = np.asarray(grid, dtype=np.float64)
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
            self._cache[key] = (theta_arr, eta_full)
            cached_grid, cached_eta = theta_arr, eta_full
        else:
            cached_grid, cached_eta = cached
        return np.interp(theta_arr, cached_grid, cached_eta)


# Threshold for the runtime safety clamp in `LearnedDynamicEtaSelector`.
# If the predicted η is out of admissible range for more than this
# fraction of the batch, raise rather than clamp.
#
# Set to 1.0 (effectively disabled) by default since the dynamic-CI
# scan's calibration guarantee — "η at each θ depends only on θ, not
# on D, so the WALDO p-value at any fixed η is U[0,1] under H0" —
# holds for ANY choice of η, including a clamped one. Clamping
# produces wider CIs at the boundary but does NOT break calibration.
#
# The original Cluster G rationale (gate it at 20%) was correct only
# under the assumption that out-of-admissible η meant "wrong checkpoint
# / undertrained." Phase G + σ₀-anchored training surfaces a benign
# extrapolation case: the audit's wide θ_grid at small σ₀ pushes the
# dynamic CI scan to query θ outside the per-slice trained range, where
# η extrapolation may drift out-of-admissible. Refusing here is overly
# conservative; the warning still flags it for inspection.
#
# Set < 1.0 to re-enable strict refusal (e.g. for unit tests that
# verify the safety net fires).
_CLAMP_FAIL_THRESHOLD = 1.0

# Phase G: map YAML class names → Python class __name__ for the
# cross-experiment guard's class-match check.
_CANONICAL_PRIOR_CLASS_NAMES: dict[str, str] = {
    "normal": "NormalDistribution",
}
_CANONICAL_MODEL_CLASS_NAMES: dict[str, str] = {
    "normal_normal": "NormalNormalModel",
}


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

    artifact: LearnedArtifact  # any Phase E v3 dual-head artifact (concrete EtaArtifact, NullArtifact stub, ...)
    name: ClassVar[str] = "learned_dynamic"
    is_dynamic: bool = True
    # Per-θ learned selector — η = MLP(θ), no D-conditioning. Calibrated
    # by construction (see learned_eta.md / dual-head training).
    is_post_selection: ClassVar[bool] = False
    # Default ON: when θ falls outside the training θ-distribution box,
    # override the network's prediction with `scheme.param_space.
    # eta_likelihood_only` — the η value that makes the procedure
    # reduce to a likelihood-only / data-only inference (e.g. η=1 in
    # power_law gives standard Wald). The network's behavior outside
    # its training distribution is unspecified; clamping to the
    # likelihood-only value gives a calibrated fallback (any fixed η
    # gives U[0,1] p-values under H0). Disable for diagnostic probes
    # that explicitly want to see the network's extrapolation behavior.
    clamp_outside_training: bool = True
    # Lazy-load latch for `artifact.load()`. Not part of equality —
    # two selectors with the same artifact are equal regardless of
    # whether either has loaded yet.
    _loaded: bool = field(default=False, init=False, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.artifact.load()
            self._loaded = True
            from .._errors import MissingArtifactError
            from ..learned.eta_artifact import CHECKPOINT_FORMAT_VERSION

            meta = self.artifact.metadata
            v = meta.get("checkpoint_format_version", None)
            if v != CHECKPOINT_FORMAT_VERSION or "experiment_config" not in meta:
                raise MissingArtifactError(
                    f"{self.artifact.name}: expected Phase E "
                    f"(v{CHECKPOINT_FORMAT_VERSION}) checkpoint with "
                    f"`experiment_config`; got format_version={v!r}. "
                    f"Re-train via `python -m scripts.train_learned_eta "
                    f"--config <experiment.yaml>`."
                )

    def _check_scheme(self, scheme: TiltingScheme) -> None:
        from .._errors import MissingArtifactError

        if not self._loaded:
            raise MissingArtifactError(
                f"{self.artifact.name} not loaded; " f"call .load() before _check_scheme()."
            )
        meta = self.artifact.metadata
        cfg = meta["experiment_config"]
        # v4 schema serialises as "scheme"; v3 used "scheme_name". Accept both.
        trained_scheme = cfg.get("scheme") or cfg.get("scheme_name")
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
        # Audit P0-16: gate on `alpha_mode` (added in Cluster E) NOT on
        # `alpha is None`. Pre-fix: a checkpoint trained with
        # loss_kind=static_width but with the alpha field accidentally
        # stripped to None (e.g. via a metadata sanitiser) would pass
        # `_check_alpha` silently. The new explicit `alpha_mode in
        # {"marginalised", "fixed"}` field carries the intended
        # contract; `is None` is no longer load-bearing.
        alpha_mode = meta.get("alpha_mode")
        if alpha_mode is None:
            # Legacy checkpoints (pre-Cluster E) lack alpha_mode; fall
            # back to the old `alpha is None → marginalised` heuristic
            # for backward compatibility, but warn so the user knows to
            # re-train at their convenience.
            stored = meta.get("alpha")
            if stored is None:
                import warnings as _w
                _w.warn(
                    f"{self.artifact.name}: legacy checkpoint without "
                    f"`alpha_mode`; falling back to alpha-is-None heuristic. "
                    f"Re-train via `python -m scripts.train_learned_eta` to "
                    f"get the explicit alpha_mode field.",
                    UserWarning,
                    stacklevel=2,
                )
                return
            trained_alpha = float(stored)
            if abs(alpha - trained_alpha) > 1e-9:
                raise MissingArtifactError(
                    f"{self.artifact.name} trained at alpha={trained_alpha}, "
                    f"but inference alpha={alpha}; retrain or use a "
                    f"marginalised checkpoint."
                )
            return
        if alpha_mode == "marginalised":
            return
        if alpha_mode == "fixed":
            stored = meta.get("alpha")
            if stored is None:
                raise MissingArtifactError(
                    f"{self.artifact.name} declares alpha_mode='fixed' but "
                    f"alpha is None — checkpoint metadata is internally "
                    f"inconsistent. Re-train."
                )
            trained_alpha = float(stored)
            if abs(alpha - trained_alpha) > 1e-9:
                raise MissingArtifactError(
                    f"{self.artifact.name} trained at alpha={trained_alpha} "
                    f"(alpha_mode=fixed), but inference alpha={alpha}; "
                    f"retrain at the right alpha or use a marginalised "
                    f"checkpoint."
                )
            return
        raise MissingArtifactError(
            f"{self.artifact.name}: unknown alpha_mode={alpha_mode!r}; "
            f"expected 'marginalised' or 'fixed'."
        )

    def _check_classes_and_range(
        self, prior: Prior, model: Model,
    ) -> None:
        """Phase G cross-experiment guard: class match + in-range hp.

        Refuses (with MissingArtifactError) if:
          - prior class != trained prior_class
          - model class != trained model_class
          - prior.hyperparams() or model.hyperparams() outside the
            HyperparamDistribution.support() the checkpoint was trained on
        """
        from .._errors import MissingArtifactError
        from ..learned.training.hyperparam_distribution import HyperparamDistribution

        meta = self.artifact.metadata["experiment_config"]
        if type(prior).__name__ != _CANONICAL_PRIOR_CLASS_NAMES.get(meta["prior_class"]):
            raise MissingArtifactError(
                f"{self.artifact.name} trained for prior_class="
                f"{meta['prior_class']!r} (expects "
                f"{_CANONICAL_PRIOR_CLASS_NAMES.get(meta['prior_class'])!r}); "
                f"got {type(prior).__name__!r}."
            )
        if type(model).__name__ != _CANONICAL_MODEL_CLASS_NAMES.get(meta["model_class"]):
            raise MissingArtifactError(
                f"{self.artifact.name} trained for model_class="
                f"{meta['model_class']!r} (expects "
                f"{_CANONICAL_MODEL_CLASS_NAMES.get(meta['model_class'])!r}); "
                f"got {type(model).__name__!r}."
            )
        distr = HyperparamDistribution.from_dict(meta["hyperparam_distribution"])
        prior_hp = prior.hyperparams()
        lik_hp = model.hyperparams()
        prior_names = type(prior).hyperparam_names()
        lik_names = type(model).hyperparam_names()
        bad = distr.first_out_of_range(prior_hp, lik_hp, prior_names, lik_names)
        if bad is not None:
            raise MissingArtifactError(
                f"{self.artifact.name}: inference {bad.name}={bad.value} is "
                f"outside trained range [{bad.low}, {bad.high}]; "
                f"train a wider checkpoint via scripts.train_learned_eta."
            )

    def _check_experiment(
        self,
        w: float | None,
        model_fingerprint: tuple | None = None,
        prior_fingerprint: tuple | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        n_data: int | None = None,
        theta_distribution_fingerprint: tuple | None = None,
    ) -> None:
        """Phase E: verify inference matches the trained experiment.

        Strict tuple-equal compare on (model.fingerprint(),
        prior.fingerprint()) is the primary safety check — and it
        is sufficient: byte-equal fingerprints rule out any
        cross-experiment use. The historical w-derived check (NN
        only) and the NN-only model/prior gate are kept conditional
        on ``trained_model_fp[0] == "normal_normal"`` and ``w is not
        None``, so future non-NN experiments can train and load
        checkpoints through the same selector.

        Phase 4 skeptic #6 (defense-in-depth on subclass collisions):
        when the inference-time ``model`` / ``prior`` are passed and
        the checkpoint records ``model_class`` / ``prior_class``
        (post-Phase-4 checkpoints), reject if the class names differ.
        Closes the "subclass with same fingerprint but overridden
        ``logpdf``" silent-acceptance hole. Older checkpoints lacking
        these keys preserve the legacy fingerprint-only check.

        Audit P1 H.2: model_fingerprint / prior_fingerprint are now
        **required** (raise on None instead of silently skipping).
        Pre-fix a misbehaving model returning ``None`` from
        ``fingerprint()`` would slip through cross-experiment use of
        a checkpoint trained on a different model.

        Audit P1 H.3: optional ``n_data`` and
        ``theta_distribution_fingerprint`` arguments. These are
        training-time concepts (the checkpoint's recorded sample
        size and θ-sampler), so callers at inference rarely have
        them. When supplied, refuse on mismatch.
        """
        from .._errors import MissingArtifactError

        meta = self.artifact.metadata["experiment_config"]
        trained_model_fp = tuple(meta["model_fingerprint"])
        trained_prior_fp = tuple(meta["prior_fingerprint"])
        trained_n_data = int(meta.get("n_data", 1))
        trained_td_fp_raw = meta.get("theta_distribution_fingerprint")
        trained_td_fp = tuple(trained_td_fp_raw) if trained_td_fp_raw is not None else None

        # Audit P1 H.2: model_fingerprint and prior_fingerprint are
        # required. A None on either side previously silently skipped
        # the cross-experiment check, which was the whole point of the
        # fingerprint contract. Refuse loudly so callers wire it up.
        if model_fingerprint is None:
            raise MissingArtifactError(
                f"{self.artifact.name}._check_experiment requires "
                f"model_fingerprint to be supplied (got None). The "
                f"fingerprint is the primary cross-experiment safety net; "
                f"silently skipping it lets a checkpoint trained on a "
                f"different model load. Pass `tuple(model.fingerprint())` "
                f"explicitly. Trained on model={trained_model_fp!r}."
            )
        if prior_fingerprint is None:
            raise MissingArtifactError(
                f"{self.artifact.name}._check_experiment requires "
                f"prior_fingerprint to be supplied (got None). "
                f"Pass `tuple(prior.fingerprint())` explicitly. "
                f"Trained with prior={trained_prior_fp!r}."
            )

        if tuple(model_fingerprint) != trained_model_fp:
            raise MissingArtifactError(
                f"{self.artifact.name} trained on model "
                f"{trained_model_fp!r}, but inference model is "
                f"{tuple(model_fingerprint)!r}. Train a new "
                f"checkpoint for this experiment."
            )
        if tuple(prior_fingerprint) != trained_prior_fp:
            raise MissingArtifactError(
                f"{self.artifact.name} trained with prior "
                f"{trained_prior_fp!r}, but inference prior is "
                f"{tuple(prior_fingerprint)!r}. Train a new "
                f"checkpoint for this experiment."
            )

        # Audit P1 H.3: optional n_data + theta_distribution_fingerprint
        # checks. These are training-time concepts; callers at inference
        # rarely have them, so the default-None path skips. When
        # supplied, refuse on mismatch — the trained checkpoint's MC
        # width loss was computed at the recorded n_data, and the η*
        # surface is calibrated for that.
        if n_data is not None and int(n_data) != trained_n_data:
            raise MissingArtifactError(
                f"{self.artifact.name} trained at n_data={trained_n_data}, "
                f"but inference n_data={int(n_data)}. The MC width loss "
                f"calibration is per-n_data; train a new checkpoint."
            )
        if (
            theta_distribution_fingerprint is not None
            and trained_td_fp is not None
            and tuple(theta_distribution_fingerprint) != trained_td_fp
        ):
            raise MissingArtifactError(
                f"{self.artifact.name} trained with theta_distribution "
                f"{trained_td_fp!r}, but inference θ-distribution is "
                f"{tuple(theta_distribution_fingerprint)!r}. The η* "
                f"surface is calibrated for the trained sampler; "
                f"train a new checkpoint."
            )

        # Phase 4 skeptic #6: class-name compare guards subclass
        # collisions where ``fingerprint()`` matches but ``logpdf`` /
        # ``sample_data`` are overridden. Pre-Phase-4 checkpoints lack
        # the class keys; treat absence as legacy (fingerprint-only).
        trained_model_cls = meta.get("model_class")
        trained_prior_cls = meta.get("prior_class")
        if trained_model_cls is not None and model is not None:
            if type(model).__name__ != trained_model_cls:
                raise MissingArtifactError(
                    f"{self.artifact.name} trained with "
                    f"model class {trained_model_cls!r}, but inference "
                    f"model is class {type(model).__name__!r}. The "
                    f"fingerprint matches but a subclass with overridden "
                    f"behaviour can produce different posteriors; refuse "
                    f"rather than silently load."
                )
        if trained_prior_cls is not None and prior is not None:
            if type(prior).__name__ != trained_prior_cls:
                raise MissingArtifactError(
                    f"{self.artifact.name} trained with "
                    f"prior class {trained_prior_cls!r}, but inference "
                    f"prior is class {type(prior).__name__!r}. The "
                    f"fingerprint matches but a subclass with overridden "
                    f"logpdf can produce different posteriors; refuse "
                    f"rather than silently load."
                )

        # NN-specific w-derived sanity check — only applies when the
        # trained checkpoint is NormalNormal + Normal prior AND the
        # caller supplied a w. For non-NN models w is meaningless and
        # the call passes ``w=None``.
        if w is None:
            return
        if trained_model_fp[0] != "normal_normal" or trained_prior_fp[0] != "normal":
            return
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
        # w → 0 / w → 1 (where the JAX port's denom-clamp distorts
        # silently).
        _W_EPS = 1e-3
        if not (_W_EPS < w < 1.0 - _W_EPS):
            raise MissingArtifactError(
                f"{self.artifact.name}: inference w={w:.6f} is outside "
                f"({_W_EPS}, {1.0 - _W_EPS}); the JAX port's "
                f"denom-clamp distorts silently in this regime."
            )

    def select(
        self,
        scheme: TiltingScheme,
        *,
        data: np.ndarray | None = None,
        model: Model | None = None,
        prior: Prior | None = None,
        alpha: float | None = None,
        statistic: TestStatistic | None = None,
    ) -> float:
        """Single-context η. Convenience for non-dynamic callers."""
        assert data is not None and model is not None and prior is not None
        assert alpha is not None and statistic is not None
        self._ensure_loaded()
        self._check_scheme(scheme)
        self._check_alpha(alpha)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        out = self.select_grid(
            np.asarray([D]),
            scheme,
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
        model: Model,
        prior: Prior,
        alpha: float,
    ):
        """Per-θ η* lookup via the trained Phase E EtaNet.

        EtaNet receives θ directly (its natural input); no |Δ|
        inversion / two-branch averaging.
        """
        self._ensure_loaded()
        self._check_scheme(scheme)
        self._check_alpha(alpha)
        self._check_classes_and_range(prior, model)

        theta_arr = np.asarray(grid, dtype=np.float64)
        prior_hp = prior.hyperparams()
        lik_hp = model.hyperparams()
        eta = self.artifact.predict_eta(theta_arr, prior_hp, lik_hp)

        # OOD-θ clamp: θ outside the training distribution → likelihood-only η.
        if self.clamp_outside_training:
            eta = self._clamp_outside_training_box(eta, theta_arr, prior, scheme)

        # NN-specific admissibility window for clamping.
        from ..models._dispatch import is_normal_normal
        from ..models.distributions import NormalDistribution
        if is_normal_normal(model) and isinstance(prior, NormalDistribution):
            w_eff = float(prior.scale) ** 2 / (
                float(model.sigma) ** 2 + float(prior.scale) ** 2
            )
        else:
            w_eff = None
        return self._maybe_clamp_eta(eta, scheme=scheme, w=w_eff, alpha=alpha)

    def _clamp_outside_training_box(self, eta, theta_arr, prior, scheme):
        """Override η = scheme.param_space.eta_likelihood_only for θ
        outside the training θ-distribution box.

        Reads the training θ-distribution from the checkpoint metadata.
        For ``sigma_anchored_uniform`` the box is ``[μ₀ − K·σ₀, μ₀ +
        K·σ₀]`` (using the per-call prior's loc/scale). For ``uniform``
        the box is ``[low, high]``. For unknown / unsupported types the
        clamp is a no-op (with a warning). For schemes whose
        ``param_space.eta_likelihood_only`` is None (e.g. identity,
        fisher_rao stub) the clamp is also a no-op.

        Calibration is preserved: any fixed η yields U[0,1] p-values
        under H₀, so swapping the network's η for `eta_likelihood_only`
        outside the training box keeps the dynamic-CI's 1-α coverage.
        """
        eta_lo_only = getattr(scheme.param_space, "eta_likelihood_only", None)
        if eta_lo_only is None:
            return eta

        meta = self.artifact.metadata
        cfg = meta.get("experiment_config") or {}
        theta_dist_spec = cfg.get("theta_distribution") or {}
        dist_type = theta_dist_spec.get("type")

        if dist_type == "sigma_anchored_uniform":
            K = float(theta_dist_spec.get("K", 5.0))
            prior_names = list(prior.hyperparam_names())
            if "loc" not in prior_names or "scale" not in prior_names:
                return eta
            prior_hp = np.asarray(prior.hyperparams(), dtype=np.float64)
            mu0 = float(prior_hp[prior_names.index("loc")])
            sigma0 = float(prior_hp[prior_names.index("scale")])
            lo = mu0 - K * sigma0
            hi = mu0 + K * sigma0
        elif dist_type == "uniform":
            lo = float(theta_dist_spec.get("low"))
            hi = float(theta_dist_spec.get("high"))
        else:
            import warnings
            warnings.warn(
                f"{self.artifact.name}: unknown theta_distribution type "
                f"{dist_type!r}; OOD-θ clamp is a no-op for this checkpoint.",
                RuntimeWarning,
                stacklevel=3,
            )
            return eta

        out_of_box = (theta_arr < lo) | (theta_arr > hi)
        if out_of_box.any():
            return np.where(out_of_box, float(eta_lo_only), eta)
        return eta

    def _maybe_clamp_eta(self, eta, *, scheme, w, alpha):
        """Runtime safety net for predicted η outside admissible range.

        For Normal-Normal experiments (``w`` is a finite float in
        ``(0, 1)``) the admissible window is scheme-specific:
        - ``ot``     : ``[0, 1]``
        - ``mixture``: ``[0, 1]`` (closed-form admissibility under the
                       structural sigmoid bound; tighter than PL's window)
        - other      : ``(-w/(1-w) + buffer, 1/(1-w) - buffer)`` (power_law)

        For non-Normal-Normal experiments (``w is None``), the
        admissibility region is learned by ``ValidityNet`` during
        training; the closed-form clamp is
        not applicable. We fall back to the ``eta_explore_box`` recorded
        in the checkpoint's ``experiment_config`` metadata, with a
        small interior buffer.
        """
        del alpha  # bounds are α-independent
        if w is None:
            # Phase G non-NN: no closed-form admissibility window;
            # ValidityNet learned the boundary. Read the training-time
            # eta_explore_box from the checkpoint metadata; this is the
            # range Head B was trained on and so the only range its
            # admissibility predictions cover.
            buffer = 1e-3
            cfg = self.artifact.metadata.get("experiment_config") or {}
            box = cfg.get("eta_explore_box")
            if box is not None and len(box) == 2:
                lo = float(box[0]) + buffer
                hi = float(box[1]) - buffer
            else:
                # Pre-v4 metadata or malformed: fall back to the historic
                # default ``[-5, 5]``.
                lo = -5.0 + buffer
                hi = 5.0 - buffer
        else:
            try:
                if not (0.0 < w < 1.0):
                    raise ValueError(f"w must lie in (0, 1); got {w!r}")
                buffer = 1e-3
                if scheme.name == "ot" or scheme.name == "mixture":
                    lo, hi = 0.0 + buffer, 1.0 - buffer
                else:
                    # power_law (and any other Normal-Normal scheme
                    # using the same admissible window).
                    lo = -w / (1.0 - w) + buffer
                    hi = 1.0 / (1.0 - w) - buffer
            except (ValueError, TiltingDomainError):
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
