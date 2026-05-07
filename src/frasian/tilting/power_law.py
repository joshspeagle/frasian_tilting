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

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..config import Config
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..models.normal_normal import weight as _weight
from ..statistics.base import TestStatistic
from .base import EtaSelector, ParamSpec
from .eta_selectors import FixedEtaSelector

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


from functools import partial

# scipy: used by the numpy-eager scalar fast path inside tilted_pvalue.
# JAX dispatch costs ~50-300 us per scalar call even with jit; numpy +
# scipy.stats.norm is ~10 us. The bulk JAX kernel below handles the
# array path (length >= 2) and is autodiff-clean for Phase 4.
from scipy import stats as _scalar_scipy_stats


def _tilted_pvalue_numpy_scalar(
    theta_f: float,
    eta_f: float,
    D_f: float,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> float:
    """Numpy-eager scalar fast path. Mirrors `_tilted_pvalue_kernel` but
    runs on Python floats + scipy.stats.norm. Used when caller passes
    scalar (theta, eta, D) — the brentq inner-loop pattern. ~10 us/call.
    """
    denom = 1.0 - eta_f * (1.0 - w)
    if statistic_name == "wald":
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))
    if statistic_name == "waldo":
        mu_eta = (w * D_f + (1.0 - eta_f) * (1.0 - w) * mu0) / denom
        norm_factor = w * sigma / denom
        a_eta = abs(mu_eta - theta_f) / norm_factor
        b_eta = (1.0 - eta_f) * (1.0 - w) * (mu0 - theta_f) / (denom * norm_factor)
        return float(
            _scalar_scipy_stats.norm.cdf(b_eta - a_eta)
            + _scalar_scipy_stats.norm.cdf(-a_eta - b_eta)
        )
    raise NotImplementedError(
        f"_tilted_pvalue_numpy_scalar not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo'."
    )


@partial(jax.jit, static_argnames=("statistic_name",))
def _tilted_pvalue_kernel(
    theta: jax.Array,
    eta: jax.Array,
    D: jax.Array,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> jax.Array:
    """Pure JAX arithmetic kernel for the tilted p-value.

    Autodiff-clean (no validation, no Python control flow except the
    static `statistic_name` dispatch). Phase 4's learned-eta `loss`
    closes over this directly inside `@jax.jit`. The public
    `PowerLawTilting.tilted_pvalue` validates inputs once, then
    delegates here.

    JIT note (style guide allows: profile shows it pays off): the bulk
    path is called by `dynamic_ci_scan` (theta-grid shape (n_grid,))
    and by selectors / experiments with a small set of recurring
    shapes. JAX's jit cache is process-wide, so within one test sweep
    the first call per (shape, statistic_name) pays ~150 ms compile
    and every subsequent call is ~0.5 ms; eager mode without jit
    would be ~2 ms per call regardless. Across the L0 sweep that
    difference is the ~30 s vs >120 s gap.

    Scalar inputs do NOT reach this kernel — the public method's
    shape dispatch routes them to the numpy fast path (~10 us/call,
    much faster than either jit'd or eager JAX would be on a 0-d).
    """
    denom = 1.0 - eta * (1.0 - w)
    if statistic_name == "wald":
        # eta-independent: 2 * (1 - Phi(|D - theta| / sigma)).
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - jsp_stats.norm.cdf(z))
    if statistic_name == "waldo":
        mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * mu0) / denom
        norm_factor = w * sigma / denom
        a_eta = jnp.abs(mu_eta - theta) / norm_factor
        b_eta = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / (denom * norm_factor)
        return jsp_stats.norm.cdf(b_eta - a_eta) + jsp_stats.norm.cdf(-a_eta - b_eta)
    raise NotImplementedError(
        f"_tilted_pvalue_kernel not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo'."
    )


def _require_gaussian(
    posterior: Posterior, prior: Prior, likelihood: Likelihood
) -> tuple[NormalDistribution, NormalDistribution, GaussianLikelihood]:
    """Recognize a Normal-Normal context. Raise if any input is not Gaussian."""
    if not isinstance(posterior, NormalDistribution):
        raise NotImplementedError(
            "PowerLawTilting requires a NormalDistribution posterior; "
            f"got {type(posterior).__name__!r}. Numerical fallback is a "
            f"future extension."
        )
    if not isinstance(prior, NormalDistribution):
        raise NotImplementedError(
            "PowerLawTilting requires a NormalDistribution prior; " f"got {type(prior).__name__!r}."
        )
    if not isinstance(likelihood, GaussianLikelihood):
        raise NotImplementedError(
            "PowerLawTilting requires a GaussianLikelihood; " f"got {type(likelihood).__name__!r}."
        )
    return posterior, prior, likelihood


def _denom(w: float, eta: float) -> float:
    return 1.0 - eta * (1.0 - w)


def _data_to_scalar_D(data: NDArray[np.float64]) -> float:
    """Coerce ``data`` to a single scalar D for the n=1 sandbox.

    The framework's Normal-Normal contract is single-observation
    (CLAUDE.md "1D conjugate Normal-Normal sandbox"). Earlier code
    silently used ``data.mean()`` which produced wrong CI widths for
    n>1 (the effective σ would shrink as σ/√n, not σ). Make the
    assumption explicit: refuse n>1 with a clear message rather than
    silently mis-scaling. Tier 1.5-O8 in the audit.
    """
    arr = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if arr.size != 1:
        raise NotImplementedError(
            f"Normal-Normal sandbox is single-observation (n=1); got "
            f"data.size={arr.size}. For n>1, use sigma_eff=sigma/sqrt(n) "
            f"and pass data.mean() with a model whose sigma is sigma_eff."
        )
    # Use ``arr.item()`` so shape-(1,1) and other non-flat single-element
    # inputs reduce cleanly. ``float(arr[0])`` would crash with
    # ``TypeError: only 0-dimensional arrays can be converted to Python
    # scalars`` on a shape-(1,1) input where ``arr[0]`` is a shape-(1,)
    # array. Phase 5 skeptic vector #3.
    return float(arr.item())


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

    name: ClassVar[str] = "power_law"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description="eta=0 recovers WALDO; eta=1 recovers Wald.",
    )
    selector: EtaSelector = field(default_factory=lambda: FixedEtaSelector(eta=0.0))

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

    def tilt(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, eta: ArrayLike
    ) -> NormalDistribution:
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
        sigma_eta_sq = w * lik.sigma**2 / denom
        sigma_eta = float(np.sqrt(sigma_eta_sq))
        return NormalDistribution(loc=float(mu_eta), scale=sigma_eta)

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[NormalDistribution]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, t)

    def is_identity(self, eta: float) -> bool:
        return eta == self.param_space.eta_identity

    # ----- (TiltingScheme, TestStatistic) cross-product specialisations -----
    #
    # The smoothness experiment needs `tilted p-value` and `tilted CI` —
    # quantities that depend on *both* the tilting scheme and the test
    # statistic. The cleanest factoring would be multiple dispatch on
    # (scheme, statistic) types; for now we dispatch on `statistic.name`
    # inside the scheme. Documented as a temporary bridge in
    # `docs/methods/power_law.md`. The generalisation lands when a
    # second non-Wald/WALDO statistic gets implemented.

    def tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float | NDArray[np.float64],
        model: object,
        prior: NormalDistribution,
        eta: ArrayLike,
        statistic_name: str,
    ) -> jax.Array:
        """p-value of `statistic_name` evaluated against the tilted posterior.

        Specialized for (power_law, waldo) — Theorem 8 closed form — and
        (power_law, wald) — eta-independent two-sided Wald.

        ``eta`` accepts either a scalar (the historical contract) or an
        array broadcastable to ``theta``. The array path is what
        ``dynamic_ci_scan`` (and ``dynamic_tilted_pvalue``) use to evaluate
        a per-θ varying η in one bulk JAX call (Tier 1.3 N1/N3).

        ``D`` accepts either a scalar (the historical contract) or an
        array broadcastable to ``theta_arr``. The closed-form formulas
        are pure broadcasting arithmetic.

        Returns ``jax.Array`` for bulk inputs (length >= 2 in any of
        theta/eta/D); returns Python float for the all-scalar fast
        path. The relaxed scalar return type is deliberate: wrapping a
        scalar in ``jnp.asarray(...)`` costs ~200 us (dwarfs the ~10 us
        scalar arithmetic), defeating the dispatch. Callers that need
        a uniform return type wrap in ``float(...)`` or
        ``jnp.asarray(...)`` at their own boundary; this is what the
        brentq closures in ``tilted_confidence_interval`` and
        ``dynamic_tilted_confidence_interval`` already do, and what
        ``dynamic_tilted_pvalue`` already does via ``np.asarray(...)``.
        See ``docs/jax_style.md``.
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
        w = sigma0**2 / (sigma**2 + sigma0**2)

        # Validation runs in numpy (JAX can't raise mid-trace; validation
        # lives outside the autodiff-clean kernel). Convert eta to numpy
        # exactly once for the finiteness + admissibility checks; the
        # JAX kernel re-asarrays internally.
        eta_np = np.asarray(eta, dtype=np.float64)
        mask_invalid = ~np.isfinite(eta_np)
        if np.any(mask_invalid):
            bad_idx = int(np.argmax(mask_invalid))
            raise TiltingDomainError(
                f"PowerLawTilting.tilted_pvalue requires finite eta; "
                f"offending index {bad_idx} eta="
                f"{float(eta_np.flat[bad_idx])!r}"
            )
        denom_np = 1.0 - eta_np * (1.0 - w)
        if np.any(denom_np <= 0.0):
            if eta_np.ndim == 0:
                raise TiltingDomainError(
                    f"eta={float(eta_np)!r} drives denom to "
                    f"{float(denom_np)!r} <= 0 with w={w!r}."
                )
            bad = int(np.argmax(denom_np <= 0.0))
            raise TiltingDomainError(
                f"PowerLawTilting.tilted_pvalue: eta drives denom <= 0 at "
                f"index {bad} (eta={float(eta_np.flat[bad])!r}, "
                f"denom={float(np.asarray(denom_np).flat[bad])!r}, w={w!r})."
            )

        # Shape dispatch: scalar inputs route through the numpy fast path
        # (~10 us per call), bulk inputs through the autodiff-clean JAX
        # kernel (~10 us per call after jit warmup; the first call per
        # unique shape pays a ~50 ms compile). The criterion is "all of
        # theta, eta, D are scalar" — when even one is an array, the JAX
        # broadcast handles the result. Both branches return jax.Array.
        theta_np = np.asarray(theta, dtype=np.float64)
        D_np = np.asarray(D, dtype=np.float64)
        if theta_np.size == 1 and eta_np.size == 1 and D_np.size == 1:
            # Scalar fast path: return Python float directly, NOT
            # jnp.asarray(float). Wrapping in jnp.asarray would cost
            # ~200 us per call (dwarfing the ~10 us scalar arithmetic
            # below) and defeat the dispatch. Callers wrap in float()
            # or jnp.asarray() at their own boundary; the protocol's
            # nominal jax.Array return type is honoured by the bulk
            # path and explicitly relaxed for this scalar fast path
            # (documented in the docstring above).
            return _tilted_pvalue_numpy_scalar(  # type: ignore[return-value]
                # .item() handles shape-(), shape-(1,), shape-(1,1) uniformly;
                # plain float(arr) fails on >0-d single-element arrays.
                float(theta_np.item()),
                float(eta_np.item()),
                float(D_np.item()),
                w,
                mu0,
                sigma,
                statistic_name,
            )
        # Bulk JAX kernel — autodiff-clean, no Python control flow.
        theta_arr = jnp.asarray(theta, dtype=jnp.float64)
        eta_arr = jnp.asarray(eta, dtype=jnp.float64)
        D_arr = jnp.asarray(D, dtype=jnp.float64)
        return _tilted_pvalue_kernel(theta_arr, eta_arr, D_arr, w, mu0, sigma, statistic_name)

    def dynamic_tilted_pvalue(
        self,
        theta: ArrayLike,
        D: float,
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
        # Vectorised path: `tilted_pvalue` now returns jax.Array; convert
        # to numpy at the boundary because `_dynamic.py`'s scan algorithm
        # is numpy-only by design (Brent root-find inner loops). One bulk
        # call replaces the scalar Python loop; an invalid eta in any
        # slot still raises TiltingDomainError with the offending index
        # identified (Tier 1.3 N3).
        out = np.asarray(
            self.tilted_pvalue(theta_arr, D, model, prior, eta_arr, statistic_name),
            dtype=np.float64,
        )
        return out if out.size > 1 else np.asarray(float(out.item()))

    def dynamic_tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
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
                "dynamic_tilted_confidence_interval currently requires " "NormalNormalModel."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        def _tilted_pvalue_fn(theta: float, eta: float) -> float:
            return float(
                self.tilted_pvalue(
                    theta,
                    D,
                    model,
                    prior,
                    eta,
                    statistic_name,
                )
            )

        def _tilted_pvalue_vec_fn(
            theta_arr: np.ndarray, eta_arr: np.ndarray
        ) -> np.ndarray:
            # Bulk path: tilted_pvalue broadcasts over array eta when
            # eta has the same shape as theta. Single numpy/scipy call
            # replaces the scalar Python loop in dynamic_ci_scan
            # (Tier 1.3 N1).
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
            model=model,
            prior=prior,
        )

    # ----- Uniform CI / regions / pvalue interface (called by experiments) -----

    def _require_normal_sandbox(self, model: Model, prior: Prior) -> None:
        from ..models.normal_normal import NormalNormalModel

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "PowerLawTilting requires NormalNormalModel; " f"got {type(model).__name__!r}."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "PowerLawTilting requires a NormalDistribution prior; "
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
        multi-element for dynamic selectors at conflict-band D where the
        dynamic p-value is multimodal.

        ``config`` (optional, kw-only): when supplied, the dynamic-CI
        scan reads ``n_grid`` / ``coarse_n`` / ``search_mult`` from
        ``Config.dynamic_*``. When ``None`` (default), falls back to
        the selector's own attributes (or the function-default
        constants when the selector lacks them) for backward
        compatibility. Skeptic Phase 5 vector #2: previously the
        Config fields fed only the cache fingerprint, never the
        runtime path; ``config`` plumbing closes that gap.
        """
        self._require_normal_sandbox(model, prior)
        # Narrow types after the dispatch check (mypy can't infer through it).
        # `cast` is `-O`-safe; the runtime gate is `_require_normal_sandbox` above.
        from ..models.normal_normal import NormalNormalModel

        model = cast(NormalNormalModel, model)
        prior = cast(NormalDistribution, prior)
        D = _data_to_scalar_D(data)
        sigma = model.sigma
        sigma0 = prior.scale
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
            # Config wins when supplied; selector attrs are the legacy
            # fallback (preserves existing behaviour for callers that
            # don't yet thread Config through).
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

        eta = float(
            self.selector.select(
                self,
                data=data,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=statistic,
            )
        )
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
        """Convex hull of `confidence_regions` — single (lo, hi) summary.

        Multi-region cells (e.g. dynamic-η Dyn-WALDO at low |Δ|) collapse
        to `(min lo, max hi)` here; consumers that need union semantics
        should call `confidence_regions` directly.

        ``config`` is forwarded to `confidence_regions`; see that method
        for the dynamic-CI scan semantics.
        """
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
        # Narrow types after the dispatch check (mypy can't infer through it).
        # `cast` is `-O`-safe; the runtime gate is `_require_normal_sandbox` above.
        from ..models.normal_normal import NormalNormalModel

        model = cast(NormalNormalModel, model)
        prior = cast(NormalDistribution, prior)
        from ..config import Config
        from .eta_selectors import _NamedStatistic

        alpha = float(Config.default().alpha)
        D = _data_to_scalar_D(data)
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        sigma = model.sigma
        sigma0 = prior.scale
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
            # Phase 3a-1: dynamic selectors are θ-indexed natively. Build a
            # coarse θ-grid spanning the inference θ values and call
            # `select_grid(theta_grid, model=, prior=, ...)` directly.
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            theta_lo = float(theta_arr.min())
            theta_hi = float(theta_arr.max())
            # Pad the bounds slightly so np.interp at the extremes does
            # not extrapolate.
            half_pad = 1e-6 * max(1.0, abs(theta_hi - theta_lo))
            coarse_grid = np.linspace(theta_lo - half_pad, theta_hi + half_pad, coarse_n)
            coarse_eta = self.selector.select_grid(  # type: ignore[attr-defined]
                coarse_grid,
                self,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=_NamedStatistic(statistic.name),
            )
            eta_at_theta = np.interp(theta_arr, coarse_grid, coarse_eta)
            return self.dynamic_tilted_pvalue(
                theta_arr,
                D,
                model,
                prior,
                statistic.name,
                eta_at_theta,
            )

        # Static selector: single η for the whole evaluation.
        eta = float(
            self.selector.select(
                self,
                data=data,
                model=model,
                prior=prior,
                alpha=alpha,
                statistic=statistic,
            )
        )
        return self.tilted_pvalue(theta_arr, D, model, prior, eta, statistic.name)

    def tilted_confidence_interval(
        self,
        alpha: float,
        D: float,
        model: object,
        prior: NormalDistribution,
        eta: float,
        statistic_name: str,
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
        w = sigma0**2 / (sigma**2 + sigma0**2)

        # Use the tilted posterior mean as the bracket midpoint (CI mode for WALDO).
        denom = _denom(w, eta)
        if denom <= 0.0:
            raise TiltingDomainError(f"eta={eta!r} drives denom to {denom!r} <= 0 with w={w!r}.")
        mu_eta = (w * D + (1.0 - eta) * (1.0 - w) * prior.loc) / denom
        if statistic_name == "wald":
            mu_eta = D  # Wald CI is centred on D, not mu_eta.

        def f(theta_val: float) -> float:
            return (
                float(self.tilted_pvalue(float(theta_val), D, model, prior, eta, statistic_name))
                - alpha
            )

        half = 4.0 * sigma
        lo = brentq_with_doubling(f, midpoint=float(mu_eta), initial_half_width=half, direction=-1)
        hi = brentq_with_doubling(f, midpoint=float(mu_eta), initial_half_width=half, direction=+1)
        return (lo, hi)
