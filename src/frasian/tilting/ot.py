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

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar, cast

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
from numpy.typing import ArrayLike, NDArray

# scipy: used by the numpy-eager scalar fast path inside tilted_pvalue.
# Same boundary discipline as power_law.py — see that file's comments
# for the rationale (~10 us scalar vs ~200 us under JAX dispatch).
from scipy import stats as _scalar_scipy_stats

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .._errors import TiltingDomainError
from .._registry import register_tilting
from ..models.base import Likelihood, Model, Posterior, Prior
from ..models.distributions import GaussianLikelihood, NormalDistribution
from ..statistics.base import TestStatistic
from .base import EtaSelector, ParamSpec, TiltingContext
from .eta_selectors import FixedEtaSelector
from .quantile_mixture import QuantileMixturePath

if TYPE_CHECKING:
    from ..config import Config

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _ot_tilted_pvalue_numpy_scalar(
    theta_f: float,
    eta_f: float,
    D_f: float,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> float:
    """Numpy-eager scalar fast path. Mirrors `_ot_tilted_pvalue_kernel`
    but runs on Python floats + scipy.stats.norm. Used when caller
    passes scalar (theta, eta, D) — the brentq inner-loop pattern.
    ~10 us/call.
    """
    if statistic_name == "wald":
        z = abs(D_f - theta_f) / sigma
        return float(2.0 * _scalar_scipy_stats.norm.sf(z))
    if statistic_name == "waldo":
        mu_n = w * D_f + (1.0 - w) * mu0
        mu_t = (1.0 - eta_f) * mu_n + eta_f * D_f
        s_t = (w + eta_f * (1.0 - w)) * sigma
        a = abs(mu_t - theta_f) / s_t
        b = (1.0 - eta_f) * (1.0 - w) * (mu0 - theta_f) / s_t
        return float(
            _scalar_scipy_stats.norm.cdf(b - a) + _scalar_scipy_stats.norm.cdf(-a - b)
        )
    raise NotImplementedError(
        f"_ot_tilted_pvalue_numpy_scalar not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo'."
    )


@partial(jax.jit, static_argnames=("statistic_name",))
def _ot_tilted_pvalue_kernel(
    theta: jax.Array,
    eta: jax.Array,
    D: jax.Array,
    w: float,
    mu0: float,
    sigma: float,
    statistic_name: str,
) -> jax.Array:
    """Pure JAX arithmetic kernel for the OT-tilted p-value.

    Same shape / discipline / jit rationale as
    `power_law._tilted_pvalue_kernel`. Scalar inputs do NOT reach
    this kernel — the public method's shape dispatch routes them to
    the numpy fast path.
    """
    if statistic_name == "wald":
        z = jnp.abs(D - theta) / sigma
        return 2.0 * (1.0 - jsp_stats.norm.cdf(z))
    if statistic_name == "waldo":
        mu_n = w * D + (1.0 - w) * mu0
        mu_t = (1.0 - eta) * mu_n + eta * D
        s_t = (w + eta * (1.0 - w)) * sigma
        a = jnp.abs(mu_t - theta) / s_t
        b = (1.0 - eta) * (1.0 - w) * (mu0 - theta) / s_t
        return jsp_stats.norm.cdf(b - a) + jsp_stats.norm.cdf(-a - b)
    raise NotImplementedError(
        f"_ot_tilted_pvalue_kernel not implemented for statistic={statistic_name!r}; "
        f"supported: 'wald', 'waldo'."
    )


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


@register_tilting(name="ot", brief="docs/methods/ot.md")
@dataclass(frozen=True)
class OTTilting:
    """Wasserstein-2 geodesic tilting (general 1D, Gaussian fast path)."""

    name: ClassVar[str] = "ot"
    param_space: ParamSpec = ParamSpec(
        eta_default=0.0,
        eta_identity=0.0,
        description=(
            "t in [0, 1] along the W2 geodesic between posterior (t=0) "
            "and likelihood-induced Gaussian N(D, sigma^2) (t=1)."
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
    ) -> Posterior:
        """W2-geodesic tilt between posterior and likelihood-as-distribution."""
        eta_arr = np.asarray(eta, dtype=np.float64)
        if eta_arr.ndim != 0:
            raise NotImplementedError(
                "tilt() expects scalar eta; vectorised eta is consumed via "
                "repeated scalar calls (see `path`)."
            )
        t = float(eta_arr)
        if not (0.0 <= t <= 1.0):
            raise TiltingDomainError(f"OTTilting requires eta in [0, 1], got {t!r}.")

        # Gaussian fast path: linear interpolation in (mu, sigma).
        if isinstance(posterior, NormalDistribution) and isinstance(likelihood, GaussianLikelihood):
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

    def path(
        self, posterior: Posterior, prior: Prior, likelihood: Likelihood, ts: NDArray[np.float64]
    ) -> Iterable[Posterior]:
        for t in np.asarray(ts, dtype=np.float64):
            yield self.tilt(posterior, prior, likelihood, float(t))

    def is_identity(self, eta: float) -> bool:
        return float(eta) == self.param_space.eta_identity

    def admissible_range(self, context: TiltingContext) -> tuple[float, float]:
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
    ) -> jax.Array:
        """Tilted p-value evaluated against the W2-tilted Gaussian.

        Specialized for (ot, waldo) and (ot, wald) on Normal-Normal:

          (ot, wald): eta-independent two-sided Wald, 2 * Phi(-|D-theta|/sigma)
          (ot, waldo): closed form derived in docs/methods/ot.md, with the
            standard error s_t = (w + eta*(1-w))*sigma.

        Endpoint sanity: at eta=0 reduces to bare WALDO; at eta=1 reduces
        to bare Wald (s_t -> sigma, mu_t -> D, b -> 0, a -> |D-theta|/sigma).

        Returns ``jax.Array`` for bulk inputs (length >= 2 in any of
        theta/eta/D); returns Python float for the all-scalar fast
        path. Same shape-dispatch + return-type discipline as
        `PowerLawTilting.tilted_pvalue` — see that method's docstring
        for the rationale.
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
        w = sigma0**2 / (sigma**2 + sigma0**2)

        # Validation runs in numpy (JAX can't raise mid-trace). Convert
        # eta to numpy exactly once for the [0, 1] + finiteness check.
        eta_np = np.asarray(eta, dtype=np.float64)
        invalid = ~(np.isfinite(eta_np) & (eta_np >= 0.0) & (eta_np <= 1.0))
        if np.any(invalid):
            if eta_np.ndim == 0:
                raise TiltingDomainError(
                    f"OTTilting.tilted_pvalue requires eta in [0, 1], got "
                    f"{float(eta_np)!r}."
                )
            bad = int(np.argmax(invalid))
            raise TiltingDomainError(
                f"OTTilting.tilted_pvalue requires eta in [0, 1] and finite; "
                f"offending index {bad} eta={float(eta_np.flat[bad])!r}."
            )

        # Shape dispatch: scalar -> numpy fast path; bulk -> jit'd kernel.
        theta_np = np.asarray(theta, dtype=np.float64)
        D_np = np.asarray(D, dtype=np.float64)
        if theta_np.size == 1 and eta_np.size == 1 and D_np.size == 1:
            # Returns Python float, NOT jnp.asarray(float) — see the
            # corresponding note in PowerLawTilting.tilted_pvalue.
            return _ot_tilted_pvalue_numpy_scalar(  # type: ignore[return-value]
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
        return _ot_tilted_pvalue_kernel(theta_arr, eta_arr, D_arr, w, mu0, sigma, statistic_name)

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
                "OTTilting.tilted_confidence_interval currently requires " "NormalNormalModel."
            )
        if not isinstance(prior, NormalDistribution):
            raise NotImplementedError(
                "OTTilting.tilted_confidence_interval currently requires a "
                "NormalDistribution prior."
            )
        sigma = float(model.sigma)
        mu0 = float(prior.loc)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)
        mu_n = w * D + (1.0 - w) * mu0

        mid = float(D) if statistic_name == "wald" else float((1.0 - eta) * mu_n + eta * D)  # waldo

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
        # eta validation now lives in `tilted_pvalue` (single source of
        # truth; was previously duplicated here for the Phase 1 skeptic
        # vector #5 contract). Removing the upfront check eliminates
        # ~5 µs of redundant work per scan and keeps error messages
        # consistent regardless of whether a caller hits this dynamic
        # entry-point or the inner `tilted_pvalue` directly (e.g. through
        # the vec_fn callback at `_dynamic.py:151`).
        # Vectorised path: `tilted_pvalue` now broadcasts over array eta,
        # so one bulk call replaces the scalar Python loop (Tier 1.3 N3).
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

        Delegates to
        `frasian.tilting._dynamic.dynamic_ci_scan`; the scheme-specific
        bit is the `tilted_pvalue` closure.
        """
        from ..models.normal_normal import NormalNormalModel
        from ._dynamic import dynamic_ci_scan

        if not isinstance(model, NormalNormalModel):
            raise NotImplementedError(
                "OTTilting.dynamic_tilted_confidence_interval currently "
                "requires NormalNormalModel."
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
                f"OTTilting requires a NormalDistribution prior; " f"got {type(prior).__name__!r}."
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
        the selector's own attributes for backward compatibility.
        Skeptic Phase 5 vector #2.
        """
        self._require_normal_sandbox(model, prior)
        # Narrow types after the dispatch check (mypy can't infer through it).
        # `cast` is `-O`-safe; the runtime gate is `_require_normal_sandbox` above.
        from ..models.normal_normal import NormalNormalModel

        model = cast(NormalNormalModel, model)
        prior = cast(NormalDistribution, prior)
        D = _data_to_scalar_D(data)
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

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
        """Convex hull of `confidence_regions`; ``config`` forwarded
        to that method for the dynamic-CI scan resolution.
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
        sigma = float(model.sigma)
        sigma0 = float(prior.scale)
        w = sigma0**2 / (sigma**2 + sigma0**2)

        if getattr(self.selector, "is_dynamic", False):
            # Phase 3a-1: θ-space directly.
            coarse_n = int(getattr(self.selector, "coarse_n", 25))
            theta_lo = float(theta_arr.min())
            theta_hi = float(theta_arr.max())
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
