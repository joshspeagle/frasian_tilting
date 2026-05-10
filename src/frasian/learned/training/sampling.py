"""Phase G experiment-config + sampling primitives.

- ``ThetaDistribution`` — protocol for any 1D distribution over θ
  exposing ``sample(n, rng)``, ``support()``, and ``fingerprint()``.
- ``UniformThetaDistribution`` — concrete uniform-on-[low, high].
- ``THETA_DISTRIBUTION_REGISTRY`` — name → class lookup for YAML.
- ``ExperimentConfig`` — frozen dataclass binding (scheme, statistic,
  prior_cls, model_cls, hyperparam_distribution, theta_distribution)
  into a single self-describing object that drives both training and
  selector validation. Phase G change: prior + model are CLASSES (not
  instances); per-batch hyperparams are sampled from
  ``hyperparam_distribution`` during training.
- ``lhs_1d(theta_dist, n, seed)`` — 1D Latin Hypercube Sampling on a
  ``ThetaDistribution``'s support.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.stats import qmc

from ..._registry import registry as _registry

if TYPE_CHECKING:
    from .hyperparam_distribution import HyperparamDistribution


@runtime_checkable
class ThetaDistribution(Protocol):
    """A distribution over θ-space."""

    name: str

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]: ...
    def support(self) -> tuple[float, float]: ...
    def fingerprint(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class UniformThetaDistribution:
    """Uniform(low, high) over θ."""

    low: float
    high: float

    name: ClassVar[str] = "uniform"
    is_anchored: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if not (np.isfinite(self.low) and np.isfinite(self.high)):
            raise ValueError(f"low and high must be finite; got ({self.low}, {self.high})")
        if self.high <= self.low:
            raise ValueError(f"high must exceed low; got ({self.low}, {self.high})")

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]:
        return rng.uniform(low=self.low, high=self.high, size=n).astype(np.float64)

    def support(self) -> tuple[float, float]:
        return (float(self.low), float(self.high))

    def fingerprint(self) -> tuple[Any, ...]:
        return ("uniform", float(self.low), float(self.high))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "uniform", "low": float(self.low), "high": float(self.high)}


@dataclass(frozen=True)
class Sigma0AnchoredUniformThetaDistribution:
    """Per-batch-element uniform anchored to the **PRIOR's σ₀**:
    θ ~ U(μ₀ − K·σ₀, μ₀ + K·σ₀).

    Naming clarification (read this first):
    The "Sigma" / "sigma" in this class refers to **σ₀ — the prior's
    standard deviation**, NOT the likelihood's σ. The framework uses
    both σ and σ₀ heavily; only σ₀ is used for anchoring θ_true here.
    A formerly-confusing shorthand "σ-anchored" appears in older
    docstrings/notes/comments — those mean σ₀-anchored. The frozen
    serialization identifier remains ``sigma_anchored_uniform`` for
    backwards compat with existing v4 checkpoints; the canonical
    Python class name is ``Sigma0AnchoredUniformThetaDistribution``.

    The training loop sees a *relative* sample u ~ U(-K, K) from
    ``sample(n, rng)`` and reconstructs the actual θ per batch element
    as ``θ_actual[i] = μ₀_i + σ₀_i · u[i]``. This concentrates training
    samples where the prior is informative (within K standard
    deviations of the prior center), which lets the conditional
    EtaNet learn a meaningful V-shaped η(θ) — wide-θ (uniform on a
    fixed range) puts most samples in the no-prior-coverage tail,
    causing the model to collapse to η ≈ 1 (Wald) on average.

    K=5 is the recommended default — covers ±5σ₀ of the prior, the
    same range used for typical prior coverage tests.

    The integrated-p loss's θ-grid stays in absolute units (sized to
    cover the union of all per-element θ ranges; set explicitly via
    ``ExperimentConfig.theta_grid_lo`` / ``theta_grid_hi``).

    Implication for evaluation: the data marginal D under this
    sampling is concentrated near μ₀ (within ~μ₀ ± K·σ₀ + σ). A
    network trained against this distribution is calibrated and near-
    optimal for inference whose data also comes from this same
    distribution; under a broader (e.g. likelihood-σ-anchored) data
    distribution the network's η values inside the prior support
    box are no longer optimal. The OOD-θ clamp on
    ``LearnedDynamicEtaSelector`` covers θ_test outside the σ₀ box;
    the data-marginal mismatch is a separate failure mode (handle
    by re-training on the target data distribution).
    """

    K: float = 5.0

    # Frozen serialization identifier — DO NOT change without breaking
    # backwards compat with checkpoints whose metadata records this
    # string (and whose fingerprint pins it).
    name: ClassVar[str] = "sigma_anchored_uniform"
    is_anchored: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not (np.isfinite(self.K) and self.K > 0):
            raise ValueError(f"K must be a positive finite float; got {self.K}")

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]:
        """Return relative samples u ~ U(-K, K). Anchoring to per-element
        prior happens at use time in the training loop."""
        return rng.uniform(low=-float(self.K), high=float(self.K), size=n).astype(np.float64)

    def support(self) -> tuple[float, float]:
        """Relative support of the sampler (units of σ₀ from μ₀)."""
        return (-float(self.K), float(self.K))

    def fingerprint(self) -> tuple[Any, ...]:
        return ("sigma_anchored_uniform", float(self.K))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "sigma_anchored_uniform", "K": float(self.K)}


THETA_DISTRIBUTION_REGISTRY: dict[str, Any] = {
    "uniform": UniformThetaDistribution,
    "sigma_anchored_uniform": Sigma0AnchoredUniformThetaDistribution,
}

# Backwards-compat alias for any external code that imported the
# previously-confusing class name. The class itself was renamed
# 2026-05-10 to make explicit that the "Sigma" is σ₀ (the prior's
# scale), not σ (the likelihood's scale). New code should use the
# canonical name.
SigmaAnchoredUniformThetaDistribution = Sigma0AnchoredUniformThetaDistribution

_THETA_DIST_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    "uniform": frozenset({"low", "high"}),
    "sigma_anchored_uniform": frozenset({"K"}),
}


def _filter_kwargs(
    spec: Mapping[str, Any],
    allowed: frozenset[str],
    type_: str,
    kind: str,
) -> dict[str, Any]:
    """Strip keys outside ``allowed``; raise on extras to surface typos."""
    extras = set(spec.keys()) - allowed
    if extras:
        raise ValueError(
            f"Unexpected {kind} kwargs for type={type_!r}: {sorted(extras)} "
            f"(allowed: {sorted(allowed)})"
        )
    return {k: v for k, v in spec.items() if k in allowed}


def anchor_theta_to_prior(
    theta_relative_np: NDArray[np.float64],
    prior_hp_batch_np: NDArray[np.float64],
    prior_names: tuple[str, ...],
    theta_distribution: ThetaDistribution,
) -> NDArray[np.float64]:
    """Convert relative θ samples to absolute given per-element prior_hp.

    When ``theta_distribution.is_anchored`` is True (e.g.
    ``Sigma0AnchoredUniformThetaDistribution``), the sampler emits values
    in σ₀-units (e.g. U(-K, K)). This helper converts each relative
    sample to absolute θ via ``μ₀_i + σ₀_i · u_i``, using the per-element
    prior hyperparams. Requires the prior class to have ``loc`` and
    ``scale`` in its ``hyperparam_names()``.

    For non-anchored distributions, returns ``theta_relative_np`` unchanged.
    """
    if not getattr(theta_distribution, "is_anchored", False):
        return theta_relative_np
    if "loc" not in prior_names or "scale" not in prior_names:
        raise ValueError(
            f"theta_distribution {theta_distribution.name!r} requires the "
            f"prior class to expose 'loc' and 'scale' in hyperparam_names; "
            f"got names={prior_names!r}. Anchored sampling is currently "
            "supported only for location-scale priors (e.g. NormalDistribution)."
        )
    loc_idx = prior_names.index("loc")
    scale_idx = prior_names.index("scale")
    mu0 = prior_hp_batch_np[:, loc_idx]
    sigma0 = prior_hp_batch_np[:, scale_idx]
    return (mu0 + sigma0 * theta_relative_np).astype(np.float64)


def _build_theta_distribution_from_dict(
    d: Mapping[str, Any],
) -> ThetaDistribution:
    spec = dict(d)
    type_ = spec.pop("type")
    if type_ not in THETA_DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unknown theta_distribution type {type_!r}; expected one of: "
            f"{list(THETA_DISTRIBUTION_REGISTRY)}"
        )
    cls = THETA_DISTRIBUTION_REGISTRY[type_]
    allowed = _THETA_DIST_ALLOWED_KWARGS.get(type_, frozenset(spec.keys()))
    kwargs = _filter_kwargs(spec, allowed, type_, "theta_distribution")
    instance: ThetaDistribution = cls(**kwargs)
    return instance


# ----- Phase G class resolvers -----

_PRIOR_CLASSES: dict[str, type | None] = {
    "normal": None,
    "beta":   None,
}
_MODEL_CLASSES: dict[str, type | None] = {
    "normal_normal": None,
    "bernoulli":     None,
}


def _resolve_prior_class(type_str: str) -> type:
    if _PRIOR_CLASSES["normal"] is None:
        from ...models.distributions import BetaDistribution, NormalDistribution
        _PRIOR_CLASSES["normal"] = NormalDistribution
        _PRIOR_CLASSES["beta"] = BetaDistribution
    if type_str not in _PRIOR_CLASSES or _PRIOR_CLASSES[type_str] is None:
        raise ValueError(
            f"Unknown prior_class {type_str!r}; known: {sorted(_PRIOR_CLASSES)}"
        )
    return _PRIOR_CLASSES[type_str]


def _resolve_model_class(type_str: str) -> type:
    if _MODEL_CLASSES["normal_normal"] is None:
        from ...models.bernoulli import BernoulliModel
        from ...models.normal_normal import NormalNormalModel
        _MODEL_CLASSES["normal_normal"] = NormalNormalModel
        _MODEL_CLASSES["bernoulli"] = BernoulliModel
    if type_str not in _MODEL_CLASSES or _MODEL_CLASSES[type_str] is None:
        raise ValueError(
            f"Unknown model_class {type_str!r}; known: {sorted(_MODEL_CLASSES)}"
        )
    return _MODEL_CLASSES[type_str]


def _prior_class_name(prior_cls: type) -> str:
    """Reverse-lookup the YAML name for a prior class."""
    _resolve_prior_class("normal")  # populate
    for k, v in _PRIOR_CLASSES.items():
        if v is prior_cls:
            return k
    raise ValueError(f"Unknown prior_class {prior_cls!r}")


def _model_class_name(model_cls: type) -> str:
    _resolve_model_class("normal_normal")  # populate
    for k, v in _MODEL_CLASSES.items():
        if v is model_cls:
            return k
    raise ValueError(f"Unknown model_class {model_cls!r}")


def _parse_eta_explore_box(raw: Any | None) -> tuple[float, float]:
    """Coerce a YAML/dict eta_explore_box entry to a (lo, hi) tuple."""
    if raw is None:
        return (-5.0, 5.0)
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise ValueError(
            f"eta_explore_box must be a 2-element list/tuple [lo, hi]; "
            f"got {raw!r}."
        )
    return (float(raw[0]), float(raw[1]))


@dataclass(frozen=True)
class ExperimentConfig:
    """Self-describing experiment for the conditional learned-η training loop (v4).

    Phase G change: prior + model are now CLASSES (not instances); the
    per-batch (prior_hp, lik_hp) are sampled from
    ``hyperparam_distribution`` during training. At inference, the
    LearnedDynamicEtaSelector extracts hyperparams from the (prior, model)
    instance passed at runtime and dispatches the conditional EtaNet.
    """

    scheme_name: str
    statistic_name: str
    prior_cls: type
    model_cls: type
    hyperparam_distribution: "HyperparamDistribution"
    theta_distribution: ThetaDistribution
    n_grid: int = 401
    n_lhs: int = 10000
    seed: int = 42
    name: str = ""
    description: str = ""
    n_data: int = 1
    eta_explore_box: tuple[float, float] = (-5.0, 5.0)
    # Absolute θ-grid bounds for the integrated-p loss + dynamic CI scan.
    # If both None, defaults to theta_distribution.support() (legacy
    # behavior; works only for non-anchored ThetaDistributions).
    # For anchored distributions (sigma_anchored_uniform), MUST be set
    # to cover the union of per-element θ ranges
    # (μ₀ ± K·σ₀ + buffer for integrated-p tails).
    theta_grid_lo: float | None = None
    theta_grid_hi: float | None = None

    def __post_init__(self) -> None:
        if self.scheme_name not in _registry.tiltings:
            raise ValueError(
                f"scheme_name {self.scheme_name!r} not in registry; "
                f"known: {list(_registry.tiltings)}"
            )
        if self.statistic_name not in _registry.statistics:
            raise ValueError(
                f"statistic_name {self.statistic_name!r} not in registry; "
                f"known: {list(_registry.statistics)}"
            )
        if self.n_grid < 3:
            raise ValueError(f"n_grid must be >= 3, got {self.n_grid}")
        if self.n_lhs < 20:
            raise ValueError(
                f"n_lhs must be >= 20 (training carves off 10% for "
                f"held-out validation; below this the loop trains on "
                f"essentially nothing); got {self.n_lhs}."
            )
        if self.n_data < 1:
            raise ValueError(
                f"n_data must be >= 1 (number of likelihood draws per θ "
                f"in the MC width loss); got {self.n_data}."
            )
        eta_lo, eta_hi = self.eta_explore_box
        if not (eta_lo < eta_hi):
            raise ValueError(
                f"eta_explore_box must satisfy lo < hi; got "
                f"({eta_lo}, {eta_hi})."
            )
        sup_lo, sup_hi = self.theta_distribution.support()
        if not (np.isfinite(sup_lo) and np.isfinite(sup_hi)):
            raise ValueError(
                f"theta_distribution.support() must be finite for the "
                f"learned-η training loop; got ({sup_lo}, {sup_hi})."
            )
        if getattr(self.theta_distribution, "is_anchored", False):
            if self.theta_grid_lo is None or self.theta_grid_hi is None:
                raise ValueError(
                    f"theta_distribution {self.theta_distribution.name!r} is "
                    "anchored to per-element prior; absolute theta_grid_lo "
                    "and theta_grid_hi MUST be set in the YAML to define the "
                    "absolute integration grid (cover μ₀ ± K·σ₀ + buffer)."
                )
        if self.theta_grid_lo is not None and self.theta_grid_hi is not None:
            if self.theta_grid_hi <= self.theta_grid_lo:
                raise ValueError(
                    f"theta_grid_hi must exceed theta_grid_lo; got "
                    f"({self.theta_grid_lo}, {self.theta_grid_hi})."
                )
        scheme = _registry.tiltings[self.scheme_name]()
        statistic = _registry.statistics[self.statistic_name]()
        if hasattr(statistic, "accepts_tilting") and not statistic.accepts_tilting(scheme):
            raise ValueError(
                f"statistic {self.statistic_name!r} does not accept "
                f"tilting {self.scheme_name!r}."
            )
        # n_data > 1 on NN is unsupported (closed-form pvalue port assumes
        # single observation D per θ).
        if self.n_data > 1 and self.model_cls.__name__ == "NormalNormalModel":
            raise ValueError(
                f"ExperimentConfig with NormalNormalModel requires "
                f"n_data == 1; got n_data={self.n_data}. Use a non-NN "
                f"model (e.g. BernoulliModel) if you need n_data > 1."
            )

    @cached_property
    def theta_grid(self) -> NDArray[np.float64]:
        """Canonical absolute θ-grid for dynamic-pvalue / integrated-p loss.

        Always returns absolute θ values (used by the loss integration and
        the dynamic CI scan). For non-anchored ThetaDistributions defaults
        to the distribution's support; for anchored ones uses the explicit
        ``theta_grid_lo``/``theta_grid_hi`` set in the YAML.
        """
        if self.theta_grid_lo is not None and self.theta_grid_hi is not None:
            return np.linspace(self.theta_grid_lo, self.theta_grid_hi, self.n_grid)
        lo, hi = self.theta_distribution.support()
        return np.linspace(lo, hi, self.n_grid)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable JSON-friendly serialisation."""
        return {
            "prior_class": _prior_class_name(self.prior_cls),
            "model_class": _model_class_name(self.model_cls),
            "hyperparam_distribution": self.hyperparam_distribution.to_dict(),
            "theta_distribution": self.theta_distribution.to_dict(),
            "scheme": self.scheme_name,
            "statistic": self.statistic_name,
            "n_grid": self.n_grid,
            "n_lhs": self.n_lhs,
            "seed": self.seed,
            "name": self.name,
            "description": self.description,
            "n_data": self.n_data,
            "eta_explore_box": list(self.eta_explore_box),
            "theta_grid_lo": self.theta_grid_lo,
            "theta_grid_hi": self.theta_grid_hi,
        }

    def fingerprint(self) -> str:
        """Stable hash of the config — used in the cache key + checkpoint
        cross-experiment guard."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExperimentConfig:
        from .hyperparam_distribution import HyperparamDistribution

        # Required v4 keys.
        for required in ("prior_class", "model_class", "hyperparam_distribution"):
            if required not in d:
                raise KeyError(
                    f"ExperimentConfig.from_dict missing required key {required!r}. "
                    f"v4 schema replaces v3's `prior:` / `model:` blocks with "
                    f"`prior_class:` + `model_class:` + `hyperparam_distribution:`. "
                    f"See docs/methods/learned_eta.md for the migration."
                )
        # Accept both v4 short keys (scheme/statistic) and v3 long keys
        # (scheme_name/statistic_name) for the scheme + statistic.
        scheme_name = d.get("scheme") or d.get("scheme_name")
        statistic_name = d.get("statistic") or d.get("statistic_name")
        if scheme_name is None or statistic_name is None:
            raise KeyError(
                "ExperimentConfig.from_dict requires `scheme` and `statistic` "
                "keys (or the v3 `scheme_name` / `statistic_name`)."
            )
        prior_cls = _resolve_prior_class(d["prior_class"])
        model_cls = _resolve_model_class(d["model_class"])
        hp_distr = HyperparamDistribution.from_dict(d["hyperparam_distribution"])
        return cls(
            scheme_name=str(scheme_name),
            statistic_name=str(statistic_name),
            prior_cls=prior_cls,
            model_cls=model_cls,
            hyperparam_distribution=hp_distr,
            theta_distribution=_build_theta_distribution_from_dict(d["theta_distribution"]),
            n_grid=int(d.get("n_grid", 401)),
            n_lhs=int(d.get("n_lhs", 10000)),
            seed=int(d.get("seed", 42)),
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            n_data=int(d.get("n_data", 1)),
            eta_explore_box=_parse_eta_explore_box(d.get("eta_explore_box")),
            theta_grid_lo=(
                float(d["theta_grid_lo"]) if d.get("theta_grid_lo") is not None else None
            ),
            theta_grid_hi=(
                float(d["theta_grid_hi"]) if d.get("theta_grid_hi") is not None else None
            ),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "ExperimentConfig.from_yaml requires PyYAML; install with `pip install pyyaml`."
            ) from e
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


def lhs_1d(
    theta_dist: ThetaDistribution,
    n: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """1D Latin Hypercube Sample on ``theta_dist``'s support."""
    lo, hi = theta_dist.support()
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"lhs_1d requires a finitely-supported theta_dist; got ({lo}, {hi}).")
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n=n).reshape(-1)
    out: NDArray[np.float64] = (lo + u * (hi - lo)).astype(np.float64)
    return out


__all__ = [
    "ThetaDistribution",
    "UniformThetaDistribution",
    "Sigma0AnchoredUniformThetaDistribution",
    # Backwards-compat alias for the canonical name (renamed 2026-05-10).
    "SigmaAnchoredUniformThetaDistribution",
    "anchor_theta_to_prior",
    "ExperimentConfig",
    "lhs_1d",
    "THETA_DISTRIBUTION_REGISTRY",
]
