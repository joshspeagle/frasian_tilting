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


THETA_DISTRIBUTION_REGISTRY: dict[str, Any] = {
    "uniform": UniformThetaDistribution,
}

_THETA_DIST_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    "uniform": frozenset({"low", "high"}),
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
        sup_lo, sup_hi = self.theta_distribution.support()
        if not (np.isfinite(sup_lo) and np.isfinite(sup_hi)):
            raise ValueError(
                f"theta_distribution.support() must be finite for the "
                f"learned-η training loop; got ({sup_lo}, {sup_hi})."
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
        """Canonical grid for dynamic-pvalue evaluation + CI inversion."""
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
    "ExperimentConfig",
    "lhs_1d",
    "THETA_DISTRIBUTION_REGISTRY",
]
