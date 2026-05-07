"""Phase E experiment-config + sampling primitives.

- ``ThetaDistribution`` — protocol for any 1D distribution over θ
  exposing ``sample(n, rng)``, ``support()``, and ``fingerprint()``.
- ``UniformThetaDistribution`` — concrete uniform-on-[low, high].
- ``THETA_DISTRIBUTION_REGISTRY`` — name → class lookup for YAML.
- ``ExperimentConfig`` — frozen dataclass binding (scheme, statistic,
  prior, model, theta_distribution) into a single self-describing
  object that drives both training and selector validation. Round-
  trips through YAML and embeds in checkpoints.
- ``lhs_1d(theta_dist, n, seed)`` — 1D Latin Hypercube Sampling on a
  ``ThetaDistribution``'s support. One-shot stratified sample at
  training start.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.stats import qmc

from ..._registry import registry as _registry
from ...models.base import Model, Prior


@runtime_checkable
class ThetaDistribution(Protocol):
    """A distribution over θ-space.

    Used both as the source of training θ samples (LHS at startup,
    i.i.d. for boundary-probing aux samples) and as the bounds of
    the canonical inversion grid in ``ExperimentConfig.theta_grid``.
    """

    name: str

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]: ...
    def support(self) -> tuple[float, float]: ...
    def fingerprint(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class UniformThetaDistribution:
    """Uniform(low, high) over θ.

    `name` is a class-level constant (not a kwarg) so a constructed
    instance cannot lie about its identity past the fingerprint check.
    """

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


# Local registry mapping YAML "type" strings to ThetaDistribution
# constructors. (Priors / models go through their own dispatch in
# `_build_*_from_dict`; consolidating those into a registry too is a
# follow-up — for now the dispatch is small enough to be inline.)
THETA_DISTRIBUTION_REGISTRY: dict[str, Any] = {
    "uniform": UniformThetaDistribution,
}


_PRIOR_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    "normal": frozenset({"loc", "scale"}),
    "beta": frozenset({"alpha", "beta"}),
}
_MODEL_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    # Class-level identifiers (`name`, `param_dim`) are not overridable
    # from YAML — only true instance state is accepted.
    "normal_normal": frozenset({"sigma"}),
    "bernoulli": frozenset(),
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


def _build_prior_from_dict(d: Mapping[str, Any]) -> Prior:
    """Construct a Prior from a YAML dict ``{"type": ..., kwargs}``."""
    spec = dict(d)
    type_ = spec.pop("type")
    if type_ not in _PRIOR_ALLOWED_KWARGS:
        raise ValueError(
            f"Unknown prior type {type_!r}; " f"expected one of: {sorted(_PRIOR_ALLOWED_KWARGS)}"
        )
    kwargs = _filter_kwargs(spec, _PRIOR_ALLOWED_KWARGS[type_], type_, "prior")
    if type_ == "normal":
        from ...models.distributions import NormalDistribution

        return NormalDistribution(**kwargs)
    if type_ == "beta":
        from ...models.distributions import BetaDistribution

        return BetaDistribution(**kwargs)
    raise AssertionError("unreachable")  # registry-checked above


def _build_model_from_dict(d: Mapping[str, Any]) -> Model:
    """Construct a Model from a YAML dict ``{"type": ..., kwargs}``."""
    spec = dict(d)
    type_ = spec.pop("type")
    if type_ not in _MODEL_ALLOWED_KWARGS:
        raise ValueError(
            f"Unknown model type {type_!r}; " f"expected one of: {sorted(_MODEL_ALLOWED_KWARGS)}"
        )
    kwargs = _filter_kwargs(spec, _MODEL_ALLOWED_KWARGS[type_], type_, "model")
    if type_ == "normal_normal":
        from ...models.normal_normal import NormalNormalModel

        return NormalNormalModel(**kwargs)
    if type_ == "bernoulli":
        from ...models.bernoulli import BernoulliModel

        return BernoulliModel(**kwargs)
    raise AssertionError("unreachable")


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


def _prior_to_dict(prior: Prior) -> dict[str, Any]:
    fp = prior.fingerprint()
    if fp[0] == "normal":
        return {"type": "normal", "loc": fp[1], "scale": fp[2]}
    if fp[0] == "beta":
        return {"type": "beta", "alpha": fp[1], "beta": fp[2]}
    raise ValueError(f"Cannot serialise prior with fingerprint {fp!r}")


def _model_to_dict(model: Model) -> dict[str, Any]:
    fp = model.fingerprint()
    if fp[0] == "normal_normal":
        return {"type": "normal_normal", "sigma": fp[1]}
    if fp[0] == "bernoulli":
        return {"type": "bernoulli"}
    raise ValueError(f"Cannot serialise model with fingerprint {fp!r}")


def _theta_distribution_to_dict(td: ThetaDistribution) -> dict[str, Any]:
    fp = td.fingerprint()
    if fp[0] == "uniform":
        return {"type": "uniform", "low": fp[1], "high": fp[2]}
    raise ValueError(f"Cannot serialise theta_distribution with fingerprint {fp!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Self-describing experiment for the learned-η training loop.

    Binds a tilting scheme, test statistic, prior, model, and
    θ-distribution into a single object. Drives both the training
    loop (no scheme-specific code paths) and the selector's
    inference-time experiment-match check (fingerprints embedded
    in the checkpoint).
    """

    scheme_name: str
    statistic_name: str
    prior: Prior
    model: Model
    theta_distribution: ThetaDistribution
    n_grid: int = 401
    n_lhs: int = 10000
    eta_explore_box: tuple[float, float] = (-5.0, 5.0)
    seed: int = 42
    name: str = ""
    description: str = ""
    # Number of likelihood draws ``D`` per θ used in the training-time
    # MC width-loss sample. Default 1 preserves byte-equality with the
    # pre-Phase-4c Normal-Normal pipeline (where each θ in the
    # minibatch is paired with one D draw). For non-Normal models the
    # likelihood from a single observation is too diffuse to yield a
    # discriminating learned-η selector — Bernoulli + Beta in
    # particular requires ``n_data`` of order ~16-64. The setting is
    # part of the ExperimentConfig fingerprint round-trip so the
    # selector can refuse a checkpoint trained at a different ``n_data``.
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
        # n_lhs minimum: the training loop carves off ~10% for held-out
        # validation, then iterates batch_size-sized minibatches. Below
        # ~20 LHS samples the held-out set is degenerate and Head A
        # trains on essentially nothing — refuse loudly.
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
        lo, hi = self.eta_explore_box
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(
                f"eta_explore_box must be a finite (low, high) with "
                f"high > low; got {self.eta_explore_box}"
            )
        # Validate theta_distribution support is finite at construct
        # time, not lazily on first theta_grid access.
        sup_lo, sup_hi = self.theta_distribution.support()
        if not (np.isfinite(sup_lo) and np.isfinite(sup_hi)):
            raise ValueError(
                f"theta_distribution.support() must be finite for the "
                f"learned-η training loop; got ({sup_lo}, {sup_hi}). "
                f"Use a compactly supported θ distribution (e.g., "
                f"UniformThetaDistribution)."
            )
        # Compatibility: refuse incompatible (scheme, statistic) cells
        # up front rather than failing mid-training.
        scheme = _registry.tiltings[self.scheme_name]()
        statistic = _registry.statistics[self.statistic_name]()
        if hasattr(statistic, "accepts_tilting") and not statistic.accepts_tilting(scheme):
            raise ValueError(
                f"statistic {self.statistic_name!r} does not accept "
                f"tilting {self.scheme_name!r}. Pair the scheme with a "
                f"compatible statistic (e.g., 'waldo')."
            )

    @cached_property
    def theta_grid(self) -> NDArray[np.float64]:
        """Canonical grid for dynamic-pvalue evaluation + CI inversion."""
        lo, hi = self.theta_distribution.support()
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                f"theta_distribution.support() must be finite for the "
                f"inversion grid; got ({lo}, {hi}). Provide a "
                f"compactly supported θ distribution."
            )
        return np.linspace(lo, hi, self.n_grid)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable JSON-friendly serialisation.

        Includes both the constructor kwargs (so ``from_dict`` can
        rebuild the object) and the fingerprints (so callers reading
        a saved config can compare against in-memory objects without
        rebuilding them).
        """
        return {
            "scheme_name": self.scheme_name,
            "statistic_name": self.statistic_name,
            "prior": _prior_to_dict(self.prior),
            "model": _model_to_dict(self.model),
            "theta_distribution": _theta_distribution_to_dict(self.theta_distribution),
            "n_grid": self.n_grid,
            "n_lhs": self.n_lhs,
            "n_data": self.n_data,
            "eta_explore_box": list(self.eta_explore_box),
            "seed": self.seed,
            "name": self.name,
            "description": self.description,
            # Convenience fingerprints for selector validation.
            "prior_fingerprint": list(self.prior.fingerprint()),
            "model_fingerprint": list(self.model.fingerprint()),
            "theta_distribution_fingerprint": list(self.theta_distribution.fingerprint()),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExperimentConfig:
        # Strict allowlist on top-level keys so YAML typos (e.g. ``n_gird``)
        # surface as a loud error instead of being silently dropped to a
        # default. Fingerprint keys produced by ``to_dict`` are also
        # accepted (and ignored on reconstruction — they're round-trip
        # convenience metadata, not constructor inputs).
        allowed = {
            "scheme_name",
            "statistic_name",
            "prior",
            "model",
            "theta_distribution",
            "n_grid",
            "n_lhs",
            "n_data",
            "eta_explore_box",
            "seed",
            "name",
            "description",
            "prior_fingerprint",
            "model_fingerprint",
            "theta_distribution_fingerprint",
        }
        extras = set(d.keys()) - allowed
        if extras:
            raise ValueError(
                f"Unexpected ExperimentConfig keys: {sorted(extras)} "
                f"(allowed: {sorted(allowed)})"
            )
        return cls(
            scheme_name=str(d["scheme_name"]),
            statistic_name=str(d["statistic_name"]),
            prior=_build_prior_from_dict(d["prior"]),
            model=_build_model_from_dict(d["model"]),
            theta_distribution=_build_theta_distribution_from_dict(d["theta_distribution"]),
            n_grid=int(d.get("n_grid", 401)),
            n_lhs=int(d.get("n_lhs", 10000)),
            n_data=int(d.get("n_data", 1)),
            eta_explore_box=tuple(d.get("eta_explore_box", (-5.0, 5.0))),
            seed=int(d.get("seed", 42)),
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "ExperimentConfig.from_yaml requires PyYAML; " "install with `pip install pyyaml`."
            ) from e
        with open(path) as f:
            d = yaml.safe_load(f)
        # Allow YAML to use "scheme" / "statistic" as shorthand.
        if "scheme" in d and "scheme_name" not in d:
            d["scheme_name"] = d.pop("scheme")
        if "statistic" in d and "statistic_name" not in d:
            d["statistic_name"] = d.pop("statistic")
        return cls.from_dict(d)


def lhs_1d(
    theta_dist: ThetaDistribution,
    n: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """1D Latin Hypercube Sample on ``theta_dist``'s support.

    One-shot stratified sample at training start; aux samples
    elsewhere in the loop are drawn i.i.d. via
    ``theta_dist.sample(n, rng)``.
    """
    lo, hi = theta_dist.support()
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"lhs_1d requires a finitely-supported theta_dist; got ({lo}, {hi}).")
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n=n).reshape(-1)  # (n,)
    out: NDArray[np.float64] = (lo + u * (hi - lo)).astype(np.float64)
    return out


__all__ = [
    "ThetaDistribution",
    "UniformThetaDistribution",
    "ExperimentConfig",
    "lhs_1d",
    "THETA_DISTRIBUTION_REGISTRY",
]
