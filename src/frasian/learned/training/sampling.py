"""Training distribution and experiment-config primitives.

Phase E adds:

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

The legacy ``TrainingDistribution`` / ``lhs_sample`` /
``draw_data_batch`` continue to live here while ``train.py`` still
references them; they are removed in E.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.stats import qmc

from ..._registry import registry as _registry
from ...models.base import Model, Prior


@dataclass(frozen=True)
class TrainingDistribution:
    """Configurable π over (w, θ_true) for the Normal-Normal sandbox.

    Attributes
    ----------
    w_range
        `(w_min, w_max)` — uniform sampling interval.
    theta_true_half_width
        Half-width of `θ_true ~ Uniform(μ₀ - X·σ, μ₀ + X·σ)`.
        Default 10 covers the conflict band and asymptotic regime.
    mu0
        Prior mean (default 0.0).
    sigma
        Likelihood std (default 1.0).
    """

    w_range: Tuple[float, float] = (0.05, 0.95)
    theta_true_half_width: float = 5.0
    mu0: float = 0.0
    sigma: float = 1.0

    @classmethod
    def normal_normal_default(cls) -> "TrainingDistribution":
        """Canonical default: `Uniform(0.05, 0.95)` × `Uniform(±5σ)`."""
        return cls()

    def to_dict(self) -> dict:
        """Serialise for embedding in a checkpoint."""
        return {
            "w_range": list(self.w_range),
            "theta_true_half_width": self.theta_true_half_width,
            "mu0": self.mu0,
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingDistribution":
        return cls(
            w_range=tuple(d["w_range"]),
            theta_true_half_width=d["theta_true_half_width"],
            mu0=d["mu0"],
            sigma=d["sigma"],
        )


def lhs_sample(
    distribution: TrainingDistribution,
    n_lhs: int,
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Latin-Hypercube sample (w, θ_true) from `distribution`.

    Returns `(w_arr, theta_true_arr)` each of shape `(n_lhs,)`.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n=n_lhs)                                # (n_lhs, 2) in [0, 1)

    w_lo, w_hi = distribution.w_range
    w = w_lo + u[:, 0] * (w_hi - w_lo)

    half = distribution.theta_true_half_width * distribution.sigma
    theta_true = (distribution.mu0 - half) + u[:, 1] * 2.0 * half

    return w.astype(np.float64), theta_true.astype(np.float64)


def draw_data_batch(
    distribution: TrainingDistribution,
    w_batch: NDArray[np.float64],
    theta_true_batch: NDArray[np.float64],
    n_mc: int,
    rng: np.random.Generator,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Generate `(D, w, theta_true)` torch tensors for a mini-batch.

    For each `(w_i, θ_i)` in the input batch, draws `n_mc` independent
    `D_ij ~ N(θ_i, σ²)`. The returned tensors are flattened so each
    row of the batch is one `(D, w, θ_true)` tuple — simpler for the
    training loop's loss computation.

    Returns a dict with keys:
      D:          (B*n_mc,) tensor
      w:          (B*n_mc,) tensor
      theta_true: (B*n_mc,) tensor
      mu0, sigma: scalar tensors

    where B = len(w_batch).
    """
    B = len(w_batch)
    assert len(theta_true_batch) == B, "w and theta_true batch sizes must match"
    sigma = distribution.sigma
    mu0 = distribution.mu0

    # Each (w_i, theta_i) gets n_mc D draws.
    D = rng.normal(
        loc=theta_true_batch.repeat(n_mc),
        scale=sigma,
        size=B * n_mc,
    )
    w_full = np.tile(w_batch, n_mc)
    theta_full = np.tile(theta_true_batch, n_mc)

    return {
        "D": torch.tensor(D, dtype=dtype, device=device),
        "w": torch.tensor(w_full, dtype=dtype, device=device),
        "theta_true": torch.tensor(theta_full, dtype=dtype, device=device),
        "mu0": torch.tensor(mu0, dtype=dtype, device=device),
        "sigma": torch.tensor(sigma, dtype=dtype, device=device),
    }


# ---------------------------------------------------------------------------
# Phase E: ThetaDistribution protocol, ExperimentConfig, factory registry
# ---------------------------------------------------------------------------


@runtime_checkable
class ThetaDistribution(Protocol):
    """A distribution over θ-space.

    Used both as the source of training θ samples (LHS at startup,
    i.i.d. for boundary-probing aux samples) and as the bounds of
    the canonical inversion grid in ``ExperimentConfig.theta_grid``.
    """

    name: str

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]: ...
    def support(self) -> Tuple[float, float]: ...
    def fingerprint(self) -> tuple: ...


@dataclass(frozen=True)
class UniformThetaDistribution:
    """Uniform(low, high) over θ."""

    low: float
    high: float
    name: str = "uniform"

    def __post_init__(self) -> None:
        if not (np.isfinite(self.low) and np.isfinite(self.high)):
            raise ValueError(
                f"low and high must be finite; got ({self.low}, {self.high})"
            )
        if self.high <= self.low:
            raise ValueError(
                f"high must exceed low; got ({self.low}, {self.high})"
            )

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]:
        return rng.uniform(low=self.low, high=self.high, size=n).astype(
            np.float64
        )

    def support(self) -> Tuple[float, float]:
        return (float(self.low), float(self.high))

    def fingerprint(self) -> tuple:
        return ("uniform", float(self.low), float(self.high))


# Factory registries mapping YAML "type" strings to constructors.
# The framework's main `_registry` already names tilting schemes,
# statistics, and models by their decorator; we add a small local
# registry for priors and θ-distributions which are passed by
# parameters rather than registered via decorator today.
PRIOR_REGISTRY: dict[str, Any] = {}
MODEL_REGISTRY: dict[str, Any] = {}
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
            f"Unknown prior type {type_!r}; "
            f"expected one of: {sorted(_PRIOR_ALLOWED_KWARGS)}"
        )
    kwargs = _filter_kwargs(
        spec, _PRIOR_ALLOWED_KWARGS[type_], type_, "prior"
    )
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
            f"Unknown model type {type_!r}; "
            f"expected one of: {sorted(_MODEL_ALLOWED_KWARGS)}"
        )
    kwargs = _filter_kwargs(
        spec, _MODEL_ALLOWED_KWARGS[type_], type_, "model"
    )
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
    return cls(**kwargs)


def _prior_to_dict(prior: Prior) -> dict:
    fp = prior.fingerprint()
    if fp[0] == "normal":
        return {"type": "normal", "loc": fp[1], "scale": fp[2]}
    if fp[0] == "beta":
        return {"type": "beta", "alpha": fp[1], "beta": fp[2]}
    raise ValueError(f"Cannot serialise prior with fingerprint {fp!r}")


def _model_to_dict(model: Model) -> dict:
    fp = model.fingerprint()
    if fp[0] == "normal_normal":
        return {"type": "normal_normal", "sigma": fp[1]}
    if fp[0] == "bernoulli":
        return {"type": "bernoulli"}
    raise ValueError(f"Cannot serialise model with fingerprint {fp!r}")


def _theta_distribution_to_dict(td: ThetaDistribution) -> dict:
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
    eta_explore_box: Tuple[float, float] = (-5.0, 5.0)
    seed: int = 42
    name: str = ""
    description: str = ""

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
        if self.n_lhs < 1:
            raise ValueError(f"n_lhs must be >= 1, got {self.n_lhs}")
        lo, hi = self.eta_explore_box
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(
                f"eta_explore_box must be a finite (low, high) with "
                f"high > low; got {self.eta_explore_box}"
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

    def to_dict(self) -> dict:
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
            "theta_distribution": _theta_distribution_to_dict(
                self.theta_distribution
            ),
            "n_grid": self.n_grid,
            "n_lhs": self.n_lhs,
            "eta_explore_box": list(self.eta_explore_box),
            "seed": self.seed,
            "name": self.name,
            "description": self.description,
            # Convenience fingerprints for selector validation.
            "prior_fingerprint": list(self.prior.fingerprint()),
            "model_fingerprint": list(self.model.fingerprint()),
            "theta_distribution_fingerprint": list(
                self.theta_distribution.fingerprint()
            ),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            scheme_name=str(d["scheme_name"]),
            statistic_name=str(d["statistic_name"]),
            prior=_build_prior_from_dict(d["prior"]),
            model=_build_model_from_dict(d["model"]),
            theta_distribution=_build_theta_distribution_from_dict(
                d["theta_distribution"]
            ),
            n_grid=int(d.get("n_grid", 401)),
            n_lhs=int(d.get("n_lhs", 10000)),
            eta_explore_box=tuple(d.get("eta_explore_box", (-5.0, 5.0))),
            seed=int(d.get("seed", 42)),
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "ExperimentConfig.from_yaml requires PyYAML; "
                "install with `pip install pyyaml`."
            ) from e
        with open(path, "r") as f:
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
        raise ValueError(
            f"lhs_1d requires a finitely-supported theta_dist; got ({lo}, {hi})."
        )
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n=n).reshape(-1)                          # (n,)
    return (lo + u * (hi - lo)).astype(np.float64)


__all__ = [
    # Phase E additions
    "ThetaDistribution",
    "UniformThetaDistribution",
    "ExperimentConfig",
    "lhs_1d",
    "PRIOR_REGISTRY",
    "MODEL_REGISTRY",
    "THETA_DISTRIBUTION_REGISTRY",
    # Legacy (still imported by train.py until E.2 cuts it over)
    "TrainingDistribution",
    "lhs_sample",
    "draw_data_batch",
]
