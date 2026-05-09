"""HyperparamDistribution — per-batch sampler for the Phase G
conditional learned-η training loop.

Each training step samples (μ₀, σ₀, σ) per batch element from this
distribution; the network learns η_φ(θ | μ₀, σ₀, σ) as a conditional
function. At inference, the cross-experiment guard refuses any
(prior_hp, lik_hp) outside the trained `support()`.

Two distribution kinds supported per scalar hyperparameter:
  * `uniform(low, high)` — `rng.uniform(low, high)`.
  * `loguniform(low, high)` — `exp(rng.uniform(log low, log high))`.
    Natural for positive scale-like quantities (σ₀, σ, α, β).

YAML schema:
    hyperparam_distribution:
      prior:
        loc:   { dist: uniform,    low: -2.0, high: 2.0 }
        scale: { dist: loguniform, low: 0.2,  high: 5.0 }
      lik:
        sigma: { dist: loguniform, low: 0.5,  high: 2.0 }
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


_VALID_KINDS = ("uniform", "loguniform")


@dataclass(frozen=True)
class ScalarDist:
    """Per-hyperparam sampling spec.

    Inclusive endpoints on `in_range`: `low ≤ x ≤ high` is accepted.
    """

    kind: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"ScalarDist kind must be one of {_VALID_KINDS}; got {self.kind!r}."
            )
        if not (self.low < self.high):
            raise ValueError(
                f"ScalarDist requires low < high; got {self.low!r}, {self.high!r}."
            )
        if self.kind == "loguniform" and self.low <= 0.0:
            raise ValueError(
                f"loguniform requires low > 0; got {self.low!r}."
            )

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]:
        if self.kind == "uniform":
            return rng.uniform(self.low, self.high, size=int(n)).astype(np.float64)
        log_lo, log_hi = float(np.log(self.low)), float(np.log(self.high))
        return np.exp(rng.uniform(log_lo, log_hi, size=int(n))).astype(np.float64)

    def in_range(self, x: float) -> bool:
        return float(self.low) <= float(x) <= float(self.high)

    def feature_stats(self) -> tuple[float, float, bool]:
        """Return `(loc, scale, log_flag)` for input z-score normalization.

        For ``uniform``: linear-space mean and std of U(low, high) →
        ``loc = (low+high)/2``, ``scale = (high-low)/sqrt(12)``, log=False.
        For ``loguniform``: log-space mean and std of LogU(low, high) →
        ``loc = (log low + log high)/2``, ``scale = (log high - log low)/sqrt(12)``,
        log=True. The Phase G EtaNet/ValidityNet apply
        ``x' = (log(x) if log else x - loc) / scale`` per feature.
        """
        if self.kind == "uniform":
            loc = 0.5 * (self.low + self.high)
            scale = (self.high - self.low) / float(np.sqrt(12.0))
            return float(loc), float(scale), False
        log_lo = float(np.log(self.low))
        log_hi = float(np.log(self.high))
        loc = 0.5 * (log_lo + log_hi)
        scale = (log_hi - log_lo) / float(np.sqrt(12.0))
        return float(loc), float(scale), True


@dataclass(frozen=True)
class ScalarOutOfRange:
    """Diagnostic record for `first_out_of_range`."""

    name: str
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class HyperparamDistribution:
    """Per-batch joint sampler for (prior_hp, lik_hp).

    Keys in `prior_specs` MUST match `Prior.hyperparam_names()` exactly.
    Same for `lik_specs` and `Model.hyperparam_names()`. Sample columns
    are emitted in `hyperparam_names()` order.
    """

    prior_specs: Mapping[str, ScalarDist]
    lik_specs: Mapping[str, ScalarDist]

    def sample(
        self,
        n: int,
        rng: Generator,
        *,
        prior_names: tuple[str, ...],
        lik_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns `(prior_hp_batch, lik_hp_batch)` shapes
        `(n, len(prior_names))`, `(n, len(lik_names))`."""
        self._validate_names(prior_names, lik_names)
        prior_cols = [self.prior_specs[name].sample(n, rng) for name in prior_names]
        lik_cols = [self.lik_specs[name].sample(n, rng) for name in lik_names]
        prior_b = (
            np.stack(prior_cols, axis=-1)
            if prior_cols else np.empty((int(n), 0), dtype=np.float64)
        )
        lik_b = (
            np.stack(lik_cols, axis=-1)
            if lik_cols else np.empty((int(n), 0), dtype=np.float64)
        )
        return prior_b, lik_b

    def support(self) -> tuple[
        dict[str, tuple[float, float]],
        dict[str, tuple[float, float]],
    ]:
        prior_supp = {n: (s.low, s.high) for n, s in self.prior_specs.items()}
        lik_supp = {n: (s.low, s.high) for n, s in self.lik_specs.items()}
        return prior_supp, lik_supp

    def feature_stats(
        self,
        prior_names: tuple[str, ...],
        lik_names: tuple[str, ...],
    ) -> tuple[list[float], list[float], list[bool]]:
        """Per-column ``(loc, scale, log_flag)`` for input normalization.

        Concatenates ``prior_specs[name]`` for ``prior_names`` then
        ``lik_specs[name]`` for ``lik_names``, in that order.
        """
        self._validate_names(prior_names, lik_names)
        locs: list[float] = []
        scales: list[float] = []
        logs: list[bool] = []
        for name in prior_names:
            loc, scale, log_flag = self.prior_specs[name].feature_stats()
            locs.append(loc); scales.append(scale); logs.append(log_flag)
        for name in lik_names:
            loc, scale, log_flag = self.lik_specs[name].feature_stats()
            locs.append(loc); scales.append(scale); logs.append(log_flag)
        return locs, scales, logs

    def in_range(
        self,
        prior_hp: NDArray[np.float64],
        lik_hp: NDArray[np.float64],
        prior_names: tuple[str, ...],
        lik_names: tuple[str, ...],
    ) -> bool:
        return self.first_out_of_range(prior_hp, lik_hp, prior_names, lik_names) is None

    def first_out_of_range(
        self,
        prior_hp: NDArray[np.float64],
        lik_hp: NDArray[np.float64],
        prior_names: tuple[str, ...],
        lik_names: tuple[str, ...],
    ) -> ScalarOutOfRange | None:
        self._validate_names(prior_names, lik_names)
        prior_arr = np.atleast_1d(np.asarray(prior_hp, dtype=np.float64))
        lik_arr = np.atleast_1d(np.asarray(lik_hp, dtype=np.float64))
        for i, name in enumerate(prior_names):
            spec = self.prior_specs[name]
            if not spec.in_range(prior_arr[i]):
                return ScalarOutOfRange(
                    name=name, value=float(prior_arr[i]),
                    low=spec.low, high=spec.high,
                )
        for j, name in enumerate(lik_names):
            spec = self.lik_specs[name]
            if not spec.in_range(lik_arr[j]):
                return ScalarOutOfRange(
                    name=name, value=float(lik_arr[j]),
                    low=spec.low, high=spec.high,
                )
        return None

    def fingerprint(self) -> str:
        """Stable hash for cache invalidation. 16 hex chars."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior": {n: {"dist": s.kind, "low": s.low, "high": s.high}
                       for n, s in self.prior_specs.items()},
            "lik":   {n: {"dist": s.kind, "low": s.low, "high": s.high}
                       for n, s in self.lik_specs.items()},
        }

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> "HyperparamDistribution":
        if "prior" not in spec or "lik" not in spec:
            raise ValueError(
                f"HyperparamDistribution.from_dict expects keys 'prior' "
                f"and 'lik'; got {sorted(spec)}."
            )
        return cls(
            prior_specs={n: cls._scalar_from_dict(d) for n, d in spec["prior"].items()},
            lik_specs={n: cls._scalar_from_dict(d) for n, d in spec["lik"].items()},
        )

    @staticmethod
    def _scalar_from_dict(d: Mapping[str, Any]) -> ScalarDist:
        return ScalarDist(kind=str(d["dist"]), low=float(d["low"]), high=float(d["high"]))

    def _validate_names(
        self, prior_names: tuple[str, ...], lik_names: tuple[str, ...],
    ) -> None:
        prior_keys = set(self.prior_specs.keys())
        if prior_keys != set(prior_names):
            raise ValueError(
                f"HyperparamDistribution.prior_specs keys {sorted(prior_keys)} "
                f"don't match prior_names {sorted(prior_names)}."
            )
        lik_keys = set(self.lik_specs.keys())
        if lik_keys != set(lik_names):
            raise ValueError(
                f"HyperparamDistribution.lik_specs keys {sorted(lik_keys)} "
                f"don't match lik_names {sorted(lik_names)}."
            )


__all__ = ["ScalarDist", "ScalarOutOfRange", "HyperparamDistribution"]
