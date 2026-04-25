"""Single source of truth for framework-level constants.

Everything that used to be a magic number in the legacy code (alpha, grid sizes,
eps, seeds, eta-buffer) is read from a frozen `Config` instance. Tests override
explicitly via `Config.from_overrides(...)` rather than monkey-patching globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Specification for a parameter sweep grid."""

    name: str
    low: float
    high: float
    n_points: int
    log_spaced: bool = False

    def to_array(self) -> np.ndarray:
        if self.log_spaced:
            return np.logspace(np.log10(self.low), np.log10(self.high), self.n_points)
        return np.linspace(self.low, self.high, self.n_points)


@dataclass(frozen=True)
class Config:
    """Framework-wide configuration.

    Frozen by design. Any test or script that needs a different value calls
    `Config.from_overrides(alpha=0.10)` to get a derived instance.
    """

    # --- statistical ---
    alpha: float = 0.05
    seed: int = 0
    n_reps: int = 10_000

    # --- numerics ---
    eta_min_buffer: float = 1e-3
    brentq_xtol: float = 1e-9
    brentq_rtol: float = 1e-9
    brentq_maxiter: int = 200
    bracket_doubling_max: int = 16

    # --- grids (defaults; experiments override) ---
    delta_grid: GridSpec = field(
        default_factory=lambda: GridSpec("abs_delta", 0.0, 5.0, 51)
    )
    w_grid: GridSpec = field(
        default_factory=lambda: GridSpec("w", 0.05, 0.95, 19)
    )
    theta_grid: GridSpec = field(
        default_factory=lambda: GridSpec("theta", -4.0, 6.0, 21)
    )

    # --- IO ---
    cache_enabled: bool = True

    @classmethod
    def default(cls) -> "Config":
        return cls()

    @classmethod
    def fast(cls) -> "Config":
        """Smoke-mode config: small enough to run end-to-end in seconds.

        Use this for `--fast` CLI flag, examples, and sanity checks. The
        L3 statistical layer (KS / coverage at nominal level) requires
        `n_reps >= 1000`; tests that need that ask for it explicitly via
        `Config.fast().from_overrides(n_reps=...)`.
        """
        return cls(
            n_reps=200,
            delta_grid=GridSpec("abs_delta", 0.0, 5.0, 11),
            w_grid=GridSpec("w", 0.2, 0.8, 5),
            theta_grid=GridSpec("theta", -3.0, 4.0, 8),
        )

    def from_overrides(self, **overrides) -> "Config":
        return replace(self, **overrides)

    def fingerprint(self) -> str:
        """Deterministic short hash used for cache invalidation."""
        import hashlib
        import json

        payload = json.dumps(self._serializable(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _serializable(self) -> Mapping[str, object]:
        return {
            "alpha": self.alpha,
            "seed": self.seed,
            "n_reps": self.n_reps,
            "eta_min_buffer": self.eta_min_buffer,
            "brentq_xtol": self.brentq_xtol,
            "brentq_rtol": self.brentq_rtol,
            "brentq_maxiter": self.brentq_maxiter,
            "bracket_doubling_max": self.bracket_doubling_max,
            "delta_grid": self.delta_grid.__dict__,
            "w_grid": self.w_grid.__dict__,
            "theta_grid": self.theta_grid.__dict__,
            "cache_enabled": self.cache_enabled,
        }
