"""Single source of truth for framework-level constants.

Everything that used to be a magic number in the legacy code (alpha, grid sizes,
eps, seeds, eta-buffer) is read from a frozen `Config` instance. Tests override
explicitly via `Config.from_overrides(...)` rather than monkey-patching globals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

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
        """Materialise the grid: linspace by default, logspace if `log_spaced`."""
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

    # --- dynamic CI scan params (consumed by tilting._dynamic.dynamic_ci_scan) ---
    # Live in Config so Config.fingerprint() invalidates the cache when
    # they change (Tier 1.7-C2 in the audit). Function-default fall-backs
    # in dynamic_ci_scan match these values for backward compatibility.
    dynamic_n_grid: int = 401
    dynamic_coarse_n: int = 25
    dynamic_search_mult: float = 8.0

    # --- grids (defaults; experiments override) ---
    delta_grid: GridSpec = field(default_factory=lambda: GridSpec("abs_delta", 0.0, 5.0, 51))
    w_grid: GridSpec = field(default_factory=lambda: GridSpec("w", 0.05, 0.95, 19))
    theta_grid: GridSpec = field(default_factory=lambda: GridSpec("theta", -4.0, 6.0, 21))

    # --- IO ---
    cache_enabled: bool = True

    # --- Parallelism ---
    # Workers used by the per-replicate `parallel_map` inside
    # coverage / width / confidence_distribution. Default `1` keeps the
    # serial behaviour byte-reproducible. `-1` uses all cores. Process-
    # based workers (joblib loky) so JAX state stays per-worker; first
    # dispatch per worker pays a ~1-2 s import + JAX trace cost, so set
    # `n_jobs > 1` only when the per-replicate work is large enough to
    # amortise that (rule of thumb: per-CI cost > ~50 ms; safe for
    # generic-MC WALDO and any OT-tilted cell, wasteful for closed-form
    # Wald / WALDO).
    #
    # Fingerprint deliberately EXCLUDES n_jobs — the same Config with
    # n_jobs=1 and n_jobs=8 must hit the same cache slot since the
    # numerical result is identical (per-replicate D values come from a
    # pre-generated raw stream; parallelism reorders the *evaluation*
    # not the *seeding*). Verified by `test_parallelism_bitwise.py`.
    n_jobs: int = 1

    @classmethod
    def default(cls) -> Config:
        """Production-resolution config (n_reps=10_000, full grids)."""
        return cls()

    @classmethod
    def fast(cls) -> Config:
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

    def from_overrides(self, **overrides) -> Config:
        """Return a derived `Config` with the named fields replaced.

        The fingerprint changes accordingly, so cached results computed
        under the original config are invalidated automatically.
        """
        return replace(self, **overrides)

    def fingerprint(self) -> str:
        """Deterministic short hash used for cache invalidation.

        Audit P1 L.6: includes major.minor versions of `numpy` and
        `jax` so an environment upgrade (e.g. `jax 0.4.x` → `jax 0.5.x`,
        which changed PRNG semantics on some operations) invalidates
        the cache. Patch versions are excluded because they're
        ABI-stable by SemVer convention; if a patch release ever
        changes numerical output, callers can `--force` to bypass.
        """
        import hashlib
        import json

        payload = json.dumps(self._serializable(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _major_minor(version_str: str) -> str:
        """Return `"major.minor"` from a `"major.minor.patch[...]"` string."""
        parts = version_str.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else version_str

    def _serializable(self) -> Mapping[str, object]:
        # Audit P1 L.6: include numpy + jax major.minor versions.
        # Patch versions are excluded (SemVer-stable ABI by convention).
        try:
            import numpy as _np_pkg
            numpy_ver = self._major_minor(_np_pkg.__version__)
        except Exception:
            numpy_ver = "unknown"
        try:
            import jax as _jax_pkg
            jax_ver = self._major_minor(_jax_pkg.__version__)
        except Exception:
            jax_ver = "unknown"
        return {
            "alpha": self.alpha,
            "seed": self.seed,
            "n_reps": self.n_reps,
            "eta_min_buffer": self.eta_min_buffer,
            "brentq_xtol": self.brentq_xtol,
            "brentq_rtol": self.brentq_rtol,
            "brentq_maxiter": self.brentq_maxiter,
            "bracket_doubling_max": self.bracket_doubling_max,
            "dynamic_n_grid": self.dynamic_n_grid,
            "dynamic_coarse_n": self.dynamic_coarse_n,
            "dynamic_search_mult": self.dynamic_search_mult,
            "delta_grid": self.delta_grid.__dict__,
            "w_grid": self.w_grid.__dict__,
            "theta_grid": self.theta_grid.__dict__,
            "cache_enabled": self.cache_enabled,
            "_numpy_version": numpy_ver,
            "_jax_version": jax_ver,
        }
