"""Frasian inference framework — research scaffold.

Public surface:
    - `Config`, `GridSpec`     — framework-wide configuration
    - `registry`               — singleton plugin registry
    - `register_*`             — decorators to add concrete implementations
    - `run_experiment`         — cross-product runner
    - `list_methods`           — enumerate registered methods
    - Errors: `FrasianError`, `EmptyRegistryError`, `TiltingDomainError`,
      `RegistryConflictError`, `MissingArtifactError`

Importing this module is side-effect-free; concrete implementations are
loaded lazily via `_registry_bootstrap` on first call to `run_experiment`
or `list_methods`.
"""

from __future__ import annotations

from ._errors import (
    EmptyRegistryError,
    FrasianError,
    MissingArtifactError,
    RegistryConflictError,
    TiltingDomainError,
)
from ._registry import (
    Registry,
    RegistryEntry,
    register_diagnostic,
    register_experiment,
    register_model,
    register_statistic,
    register_tilting,
    registry,
)
from ._default_cells import (
    default_cells,
    default_smoothness_tiltings,
    default_statistics,
    default_tiltings,
    post_selection_demo_tiltings,
)
from ._runner import RunSummary, list_methods, run_experiment
from .cd import (
    CDValidityIssue,
    ConfidenceDistribution,
    GridConfidenceDistribution,
    build_cd_from_pvalue,
    total_variation,
    wasserstein_1,
    wasserstein_2,
)
from .config import Config, GridSpec

__all__ = [
    "Config",
    "GridSpec",
    "Registry",
    "RegistryEntry",
    "RunSummary",
    # CD public surface
    "ConfidenceDistribution",
    "GridConfidenceDistribution",
    "CDValidityIssue",
    "build_cd_from_pvalue",
    "wasserstein_1",
    "wasserstein_2",
    "total_variation",
    # default cells
    "default_cells",
    "default_smoothness_tiltings",
    "default_statistics",
    "default_tiltings",
    "post_selection_demo_tiltings",
    # registry
    "registry",
    "register_diagnostic",
    "register_experiment",
    "register_model",
    "register_statistic",
    "register_tilting",
    "run_experiment",
    "list_methods",
    # errors
    "FrasianError",
    "EmptyRegistryError",
    "TiltingDomainError",
    "RegistryConflictError",
    "MissingArtifactError",
]

__version__ = "0.2.0.dev0"
