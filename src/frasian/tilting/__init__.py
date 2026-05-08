"""Tilting schemes.

Only the protocol surface (`TiltingScheme`, `EtaSelector`,
`ParamSpec`) and the shared `GridDistribution` helper are
re-exported here. Concrete schemes (`PowerLawTilting`,
`OTTilting`, `IdentityTilting`, ...) are NOT re-exported by
design — they self-register via `@register_tilting` when
`frasian._registry_bootstrap.bootstrap()` imports them.
Consumers fetch them via the registry
(`registry.tiltings[name]()`) rather than hard-coding imports,
which keeps the cross-product runner extensible and the import
graph one-way (consumers depend on the registry, not on the
concrete classes).
"""

from ._grid_distribution import GridDistribution, grid_distribution_from_log_density
from .base import EtaSelector, ParamSpec, TiltingScheme

__all__ = [
    "EtaSelector",
    "GridDistribution",
    "ParamSpec",
    "TiltingScheme",
    "grid_distribution_from_log_density",
]
