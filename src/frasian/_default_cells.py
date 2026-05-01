"""Canonical cell list for `(TiltingScheme x TestStatistic)` runs.

The runner used to enumerate `registry.tiltings.implemented()` x
`registry.statistics.implemented()` and silently produce numerically
duplicate cells (e.g. `(power_law, wald)` collapsing to plain Wald).
The refactored runner takes *instances* and gates by
`accepts_tilting`, so the canonical baseline cell list is now an
explicit, configurable set of three cells:

    (identity, wald)                       — plain Wald
    (identity, waldo)                      — plain (static) WALDO
    (power_law[dynamic_numerical], waldo)  — Dynamic-WALDO

The redundant `(power_law[fixed-η=0], waldo)` cell is omitted by
design: it is numerically identical to `(identity, waldo)` (η=0
recovers WALDO).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .statistics.base import TestStatistic
    from .tilting.base import TiltingScheme


def default_tiltings(*, sigma: float = 1.0, mu0: float = 0.0,
                     n_grid: int = 401, coarse_n: int = 25,
                     ) -> list["TiltingScheme"]:
    """For coverage / width: tiltings *with their selector baked in*.

    Identity + power_law[dynamic_numerical]. The fixed-η=0 power_law
    cell is omitted as numerically redundant with identity. Future
    smooth schemes (OT / geodesic / mixture / exp_family) plug in here
    once their stubs are implemented.
    """
    # Imports inside the function body keep `import frasian` side-effect-
    # free: only callers that actually need the cell instances trigger
    # the tilting/statistic module registrations.
    from .tilting.eta_selectors import DynamicNumericalEtaSelector
    from .tilting.identity import IdentityTilting
    from .tilting.power_law import PowerLawTilting
    return [
        IdentityTilting(),
        PowerLawTilting(
            selector=DynamicNumericalEtaSelector(
                sigma=sigma, mu0=mu0,
                n_grid=n_grid, coarse_n=coarse_n,
            ),
        ),
    ]


def default_smoothness_tiltings() -> list["TiltingScheme"]:
    """For smoothness: tilting *families* whose η*(|Δ|) curve we want to
    characterise. Smoothness sweeps the parameter via its own internal
    `NumericalEtaSelector`, so the cell's selector is irrelevant — we
    pass the bare family instances to avoid duplicate output.
    """
    from .tilting.identity import IdentityTilting
    from .tilting.power_law import PowerLawTilting
    return [IdentityTilting(), PowerLawTilting()]


def default_statistics() -> list["TestStatistic"]:
    """Wald + WALDO."""
    from .statistics.wald import WaldStatistic
    from .statistics.waldo import WaldoStatistic
    return [WaldStatistic(), WaldoStatistic()]


def default_cells(*, experiment: str = "coverage",
                   **tilting_kwargs) -> tuple[list[TiltingScheme],
                                              list[TestStatistic]]:
    """`(tiltings, statistics)` for the runner, dispatched by experiment.

    `coverage` / `width` use selector-baked tiltings (so `power_law`
    contributes the dynamic-η cell). `smoothness` uses bare families
    so the cell list does not duplicate the η-sweep output.
    """
    if experiment == "smoothness":
        return default_smoothness_tiltings(), default_statistics()
    return default_tiltings(**tilting_kwargs), default_statistics()
