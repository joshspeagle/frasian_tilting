"""Canonical cell list for `(TiltingScheme x TestStatistic)` runs.

The runner used to enumerate `registry.tiltings.implemented()` x
`registry.statistics.implemented()` and silently produce numerically
duplicate cells (e.g. `(power_law, wald)` collapsing to plain Wald).
The refactored runner takes *instances* and gates by
`accepts_tilting`, so the canonical baseline cell list is now an
explicit, configurable set of three cells:

    (identity, wald)                       — plain Wald
    (identity, waldo)                      — plain (static) WALDO
    (power_law[dynamic_numerical], waldo)  — Dynamic-WALDO (calibrated)

The redundant `(power_law[fixed-η=0], waldo)` cell is omitted by
design: it is numerically identical to `(identity, waldo)` (η=0
recovers WALDO).

Why the dynamic-η selector and not the static η*-opt?
=====================================================

`DynamicNumericalEtaSelector` (η varying per θ during inversion) has
exact 1-α coverage by construction. `NumericalEtaSelector` (single
η minimising CI width per D) gives strictly narrower CIs but
**undercovers** by ~2 points at α=0.05 because of post-selection
inference. The framework's default insists on calibration, so the
dynamic selector is the headline. The static selector is exposed via
`post_selection_demo_tiltings()` purely as a baseline for studying
the coverage / width trade-off — running it on `coverage` is the way
to *see* the calibration loss empirically.

See `NumericalEtaSelector` and `DynamicNumericalEtaSelector` docstrings
for details, and `tests/regression/test_post_selection_coverage.py`
for the regression that pins the empirical shortfall.
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
    cell is omitted as numerically redundant with identity. Smoother
    geodesic schemes (`ot`, `fisher_rao`, `mixture`) plug in here as
    they are implemented; `ot[dynamic_numerical]` is the W2-tilted
    counterpart of Dynamic-WALDO.
    """
    # Imports inside the function body keep `import frasian` side-effect-
    # free: only callers that actually need the cell instances trigger
    # the tilting/statistic module registrations.
    from .tilting.eta_selectors import DynamicNumericalEtaSelector
    from .tilting.identity import IdentityTilting
    from .tilting.ot import OTTilting
    from .tilting.power_law import PowerLawTilting
    return [
        IdentityTilting(),
        PowerLawTilting(
            selector=DynamicNumericalEtaSelector(
                sigma=sigma, mu0=mu0,
                n_grid=n_grid, coarse_n=coarse_n,
            ),
        ),
        OTTilting(
            selector=DynamicNumericalEtaSelector(
                sigma=sigma, mu0=mu0,
                n_grid=n_grid, coarse_n=coarse_n,
            ),
        ),
    ]


def post_selection_demo_tiltings(*, sigma: float = 1.0, mu0: float = 0.0,
                                   ) -> list["TiltingScheme"]:
    """Tiltings for the **post-selection coverage demo**.

    Returns `[IdentityTilting(), PowerLawTilting(NumericalEtaSelector())]`.
    Run this through the `coverage` / `width` experiments to demonstrate
    that the static η*-opt selector achieves narrower-than-WALDO CIs
    (in fact ≤ Wald asymptotically) but at the cost of ~2 points of
    nominal coverage — a textbook post-selection inference effect.

    NOT for production CI estimation. The framework's calibrated
    default is `default_tiltings()` (which uses
    `DynamicNumericalEtaSelector`).
    """
    from .tilting.eta_selectors import NumericalEtaSelector
    from .tilting.identity import IdentityTilting
    from .tilting.power_law import PowerLawTilting
    return [
        IdentityTilting(),
        PowerLawTilting(
            selector=NumericalEtaSelector(sigma=sigma, mu0=mu0),
        ),
    ]


def default_smoothness_tiltings() -> list["TiltingScheme"]:
    """For smoothness: tilting *families* whose η*(|Δ|) curve we want to
    characterise. Smoothness sweeps the parameter via its own internal
    `NumericalEtaSelector`, so the cell's selector is irrelevant — we
    pass the bare family instances to avoid duplicate output.
    """
    from .tilting.identity import IdentityTilting
    from .tilting.ot import OTTilting
    from .tilting.power_law import PowerLawTilting
    return [IdentityTilting(), PowerLawTilting(), OTTilting()]


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
