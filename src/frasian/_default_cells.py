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


def _resolve_dynamic_eta_mode() -> str:
    """Read FRASIAN_DEFAULT_DYNAMIC_ETA env var.

    Values:
      - "numerical" (default): `DynamicNumericalEtaSelector` wrapping
        `NumericalEtaSelector`. Calibrated by construction but
        inflates CI width at conflict (kinky η*(|Δ|) curve).
      - "learned": `LearnedDynamicEtaSelector` reading the trained
        Phase E `EtaArtifact` at the canonical YAML's smoke /
        production path. Calibrated AND narrow (the headline claim).

    Default is `numerical`.

    Audit P2 (Cluster F): the env var is read on every call rather
    than cached at import time. This is deliberate so tests can
    flip the mode via `monkeypatch.setenv(...)`, but the caller
    should be aware that flipping the var mid-process produces a
    **fresh** `EtaArtifact` with a cold artifact cache — every
    new call to `default_tiltings()` will re-read the .eqx file
    from disk. Set the env var once before launching Python (or
    in a session-scoped fixture) for production runs.
    """
    import os

    return os.environ.get("FRASIAN_DEFAULT_DYNAMIC_ETA", "numerical").lower()


_PHASE_E_CHECKPOINT_FOR_SCHEME = {
    "power_law": "canonical_normal_normal_powerlaw_v4",
    "ot": "canonical_normal_normal_ot_v4",
}


def _make_learned_selector(scheme_name: str):
    """Build a `LearnedDynamicEtaSelector` for a Phase E checkpoint.

    Looks up the YAML config name for ``scheme_name`` and prefers the
    production checkpoint; falls back to the smoke checkpoint. Raises
    a clear error when neither exists — train via
    ``python -m scripts.train_learned_eta --config <yaml>``.
    """
    from pathlib import Path

    from .learned.eta_artifact import EtaArtifact
    from .tilting.eta_selectors import LearnedDynamicEtaSelector

    config_name = _PHASE_E_CHECKPOINT_FOR_SCHEME.get(scheme_name)
    if config_name is None:
        raise ValueError(
            f"No Phase E experiment config registered for scheme "
            f"{scheme_name!r}. Add an entry to "
            f"`_PHASE_E_CHECKPOINT_FOR_SCHEME` and a YAML under "
            f"`experiments/`."
        )
    # Anchor at the project root so resolution doesn't depend on CWD;
    # `_default_cells.py` lives at `<root>/src/frasian/`.
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "artifacts" / f"learned_eta_{config_name}.eqx",
    ]
    chosen = next((c for c in candidates if c.exists()), None)
    if chosen is None:
        raise FileNotFoundError(
            f"FRASIAN_DEFAULT_DYNAMIC_ETA=learned but no Phase E "
            f"checkpoint found for scheme {scheme_name!r} at any of "
            f"{candidates}. Train one via `python -m "
            f"scripts.train_learned_eta --config "
            f"experiments/{config_name}.yaml --out "
            f"{candidates[0]}`."
        )
    artifact = EtaArtifact(artifact_path=chosen)
    return LearnedDynamicEtaSelector(artifact=artifact)


def default_tiltings(
    *,
    n_grid: int = 401,
    coarse_n: int = 25,
) -> list[TiltingScheme]:
    """For coverage / width: tiltings *with their selector baked in*.

    Identity + power_law[dynamic] + ot[dynamic]. The fixed-η=0 power_law
    cell is omitted as numerically redundant with identity. The
    "dynamic" selector is gated by `FRASIAN_DEFAULT_DYNAMIC_ETA`:

      - `numerical` (default): `DynamicNumericalEtaSelector`. Calibrated
        but kinky; inflates CI width at conflict.
      - `learned`: `LearnedDynamicEtaSelector` with the trained MLP.
        Calibrated AND narrow.

    Future smoother geodesic schemes (`fisher_rao`, `mixture`) plug
    in here once their stubs land.
    """
    from .tilting.eta_selectors import DynamicNumericalEtaSelector
    from .tilting.identity import IdentityTilting
    from .tilting.ot import OTTilting
    from .tilting.power_law import PowerLawTilting

    mode = _resolve_dynamic_eta_mode()
    if mode == "learned":
        pl_selector = _make_learned_selector("power_law")
        ot_selector = _make_learned_selector("ot")
    elif mode == "numerical":
        pl_selector = DynamicNumericalEtaSelector(
            n_grid=n_grid,
            coarse_n=coarse_n,
        )
        ot_selector = DynamicNumericalEtaSelector(
            n_grid=n_grid,
            coarse_n=coarse_n,
        )
    else:
        raise ValueError(
            f"FRASIAN_DEFAULT_DYNAMIC_ETA={mode!r}; " f"expected 'numerical' or 'learned'."
        )

    return [
        IdentityTilting(),
        PowerLawTilting(selector=pl_selector),
        OTTilting(selector=ot_selector),
    ]


def post_selection_demo_tiltings() -> list[TiltingScheme]:
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
            selector=NumericalEtaSelector(),
        ),
    ]


def default_smoothness_tiltings() -> list[TiltingScheme]:
    """For smoothness: tilting *families* whose η*(|Δ|) curve we want to
    characterise. Smoothness sweeps the parameter via its own internal
    `NumericalEtaSelector`, so the cell's selector is irrelevant — we
    pass the bare family instances to avoid duplicate output.
    """
    from .tilting.identity import IdentityTilting
    from .tilting.ot import OTTilting
    from .tilting.power_law import PowerLawTilting

    return [IdentityTilting(), PowerLawTilting(), OTTilting()]


def default_statistics() -> list[TestStatistic]:
    """Wald + WALDO."""
    from .statistics.wald import WaldStatistic
    from .statistics.waldo import WaldoStatistic

    return [WaldStatistic(), WaldoStatistic()]


def default_cells(
    *, experiment: str = "coverage", **tilting_kwargs
) -> tuple[list[TiltingScheme], list[TestStatistic]]:
    """`(tiltings, statistics)` for the runner, dispatched by experiment.

    `coverage` / `width` use selector-baked tiltings (so `power_law`
    contributes the dynamic-η cell). `smoothness` uses bare families
    so the cell list does not duplicate the η-sweep output.
    """
    if experiment == "smoothness":
        return default_smoothness_tiltings(), default_statistics()
    return default_tiltings(**tilting_kwargs), default_statistics()
