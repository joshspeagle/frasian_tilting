"""Cross-product experiment runner.

Iterates the cartesian product (TiltingScheme × TestStatistic) for a given
`Experiment`, dispatches each cell, persists results, computes diagnostics,
and writes a manifest.json. Step-1 implementation handles the empty-registry
case cleanly; richer functionality (caching, manifest, parallelism) lands in
Step 3 alongside the simulation infrastructure.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ._errors import EmptyRegistryError
from ._registry import RegistryEntry
from ._registry_bootstrap import bootstrap
from .config import Config
from .experiments.base import Experiment, RawResult


@dataclass
class RunSummary:
    """Returned by `run_experiment`. Lightweight, serializable."""

    experiment: str
    config_fingerprint: str
    cells: list[dict[str, str]] = field(default_factory=list)
    out_dir: Path | None = None


def run_experiment(
    *,
    experiment: Experiment,
    tiltings: Iterable[type] | None = None,
    statistics: Iterable[type] | None = None,
    config: Config | None = None,
    out_dir: Path | None = None,
) -> RunSummary:
    """Execute `experiment` for every (tilting × statistic) cell.

    `tiltings` / `statistics` default to *all* registered classes. An empty
    cartesian product raises `EmptyRegistryError` rather than silently
    succeeding — Step 1's primary correctness guarantee.
    """
    bootstrap()
    config = config or Config.default()

    tilting_classes = list(tiltings) if tiltings is not None else []
    statistic_classes = list(statistics) if statistics is not None else []

    if not tilting_classes or not statistic_classes:
        raise EmptyRegistryError(
            f"Experiment '{experiment.name}' requires at least one tilting "
            f"and one statistic, got tiltings={len(tilting_classes)} "
            f"statistics={len(statistic_classes)}. Concrete methods are "
            f"registered starting in Step 2 of the migration."
        )

    summary = RunSummary(
        experiment=experiment.name,
        config_fingerprint=config.fingerprint(),
        out_dir=out_dir,
    )
    ctx = experiment.setup(config)
    for tilting_cls, statistic_cls in itertools.product(
        tilting_classes, statistic_classes
    ):
        tilting = tilting_cls()
        statistic = statistic_cls()
        result: RawResult = experiment.run_cell(ctx, tilting, statistic)
        summary.cells.append(
            {"tilting": result.tilting, "statistic": result.statistic}
        )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "experiment": summary.experiment,
                    "config_fingerprint": summary.config_fingerprint,
                    "cells": summary.cells,
                },
                indent=2,
            )
        )
    return summary


def list_methods() -> dict[str, list[RegistryEntry]]:
    """Used by `python -m scripts.run --list`."""
    bootstrap()
    from ._registry import registry

    return {
        "models": registry.models.entries(),
        "tiltings": registry.tiltings.entries(),
        "statistics": registry.statistics.entries(),
        "experiments": registry.experiments.entries(),
        "diagnostics": registry.diagnostics.entries(),
    }
