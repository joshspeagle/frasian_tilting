"""Cross-product experiment runner.

Iterates the cartesian product (TiltingScheme x TestStatistic) for a given
`Experiment`, dispatches each cell, persists results through the cache,
runs the experiment's diagnostics, and emits a manifest.json that is
byte-reproducible at the same git-sha and Config fingerprint.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from ._errors import EmptyRegistryError
from ._registry_bootstrap import bootstrap
from .config import Config
from .experiments.base import Experiment, RawResult
from .simulation.cache import git_sha
from .simulation.runner import persist_cell
from .statistics.base import accepts_tilting as _accepts_tilting


@dataclass
class CellSummary:
    """One row of `RunSummary.cells`: the (tilting, statistic) cell and the
    relative path to its persisted result inside the run's `out_dir`.

    `cache_path` is empty for cells skipped due to `accepts_tilting`
    incompatibility; `status` carries the reason ('ok' | 'incompatible').
    """

    tilting: str
    statistic: str
    cache_path: str
    status: str = "ok"
    reason: str = ""


@dataclass
class RunSummary:
    """Lightweight, JSON-serialisable summary returned by `run_experiment`.

    `out_dir/manifest.json` is built from this struct; downstream tooling
    (figures.py, the completeness checker) consumes the manifest to find
    each cell's persisted arrays.
    """

    experiment: str
    config_fingerprint: str
    git_sha: str
    cells: list[CellSummary] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    out_dir: Path | None = None
    figures: list[str] = field(default_factory=list)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, CellSummary):
        return {"tilting": obj.tilting, "statistic": obj.statistic,
                "cache_path": obj.cache_path,
                "status": obj.status, "reason": obj.reason}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _instantiate(obj: Any) -> Any:
    """Accept a class or instance; return an instance.

    The runner used to take iterables of classes; the refactor allows
    iterables of instances so callers can configure tilting selectors.
    Bare classes still work as long as they accept zero-arg construction.
    """
    return obj() if isinstance(obj, type) else obj


def _cell_name(tilting: Any) -> str:
    """Display name used in the manifest. Tiltings may opt into a
    `cell_name` property to encode their selector; fall back to `name`."""
    return getattr(tilting, "cell_name", None) or getattr(tilting, "name", "?")


def run_experiment(
    *,
    experiment: Experiment,
    tiltings: Iterable[type] | None = None,
    statistics: Iterable[type] | None = None,
    config: Config | None = None,
    out_dir: Path | None = None,
    cache_root: Path | None = None,
) -> RunSummary:
    """Execute `experiment` for every (tilting x statistic) cell.

    `tiltings` / `statistics` default to all registered classes. Empty
    cartesian product raises `EmptyRegistryError`.

    `cache_root` defaults to `<out_dir>/cache` (or a sibling of `out_dir`
    if `out_dir` is given). The manifest at `<out_dir>/manifest.json` lists
    every cell's cache path so a downstream `figures.py` script can find them.
    """
    bootstrap()
    config = config or Config.default()

    tilting_classes = list(tiltings) if tiltings is not None else []
    statistic_classes = list(statistics) if statistics is not None else []
    if not tilting_classes or not statistic_classes:
        raise EmptyRegistryError(
            f"Experiment '{experiment.name}' requires at least one tilting "
            f"and one statistic, got tiltings={len(tilting_classes)} "
            f"statistics={len(statistic_classes)}."
        )

    if out_dir is None:
        out_dir = Path("results") / experiment.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_root is None:
        cache_root = out_dir / "cache"
    cache_root = Path(cache_root)

    sha = git_sha()
    summary = RunSummary(
        experiment=experiment.name,
        config_fingerprint=config.fingerprint(),
        git_sha=sha,
        out_dir=out_dir,
    )

    ctx = experiment.setup(config)
    raw_results: list[RawResult] = []
    for tilting_obj, statistic_obj in itertools.product(
        tilting_classes, statistic_classes
    ):
        tilting = _instantiate(tilting_obj)
        statistic = _instantiate(statistic_obj)
        cell_tilting_name = _cell_name(tilting)
        cell_statistic_name = getattr(statistic, "name", "?")

        if not _accepts_tilting(statistic, tilting):
            summary.cells.append(CellSummary(
                tilting=cell_tilting_name,
                statistic=cell_statistic_name,
                cache_path="",
                status="incompatible",
                reason=(f"{cell_statistic_name} declines pairing with "
                        f"{cell_tilting_name}"),
            ))
            continue

        result = experiment.run_cell(ctx, tilting, statistic)
        raw_fp = str(result.metadata.get("raw_fingerprint", ""))
        path = persist_cell(
            raw_result=result, config=config, cache_root=cache_root,
            raw_fingerprint=raw_fp, tilting=tilting,
        )
        raw_results.append(result)
        # Record path *relative to out_dir* so the manifest is byte-reproducible
        # across machines / temp dirs.
        try:
            rel = path.resolve().relative_to(out_dir.resolve())
        except ValueError:
            rel = path.resolve()
        summary.cells.append(CellSummary(
            tilting=result.tilting,
            statistic=result.statistic,
            cache_path=str(rel),
        ))

    # Diagnostics: each diagnostic compute()s on every cell, concatenates
    # into a single tidy DataFrame, then render()s once.
    fig_dir = out_dir / "figures"
    for diag in experiment.diagnostics():
        tables = [diag.compute(r) for r in raw_results]
        if not tables:
            continue
        merged_df = pd.concat([t.table for t in tables], ignore_index=True)
        from .diagnostics.base import DiagnosticTable

        merged = DiagnosticTable(
            name=diag.name,
            table=merged_df,
            units=tables[0].units,
            metadata=tables[0].metadata,
        )
        merged_df.to_csv(out_dir / f"{diag.name}.csv", index=False)
        path = diag.render(merged, fig_dir)
        summary.diagnostics.append(diag.name)
        try:
            rel = path.resolve().relative_to(out_dir.resolve())
        except ValueError:
            rel = path.resolve()
        summary.figures.append(str(rel))

    manifest = {
        "experiment": summary.experiment,
        "config_fingerprint": summary.config_fingerprint,
        "git_sha": summary.git_sha,
        "cells": [_to_jsonable(c) for c in summary.cells],
        "diagnostics": summary.diagnostics,
        "figures": summary.figures,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return summary


def list_methods() -> dict[str, list]:
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
