"""Regenerate figures from a results directory.

`python -m scripts.figures results/coverage`

Reads `manifest.json`, locates each cell's cache directory, loads the
arrays via `simulation.storage.load_result`, reconstructs the cells'
`RawResult`s, and re-renders the figures via the experiment's
diagnostics. Idempotent — same inputs produce the same figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from frasian._registry import registry
from frasian._registry_bootstrap import bootstrap
from frasian.experiments.base import RawResult
from frasian.simulation.storage import load_result


def _load_cells(manifest_path: Path) -> tuple[str, list[RawResult]]:
    manifest = json.loads(manifest_path.read_text())
    experiment_name = manifest["experiment"]
    raw_results: list[RawResult] = []
    base = manifest_path.parent
    for cell in manifest["cells"]:
        # Skip cells the runner gated as incompatible (no cached arrays).
        if cell.get("status", "ok") != "ok":
            continue
        cell_path = Path(cell["cache_path"])
        path = cell_path if cell_path.is_absolute() else base / cell_path
        stored = load_result(path)
        # Strip the `_cache_key` and `_schema_version` keys; everything else
        # was the experiment's RawResult.metadata at write time.
        meta = {k: v for k, v in stored.metadata.items() if not k.startswith("_")}
        raw_results.append(
            RawResult(
                experiment=experiment_name,
                tilting=cell["tilting"],
                statistic=cell["statistic"],
                arrays=stored.arrays,
                metadata=meta,
            )
        )
    return experiment_name, raw_results


def regenerate(results_dir: Path) -> list[Path]:
    bootstrap()
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")

    experiment_name, raw_results = _load_cells(manifest_path)
    if experiment_name not in registry.experiments:
        raise KeyError(f"experiment '{experiment_name}' not registered")
    experiment = registry.experiments[experiment_name]()

    # Audit P1 K.3: short-circuit on empty raw_results. A manifest with
    # zero ran cells (every cell `incompatible` or `error`) yields an
    # empty list; the diagnostic loop would silently produce no
    # figures and no CSVs, leaving the user without any signal that
    # nothing happened. Surface that explicitly.
    if not raw_results:
        import warnings as _w
        _w.warn(
            f"figures: manifest at {manifest_path} contains no ran cells "
            f"(all cells were incompatible / errored). No figures or "
            f"CSVs produced.",
            UserWarning,
            stacklevel=2,
        )
        return []

    fig_dir = results_dir / "figures"
    out: list[Path] = []
    for diag in experiment.diagnostics():
        tables = [diag.compute(r) for r in raw_results]
        if not tables:
            continue
        merged_df = pd.concat([t.table for t in tables], ignore_index=True)
        from frasian.diagnostics.base import DiagnosticTable

        merged = DiagnosticTable(
            name=diag.name,
            table=merged_df,
            units=tables[0].units,
            metadata=tables[0].metadata,
        )
        merged_df.to_csv(results_dir / f"{diag.name}.csv", index=False)
        out.append(diag.render(merged, fig_dir))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.figures")
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)
    paths = regenerate(args.results_dir)
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
