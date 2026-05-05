"""CLI entry point: `python -m scripts.run [--list] [experiment=<name>]`.

Step-1 implementation supports `--list` (enumerate registered methods) and
will print a stub message for any non-empty `experiment=` value until Step 4
ports the concrete experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frasian import EmptyRegistryError, default_cells, list_methods, registry, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.run")
    parser.add_argument("--list", action="store_true", help="enumerate registered methods and exit")
    parser.add_argument(
        "--fast", action="store_true", help="run on Config.fast() (small grids, few reps)"
    )
    parser.add_argument("experiment", nargs="?", default=None, help='e.g. "experiment=coverage"')
    args = parser.parse_args(argv)

    from frasian._registry_bootstrap import bootstrap

    bootstrap()

    if args.list:
        groups = list_methods()
        any_found = False
        for kind, entries in groups.items():
            print(f"{kind} ({len(entries)}):")
            for entry in entries:
                print(f"  - {entry.name:20s}  status={entry.status:12s}  " f"brief={entry.brief}")
                any_found = True
            if not entries:
                print("  (none)")
        if not any_found:
            print("\nNo concrete methods are registered yet. " "Step 2 of the migration adds them.")
        return 0

    if args.experiment is None:
        parser.print_help()
        return 0

    if not args.experiment.startswith("experiment="):
        parser.error("positional arg must look like experiment=<name>")

    name = args.experiment.split("=", 1)[1]
    if name not in registry.experiments:
        print(f"experiment '{name}' not registered. Try --list.", file=sys.stderr)
        return 1

    experiment_cls = registry.experiments[name]
    from frasian import Config

    cfg = Config.fast() if args.fast else Config.default()
    # Smaller scan grid + coarse-η budget for the dynamic selector under --fast
    # so the cross-product of n_reps × dynamic-CI per D stays tractable.
    n_grid, coarse_n = (201, 11) if args.fast else (401, 25)
    tiltings, statistics = default_cells(
        experiment=name,
        n_grid=n_grid,
        coarse_n=coarse_n,
    )
    try:
        summary = run_experiment(
            experiment=experiment_cls(),
            tiltings=tiltings,
            statistics=statistics,
            config=cfg,
            out_dir=Path(f"results/{name}"),
        )
    except EmptyRegistryError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    n_run = sum(1 for c in summary.cells if c.status == "ok")
    n_skip = sum(1 for c in summary.cells if c.status == "incompatible")
    print(f"ran {n_run} cells (skipped {n_skip} incompatible); " f"out_dir={summary.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
