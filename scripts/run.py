"""CLI entry point: `python -m scripts.run [--list] [experiment=<name>]`.

Step-1 implementation supports `--list` (enumerate registered methods) and
will print a stub message for any non-empty `experiment=` value until Step 4
ports the concrete experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frasian import EmptyRegistryError, list_methods, registry, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.run")
    parser.add_argument("--list", action="store_true",
                        help="enumerate registered methods and exit")
    parser.add_argument("experiment", nargs="?", default=None,
                        help='e.g. "experiment=coverage" (Step 4+)')
    args = parser.parse_args(argv)

    if args.list:
        groups = list_methods()
        any_found = False
        for kind, entries in groups.items():
            print(f"{kind} ({len(entries)}):")
            for entry in entries:
                print(f"  - {entry.name:20s}  status={entry.status:12s}  "
                      f"brief={entry.brief}")
                any_found = True
            if not entries:
                print("  (none)")
        if not any_found:
            print("\nNo concrete methods are registered yet. "
                  "Step 2 of the migration adds them.")
        return 0

    if args.experiment is None:
        parser.print_help()
        return 0

    if not args.experiment.startswith("experiment="):
        parser.error('positional arg must look like experiment=<name>')

    name = args.experiment.split("=", 1)[1]
    if name not in registry.experiments:
        print(f"experiment '{name}' not registered. Try --list.",
              file=sys.stderr)
        return 1

    experiment_cls = registry.experiments[name]
    try:
        summary = run_experiment(
            experiment=experiment_cls(),
            tiltings=registry.tiltings.all(),
            statistics=registry.statistics.all(),
            out_dir=Path(f"results/{name}"),
        )
    except EmptyRegistryError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    print(f"ran {len(summary.cells)} cells; out_dir={summary.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
