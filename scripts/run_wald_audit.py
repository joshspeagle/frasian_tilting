"""Drive the NN x (Wald | WALDO) audit (closed-form vs generic).

Run as `python -m scripts.run_wald_audit --flavor <flavor>` where
`<flavor>` is one of: wald, wald_generic, waldo, waldo_generic.
Each invocation runs all four experiments
(coverage / width / smoothness / confidence_distribution) with a single
statistic instance + IdentityTilting only, under Config.fast().

Results land at `results/wald_audit/<flavor>/<experiment>/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from frasian import Config, registry, run_experiment
from frasian._registry_bootstrap import bootstrap
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.identity import IdentityTilting


def _build_statistic(flavor: str):
    if flavor == "wald":
        return WaldStatistic(force_generic=False)
    if flavor == "wald_generic":
        return WaldStatistic(force_generic=True)
    if flavor == "waldo":
        return WaldoStatistic(force_generic=False)
    if flavor == "waldo_generic":
        return WaldoStatistic(force_generic=True)
    raise ValueError(f"unknown flavor {flavor!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flavor",
        choices=["wald", "wald_generic", "waldo", "waldo_generic"],
        required=True,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["coverage", "width", "smoothness", "confidence_distribution"],
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/wald_audit"),
    )
    args = parser.parse_args()

    bootstrap()
    config = Config.fast()
    statistic = _build_statistic(args.flavor)
    statistics = [statistic]
    tiltings = [IdentityTilting()]

    print(f"--- {args.flavor} ({type(statistic).__name__}"
          f"(force_generic={getattr(statistic, 'force_generic', False)})) ---")
    print(f"config: n_reps={config.n_reps}, theta_grid={config.theta_grid.n_points}, "
          f"w_grid={config.w_grid.n_points}, delta_grid={config.delta_grid.n_points}")

    for exp_name in args.experiments:
        out_dir = args.results_root / args.flavor / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        summary = run_experiment(
            experiment=registry.experiments[exp_name](),
            tiltings=tiltings,
            statistics=statistics,
            config=config,
            out_dir=out_dir,
        )
        dt = time.time() - t0
        rows = [(c.tilting, c.statistic, c.status) for c in summary.cells]
        print(f"\n[{exp_name}] {dt:.1f}s -> {out_dir}")
        for tilting, statistic, status in rows:
            print(f"   {tilting:30s} x {statistic:20s} {status}")


if __name__ == "__main__":
    main()
