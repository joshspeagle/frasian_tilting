"""Drive the NN x (tilting × statistic) audit.

Tilting × statistic flavors:
  * Statistic-only flavors (paired with IdentityTilting):
      wald, wald_generic, waldo, waldo_generic
  * power_law × waldo flavors (analytic Theorem-8 vs generic MC):
      pl_fixed0, pl_fixed05,
      pl_fixed0_generic, pl_fixed05_generic,
      pl_numerical, pl_numerical_intp, pl_numerical_generic,
      pl_dyn_numerical
        (pl_dyn_numerical_generic blocked by design — dynamic + force_generic
        raises NotImplementedError)
  * smoothness gets a `pl_bare` flavor: smoothness sweeps eta internally,
    so the cell's selector is irrelevant; passing the bare instance avoids
    duplicate output across the numerical / dyn_numerical variants.

Each invocation runs the four experiments at Config.fast() with `n_jobs`
parallelism. Results land at `results/wald_audit/<flavor>/<experiment>/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from frasian import Config, registry, run_experiment
from frasian._registry_bootstrap import bootstrap
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import (
    DynamicNumericalEtaSelector,
    FixedEtaSelector,
    NumericalEtaSelector,
)
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


def _build_cell(flavor: str):
    """Return ``(tilting, statistic, smoothness_tilting_override)``.

    `smoothness_tilting_override` is the tilting passed for the smoothness
    experiment (where the per-cell selector is otherwise redundant — the
    experiment internally sweeps eta via NumericalEtaSelector). When
    `None`, smoothness uses the default `tilting`. For power_law cells
    we override with the bare `PowerLawTilting()` so smoothness's results
    aren't duplicated across selector variants.
    """
    # Statistic-only (IdentityTilting)
    if flavor == "wald":
        return IdentityTilting(), WaldStatistic(force_generic=False), None
    if flavor == "wald_generic":
        return IdentityTilting(), WaldStatistic(force_generic=True), None
    if flavor == "waldo":
        return IdentityTilting(), WaldoStatistic(force_generic=False), None
    if flavor == "waldo_generic":
        return IdentityTilting(), WaldoStatistic(force_generic=True), None
    # Power-law × WALDO
    pl_bare = PowerLawTilting()
    if flavor == "pl_fixed0":
        return (PowerLawTilting(selector=FixedEtaSelector(eta=0.0)),
                WaldoStatistic(force_generic=False), pl_bare)
    if flavor == "pl_fixed05":
        return (PowerLawTilting(selector=FixedEtaSelector(eta=0.5)),
                WaldoStatistic(force_generic=False), pl_bare)
    if flavor == "pl_fixed0_generic":
        return (PowerLawTilting(selector=FixedEtaSelector(eta=0.0)),
                WaldoStatistic(force_generic=True), pl_bare)
    if flavor == "pl_fixed05_generic":
        return (PowerLawTilting(selector=FixedEtaSelector(eta=0.5)),
                WaldoStatistic(force_generic=True), pl_bare)
    if flavor == "pl_numerical":
        return (PowerLawTilting(selector=NumericalEtaSelector()),
                WaldoStatistic(force_generic=False), pl_bare)
    if flavor == "pl_numerical_intp":
        return (PowerLawTilting(selector=NumericalEtaSelector(objective="integrated_p")),
                WaldoStatistic(force_generic=False), pl_bare)
    if flavor == "pl_numerical_generic":
        return (PowerLawTilting(selector=NumericalEtaSelector()),
                WaldoStatistic(force_generic=True), pl_bare)
    if flavor == "pl_dyn_numerical":
        return (PowerLawTilting(selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)),
                WaldoStatistic(force_generic=False), pl_bare)
    if flavor == "pl_dyn_numerical_generic":
        return (PowerLawTilting(selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)),
                WaldoStatistic(force_generic=True), pl_bare)
    raise ValueError(f"unknown flavor {flavor!r}")


_FLAVORS = [
    "wald", "wald_generic", "waldo", "waldo_generic",
    "pl_fixed0", "pl_fixed05",
    "pl_fixed0_generic", "pl_fixed05_generic",
    "pl_numerical", "pl_numerical_intp", "pl_numerical_generic",
    "pl_dyn_numerical", "pl_dyn_numerical_generic",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", choices=_FLAVORS, required=True)
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Workers for per-replicate parallelism (Config.n_jobs). "
        "Use -1 for all cores. Default 1 = serial (byte-reproducible).",
    )
    args = parser.parse_args()

    bootstrap()
    config = Config.fast().from_overrides(n_jobs=args.n_jobs)
    tilting, statistic, smoothness_override = _build_cell(args.flavor)

    print(f"--- {args.flavor} ---")
    print(f"  tilting = {type(tilting).__name__}"
          f"(selector={type(getattr(tilting, 'selector', None)).__name__})")
    print(f"  statistic = {type(statistic).__name__}"
          f"(force_generic={getattr(statistic, 'force_generic', False)})")
    print(f"  config: n_reps={config.n_reps}, theta_grid={config.theta_grid.n_points}, "
          f"w_grid={config.w_grid.n_points}, delta_grid={config.delta_grid.n_points}, "
          f"n_jobs={config.n_jobs}")

    for exp_name in args.experiments:
        out_dir = args.results_root / args.flavor / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Smoothness uses the bare-tilting override when supplied, so the
        # experiment's internal eta-sweep is not double-counted across
        # selector-variant cells.
        cell_tilting = (
            smoothness_override
            if (exp_name == "smoothness" and smoothness_override is not None)
            else tilting
        )
        t0 = time.time()
        summary = run_experiment(
            experiment=registry.experiments[exp_name](),
            tiltings=[cell_tilting],
            statistics=[statistic],
            config=config,
            out_dir=out_dir,
        )
        dt = time.time() - t0
        rows = [(c.tilting, c.statistic, c.status) for c in summary.cells]
        print(f"\n[{exp_name}] {dt:.1f}s -> {out_dir}")
        for til, stat, status in rows:
            print(f"   {til:30s} x {stat:20s} {status}")


if __name__ == "__main__":
    main()
