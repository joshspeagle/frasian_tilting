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
from frasian.learned.eta_artifact import EtaArtifact
from frasian.tilting.eta_selectors import (
    DynamicNumericalEtaSelector,
    FixedEtaSelector,
    LearnedDynamicEtaSelector,
    NumericalEtaSelector,
)
from frasian.tilting.fisher_rao import FisherRaoTilting
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.mixture import MixtureTilting
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


def _learned_selector(loss: str, scheme: str = "powerlaw") -> LearnedDynamicEtaSelector:
    """Build a `LearnedDynamicEtaSelector` from a Phase G v4 per-loss checkpoint.

    Per-loss audit fixtures train the canonical NN + ``<scheme>`` v4
    YAML with one of the three loss variants (integrated_p,
    cd_variance, static_width). Train via:

        python -m scripts.train_learned_eta \\
            --config experiments/canonical_normal_normal_<scheme>_v4.yaml \\
            --loss <loss> [--alpha 0.05 if static_width] \\
            --out artifacts/learned_eta_canonical_normal_normal_<scheme>_phaseC_<loss>_v4.eqx

    ``scheme`` ∈ {``"powerlaw"``, ``"ot"``}.
    """
    art = EtaArtifact(
        artifact_path=Path(
            f"artifacts/learned_eta_canonical_normal_normal_{scheme}_phaseC_{loss}_v4.eqx"
        ),
        name="learned",
        version=f"phaseC_{scheme}_{loss}_v4",
    )
    return LearnedDynamicEtaSelector(artifact=art)


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
    # Learned-η selectors (Phase C / D) — one per loss
    if flavor == "pl_learned_intp":
        return (PowerLawTilting(selector=_learned_selector("integrated_p")),
                WaldoStatistic(force_generic=False), pl_bare)
    if flavor == "pl_learned_intp_generic":
        return (PowerLawTilting(selector=_learned_selector("integrated_p")),
                WaldoStatistic(force_generic=True), pl_bare)
    # OT non-learned dynamic variants (added 2026-05-09 for smoothness
    # comparison alongside pl_dyn_numerical and mx_dyn_numerical).
    ot_bare = OTTilting()
    if flavor == "ot_dyn_numerical":
        return (OTTilting(selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)),
                WaldoStatistic(force_generic=False), ot_bare)
    if flavor == "ot_dyn_numerical_generic":
        return (OTTilting(selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)),
                WaldoStatistic(force_generic=True), ot_bare)
    # OT learned-eta variants
    if flavor == "ot_learned_intp":
        return (OTTilting(selector=_learned_selector("integrated_p", scheme="ot")),
                WaldoStatistic(force_generic=False), ot_bare)
    if flavor == "ot_learned_cd_var":
        return (OTTilting(selector=_learned_selector("cd_variance", scheme="ot")),
                WaldoStatistic(force_generic=False), ot_bare)
    if flavor == "ot_learned_static_w":
        return (OTTilting(selector=_learned_selector("static_width", scheme="ot")),
                WaldoStatistic(force_generic=False), ot_bare)
    if flavor == "pl_learned_cd_var":
        return (PowerLawTilting(selector=_learned_selector("cd_variance")),
                WaldoStatistic(force_generic=False), pl_bare)
    if flavor == "pl_learned_static_w":
        return (PowerLawTilting(selector=_learned_selector("static_width")),
                WaldoStatistic(force_generic=False), pl_bare)
    # Mixture (m-geodesic) variants — non-learned. Learned-eta variants
    # land in Stage D after Stage C trains the checkpoints.
    mx_bare = MixtureTilting()
    if flavor == "mx_fixed0":
        return (MixtureTilting(selector=FixedEtaSelector(eta=0.0)),
                WaldoStatistic(force_generic=False), mx_bare)
    if flavor == "mx_fixed05":
        return (MixtureTilting(selector=FixedEtaSelector(eta=0.5)),
                WaldoStatistic(force_generic=False), mx_bare)
    if flavor == "mx_fixed0_generic":
        return (MixtureTilting(selector=FixedEtaSelector(eta=0.0)),
                WaldoStatistic(force_generic=True), mx_bare)
    if flavor == "mx_fixed05_generic":
        return (MixtureTilting(selector=FixedEtaSelector(eta=0.5)),
                WaldoStatistic(force_generic=True), mx_bare)
    if flavor == "mx_numerical":
        return (MixtureTilting(selector=NumericalEtaSelector()),
                WaldoStatistic(force_generic=False), mx_bare)
    if flavor == "mx_numerical_intp":
        return (MixtureTilting(selector=NumericalEtaSelector(objective="integrated_p")),
                WaldoStatistic(force_generic=False), mx_bare)
    if flavor == "mx_numerical_generic":
        return (MixtureTilting(selector=NumericalEtaSelector()),
                WaldoStatistic(force_generic=True), mx_bare)
    if flavor == "mx_dyn_numerical":
        return (MixtureTilting(selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)),
                WaldoStatistic(force_generic=False), mx_bare)
    if flavor == "mx_dyn_numerical_generic":
        return (MixtureTilting(selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)),
                WaldoStatistic(force_generic=True), mx_bare)
    # Mixture learned-η variants (added 2026-05-10).
    if flavor == "mx_learned_intp":
        return (MixtureTilting(selector=_learned_selector("integrated_p", scheme="mixture")),
                WaldoStatistic(force_generic=False), mx_bare)
    if flavor == "mx_learned_cd_var":
        return (MixtureTilting(selector=_learned_selector("cd_variance", scheme="mixture")),
                WaldoStatistic(force_generic=False), mx_bare)
    if flavor == "mx_learned_static_w":
        return (MixtureTilting(selector=_learned_selector("static_width", scheme="mixture")),
                WaldoStatistic(force_generic=False), mx_bare)
    # Fisher-Rao (Levi-Civita / information-geometric geodesic) variants —
    # mirrors OT's selector set (no fixed/numerical static; dynamic + learned only).
    fr_bare = FisherRaoTilting()
    if flavor == "fr_dyn_numerical":
        return (FisherRaoTilting(selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)),
                WaldoStatistic(force_generic=False), fr_bare)
    if flavor == "fr_dyn_numerical_generic":
        return (FisherRaoTilting(selector=DynamicNumericalEtaSelector(n_grid=401, coarse_n=25)),
                WaldoStatistic(force_generic=True), fr_bare)
    if flavor == "fr_learned_intp":
        return (FisherRaoTilting(selector=_learned_selector("integrated_p", scheme="fisher_rao")),
                WaldoStatistic(force_generic=False), fr_bare)
    if flavor == "fr_learned_cd_var":
        return (FisherRaoTilting(selector=_learned_selector("cd_variance", scheme="fisher_rao")),
                WaldoStatistic(force_generic=False), fr_bare)
    if flavor == "fr_learned_static_w":
        return (FisherRaoTilting(selector=_learned_selector("static_width", scheme="fisher_rao")),
                WaldoStatistic(force_generic=False), fr_bare)
    raise ValueError(f"unknown flavor {flavor!r}")


_FLAVORS = [
    "wald", "wald_generic", "waldo", "waldo_generic",
    "pl_fixed0", "pl_fixed05",
    "pl_fixed0_generic", "pl_fixed05_generic",
    "pl_numerical", "pl_numerical_intp", "pl_numerical_generic",
    "pl_dyn_numerical", "pl_dyn_numerical_generic",
    "pl_learned_intp", "pl_learned_cd_var", "pl_learned_static_w",
    "pl_learned_intp_generic",
    "ot_dyn_numerical", "ot_dyn_numerical_generic",
    "ot_learned_intp", "ot_learned_cd_var", "ot_learned_static_w",
    "mx_fixed0", "mx_fixed05",
    "mx_fixed0_generic", "mx_fixed05_generic",
    "mx_numerical", "mx_numerical_intp", "mx_numerical_generic",
    "mx_dyn_numerical", "mx_dyn_numerical_generic",
    # Learned-η variants. All three calibrated post-bound (sigmoid
    # squash on EtaNet output, mixture-only via train.py dispatch).
    # See `docs/notes/2026-05-10-mixture-cd-variance-instability.md`.
    "mx_learned_intp", "mx_learned_cd_var", "mx_learned_static_w",
    # Fisher-Rao (Levi-Civita / information-geometric geodesic) variants.
    # fr_dyn_numerical_generic exercises the Stage B generic-MC machinery
    # (_generic_tilt_fr + _generic_tilted_pvalue_fr + diffrax shooting BVP)
    # against the Stage A closed-form half-plane geodesic path.
    "fr_dyn_numerical", "fr_dyn_numerical_generic",
    # Stage C v4 trained heads (3 loss objectives). `_build_cell`
    # branches at `fr_learned_*` consume these. Train via
    # `python -m scripts.train_learned_eta --config
    # experiments/canonical_normal_normal_fisher_rao_v4.yaml --loss
    # {integrated_p, cd_variance, static_width} --out
    # artifacts/learned_eta_canonical_normal_normal_fisher_rao_phaseC_<loss>_v4.eqx`.
    # cd_variance requires `--lr-a 1e-4 --grad-clip-max-norm 0.5` per
    # docs/notes/2026-05-11-fisher-rao-cd-var-hyperparams.md.
    "fr_learned_intp", "fr_learned_cd_var", "fr_learned_static_w",
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
    parser.add_argument(
        "--single-w",
        type=float,
        default=None,
        help="Restrict the experiment's w_grid to a single point (e.g. 0.5). "
        "Pre-Phase-G v3 learned cells required this (they were trained at "
        "one specific prior). Phase G v4 conditional fixtures train over a "
        "range of (σ₀, σ) and accept any w whose generating (σ₀, σ) "
        "lie within that range, so the constraint is now optional — but "
        "it remains a useful knob for narrow sweeps.",
    )
    args = parser.parse_args()

    bootstrap()
    config = Config.fast().from_overrides(n_jobs=args.n_jobs)
    if args.single_w is not None:
        from frasian.config import GridSpec

        config = config.from_overrides(
            w_grid=GridSpec("w", float(args.single_w), float(args.single_w), 1)
        )
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
