"""Stage D headline comparison: η(θ) and CI-width(D) smoothness across geodesics.

Answers the framework's central empirical question: does OT / mixture /
Fisher-Rao geodesic interpolation produce smoother (lower Lipschitz / TV /
discontinuity-count / spectral roughness) families than naive power-law
tilting?

Architecture
============
For each cell in the **4 schemes × 4 selectors = 16 cells matrix**, sweep
two profiles on a fine grid at the canonical NN sandbox (μ₀=0, σ₀=σ=1,
w=0.5, α=0.05):

  - **η(θ)** : 200 θ-points in [-6, 6]; η evaluated via
    selector.select_grid (per-θ static optimum or learned MLP).
  - **CI-width(D)** : 200 D-points in [-6, 6]; CI width
    `tilting.confidence_interval(α, [D], model, prior, waldo)` for
    each D.

Then apply the existing `frasian.diagnostics.smoothness_metrics` quartet
(_local_lipschitz, _total_variation, _discontinuity_count,
_spectral_roughness) — the same metric definitions the SmoothnessExperiment
uses, so cross-scheme comparison is apples-to-apples.

Outputs
-------
- ``output/diagnostics/compare_smoothness_<git-sha>.csv`` (32 rows = 16
  cells × 2 metric_targets).
- ``output/illustrations/compare_smoothness.png`` (2-panel figure: η(θ)
  curves and CI-width(D) curves at the calibrated default
  ``learned_intp`` slice).
- Stdout summary: per-scheme median rank across 4 metrics × 2 targets,
  plus a one-line headline narrative.

Caveats encoded in this script
==============================
1. **fisher_rao[dyn_numerical]** at w=0.5 collapses to bare WALDO: the
   per-θ static optimum on FR is η = 0 (the no-tilt point), so the
   resulting η(θ) is constant and CI-width(D) is the bare-WALDO width.
   Smoothness numbers will be near-trivial for that cell — expected, not
   a bug.
2. **fisher_rao[learned_cd_var]** is known to produce wide CIs with
   negative-η at conflict (Stage C.4 note,
   ``2026-05-11-fisher-rao-cd-var-hyperparams.md``). Its smoothness
   metrics may LOOK favourable (smooth wide CIs) but its widths are
   pathological. Read the headline note for the honest interpretation.
3. **LearnedDynamicEtaSelector OOD-θ clamp** replaces η with
   ``eta_likelihood_only=1.0`` (or scheme equivalent) outside the σ₀-
   anchored training box (~5σ₀ from μ₀). For our θ ∈ [-6, 6] sweep at
   σ₀=1, the inner [-5, 5] is the trained region and the outer ±1 is
   OOD-clamped. This is acknowledged behaviour, not a bug; the η-curve
   will show a step at θ = ±5σ₀ for learned cells.

Usage
-----
``PYTHONHASHSEED=0 python -m scripts.compare_geodesic_smoothness``

Matches ``scripts.regen_headline``'s seed pin so any code path using
``hash(...)`` is reproducible.

CLI:
    --schemes power_law ot mixture fisher_rao
    --selectors dyn_numerical learned_intp learned_cd_var learned_static_w
    --n-jobs N    (1 = serial; -1 = all cores; uses frasian._parallel.parallel_map)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

# Mirror regen_headline.py: refuse to run unless PYTHONHASHSEED is pinned.
def _require_hash_seed_pinned() -> None:
    if os.environ.get("PYTHONHASHSEED") not in ("0", "random_pinned"):
        sys.stderr.write(
            "\nERROR: scripts.compare_geodesic_smoothness requires "
            "PYTHONHASHSEED=0 to be set in the environment BEFORE Python "
            "starts (matches scripts.regen_headline). Re-run with:\n\n"
            "    PYTHONHASHSEED=0 python -m scripts.compare_geodesic_smoothness\n\n"
        )
        raise SystemExit(2)


def _check_jax_available() -> None:
    try:
        import jax  # noqa: F401
        import equinox  # noqa: F401
    except ImportError as exc:
        sys.stderr.write(
            "\nERROR: scripts.compare_geodesic_smoothness requires jax + "
            "equinox to load Phase G v4 checkpoints.\n"
            f"Underlying ImportError: {exc}\n\n"
        )
        raise SystemExit(1) from exc


def _git_sha_short() -> str:
    """Short git SHA for output filenames; 'nogit' if not available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


_SCHEME_ARTIFACT_SUFFIX = {
    "power_law": "powerlaw",
    "ot": "ot",
    "mixture": "mixture",
    "fisher_rao": "fisher_rao",
}

_TILTING_CLASS = {
    "power_law": "frasian.tilting.power_law:PowerLawTilting",
    "ot": "frasian.tilting.ot:OTTilting",
    "mixture": "frasian.tilting.mixture:MixtureTilting",
    "fisher_rao": "frasian.tilting.fisher_rao:FisherRaoTilting",
}

# learned_intp / learned_cd_var / learned_static_w → trained head token.
_HEAD_FILE_TOKEN = {
    "learned_intp": "integrated_p",
    "learned_cd_var": "cd_variance",
    "learned_static_w": "static_width",
}

ALL_SCHEMES = list(_SCHEME_ARTIFACT_SUFFIX)
ALL_SELECTORS = ["dyn_numerical", "learned_intp", "learned_cd_var", "learned_static_w"]


def _learned_fixture_path(scheme: str, selector: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    suffix = _SCHEME_ARTIFACT_SUFFIX[scheme]
    return (
        project_root
        / "artifacts"
        / f"learned_eta_canonical_normal_normal_{suffix}_phaseC_{_HEAD_FILE_TOKEN[selector]}_v4.eqx"
    )


def _import_tilting_class(scheme: str):
    import importlib

    mod_name, cls_name = _TILTING_CLASS[scheme].split(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def _build_tilting(scheme: str, selector: str):
    """Construct the tilting cell with its selector wired up.

    Returns (tilt, label) or (None, label) if a learned artifact is missing.
    """
    from frasian.learned.eta_artifact import EtaArtifact
    from frasian.tilting.eta_selectors import (
        DynamicNumericalEtaSelector,
        LearnedDynamicEtaSelector,
    )

    TiltingCls = _import_tilting_class(scheme)
    label = f"{scheme}[{selector}]"
    if selector == "dyn_numerical":
        return TiltingCls(selector=DynamicNumericalEtaSelector()), label
    path = _learned_fixture_path(scheme, selector)
    if not path.exists():
        return None, label
    return (
        TiltingCls(
            selector=LearnedDynamicEtaSelector(
                artifact=EtaArtifact(artifact_path=path),
            )
        ),
        label,
    )


def _ci_width_at_D(
    D: float,
    *,
    alpha: float,
    tilting,
    statistic,
    model,
    prior,
) -> float:
    """One CI-width sample. Top-level so joblib pickles cleanly."""
    import numpy as np

    try:
        lo, hi = tilting.confidence_interval(
            alpha, np.asarray([float(D)]), model, prior, statistic
        )
        w = float(hi - lo)
        return w if (w > 0 and np.isfinite(w)) else float("nan")
    except Exception:
        return float("nan")


def _per_cell_eta_curve(
    *,
    tilting,
    selector_name: str,
    theta_grid,
    model,
    prior,
    alpha: float,
    statistic,
):
    """Evaluate η(θ) on the supplied grid using the cell's selector."""
    import numpy as np

    sel = tilting.selector
    if hasattr(sel, "select_grid"):
        try:
            etas = sel.select_grid(
                np.asarray(theta_grid, dtype=np.float64),
                tilting,
                model=model, prior=prior, alpha=alpha, statistic=statistic,
            )
            return np.asarray(etas, dtype=np.float64)
        except Exception as exc:
            print(f"[warn] {selector_name}: select_grid failed ({exc}); falling back to per-θ select")
    out = np.full(len(theta_grid), np.nan, dtype=np.float64)
    for i, th in enumerate(theta_grid):
        try:
            out[i] = float(sel.select(
                tilting, data=np.asarray([float(th)]),
                model=model, prior=prior, alpha=alpha, statistic=statistic,
            ))
        except Exception:
            pass
    return out


def _per_cell_width_curve(
    *,
    tilting,
    label: str,
    D_grid,
    model,
    prior,
    alpha,
    statistic,
    n_jobs: int,
):
    """Evaluate CI-width(D) on the supplied grid (parallelisable per-D)."""
    import numpy as np

    from frasian._parallel import parallel_map

    fn = partial(
        _ci_width_at_D,
        alpha=alpha, tilting=tilting, statistic=statistic,
        model=model, prior=prior,
    )
    widths = parallel_map(fn, list(D_grid), n_jobs=n_jobs)
    return np.asarray(widths, dtype=np.float64)


def _compute_metrics_row(
    *, scheme, selector, metric_target: str, x, y,
):
    """Apply the smoothness quartet to one (x, y) curve."""
    import numpy as np

    from frasian.diagnostics.smoothness_metrics import (
        _local_lipschitz,
        _total_variation,
        _discontinuity_count,
        _spectral_roughness,
    )
    return {
        "scheme": scheme,
        "selector": selector,
        "metric_target": metric_target,
        "lipschitz": _local_lipschitz(np.asarray(x), np.asarray(y)),
        "tv": _total_variation(np.asarray(y)),
        "discontinuity_count": int(_discontinuity_count(np.asarray(y))),
        "spectral_roughness": _spectral_roughness(np.asarray(y)),
    }


def _compute_table(
    schemes,
    selectors,
    *,
    n_jobs: int,
    n_theta: int = 200,
    n_D: int = 200,
    theta_lo: float = -6.0,
    theta_hi: float = 6.0,
    D_lo: float = -6.0,
    D_hi: float = 6.0,
):
    import numpy as np

    from frasian._registry_bootstrap import bootstrap
    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel
    from frasian.statistics.waldo import WaldoStatistic

    bootstrap()

    sigma = 1.0
    sigma0 = 1.0  # → w = 0.5
    mu0 = 0.0
    alpha = 0.05
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    waldo = WaldoStatistic()

    theta_grid = np.linspace(theta_lo, theta_hi, n_theta)
    D_grid = np.linspace(D_lo, D_hi, n_D)

    rows = []
    eta_curves = {}  # for plotting panel A
    width_curves = {}  # for plotting panel B
    skipped = []

    for scheme in schemes:
        for selector in selectors:
            tilting, label = _build_tilting(scheme, selector)
            if tilting is None:
                print(f"[skip] {label}: artifact missing")
                skipped.append(label)
                continue
            print(f"[run ] {label}", flush=True)

            # Prime cell so disk/L1 caches + JAX warmup happen once.
            try:
                tilting.confidence_interval(
                    alpha, np.array([0.0]), model, prior, waldo,
                )
            except Exception:
                pass

            # η-curve
            eta_curve = _per_cell_eta_curve(
                tilting=tilting, selector_name=label,
                theta_grid=theta_grid,
                model=model, prior=prior, alpha=alpha, statistic=waldo,
            )
            eta_curves[label] = eta_curve

            # CI-width-curve
            width_curve = _per_cell_width_curve(
                tilting=tilting, label=label, D_grid=D_grid,
                model=model, prior=prior, alpha=alpha, statistic=waldo,
                n_jobs=n_jobs,
            )
            width_curves[label] = width_curve

            rows.append(_compute_metrics_row(
                scheme=scheme, selector=selector, metric_target="eta",
                x=theta_grid, y=eta_curve,
            ))
            rows.append(_compute_metrics_row(
                scheme=scheme, selector=selector, metric_target="ci_width",
                x=D_grid, y=width_curve,
            ))

    return rows, theta_grid, D_grid, eta_curves, width_curves, skipped


def _write_csv(rows, csv_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame.from_records(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path} ({len(df)} rows)")


def _render_plot(
    *, theta_grid, D_grid, eta_curves, width_curves, png_path: Path,
    schemes, plot_selector: str = "learned_intp",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"power_law": "#D6336C", "ot": "#1864AB",
              "mixture": "#2B8A3E", "fisher_rao": "#E67700"}

    for scheme in schemes:
        label = f"{scheme}[{plot_selector}]"
        if label in eta_curves:
            axA.plot(theta_grid, eta_curves[label],
                     label=scheme, color=colors.get(scheme, None), lw=1.6)
        if label in width_curves:
            axB.plot(D_grid, width_curves[label],
                     label=scheme, color=colors.get(scheme, None), lw=1.6)

    axA.set_xlabel("θ")
    axA.set_ylabel("η(θ)")
    axA.set_title(f"Panel A: η(θ) at {plot_selector}\n(NN sandbox: μ₀=0, σ₀=σ=1, α=0.05)")
    axA.axvspan(-5.0, 5.0, alpha=0.05, color="gray", label="σ₀-anchored training box")
    axA.legend(fontsize=8, loc="best")
    axA.grid(alpha=0.3)

    axB.set_xlabel("D (data)")
    axB.set_ylabel("CI width")
    axB.set_title(f"Panel B: dynamic-WALDO CI width vs D at {plot_selector}")
    axB.legend(fontsize=8, loc="best")
    axB.grid(alpha=0.3)

    fig.suptitle(
        "Geodesic-tilting smoothness: η(θ) vs CI-width(D)",
        fontsize=11,
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")


def _print_summary(rows, schemes, selectors) -> None:
    """Per-scheme median rank across 4 metrics × 2 targets."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        print("\n[summary] no rows; nothing to rank")
        return

    metric_cols = ["lipschitz", "tv", "discontinuity_count", "spectral_roughness"]
    # Per (selector, metric_target, metric): rank schemes 1 (smoothest) - 4 (roughest)
    rank_records = []
    for sel in selectors:
        for tgt in ("eta", "ci_width"):
            sub = df[(df["selector"] == sel) & (df["metric_target"] == tgt)]
            if sub.empty:
                continue
            for m in metric_cols:
                vals = sub.set_index("scheme")[m]
                ranks = vals.rank(method="min", na_option="bottom")
                for sch in schemes:
                    if sch in ranks.index:
                        rank_records.append({
                            "scheme": sch, "selector": sel,
                            "metric_target": tgt, "metric": m,
                            "value": float(vals.get(sch, float("nan"))),
                            "rank": int(ranks.get(sch, len(schemes))),
                        })
    if not rank_records:
        return
    rank_df = pd.DataFrame.from_records(rank_records)
    print("\n=== Per-scheme median rank (1 = smoothest) ===")
    pivot = (rank_df.groupby(["scheme"])["rank"].median()).sort_values()
    for sch, med in pivot.items():
        print(f"  {sch:<12s}  median rank = {med:.2f}")

    print("\n=== Median rank by (scheme, metric_target) ===")
    pivot2 = rank_df.groupby(["scheme", "metric_target"])["rank"].median().unstack()
    print(pivot2.to_string(float_format=lambda x: f"{x:5.2f}"))

    # Simple narrative: who wins the central hypothesis test.
    # The central question is CI-width discontinuity_count.
    central = df[
        (df["metric_target"] == "ci_width") & (df["selector"] == "dyn_numerical")
    ].set_index("scheme")["discontinuity_count"]
    if not central.empty:
        ranked = central.sort_values()
        print("\n=== Headline (central hypothesis): "
              "CI-width discontinuity_count at dyn_numerical ===")
        for sch in ranked.index:
            print(f"  {sch:<12s}  discontinuity_count = {int(ranked[sch])}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.compare_geodesic_smoothness")
    parser.add_argument("--schemes", nargs="*", choices=ALL_SCHEMES, default=ALL_SCHEMES)
    parser.add_argument("--selectors", nargs="*", choices=ALL_SELECTORS, default=ALL_SELECTORS)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Workers for per-D CI-width sweep (1=serial, -1=all cores).")
    parser.add_argument("--n-theta", type=int, default=200)
    parser.add_argument("--n-D", type=int, default=200)
    parser.add_argument("--plot-selector", default="learned_intp",
                        choices=ALL_SELECTORS,
                        help="Which selector slice to plot in Panel A/B (default: learned_intp).")
    args = parser.parse_args(argv)

    _require_hash_seed_pinned()
    _check_jax_available()

    rows, theta_grid, D_grid, eta_curves, width_curves, skipped = _compute_table(
        list(args.schemes), list(args.selectors),
        n_jobs=args.n_jobs, n_theta=args.n_theta, n_D=args.n_D,
    )

    project_root = Path(__file__).resolve().parents[1]
    sha = _git_sha_short()
    csv_path = project_root / "output" / "diagnostics" / f"compare_smoothness_{sha}.csv"
    png_path = project_root / "output" / "illustrations" / "compare_smoothness.png"

    _write_csv(rows, csv_path)
    _render_plot(
        theta_grid=theta_grid, D_grid=D_grid,
        eta_curves=eta_curves, width_curves=width_curves,
        png_path=png_path, schemes=list(args.schemes),
        plot_selector=args.plot_selector,
    )
    _print_summary(rows, list(args.schemes), list(args.selectors))

    if skipped:
        print(f"\n[note] skipped {len(skipped)} cell(s) due to missing artifacts: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
