"""Regenerate the headline empirical CI-width table across schemes.

Reproduces and generalises the table cited in ``CLAUDE.md`` and
``docs/methods/learned_eta.md``. The default invocation (all 4 schemes,
all 3 learned heads) produces an 18-row table:

```
                                              θ=0    θ=1    θ=2    θ=3    θ=4
Wald                                          3.92   3.92   3.92   3.92   3.92
bare WALDO                                    3.33   3.43   3.75   4.23   4.78
power_law[numerical]                          ...
power_law[learned_intp]                       ...
power_law[learned_cd_var]                     ...
power_law[learned_static_w]                   ...
ot[numerical]                                 ...
ot[learned_intp]                              ...
... etc.
```

Numbers are NOT bit-equal to pre-port torch numbers — JAX's PRNG
primitive differs. The qualitative pattern (`<scheme>[learned_intp]`
calibrated AND ≤ Wald, narrow at conflict) is preserved.

The ``[numerical]`` rows use ``DynamicNumericalEtaSelector`` —
calibrated dynamic-η (η = η(θ), no D dependence → exact 1-α coverage),
NOT the post-selection static ``NumericalEtaSelector``. Each scheme's
dynamic-numerical η(θ) lookup is **disk-cached** at
``artifacts/eta_lookups/dyn_numerical_<hash>.npz`` (gitignored,
parallels NN artifacts). First run computes & saves; subsequent runs
load instantly.

Per-scheme admissibility (per the deriver work — see PL/OT briefs and
``tilting/eta_selectors.py:_maybe_clamp_eta``):

  - power_law:  η < 1/(1-w)              (upper-only; no finite lower)
  - ot:         η > -√w/(1-√w)           (lower-only; no finite upper)
  - mixture:    η ∈ [0, 1]               (structural sigmoid bound)
  - fisher_rao: η ∈ ℝ                    (geodesically complete; no clamp)

The runtime LearnedDynamicEtaSelector enforces these bounds; the
[numerical] rows compute their η inside admissibility by construction
(width-minimisation over the admissible range).

jax + equinox required; the script lazily imports them and prints a
clear error if either is unavailable. See ``docs/methods/learned_eta.md``
for the wider methodology.

Usage::

    python -m scripts.regen_headline                                # all 4 schemes × all 3 heads
    python -m scripts.regen_headline --fast                         # smaller MC grid
    python -m scripts.regen_headline --schemes power_law            # one scheme only
    python -m scripts.regen_headline --learned-heads intp           # one head only
    python -m scripts.regen_headline --n-jobs -1                    # all cores
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path


def _require_hash_seed_pinned() -> None:
    """Audit P0-12: refuse to run unless PYTHONHASHSEED is pinned.

    Python's default randomised hash makes any code path that uses
    `hash(...)` non-reproducible across processes. The headline
    regeneration must produce the same numbers every run; we refuse to
    proceed unless the user has pinned the seed externally. (Setting it
    in-process is too late — the interpreter has already initialised
    its hash randomisation.)
    """
    if os.environ.get("PYTHONHASHSEED") not in ("0", "random_pinned"):
        sys.stderr.write(
            "\nERROR: scripts.regen_headline requires PYTHONHASHSEED=0 to be\n"
            "set in the environment BEFORE Python starts. Otherwise any code\n"
            "path that uses Python's `hash(...)` (including the narrowness\n"
            "test's seed derivation) is process-local and the headline numbers\n"
            "drift across runs.\n\n"
            "Re-run with:\n\n"
            "    PYTHONHASHSEED=0 python -m scripts.regen_headline [args]\n\n"
        )
        raise SystemExit(2)


def _check_jax_available() -> None:
    """Lazy jax/equinox availability check with a clear error path."""
    try:
        import jax  # noqa: F401
        import equinox  # noqa: F401
    except ImportError as exc:
        sys.stderr.write(
            "\nERROR: scripts.regen_headline requires jax + equinox to load the\n"
            "Phase E v4 checkpoints. Install jax + equinox and retry.\n"
            f"Underlying ImportError: {exc}\n\n"
            "See docs/methods/learned_eta.md for the wider methodology.\n"
        )
        raise SystemExit(1) from exc


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

_HEAD_FILE_TOKEN = {
    "intp": "integrated_p",
    "cd_var": "cd_variance",
    "static_w": "static_width",
}


def _learned_fixture_path(scheme: str, head: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    suffix = _SCHEME_ARTIFACT_SUFFIX[scheme]
    return (
        project_root
        / "artifacts"
        / f"learned_eta_canonical_normal_normal_{suffix}_phaseC_{_HEAD_FILE_TOKEN[head]}_v4.eqx"
    )


def _import_tilting_class(scheme: str):
    import importlib

    mod_name, cls_name = _TILTING_CLASS[scheme].split(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def _per_d_widths(
    D: float,
    *,
    alpha: float,
    model,
    prior,
    identity,
    wald_stat,
    waldo_stat,
    cells: list,  # list of (label, tilt) tuples; statistic is always waldo
) -> dict[str, float]:
    """Compute CI widths for one D sample across all cells.

    Top-level for joblib pickling. NaN signals an exception; the
    aggregator counts NaNs as failures and aborts if any cell exceeds
    the failure threshold.
    """
    import numpy as np

    data = np.array([D])
    out: dict[str, float] = {}

    def _safe(label, tilt, stat):
        try:
            lo, hi = tilt.confidence_interval(alpha, data, model, prior, stat)
            out[label] = float(hi - lo)
        except Exception:
            out[label] = float("nan")

    _safe("Wald", identity, wald_stat)
    _safe("bare WALDO", identity, waldo_stat)
    for label, tilt in cells:
        _safe(label, tilt, waldo_stat)
    return out


def _compute_table(
    theta_grid: list[float],
    n_reps: int,
    schemes: list[str],
    learned_heads: list[str],
    n_jobs: int,
) -> tuple[list[str], dict[str, list[float]]]:
    """Compute mean CI width per (cell, θ_true) across multiple schemes.

    Row order: Wald → bare WALDO → for each scheme: <scheme>[numerical]
    then <scheme>[learned_<head>] for each available head. Missing
    learned-head artifacts are skipped with a log line.
    """
    import numpy as np

    from frasian import Config
    from frasian._parallel import parallel_map
    from frasian._registry_bootstrap import bootstrap
    from frasian.learned.eta_artifact import EtaArtifact
    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel
    from frasian.statistics.wald import WaldStatistic
    from frasian.statistics.waldo import WaldoStatistic
    from frasian.tilting.eta_selectors import (
        DynamicNumericalEtaSelector,
        LearnedDynamicEtaSelector,
    )
    from frasian.tilting.identity import IdentityTilting

    bootstrap()

    cfg = Config.default().from_overrides(n_reps=n_reps, alpha=0.05)
    sigma = 1.0
    sigma0 = 1.0  # → w = 0.5
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=0.0, scale=sigma0)

    waldo = WaldoStatistic()
    wald = WaldStatistic()
    identity = IdentityTilting()

    cells: list[tuple[str, object]] = []
    for scheme in schemes:
        TiltingCls = _import_tilting_class(scheme)
        # Numerical row: shared DynamicNumerical selector instance per scheme
        # (cache reuse across repeated invocations is via the disk L2 layer).
        cells.append(
            (f"{scheme}[numerical]", TiltingCls(selector=DynamicNumericalEtaSelector()))
        )
        # Learned rows: one per available head.
        for head in learned_heads:
            path = _learned_fixture_path(scheme, head)
            if not path.exists():
                print(f"[skip] {scheme}[learned_{head}]: artifact missing at {path}")
                continue
            cells.append(
                (
                    f"{scheme}[learned_{head}]",
                    TiltingCls(
                        selector=LearnedDynamicEtaSelector(
                            artifact=EtaArtifact(artifact_path=path),
                        )
                    ),
                )
            )

    # Prime each cell once in the main process so the in-memory state
    # (DynNumerical's dict + L2 disk write; LearnedDynamic's `_loaded`
    # latch + JAX) is set before workers pickle copies. Without this,
    # every worker would pay the same first-call cost independently.
    dummy_data = np.array([0.0])
    for label, tilt in cells:
        try:
            tilt.confidence_interval(cfg.alpha, dummy_data, model, prior, waldo)
        except Exception:
            pass

    row_labels = ["Wald", "bare WALDO", *[label for label, _ in cells]]
    out: dict[str, list[float]] = {label: [] for label in row_labels}

    max_fail_fraction = 0.05

    worker = partial(
        _per_d_widths,
        alpha=cfg.alpha,
        model=model,
        prior=prior,
        identity=identity,
        wald_stat=wald,
        waldo_stat=waldo,
        cells=cells,
    )

    rng = np.random.default_rng(cfg.seed)
    for theta_true in theta_grid:
        D_samples = rng.normal(loc=theta_true, scale=sigma, size=cfg.n_reps)
        per_d_results = parallel_map(worker, D_samples.tolist(), n_jobs=n_jobs)

        widths: dict[str, list[float]] = {k: [] for k in row_labels}
        fails: dict[str, int] = {k: 0 for k in row_labels}
        for r in per_d_results:
            for cell, w in r.items():
                if np.isnan(w):
                    fails[cell] += 1
                else:
                    widths[cell].append(w)

        for cell, n_fail in fails.items():
            frac = n_fail / max(cfg.n_reps, 1)
            if frac > max_fail_fraction:
                raise RuntimeError(
                    f"Headline regen aborting: cell {cell!r} at θ={theta_true} "
                    f"failed {n_fail}/{cfg.n_reps} reps "
                    f"({100*frac:.1f}% > {100*max_fail_fraction:.0f}% threshold). "
                    f"Silent NaN in the headline table is not acceptable; "
                    f"investigate the cell's exception path."
                )

        for cell, ws in widths.items():
            out[cell].append(float(np.mean(ws)) if ws else float("nan"))

    return row_labels, out


def _print_table(
    theta_grid: list[float], row_labels: list[str], rows: dict[str, list[float]]
) -> None:
    label_width = max(28, max(len(label) for label in row_labels) + 2)
    header = " " * label_width + "    ".join(f"θ={int(t)}" for t in theta_grid)
    print(header)
    for label in row_labels:
        vals = "   ".join(f"{v:.2f}" for v in rows[label])
        print(f"{label:<{label_width}}{vals}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.regen_headline")
    parser.add_argument(
        "--fast", action="store_true", help="run a small sweep (n_reps=50) for sanity"
    )
    parser.add_argument(
        "--n-reps", type=int, default=200, help="MC reps per θ (default 200, matches headline)"
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        choices=tuple(_SCHEME_ARTIFACT_SUFFIX),
        default=list(_SCHEME_ARTIFACT_SUFFIX),
        help=(
            "tilting schemes supplying the [numerical] and [learned_*] rows. "
            "Default: all four (power_law, ot, mixture, fisher_rao)."
        ),
    )
    parser.add_argument(
        "--learned-heads",
        nargs="*",
        choices=tuple(_HEAD_FILE_TOKEN),
        default=list(_HEAD_FILE_TOKEN),
        help=(
            "trained heads to include as separate rows per scheme. Default: "
            "all three (intp, cd_var, static_w). Missing artifacts are "
            "skipped with a log line."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Worker processes for the per-D-sample inner loop (joblib loky). "
            "Default 1 (serial). -1 = all available cores. JAX state stays "
            "per-worker; expect ~1-2s startup per worker. Use with --n-reps "
            ">= 200 to amortise."
        ),
    )
    args = parser.parse_args(argv)

    _require_hash_seed_pinned()
    _check_jax_available()

    theta_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    n_reps = 50 if args.fast else args.n_reps
    row_labels, rows = _compute_table(
        theta_grid,
        n_reps=n_reps,
        schemes=list(args.schemes),
        learned_heads=list(args.learned_heads),
        n_jobs=args.n_jobs,
    )
    _print_table(theta_grid, row_labels, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
