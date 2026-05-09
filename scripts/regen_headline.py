"""Regenerate the headline empirical CI-width table.

Reproduces the table cited in ``CLAUDE.md`` and
``docs/methods/learned_eta.md`` (post-Phase-F JAX/Equinox numbers
from the v4 checkpoint):

```
                            θ=0    θ=1    θ=2    θ=3    θ=4
Wald                        3.92   3.92   3.92   3.92   3.92
bare WALDO                  3.33   3.43   3.75   4.23   4.78
power_law[numerical]        3.36   3.49   3.91   4.54   5.24
power_law[learned]          3.63   3.64   3.68   3.75   3.82
```

Numbers above are NOT bit-equal to the pre-port torch numbers — JAX's
PRNG primitive differs from torch's even at the same nominal seed, so
re-trained Equinox weights drift within ~1× MC standard error (~0.05
across α=0.05 narrowness MC repeats). The qualitative pattern
(power_law[learned] calibrated AND ≤ Wald, narrow at conflict) is
preserved.

jax + equinox required; the script lazily imports them and prints a
clear error if either is unavailable. See ``docs/methods/learned_eta.md``
for the wider methodology.

Usage::

    python -m scripts.regen_headline                # default Config
    python -m scripts.regen_headline --fast         # smaller grid
    python -m scripts.regen_headline --n-reps 200   # custom reps

The script loads the v4 checkpoints from ``artifacts/``,
runs the canonical Normal-Normal width sweep at θ ∈ {0, 1, 2, 3, 4}
with w=0.5, α=0.05, and prints the table in the same format as the
docs. The OT learned cell is included for completeness even though
the v4 OT checkpoint is undertrained (Head B accuracy ~0.67).
"""

# jax + equinox required; see docs/methods/learned_eta.md

from __future__ import annotations

import argparse
import os
import sys
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
    """Lazy jax/equinox availability check with a clear error path.

    Avoids a hard import-time dependency so this module can be at least
    imported (and listed by `python -m scripts.regen_headline --help`)
    in environments without jax installed.
    """
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


def _check_artifacts_present() -> None:
    """Verify the local v4 checkpoints exist before running.

    Phase G v4 fixtures are not committed (gitignored as conditional
    fixtures are large and re-trainable from the YAMLs). Train them
    via ``python -m scripts.train_learned_eta --config
    experiments/canonical_*_v4.yaml``.
    """
    project_root = Path(__file__).resolve().parents[1]
    artifacts = [
        project_root / "artifacts" / "learned_eta_canonical_normal_normal_powerlaw_v4.eqx",
        project_root / "artifacts" / "learned_eta_canonical_normal_normal_ot_v4.eqx",
    ]
    missing = [p for p in artifacts if not p.exists()]
    if missing:
        sys.stderr.write(
            "\nERROR: missing v4 checkpoint(s):\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nRun `python -m scripts.train_learned_eta --config "
            "experiments/<config>_v4.yaml` to (re)train. Phase G v4\n"
            "checkpoints are gitignored — trained locally per developer.\n\n"
        )
        raise SystemExit(1)


def _compute_table(theta_grid: list[float], n_reps: int) -> dict[str, list[float]]:
    """Run the four cells (Wald, bare WALDO, power_law[numerical],
    power_law[learned]) at each θ in ``theta_grid`` and return a dict
    mapping cell name to mean CI width per θ.

    Imports happen here (after the jax availability check) so the
    module is inspectable without jax installed.
    """
    import numpy as np

    from frasian import Config
    from frasian._registry_bootstrap import bootstrap
    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel
    from frasian.statistics.wald import WaldStatistic
    from frasian.statistics.waldo import WaldoStatistic
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

    # Build power_law cells with numerical and learned dynamic selectors.
    # The env-var dispatch in default_cells handles both modes; we
    # explicitly switch via FRASIAN_DEFAULT_DYNAMIC_ETA.
    from frasian._default_cells import default_tiltings  # local import

    os.environ["FRASIAN_DEFAULT_DYNAMIC_ETA"] = "numerical"
    pl_numerical = next(
        t
        for t in default_tiltings()
        if getattr(t, "cell_name", "") == "power_law[dynamic_numerical]"
    )

    os.environ["FRASIAN_DEFAULT_DYNAMIC_ETA"] = "learned"
    pl_learned = next(
        t
        for t in default_tiltings()
        if "learned" in getattr(t, "cell_name", "")
        and getattr(t, "cell_name", "").startswith("power_law")
    )

    rng = np.random.default_rng(cfg.seed)
    out: dict[str, list[float]] = {
        "Wald": [],
        "bare WALDO": [],
        "power_law[numerical]": [],
        "power_law[learned]": [],
    }

    # Abort if more than this fraction of reps fail at any θ for any cell.
    # Silent NaN in headline regen defeats the script's purpose; a small
    # number of failures (<5%) at the conflict band can be tolerated, but
    # any larger fraction means the cell is broken or the data are
    # outside the calibrated regime — surface, don't silently emit NaN.
    max_fail_fraction = 0.05

    for theta_true in theta_grid:
        D_samples = rng.normal(loc=theta_true, scale=sigma, size=cfg.n_reps)

        widths: dict[str, list[float]] = {k: [] for k in out}
        fails: dict[str, int] = {k: 0 for k in out}
        for D in D_samples:
            data = np.array([D])
            try:
                lo, hi = identity.confidence_interval(cfg.alpha, data, model, prior, wald)
                widths["Wald"].append(hi - lo)
            except Exception:
                fails["Wald"] += 1
            try:
                lo, hi = identity.confidence_interval(cfg.alpha, data, model, prior, waldo)
                widths["bare WALDO"].append(hi - lo)
            except Exception:
                fails["bare WALDO"] += 1
            try:
                lo, hi = pl_numerical.confidence_interval(cfg.alpha, data, model, prior, waldo)
                widths["power_law[numerical]"].append(hi - lo)
            except Exception:
                fails["power_law[numerical]"] += 1
            try:
                lo, hi = pl_learned.confidence_interval(cfg.alpha, data, model, prior, waldo)
                widths["power_law[learned]"].append(hi - lo)
            except Exception:
                fails["power_law[learned]"] += 1

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

    return out


def _print_table(theta_grid: list[float], rows: dict[str, list[float]]) -> None:
    header = "                            " + "    ".join(f"θ={int(t)}" for t in theta_grid)
    print(header)
    for label in ("Wald", "bare WALDO", "power_law[numerical]", "power_law[learned]"):
        vals = "   ".join(f"{v:.2f}" for v in rows[label])
        print(f"{label:<28}{vals}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.regen_headline")
    parser.add_argument(
        "--fast", action="store_true", help="run a small sweep (n_reps=50) for sanity"
    )
    parser.add_argument(
        "--n-reps", type=int, default=200, help="MC reps per θ (default 200, matches headline)"
    )
    args = parser.parse_args(argv)

    _require_hash_seed_pinned()
    _check_jax_available()
    _check_artifacts_present()

    theta_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    n_reps = 50 if args.fast else args.n_reps
    rows = _compute_table(theta_grid, n_reps=n_reps)
    _print_table(theta_grid, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
