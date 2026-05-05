"""Regenerate the headline empirical CI-width table.

Reproduces the table cited in ``CLAUDE.md`` and
``docs/methods/learned_eta.md``:

```
                            θ=0    θ=1    θ=2    θ=3    θ=4
Wald                        3.92   3.92   3.92   3.92   3.92
bare WALDO                  3.32   3.44   3.75   4.24   4.85
power_law[numerical]        3.35   3.50   3.92   4.53   5.23
power_law[learned]          3.67   3.67   3.67   3.71   3.80
```

torch required; the script lazily imports it and prints a clear
error if it is unavailable. See ``docs/methods/learned_eta.md``
for the wider methodology.

Usage::

    python -m scripts.regen_headline                # default Config
    python -m scripts.regen_headline --fast         # smaller grid
    python -m scripts.regen_headline --n-reps 200   # custom reps

The script loads the v0_smoke checkpoints from ``artifacts/``,
runs the canonical Normal-Normal width sweep at θ ∈ {0, 1, 2, 3, 4}
with w=0.5, α=0.05, and prints the table in the same format as the
docs. The OT learned cell is included for completeness even though
the v0_smoke OT checkpoint is undertrained (Head B accuracy ~0.67).
"""

# torch required; see docs/methods/learned_eta.md

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _check_torch_available() -> None:
    """Lazy torch availability check with a clear error path.

    Avoids a hard import-time dependency so this module can be at least
    imported (and listed by `python -m scripts.regen_headline --help`)
    in environments without torch installed.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        sys.stderr.write(
            "\nERROR: scripts.regen_headline requires torch to load the\n"
            "Phase E v0_smoke checkpoints. Install torch (>=2.0) and retry.\n"
            f"Underlying ImportError: {exc}\n\n"
            "See docs/methods/learned_eta.md for the wider methodology.\n"
        )
        raise SystemExit(1) from exc


def _check_artifacts_present() -> None:
    """Verify the committed v0_smoke checkpoints exist before running."""
    project_root = Path(__file__).resolve().parents[1]
    artifacts = [
        project_root / "artifacts" / "learned_eta_canonical_normal_normal_powerlaw_v0_smoke.pt",
        project_root / "artifacts" / "learned_eta_canonical_normal_normal_ot_v0_smoke.pt",
    ]
    missing = [p for p in artifacts if not p.exists()]
    if missing:
        sys.stderr.write(
            "\nERROR: missing v0_smoke checkpoint(s):\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nRun `python -m scripts.train_learned_eta --config "
            "experiments/<config>.yaml` to (re)train, or check out the\n"
            "branch where the checkpoints are committed.\n\n"
        )
        raise SystemExit(1)


def _compute_table(theta_grid: list[float], n_reps: int) -> dict[str, list[float]]:
    """Run the four cells (Wald, bare WALDO, power_law[numerical],
    power_law[learned]) at each θ in ``theta_grid`` and return a dict
    mapping cell name to mean CI width per θ.

    Imports happen here (after the torch check) so the module is
    inspectable without torch installed.
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
        t for t in default_tiltings() if t.cell_name == "power_law[dynamic_numerical]"
    )

    os.environ["FRASIAN_DEFAULT_DYNAMIC_ETA"] = "learned"
    pl_learned = next(
        t
        for t in default_tiltings()
        if "learned" in getattr(t, "cell_name", "") and t.cell_name.startswith("power_law")
    )

    rng = np.random.default_rng(cfg.seed)
    out: dict[str, list[float]] = {
        "Wald": [],
        "bare WALDO": [],
        "power_law[numerical]": [],
        "power_law[learned]": [],
    }

    for theta_true in theta_grid:
        D_samples = rng.normal(loc=theta_true, scale=sigma, size=cfg.n_reps)

        widths = {k: [] for k in out}
        for D in D_samples:
            data = np.array([D])
            try:
                lo, hi = identity.confidence_interval(cfg.alpha, data, model, wald)
                widths["Wald"].append(hi - lo)
            except Exception:
                pass
            try:
                lo, hi = identity.confidence_interval(cfg.alpha, data, model, waldo)
                widths["bare WALDO"].append(hi - lo)
            except Exception:
                pass
            try:
                lo, hi = pl_numerical.confidence_interval(cfg.alpha, data, model, prior, waldo)
                widths["power_law[numerical]"].append(hi - lo)
            except Exception:
                pass
            try:
                lo, hi = pl_learned.confidence_interval(cfg.alpha, data, model, prior, waldo)
                widths["power_law[learned]"].append(hi - lo)
            except Exception:
                pass

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

    _check_torch_available()
    _check_artifacts_present()

    theta_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    n_reps = 50 if args.fast else args.n_reps
    rows = _compute_table(theta_grid, n_reps=n_reps)
    _print_table(theta_grid, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
