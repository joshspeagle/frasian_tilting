# Frasian Inference Framework

A research framework for studying alternatives to power-law tilting in
the Frasian inference setting. The driving question:

> Does optimal-transport or geodesic interpolation between prior,
> likelihood, and posterior produce smoother families than naive
> power-law tilting, and which test statistics exploit them best?

The framework is structured as a **(TiltingScheme × TestStatistic) cross-
product**. Every cell of the matrix produces a `RawResult` consumed by a
shared diagnostic suite (coverage, width, smoothness, dynamic-η CIs).
Adding a new tilting or statistic is a one-file change plus its brief,
property tests, and demo.

The legacy implementation that motivated this design lives under
`legacy/` for reference. See `CLAUDE.md` for the architecture and
`docs/workflows.md` for the contribution lifecycle.

## Installation

```bash
pip install -e ".[dev]"
```

Python 3.11+ recommended; the test matrix runs on 3.11 and 3.12.

## Quick start

```python
from pathlib import Path

from frasian import Config, registry, run_experiment

# Bootstrap concrete implementations into the registry
from frasian._registry_bootstrap import bootstrap
bootstrap()

# Run any registered experiment over the (Tilting × Statistic) cross-product
summary = run_experiment(
    experiment=registry.experiments["smoothness"](),
    tiltings=registry.tiltings.implemented(),    # skip stubs
    statistics=registry.statistics.implemented(),
    config=Config.fast(),                        # ~30s; Config.default() for full
    out_dir=Path("results/smoothness"),
)

print(f"ran {len(summary.cells)} cells; manifest at {summary.out_dir}/manifest.json")
```

Or via the CLI:

```bash
python -m scripts.run --list                              # discover methods
python -m scripts.run --fast experiment=coverage          # ~30s
python -m scripts.run --fast experiment=smoothness        # ~30s
python -m scripts.run --fast experiment=dynamic_ci        # ~1 min
python -m scripts.figures results/smoothness              # regenerate figures
```

## What's registered today

| Kind        | Count | Implemented                                | Stubs |
|-------------|------:|--------------------------------------------|-------|
| Models      | 2     | normal_normal, bernoulli                   | —     |
| Tiltings    | 5     | power_law                                  | ot_normal, geodesic_normal, mixture, exp_family |
| Statistics  | 5     | wald, waldo                                | lrt, signed_root, bartlett |
| Experiments | 4     | coverage, width, smoothness, dynamic_ci    | —     |

`python -m scripts.run --list` for the live status.

## The smoothness diagnostic

The framework's central technical contribution: empirically detects
whether a tilting scheme produces sharp transitions in `η*(|Δ|)`. On a
51-point sweep at w=0.5, α=0.05:

|                       | Lipschitz η* | TV(η*) | Discontinuities |
|-----------------------|-------------:|-------:|----------------:|
| (power_law, **wald**) | ~0           | ~0     | 0               |
| (power_law, **waldo**)| **17.1**     | 2.14   | **11**          |

The Wald row is the smoothness floor (η-independent). The Waldo row
shows the kink at |Δ|≈0.3 where η* leaves the admissible-range clamp.
Future tilting schemes (`ot_normal`, `geodesic_normal`, `mixture`,
`exp_family`) must beat these numbers to justify their existence.

## Adding a new method

```
/propose-method ot_normal tilting
```

This Claude Code slash command orchestrates the framework's three
subagents:
- **`deriver`** — produces a verified `Derivation` section using
  symbolic + numerical cross-check on the conjugate-Normal sandbox.
- **`literature-reviewer`** — finds real BibTeX citations.
- **`skeptic`** — adversarial review with file:line attack vectors.

It scaffolds the brief, source file, property tests, and illustration.
See `docs/workflows.md` for the full lifecycle and step-by-step
cookbook.

## Tests

```bash
python -m pytest                    # 423 tests + 32 stub-skips, ~1 min
python -m pytest -m "L0 or L1"      # math primitives + invariants
python -m pytest -m L4              # end-to-end experiments only
python tools/check_method_completeness.py    # gate every PR runs
```

Test layers (markers replace the legacy `tier1`–`tier5`):

- **L0** scalar primitives, `atol=1e-12`
- **L1** protocol invariants via hypothesis
- **L2** array-result regression vs committed baselines
- **L3** statistical (KS uniformity, coverage at nominal level)
- **L4** end-to-end Experiment runs on small grids
- **L5** cross-product cells, smoke mode, nightly only

## Repository layout

```
src/frasian/         the framework (models, tilting, statistics, cd,
                     experiments, diagnostics, simulation, learned)
tests/               L0/L1/L2 in regression+properties; L4 in experiments;
                     integration for registry contracts
docs/methods/        per-method briefs (one .md per registered class)
docs/workflows.md    /propose-method lifecycle + contribution cookbook
.claude/             subagents (skeptic, literature-reviewer, deriver)
                     and slash commands (/critique, /litreview, /derive,
                     /propose-method)
.github/workflows/   ci.yaml + method-completeness.yaml
scripts/             run.py + figures.py
tools/               check_method_completeness.py
legacy/              archived original implementation; reference only
```

## Status

Step-by-step migration from the legacy code is complete (see
`CLAUDE.md` Migration Status). The framework is in shape to host
research on alternative tilting families and richer cross-product
comparisons.

## License

MIT — see `LICENSE`.
