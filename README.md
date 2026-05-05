# Frasian Inference Framework

A research framework for studying alternatives to power-law tilting in
the Frasian inference setting. The driving question:

> Does optimal-transport or geodesic interpolation between prior,
> likelihood, and posterior produce smoother families than naive
> power-law tilting, and which test statistics exploit them best?

The framework is structured as a **(TiltingScheme × TestStatistic) cross-
product**. Every cell of the matrix produces a `RawResult` consumed by a
shared diagnostic suite (coverage, width, smoothness, and the full
**confidence distribution** — CD-median / 95% width / W₁-to-Wald /
non-monotonicity flag). Adding a new tilting or statistic is a one-file
change plus its brief, property tests, and demo.

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

from frasian import Config, default_cells, registry, run_experiment

# `default_cells` returns the calibrated (identity, wald), (identity, waldo),
# (power_law[dynamic_numerical], waldo) triple — the framework's headline
# Wald / WALDO / Dyn-WALDO comparison.
tiltings, statistics = default_cells(experiment="confidence_distribution")
summary = run_experiment(
    experiment=registry.experiments["confidence_distribution"](),
    tiltings=tiltings,
    statistics=statistics,
    config=Config.fast(),                        # ~30s coverage; ~5min CD
    out_dir=Path("results/confidence_distribution"),
)

print(f"ran {len(summary.cells)} cells; manifest at {summary.out_dir}/manifest.json")
```

Or via the CLI:

```bash
python -m scripts.run --list                                    # discover methods
python -m scripts.run --fast experiment=coverage                # ~30s
python -m scripts.run --fast experiment=width                   # ~30s
python -m scripts.run --fast experiment=smoothness              # ~30s
python -m scripts.run --fast experiment=confidence_distribution # ~5min (CD per replicate)
python -m scripts.figures results/coverage                      # regenerate figures
```

## What's registered today

| Kind        | Count | Implemented                                                  | Stubs |
|-------------|------:|--------------------------------------------------------------|-------|
| Models      | 2     | normal_normal, bernoulli                                     | —     |
| Tiltings    | 5     | identity, power_law, ot                                      | fisher_rao, mixture |
| Statistics  | 5     | wald, waldo                                                  | lrt, signed_root, bartlett |
| Experiments | 4     | coverage, width, smoothness, confidence_distribution         | —     |

`python -m scripts.run --list` for the live status.

The framework's headline cells are:

| Cell                                       | Method      |
|--------------------------------------------|-------------|
| `(identity, wald)`                         | Wald        |
| `(identity, waldo)`                        | WALDO (η=0) |
| `(power_law[dynamic_numerical], waldo)`    | Dynamic-WALDO (per-θ varying η, calibrated) |

`(power_law[numerical], waldo)` is also exposed via
`post_selection_demo_tiltings()` to demonstrate the post-selection
coverage shortfall of the static η\*-opt construction (~93% empirical
vs nominal 95%); it is not a production estimator.

## The smoothness diagnostic

The framework's central technical contribution: empirically detects
whether a tilting scheme produces sharp transitions in `η*(|Δ|)`. On a
51-point sweep at w=0.5, α=0.05:

|                       | Lipschitz η* | TV(η*) | Discontinuities |
|-----------------------|-------------:|-------:|----------------:|
| (identity, **wald**)  | ~0           | ~0     | 0               |
| (power_law, **waldo**)| **17.1**     | 2.14   | **11**          |

The `(identity, wald)` row is the smoothness floor (η-independent).
The `(power_law, waldo)` row shows the kink at |Δ|≈0.3 where η* leaves
the admissible-range clamp. Alternative tilting schemes — `ot` (W2
geodesic, implemented), `fisher_rao` (Levi-Civita geodesic, stub), and
`mixture` (m-geodesic, stub) — must beat these numbers to justify
their existence.

## The confidence distribution

The framework promotes per-α coverage/width to the full **CD** —
constructed from the cell's `tilting.pvalue(...)` via Schweder–Hjort
density `c(θ) = ½|dp/dθ|` (renormalised), with the cdf derived as the
cumulative integral of the non-negative density (always monotone, even
for multimodal p-values like Dyn-WALDO under conflict). Distance to a
Wald reference CD is reported as **W₁** (CDF-form trapezoidal — exact
to ~1e-7 for σ-mismatched Gaussians) with **W₂** available via Gauss–
Hermite quadrature on `z = Φ⁻¹(u)` (matches Olkin–Pukelsheim closed
form to ~1e-5).

For Dyn-WALDO, the inversion-based confidence curve C(θ) becomes
*non-monotone* at conflict — the smoothness pathology surfacing in
distributional form. The CD experiment reports `nonmonotone_fraction`
per cell; on `Config.fast()` it lights up to ~100% in heavy-conflict
corners (large |θ_true|, small w).

## Adding a new method

```
/propose-method ot tilting
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
python -m pytest                    # ~830 tests across L0-L4 (37 stub-skips), ~2 min
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
