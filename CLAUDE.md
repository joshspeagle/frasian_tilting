# Frasian Inference Framework

## Project Overview

A research framework for studying alternatives to power-law tilting in the
Frasian inference setting. The driving question: **does optimal-transport
or geodesic interpolation between prior, likelihood, and posterior produce
smoother families than naive power-law tilting, and which test statistics
exploit them best?**

The framework is structured as a **(TiltingScheme × TestStatistic) cross-
product**. Every cell of the matrix produces a `RawResult` consumed by a
shared diagnostic suite (coverage, width, smoothness). Adding a new tilting
or statistic is a one-file change plus its brief, property tests, and demo.

The legacy code that motivated this design lives under `legacy/` for
reference. The migration status (which Steps are done) is at the bottom.

## The Conjugate Normal Sandbox

All experiments operate on the 1D conjugate Normal-Normal model:

- **Prior**: θ ~ N(μ₀, σ₀²)
- **Likelihood**: D | θ ~ N(θ, σ²)
- **Posterior**: θ | D ~ N(μₙ, σₙ²)
- **Data weight**: w = σ₀²/(σ² + σ₀²) ∈ (0, 1)
- **Posterior mean**: μₙ = wD + (1-w)μ₀
- **Posterior std**: σₙ = √w · σ

**Key quantities** used by tilting / WALDO / smoothness:
- Scaled prior-data conflict: Δ = (1-w)(μ₀ - D)/σ
- Prior residual: δ(θ) = (θ - μ₀)/σ₀
- Non-centrality: λ(θ) = (1-w)²(μ₀ - θ)² / (w² σ²)

The protocols in `src/frasian/models/base.py` are designed so future work
can add non-conjugate or non-Gaussian models without rewriting consumers,
but **everything implemented today is Normal-Normal only**. Generic
posterior inference is a flagged future extension.

## Architecture: Six Protocols

Every concrete implementation conforms to one of:

| Protocol                | Purpose                              | Module                            |
|-------------------------|--------------------------------------|-----------------------------------|
| `Model`                 | Sample data, build posterior, MLE    | `frasian/models/base.py`          |
| `TiltingScheme`         | tilt(posterior, prior, lik, η)→post  | `frasian/tilting/base.py`         |
| `TestStatistic`         | evaluate, pvalue, CI, accept region  | `frasian/statistics/base.py`      |
| `ConfidenceDistribution`| pdf/cdf/quantile/interval/mean/mode  | `frasian/cd/base.py`              |
| `Experiment`            | grid → cells → diagnostics           | `frasian/experiments/base.py`     |
| `Diagnostic`            | RawResult → DiagnosticTable + figure | `frasian/diagnostics/base.py`     |

Plus `EtaSelector` and `LearnedArtifact` for the support layer.

### Plugin Registry

Concrete implementations register via decorators:

```python
@register_tilting(name="power_law", brief="docs/methods/power_law.md")
class PowerLawTilting:
    ...
```

The decorator records the brief path so `tools/check_method_completeness.py`
can verify every registered class ships with documentation, property tests,
and an illustration. A single `_registry_bootstrap.py` enumerates the
imports so static tooling can find every method.

User-facing call:

```python
from frasian import run_experiment, registry

summary = run_experiment(
    experiment=registry.experiments["coverage"](),
    tiltings=registry.tiltings.all(),
    statistics=registry.statistics.all(),
    config=Config.fast(),                # or Config.default()
    out_dir=Path("results/coverage"),
)
```

The runner persists each cell's `RawResult` through the cache layer (keyed
on Config fingerprint + git sha + raw fingerprint), runs the experiment's
diagnostics, and writes a `manifest.json` with relative cache paths.

## Tilting Schemes (status)

| Name           | Status        | Notes                                        |
|----------------|---------------|----------------------------------------------|
| `power_law`    | implemented   | Ported Theorem 6 from legacy `tilting.py`    |
| `ot_normal`    | scheduled stub| Optimal transport between Gaussians          |
| `geodesic_normal` | scheduled stub | Fisher-Rao geodesic for Gaussians         |
| `mixture`      | scheduled stub| Mixture-path interpolation                   |
| `exp_family`   | scheduled stub| Exponential-family interpolation             |

## Test Statistics (status)

| Name      | Status        | Notes                                         |
|-----------|---------------|-----------------------------------------------|
| `wald`    | implemented   | Closed-form CI: D ± z·σ                       |
| `waldo`   | implemented   | p(θ) = Φ(b−a) + Φ(−a−b); numerical CI inversion |
| `lrt`     | scheduled stub|                                                 |
| `signed_root` | scheduled stub|                                             |
| `bartlett` | scheduled stub| Implemented as decorator over LRT             |

## Experiments (status)

| Name        | Status        | Output                                              |
|-------------|---------------|-----------------------------------------------------|
| `coverage`  | implemented   | Empirical coverage on (θ_true, w) grid              |
| `width`     | implemented   | Mean CI width on (θ_true, w) grid                   |
| `smoothness`| scheduled     | Lipschitz / spectral / W₁ path-length on η-sweep    |
| `dynamic_ci`| scheduled stub| Dynamic-η CIs from the legacy framework             |

The `coverage` and `width` cells currently use `eta = scheme.param_space.
eta_identity` for every tilting; the Tilting dimension becomes load-bearing
when the smoothness experiment lands.

## Project Structure

```
src/frasian/
  __init__.py                # public surface: run_experiment, registry, Config
  config.py                  # frozen settings (alpha, grids, eps, seeds)
  _registry.py               # decorators + Registry singleton
  _registry_bootstrap.py     # explicit imports of every concrete impl
  _runner.py                 # cross-product runner + manifest writer
  _errors.py                 # FrasianError, EmptyRegistryError, etc.

  models/
    base.py                  # Model, Prior, Posterior, Likelihood protocols
    distributions.py         # NormalDistribution, GaussianLikelihood
    normal_normal.py         # NormalNormalModel + math primitives

  tilting/
    base.py                  # TiltingScheme, EtaSelector, TiltingDomainError
    _solvers.py              # ONE brentq_with_doubling
    power_law.py             # PowerLawTilting (Theorem 6)
    {ot_normal,geodesic_normal,mixture,exp_family}.py  # planned stubs

  statistics/
    base.py                  # TestStatistic + AsymptoticDistribution
    wald.py                  # WaldStatistic
    waldo.py                 # WaldoStatistic
    {lrt,signed_root,bartlett}.py  # planned stubs

  cd/
    base.py                  # ConfidenceDistribution, CDFamily
    factory.py               # planned: build_cd dispatcher
    {from_pvalue,from_closed_form,from_tilted}.py  # planned

  experiments/
    base.py                  # Experiment, ExperimentContext, RawResult
    coverage.py              # CoverageExperiment
    width.py                 # WidthExperiment
    smoothness.py            # planned (Step 5)
    illustrations/           # one demo per registered method
      {wald,waldo,power_law}_demo.py

  diagnostics/
    base.py                  # Diagnostic + DiagnosticTable
    coverage_table.py        # CoverageRateDiagnostic
    width_table.py           # MeanWidthDiagnostic
    smoothness_metrics.py    # planned (Step 5)

  simulation/
    storage.py               # npz + json sidecar I/O
    cache.py                 # mandatory content-keyed cache
    raw.py                   # generate_normal_D_samples + RawSamples
    processing.py            # ProcessedResult dataclass
    runner.py                # persist_cell helper

  learned/
    base.py                  # LearnedArtifact protocol
    null.py                  # NullArtifact (for tests)
    eta_lookup.py            # planned: monotonic η* MLP port

  plotting/                  # planned: shared style + primitives

tests/
  conftest.py                # autouse registry isolation + bootstrapped fixture
  properties/                # hypothesis-based protocol invariants (L1)
  regression/                # tight-tolerance baselines (L0/L2)
  experiments/               # end-to-end Experiment runs (L4)
  integration/               # registry + empty-registry checks

experiments/                 # planned: versioned analysis configs (yaml)

docs/
  methods/                   # one .md brief per registered method
    _template.md             # the skeleton /propose-method writes against
    {normal_normal,wald,waldo,power_law,coverage_experiment,width_experiment}.md
  architecture.md            # planned: rationale doc
  workflows.md               # planned: explains /critique, /derive, /propose-method

scripts/
  run.py                     # python -m scripts.run [--list] [--fast] experiment=<name>
  figures.py                 # python -m scripts.figures <results_dir>

tools/
  check_method_completeness.py  # verify brief + tests + illustration per method

.claude/                     # planned (Step 7)
  agents/                    # skeptic, literature-reviewer, deriver
  commands/                  # /critique, /litreview, /derive, /propose-method

.github/workflows/           # planned (Step 7): CI gating method completeness

legacy/                      # archived original implementation; reference only
```

## Test Layering

| Layer | Purpose                                                  | Tools                  |
|-------|----------------------------------------------------------|------------------------|
| L0    | Scalar math primitives (atol 1e-12)                      | pytest                 |
| L1    | Protocol invariants (continuity, identity, calibration)  | pytest + hypothesis    |
| L2    | Array-result regression vs committed baselines           | pytest + .npz / formula|
| L3    | Statistical (KS uniformity, coverage at nominal level)   | pytest + seeded RNG    |
| L4    | End-to-end Experiment runs on small grids                | pytest                 |
| L5    | Cross-product cells, smoke mode (nightly)                | pytest + parallel      |

Markers replace the legacy `tier1`...`tier5`. Hypothesis lives only in L1.
The L3 calibration check (Wald p-values uniform under H0) lives in
`tests/properties/test_wald_invariants.py`.

## Method Brief Discipline

Every registered class ships with a markdown brief at the path declared in
its decorator. Briefs follow `docs/methods/_template.md` with these
required sections (CI-checked by `tools/check_method_completeness.py`):

1. Summary
2. Motivation
3. Definition
4. Derivation
5. Predicted behavior
6. Failure modes
7. Invariants (mirrored as property tests)
8. Literature
9. Links

Step 7 wires the `.claude/` subagents (`skeptic`, `literature-reviewer`,
`deriver`) and slash commands (`/propose-method`, `/critique`, `/derive`,
`/litreview`) that orchestrate brief authoring + critique. Until then,
briefs are written by hand from the template.

## Running Things

```bash
# List registered methods
python -m scripts.run --list

# Run an experiment end-to-end (Config.fast() ~ 30s for coverage)
python -m scripts.run --fast experiment=coverage
python -m scripts.run --fast experiment=width

# Regenerate figures + CSVs from a persisted results dir
python -m scripts.figures results/coverage

# Run the test suite
python -m pytest                    # all 245 tests
python -m pytest -m L0              # math primitives only
python -m pytest -m "L0 or L1"      # core + properties
python -m pytest -m L4              # end-to-end
python -m pytest -n auto            # parallel (pytest-xdist)

# Verify method-completeness (briefs, tests, illustrations)
python tools/check_method_completeness.py

# Run an illustration
python -m frasian.experiments.illustrations.power_law_demo --smoke
```

## Cache Discipline

The cache at `<results_dir>/cache/` keys on `(experiment, tilting,
statistic, config_fingerprint, git_sha, raw_fingerprint, extra)` →
24-char SHA-256 digest. **Dirty git trees never hit the cache** —
uncommitted changes always recompute. Same `(config, sha)` on a
clean tree is byte-reproducible.

## Migration Status

| Step | Status | Description                                                     |
|------|--------|-----------------------------------------------------------------|
| 1    | done   | Scaffolding: protocols, registry, Config, empty-registry runner |
| 2    | done   | Wald + WALDO + power_law + NormalNormalModel + briefs + demos   |
| 3    | done   | simulation/ (storage, cache, raw, runner) + LearnedArtifact     |
| 4    | done   | CoverageExperiment + WidthExperiment + diagnostics + figures.py |
| 5    | next   | SmoothnessExperiment quantifying the power-law discontinuity    |
| 6    | next   | Stub OT / geodesic / mixture / LRT / signed-root / Bartlett     |
| 7    | next   | .claude/ subagents + slash commands + GitHub Actions CI gates   |

## Key Anti-Patterns to Avoid

- Importing from `frasian.models.normal_normal` outside `frasian/models/`.
  Other modules go through the `Model` protocol; the discipline keeps the
  framework extensible.
- Module-level mutable state. The legacy `_get_optimal_eta_predictor`
  global is replaced by an injected `LearnedArtifact`.
- Bypassing the cache. Every script goes through `simulation.cache.
  get_or_compute`; the legacy plot scripts that recomputed grids in the
  module body are gone.
- Using α≠0.05 in one place and 0.05 elsewhere. All α values come from
  `Config.alpha`; tests override via `Config.from_overrides(alpha=...)`.
- Adding a method without its brief, property tests, and illustration —
  the completeness check blocks merge.
