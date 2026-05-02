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
from frasian import default_cells, run_experiment, registry, Config

tiltings, statistics = default_cells(experiment="coverage")
summary = run_experiment(
    experiment=registry.experiments["coverage"](),
    tiltings=tiltings,                   # [IdentityTilting(), PowerLawTilting(selector=DynamicNumericalEtaSelector(...))]
    statistics=statistics,               # [WaldStatistic(), WaldoStatistic()]
    config=Config.fast(),                # or Config.default()
    out_dir=Path("results/coverage"),
)
```

`default_cells(experiment="smoothness")` returns the bare-family
tiltings instead, since the smoothness experiment sweeps the parameter
itself and would produce duplicate output if the cell carried a
non-default selector.

The runner persists each cell's `RawResult` through the cache layer (keyed
on Config fingerprint + git sha + raw fingerprint), runs the experiment's
diagnostics, and writes a `manifest.json` with relative cache paths.

## Models (status)

| Name           | Status        | Notes                                     |
|----------------|---------------|-------------------------------------------|
| `normal_normal`| implemented   | 1D conjugate Gaussian; framework's sandbox |
| `bernoulli`    | implemented   | Beta-conjugate; non-Normal protocol check  |

Pairings of `bernoulli` with `Wald` / `WALDO` / `power_law` raise
`NotImplementedError` (by design — the Normal-only methods name the
restriction explicitly via `models/_dispatch.require_model`).

## Tilting Schemes (status)

| Name         | Status        | Notes                                                                                |
|--------------|---------------|--------------------------------------------------------------------------------------|
| `identity`   | implemented   | No-op tilting; identity element of the matrix                                        |
| `power_law`  | implemented   | e-geodesic / log-linear (geometric mean); Theorem 6 closed form on Normal-Normal     |
| `ot`         | implemented   | W2 geodesic; general 1D quantile-mixture, Gaussian fast path; posterior↔likelihood   |
| `mixture`    | stub          | m-geodesic / linear-density (arithmetic mean); dual partner of `power_law`           |
| `fisher_rao` | stub          | Levi-Civita geodesic (intrinsic Riemannian); Gaussian half-plane closed form planned |

These four cover the canonical geodesic taxonomy: e-/m-geodesics
(`power_law`/`mixture` — the dually-flat pair under the Fisher
metric), the Levi-Civita / Fisher-Rao geodesic (`fisher_rao`), and
the Wasserstein geodesic (`ot`). An `exp_family` natural-parameter
scheme would be redundant with `power_law` on conjugate exponential
families and is omitted by design.

Each `TiltingScheme` owns its **selector** (`EtaSelector`) as a
constructor argument: `FixedEtaSelector` is the static identity,
`NumericalEtaSelector` is static-with-context (post-selection — see
caveat below), and `DynamicNumericalEtaSelector` varies η per θ. The
cell name picks up the selector when non-default, so
`power_law[dynamic_numerical]` is the **Dynamic-WALDO** cell when
paired with `waldo`. The `identity` tilting + a uniform
`tilting.confidence_interval(alpha, data, model, prior, statistic)`
API let coverage / width / smoothness all share one cell loop.

**Calibration caveat — important.** The framework's calibrated default
is `DynamicNumericalEtaSelector`: η at each θ depends only on θ (not
on D), so the WALDO p-value at any fixed η is U[0,1] under H0 and the
CI achieves exact 1-α coverage. The static `NumericalEtaSelector`
gives strictly narrower CIs (≤ Wald asymptotically) but is post-
selection inference and **undercovers by ~2 points at α=0.05** —
pinned by `tests/regression/test_post_selection_coverage.py`. The
static selector is exposed via `post_selection_demo_tiltings()` only
to make the trade-off measurable; never use it for production CI
estimation.

## Test Statistics (status)

| Name          | Status        | Notes                                          |
|---------------|---------------|------------------------------------------------|
| `wald`        | implemented   | Closed-form CI: D ± z·σ; identity-tilting only |
| `waldo`       | implemented   | p(θ) = Φ(b−a) + Φ(−a−b); numerical CI inversion |
| `lrt`         | stub          | -2 log Λ; on Normal-location reduces to Wald   |
| `signed_root` | stub          | sign·√LRT; on Normal-location equals Wald      |
| `bartlett`    | stub          | LRT/E[LRT]; decorator over base LRT (planned)  |

Each statistic declares which tiltings it accepts via
`accepts_tilting(tilting)`. Wald accepts only `identity` (it ignores
the prior, so non-identity cells are degenerate duplicates). Other
statistics accept any tilting by default. The runner gates incompatible
cells before `run_cell` and records them in the manifest with
`status="incompatible"`.

## Experiments (status)

| Name                       | Status      | Output                                              |
|----------------------------|-------------|-----------------------------------------------------|
| `coverage`                 | implemented | Empirical coverage on (θ_true, w) grid              |
| `width`                    | implemented | Mean CI width on (θ_true, w) grid                   |
| `smoothness`               | implemented | η*(|Δ|) Lipschitz / TV / discontinuity / spectral   |
| `confidence_distribution`  | implemented | CD median / 95-width / W₁-to-Wald / non-monotone fraction |

All three experiments dispatch CI computation through
`tilting.confidence_interval(alpha, data, model, prior, statistic)` —
the tilting owns its selector and so produces plain WALDO at
`(identity, waldo)` and Dynamic-WALDO at `(power_law[dynamic_numerical],
waldo)` from the same loop. The legacy `dynamic_ci` experiment was
folded into `coverage` / `width` (it was effectively those two
diagnostics on a different selector); the underlying
`PowerLawTilting.dynamic_tilted_*` engine still exists.

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
    eta_selectors.py         # Fixed / Numerical / DynamicNumerical EtaSelectors
    identity.py              # IdentityTilting (no-op identity element)
    power_law.py             # PowerLawTilting (e-geodesic / Theorem 6)
    ot.py                    # OTTilting (W2 geodesic, general 1D + Gaussian fast path)
    quantile_mixture.py      # QuantileMixturePath: 1D W2-geodesic Distribution wrapper
    {fisher_rao,mixture}.py  # planned stubs

  statistics/
    base.py                  # TestStatistic + AsymptoticDistribution
    wald.py                  # WaldStatistic
    waldo.py                 # WaldoStatistic
    {lrt,signed_root,bartlett}.py  # planned stubs

  cd/
    base.py                  # ConfidenceDistribution protocol
    grid.py                  # GridConfidenceDistribution (concrete)
    from_pvalue.py           # universal constructor (p-value → density → CDF)
    from_closed_form.py      # closed-form Wald/WALDO CDs (test fixtures)
    distances.py             # wasserstein_1, wasserstein_2, total_variation

  experiments/
    base.py                  # Experiment, ExperimentContext, RawResult
    coverage.py              # CoverageExperiment
    width.py                 # WidthExperiment
    smoothness.py            # SmoothnessExperiment
    confidence_distribution.py  # ConfidenceDistributionExperiment
    illustrations/           # one demo per registered method
      {identity,wald,waldo,power_law,smoothness,confidence_distribution}_demo.py

  diagnostics/
    base.py                  # Diagnostic + DiagnosticTable
    coverage_table.py        # CoverageRateDiagnostic
    width_table.py           # MeanWidthDiagnostic
    smoothness_metrics.py    # SmoothnessDiagnostic
    cd_summary.py            # CDSummaryDiagnostic

  simulation/
    storage.py               # npz + json sidecar I/O
    cache.py                 # mandatory content-keyed cache
    raw.py                   # generate_normal_D_samples + RawSamples
    processing.py            # ProcessedResult dataclass
    runner.py                # persist_cell helper

  learned/
    base.py                  # LearnedArtifact protocol (unused; future hook)
    null.py                  # NullArtifact (for tests)

tests/
  conftest.py                # autouse registry isolation + bootstrapped fixture
  properties/                # hypothesis-based protocol invariants (L1)
  regression/                # tight-tolerance baselines (L0/L2)
  experiments/               # end-to-end Experiment runs (L4)
  integration/               # registry + empty-registry checks

docs/
  methods/                   # one .md brief per registered method
    _template.md             # the skeleton /propose-method writes against
    {normal_normal,bernoulli,wald,waldo,power_law,...}.md
  workflows.md               # /propose-method lifecycle + cookbook

scripts/
  run.py                     # python -m scripts.run [--list] [--fast] experiment=<name>
  figures.py                 # python -m scripts.figures <results_dir>

tools/
  check_method_completeness.py  # verify brief + tests + illustration per method

.claude/
  agents/                    # skeptic, literature-reviewer, deriver
  commands/                  # /critique, /litreview, /derive, /propose-method

.github/workflows/
  ci.yaml                    # pytest L0-L4 on Python 3.11/3.12
  method-completeness.yaml   # check_method_completeness + smoke runs

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

The `.claude/` subagents (`skeptic`, `literature-reviewer`, `deriver`),
slash commands (`/propose-method`, `/critique`, `/derive`, `/litreview`),
and GitHub Actions workflows (`ci.yaml` running L0-L4 tests +
`method-completeness.yaml` running `tools/check_method_completeness.py`
plus illustration smoke runs) gate brief / property-test / illustration
completeness on every PR. See `docs/workflows.md` for the full lifecycle.

## Running Things

```bash
# List registered methods
python -m scripts.run --list

# Run an experiment end-to-end (Config.fast() ~ 30s for coverage)
python -m scripts.run --fast experiment=coverage
python -m scripts.run --fast experiment=width
python -m scripts.run --fast experiment=smoothness

# Regenerate figures + CSVs from a persisted results dir
python -m scripts.figures results/coverage

# Run the test suite
python -m pytest                    # all 406 passing + 32 stub-skipped
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
| 5    | done   | SmoothnessExperiment quantifying the power-law discontinuity    |
| 6    | done   | Stubs: OT / geodesic / mixture / exp_family / LRT / SR / BCLRT  |
| 7    | done   | .claude/ subagents + slash commands + GitHub Actions CI gates   |
| 8    | done   | Selector-as-tilting-member refactor: IdentityTilting + accepts_tilting + uniform `tilting.confidence_interval`; dynamic_ci subsumed by coverage/width |
| 9    | done   | Geodesic taxonomy refactor: `ot_normal`→`ot` (general 1D W2 + Gaussian fast path implemented); `geodesic_normal`→`fisher_rao` (renamed stub); `exp_family` dropped (redundant with `power_law` on conjugate exp-families); `mixture` reframed as m-geodesic / dual partner of `power_law`'s e-geodesic |

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
