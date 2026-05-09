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

Pairings of `bernoulli` with `Wald` and `WALDO` now run via the
**generic numerical paths** added in Phase 2 (see `docs/methods/wald.md`
and `waldo.md`): Wald uses `tau = (mle - theta)^2 * I(theta)` with
chi^2_1 calibration; WALDO uses an MC reference distribution under
H_0 sampled via `model.sample_data`. Pairings of `bernoulli` with
`power_law` still raise `NotImplementedError` — generic tilting is
Phase 3 work, separately tracked.

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

**CI region semantics — important.** Dynamic-η inversion can produce
a CI that is the union of multiple disjoint regions (the dynamic
p-value is non-monotone in θ at conflict). Two methods on
`TiltingScheme` resolve this:

- `confidence_regions(...) -> list[(lo, hi)]` returns the **actual
  union**: every disjoint accept-region, sorted. This is what
  `coverage` and `width` experiments call. `width` records
  `sum(hi - lo)` — true union width, NOT the convex hull.
- `confidence_interval(...) -> (lo, hi)` returns the **convex hull**
  `(min lo, max hi)` as a single-tuple summary. Use this only when a
  caller needs a single interval.

Single-region cells (Wald, identity, fixed-η static cells) have
`len(regions) == 1` so the two coincide. Multi-region cells (e.g.
`power_law[dynamic_numerical]` at extreme conflict) — `confidence_regions`
gives the honest answer, `confidence_interval` over-counts the gaps.

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

**Calibrated AND narrow: `LearnedDynamicEtaSelector`.** The
`numerical` dynamic selector is calibrated but inflates width by
~30 % at the conflict band (|Δ|≥2) because the inner
`NumericalEtaSelector` slams η to the lower clamp at small |Δ|.
The Phase G **conditional dual-head** learned selector (`EtaNet`
+ `ValidityNet`, in `src/frasian/learned/eta_artifact.py` +
`learned/training/`) trains a smooth GELU-MLP on
`(θ, prior_hp, lik_hp)` directly — no monotonicity prior, no
bounded sigmoid, no `(w, |Δ|)` features. Validity is enforced via
a `-log P(valid | θ, prior_hp, lik_hp, η)` boundary penalty driven
by Head B (`ValidityNet`), which learns the admissible region
from observed `(θ, prior_hp, lik_hp, η, valid)` triples during
training. **Conditional**: each checkpoint is trained over a
*range* of `(prior_hp, lik_hp)` (the `hyperparam_distribution` block
in the v4 YAML); the selector refuses inference when the observed
class doesn't match the trained `prior_class` / `model_class`, or
when the observed hyperparams fall outside the trained range. v4
fixtures are gitignored (`artifacts/learned_eta_*_v4.eqx`) — train
via `scripts.train_learned_eta --config
experiments/<config>_v4.yaml`. Activate via env var:

```
export FRASIAN_DEFAULT_DYNAMIC_ETA=learned   # vs default "numerical"
```

Headline empirical result on the canonical sandbox (w=0.5 — i.e.
σ=σ₀=1, μ₀=0, in the trained range; n_reps=200, α=0.05; Phase G
v4 fixture, smoke training budget):

```
                            θ=0    θ=1    θ=2    θ=3    θ=4
Wald                        3.92   3.92   3.92   3.92   3.92
bare WALDO                  3.33   3.43   3.75   4.23   4.78
power_law[numerical]        3.37   3.49   3.91   4.54   5.24   ← legacy: inflates
power_law[learned]          3.47   3.51   3.65   3.87   4.13   ← calibrated AND ≤ Wald
```

Numbers are from a fresh local v4 training (n_lhs=10k, batch=256,
~10 epochs early-stopped). They are **not bit-equal** to the
pre-Phase-G v3 fixed-prior fixture — the conditional architecture
trains over μ₀ ∈ [-2, 2], σ₀ ∈ [0.2, 5], σ ∈ [0.5, 2] instead of
the single (0, 1, 1) point, so it doesn't specialize as tightly to
the demo slice. The pattern (calibrated AND ≤ Wald, narrow at
conflict) is preserved; per-θ widths drift within ~0.1-0.2 vs the
v3 fixture, which is the cost of generality. Production retraining
(longer epochs / λ schedule) tightens the conflict-band tail.

To regenerate, run
`PYTHONHASHSEED=0 python -m scripts.regen_headline` (requires jax
+ equinox + the local `.eqx` fixtures).

The headline table is for `power_law` only. `ot[learned]` is wired
into `default_tiltings()` and runs through the coverage/width
experiments. See `docs/methods/learned_eta.md` for the full method
brief.

## Test Statistics (status)

| Name          | Status        | Notes                                          |
|---------------|---------------|------------------------------------------------|
| `wald`        | implemented   | Closed-form CI on Normal-Normal: D ± z·σ; identity-tilting only. Generic path on any Model: τ=(mle−θ)²·I(θ), χ²₁ calibration. |
| `waldo`       | implemented   | Closed-form on Normal-Normal+Normal: p(θ)=Φ(b−a)+Φ(−a−b). Generic path on any (Model, Prior): MC reference under H_0 via model.sample_data, knobs n_mc/seed (CRN across brentq). |
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

  _default_cells.py          # default (tiltings, statistics) cell list, env-var dispatch

  models/
    base.py                  # Model, Prior, Posterior, Likelihood protocols (incl. fingerprint())
    _dispatch.py             # require_model decorator (Normal-only gating)
    distributions.py         # NormalDistribution, BetaDistribution, Gaussian/Bernoulli likelihoods
    normal_normal.py         # NormalNormalModel + math primitives
    bernoulli.py             # BernoulliModel + Beta-conjugate posterior

  tilting/
    __init__.py              # Re-exports `TiltingScheme` / `EtaSelector` / `TiltingDomainError` only; concrete schemes are imported via `frasian._registry_bootstrap`.
    base.py                  # TiltingScheme, EtaSelector, TiltingDomainError
    _solvers.py              # ONE brentq_with_doubling
    _dynamic.py              # shared dynamic-η-per-θ CI scan (used by power_law/ot)
    _generic_pvalue.py       # shared generic-MC tilted_pvalue helpers (CRN seed, support resolver)
    _grid_distribution.py    # GridDistribution: tilted-density grid materialiser
    eta_selectors.py         # Fixed / Numerical / DynamicNumerical / LearnedDynamic
    identity.py              # IdentityTilting (no-op identity element)
    power_law.py             # PowerLawTilting (e-geodesic / Theorem 6)
    ot.py                    # OTTilting (W2 geodesic, general 1D + Gaussian fast path)
    quantile_mixture.py      # QuantileMixturePath: 1D W2-geodesic Distribution wrapper
    fisher_rao.py            # planned stub
    mixture.py               # planned stub

  statistics/
    base.py                  # TestStatistic + AsymptoticDistribution
    wald.py                  # WaldStatistic
    waldo.py                 # WaldoStatistic
    lrt.py                   # planned stub
    signed_root.py           # planned stub
    bartlett.py              # planned stub

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
      identity_demo.py, wald_demo.py, waldo_demo.py,
      power_law_demo.py, ot_demo.py, learned_eta_demo.py,
      smoothness_demo.py, confidence_distribution_demo.py

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
    base.py                  # LearnedArtifact protocol
    null.py                  # NullArtifact (for tests)
    eta_artifact.py          # Phase G EtaArtifact (loads dual-head v4 Equinox checkpoint)
    training/
      __init__.py            # public surface: EtaNet, ValidityNet, fit_eta_artifact, ...
      architecture.py        # EtaNet ((θ, prior_hp, lik_hp) → η) + ValidityNet ((θ, prior_hp, lik_hp, η) → logit), GELU MLPs
      losses.py              # integrated_p / cd_variance / static_width + boundary_penalty_from_validity
      validity.py            # is_pair_valid, validity_mask, compute_pvalues_per_sample{,_with_hp}
      sampling.py            # ExperimentConfig (v4), ThetaDistribution, lhs_1d
      hyperparam_distribution.py  # HyperparamDistribution: per-batch (prior_hp, lik_hp) sampler
      pvalue_jax.py          # JAX ports of tilted_pvalue per scheme (autograd path; post-Phase-F port from torch)
      cd_jax.py              # JAX port of CD density (for cd_variance loss)
      train.py               # fit_eta_artifact: dual-head training loop
      _setup.py              # device resolver / x64 / RNG plumbing for fit_eta_artifact
      _train_loop.py         # the training step + eval loop driving optax
      _losses_compose.py     # per-scheme loss composition (NN closed-form + generic-grid)
      _checkpoint.py         # Equinox .eqx file I/O (CHECKPOINT_FORMAT_VERSION = 4)
      _validity_data.py      # offline (θ, prior_hp, lik_hp, η, valid) sampler used to seed Head B

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
  train_learned_eta.py       # train Phase G EtaNet+ValidityNet from a v4 experiment YAML

experiments/
  canonical_normal_normal_powerlaw_v4.yaml  # Phase G v4 conditional fixture (NN + power_law)
  canonical_normal_normal_ot_v4.yaml        # Phase G v4 conditional fixture (NN + ot)
  canonical_bernoulli_powerlaw_v4.yaml      # Phase G v4 conditional fixture (Bernoulli + power_law)

artifacts/                   # trained Phase G v4 checkpoints; gitignored — train locally

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
python -m pytest                    # ~1530 collected (~25 stub-skips), ~3 min
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

**Gitignored-artifact policy.** `git status --porcelain` ignores
gitignored paths, so swapping a checkpoint at
`artifacts/learned_eta_*_v1.eqx` (a gitignored production
artifact) does **not** flip the tree to dirty and the cache key
does not move on `git_sha` alone. Invalidation in that case hinges
on `selector_artifact_fingerprint` (24-char digest of the artifact
file) being plumbed into the cache key via the cell's `extra`
dict — the `_runner` does this automatically for any selector
that exposes `.artifact.fingerprint()`. If you train a fresh
`v1` artifact, the next cache lookup returns a miss and recomputes;
if the runner cannot read the fingerprint (artifact removed mid-
run, fingerprint method raises), it bails to a recompute. Never
edit a checkpoint file in place — replace atomically. Pinned by
`tests/regression/test_cache_learned_invalidation.py`.

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
| 10   | done   | Phase E learned-η rewrite: model-agnostic dual-head selector (`EtaNet` + `ValidityNet`) trained per-experiment; replaces Phase D `MonotonicEtaNet` (deleted). `EtaNet`: smooth GELU-MLP from raw θ → η, no monotonicity prior, no bounded sigmoid. `ValidityNet`: learns `P(valid \| θ, η)` from observed `(θ, η, valid)` triples; provides `-log P(valid)` boundary penalty for Head A. Per-experiment fingerprints (`Prior.fingerprint()`/`Model.fingerprint()`) + strict cross-experiment refusal. New `ExperimentConfig` YAML schema. Smoke fixtures `learned_eta_canonical_normal_normal_<scheme>_v0_smoke.eqx` committed for `power_law` and `ot` (Equinox `.eqx` format after the Phase F JAX port; `.pt` torch checkpoints are gone). |
| 11   | done   | Phase G conditional learned-η: `EtaNet` and `ValidityNet` now take `(θ, prior_hp, lik_hp)` / `(θ, prior_hp, lik_hp, η)` instead of θ / (θ, η). Each checkpoint is trained over a *range* of `(prior_hp, lik_hp)` (the `hyperparam_distribution` block in the v4 YAML); the selector dispatches per-inference and refuses out-of-range hyperparams. Checkpoint format bumped 3 → 4. v3 YAMLs / fixtures / 14 v3-bound tests deleted; v4 YAMLs added; v4 fixtures gitignored (re-train via `scripts.train_learned_eta`). `eta_explore_box` restored to ExperimentConfig (per-config; defaults to `[-5, 5]`). Headline numbers in this file updated to the v4 fixture. |

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
