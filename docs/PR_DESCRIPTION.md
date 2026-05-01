# Refactor: clean-slate framework for Frasian inference research

Replaces the legacy single-purpose code with a research framework
organised around the **(TiltingScheme × TestStatistic) cross-product**
as the central object of study. The legacy implementation is preserved
verbatim under `legacy/` for reference.

## Why

The original codebase was built around one specific artifact (power-law
η-tilting + WALDO statistic on the conjugate-Normal sandbox) and could
not host the comparative research the project actually wants:

- 894-line `tilting.py` with three near-identical brentq blocks and
  three overlapping optimal-η solvers.
- Module-level mutable MLP cache (`_get_optimal_eta_predictor`) that
  broke parallelism and testability.
- Plot scripts with α=0.2 in one and α=0.05 in others.
- No abstraction for "swap in a different tilting scheme and re-run
  the diagnostics."

This PR replaces that with six runtime-checkable protocols, a plugin
registry, a content-keyed mandatory cache, and a workflow integration
(subagents + slash commands + CI gates) that makes adding a new
mathematical method a one-file change plus its brief, property tests,
and illustration.

## What's in the design matrix

| Kind        | Implemented                              | Scheduled stubs |
|-------------|------------------------------------------|-----------------|
| Models      | normal_normal, bernoulli                 | —               |
| Tiltings    | identity, power_law                      | ot_normal, geodesic_normal, mixture, exp_family |
| Statistics  | wald, waldo                              | lrt, signed_root, bartlett |
| Experiments | coverage, width, smoothness              | —               |

Stubs are first-class: each is registered, briefed (9 required
sections), and has skipped property tests. Flipping skip → pass is the
unit of progress in Phase 3.

## The smoothness diagnostic — central technical contribution

The framework's load-bearing claim is now falsifiable. On a 51-point
sweep at w=0.5, α=0.05:

|                       | Lipschitz η* | TV(η*) | Discontinuities | Spectral roughness |
|-----------------------|-------------:|-------:|----------------:|-------------------:|
| (power_law, **wald**) | ~0           | ~0     | 0               | ~0                 |
| (power_law, **waldo**)| **17.1**     | 2.14   | **11**          | 0.11               |

The kink near |Δ|≈0.3 (where η* leaves the admissible-range clamp at
−w/(1−w)) is now quantitatively detectable. Future tilting schemes
(`ot_normal`, `geodesic_normal`, etc.) are evaluated against these
numbers — there's a hard L4 test that asserts the (power_law, waldo)
cell *must* show Lipschitz>1 and at least one discontinuity outlier.

## Migration steps (each commit on the branch)

| Step | What it landed |
|-----:|----------------|
| 1    | Six protocols, decorator registry, Config, empty-registry runner, conftest fixtures |
| 2    | Math ported byte-for-byte (atol=1e-12 regression tests against legacy formulas); 3 brentq copies → one solver; module-global MLP cache → injected `LearnedArtifact`; 4 method briefs |
| 3    | npz+json result storage; mandatory content-keyed cache (git-sha + Config fingerprint + raw fingerprint); `RawSamples`; `LearnedArtifact` protocol |
| 4    | `CoverageExperiment` + `WidthExperiment` + their diagnostics; `confidence_interval` added to `TestStatistic`; `scripts/run.py --fast` and `scripts/figures.py`; manifests byte-reproducible at same `(Config, sha)` |
| 5    | `SmoothnessExperiment` + `NumericalEtaSelector` + smoothness diagnostic with Lipschitz / TV / discontinuity-count / spectral-roughness metrics |
| 6    | 7 scheduled stubs (4 tiltings + 3 statistics) with full briefs + skipped property tests + `registry.X.implemented()` filter |
| 7    | `.claude/agents/` (skeptic, literature-reviewer, deriver), `.claude/commands/` (/critique, /litreview, /derive, /propose-method), `.github/workflows/` (ci.yaml + method-completeness.yaml) |

Phase-3 follow-ups bundled in this PR:

- **(3)** CI workflows triggered on push to `claude/**` branches; locally verified.
- **(4)** `PowerLawTilting.dynamic_tilted_*` — port of the legacy dynamic-η CI engine. The Phase-8 refactor folded the formerly-separate `DynamicCIExperiment` into the `coverage` / `width` cells (via `power_law[dynamic_numerical]`).
- **(5)** `BernoulliModel` + `BetaDistribution` + `models/_dispatch.py` proving the Model protocol generalises beyond Normal-Normal; pairings with Normal-only methods raise uniform `NotImplementedError`s.

## Verification

```bash
python -m pytest                                  # 423 passed, 32 skipped
python tools/check_method_completeness.py         # 16 entries OK
python -m scripts.run --fast experiment=smoothness # full pipeline ~30s
python -m scripts.figures results/smoothness      # regenerate figures
```

Coverage: 92% with both pytest and the completeness checker as gates.
The 8% gap is mostly `raise NotImplementedError` in scheduled stubs.

## What this PR does not do

- Implement the four stub tilting schemes (OT, geodesic, mixture, exp-family). Each is a `/propose-method <name> tilting` exercise.
- Implement the three stub statistics (LRT, signed-root, Bartlett). On the Normal-location sandbox they reduce to Wald; the value is for non-Gaussian extensions.
- Generalise `WaldStatistic` / `WaldoStatistic` / `PowerLawTilting` to non-Normal models. The protocols accommodate this; the math is its own research.

These are the obvious Phase-3 entry points and are intentionally left as scaffolded stubs so the next contributor (human or `/propose-method`) has a clear target.

## Files

13 commits, ~10k LOC added (~5k tests + ~3k impl + ~2k docs). The
`legacy/` directory contains the original ~5k LOC unchanged.

## Reviewing

If reading commit-by-commit, Steps 1, 5, and 7 are the highest-signal
architectural commits. If reading by area:

- **Protocols**: `src/frasian/{models,tilting,statistics,cd,experiments,diagnostics}/base.py`
- **Workflow integration**: `.claude/agents/`, `.claude/commands/`, `docs/workflows.md`
- **Smoothness contribution**: `src/frasian/experiments/smoothness.py`,
  `src/frasian/diagnostics/smoothness_metrics.py`,
  `tests/experiments/test_smoothness_experiment.py`
- **Cache discipline**: `src/frasian/simulation/cache.py`,
  `tests/regression/test_cache.py`

For a quick smell-test: `python -m scripts.run --list` enumerates the 16
registered methods; `python -m frasian.experiments.illustrations.smoothness_demo --smoke` produces the η*(|Δ|) figure with the kink visible.
