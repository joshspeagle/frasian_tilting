# Workflows

How the framework's critique / literature-review / derivation /
proposal workflow fits together. The artifacts under `.claude/`,
`.github/workflows/`, `docs/methods/`, and `tools/` form a single
integrated discipline; this doc explains the rationale.

## The discipline

Every registered method (`TiltingScheme`, `TestStatistic`,
`ConfidenceDistribution`, `Experiment`, `Diagnostic`) ships with
*four* artifacts:

1. **Source** at `src/frasian/<area>/<name>.py`, registered via the
   appropriate decorator with a `brief=` path.
2. **Brief** at `docs/methods/<name>.md` with the nine sections
   required by `tools/check_method_completeness.py`.
3. **Property tests** at `tests/properties/test_<name>_*.py` (or
   `tests/experiments/test_<name>_*.py` for experiments) — the
   invariants from the brief, mirrored as executable assertions.
   Stubs may have `pytest.mark.skip` placeholders.
4. **Illustration** at `src/frasian/experiments/illustrations/<name>_demo.py`
   that produces a PNG in `--smoke` mode. (Stubs may skip this.)

CI gates merges on all four. A PR that adds
`src/frasian/tilting/foo.py` without `docs/methods/foo.md`,
`tests/properties/test_foo_*.py`, and `experiments/illustrations/foo_demo.py`
fails the `method-completeness` workflow.

## Subagents

Three subagents live in `.claude/agents/`. Each has a narrow,
opinionated job and is invoked by Claude Code on demand:

- **skeptic** — adversarial reviewer. Reads a method file or brief
  and produces a numbered list of attack vectors with file:line
  refs, ending with `block` / `accept-with-caveats` / `accept`. Has
  read-only tools (Read, Grep, Glob, Bash). Used by `/critique` and
  by `/propose-method`'s skeptic step.

- **literature-reviewer** — finds real citations. Returns BibTeX
  entries classified as foundational / closely-related / contrasting,
  and audits the brief for uncited claims. Has WebFetch + WebSearch.
  Used by `/litreview` and `/propose-method`.

- **deriver** — produces or verifies the brief's Derivation
  section. Cross-checks every step symbolically (sympy) or
  numerically on the conjugate-Normal sandbox. Output is markdown
  ready to paste into the brief. Used by `/derive` and
  `/propose-method`.

## Slash commands

Four commands in `.claude/commands/` orchestrate the subagents:

- **/critique &lt;path&gt;** — runs the skeptic on a single file. Use
  when a method has just been written and you want a hostile read
  before opening a PR.

- **/litreview &lt;name&gt; [&lt;description&gt;]** — runs the literature-reviewer
  on a method. The output is a markdown block ready to paste into
  the brief's Literature section.

- **/derive &lt;name&gt;** — runs the deriver on a method whose Definition
  section is filled in. Output is a Derivation block plus an
  Invariants block.

- **/propose-method &lt;name&gt; &lt;kind&gt;** — the orchestrator. Walks the
  user through brief scaffolding, deriver invocation, literature
  review, source/test/illustration generation, and a final skeptic
  pass. Pauses for review at each step. The expected entry point
  for any new method.

## Typical lifecycle

A new method goes from zero to merged via:

```
/propose-method ot tilting

  ↓ deriver fills Derivation
  ↓ literature-reviewer fills Literature
  ↓ scaffolds src/, tests/, brief/, illustration/
  ↓ skeptic returns "accept-with-caveats"
  ↓ user fixes caveats

# verify locally
python -m pytest tests/properties/test_ot_invariants.py
python tools/check_method_completeness.py
python -m frasian.experiments.illustrations.ot_demo --smoke

# open PR
git push -u origin feat/ot

  ↓ ci.yaml runs L0-L4 pytest
  ↓ method-completeness.yaml runs the checker + smoke-runs all demos
  ↓ both green → merge
```

A subsequent refinement (e.g. flipping a `pytest.mark.skip` to a
passing assertion when an invariant becomes provable):

```
/critique src/frasian/tilting/ot.py

  ↓ skeptic flags missing edge case at sigma_a → sigma_b
  ↓ user adds the corresponding property test
  ↓ skeptic re-runs → "accept"
  ↓ commit + PR
```

## CI gates (`.github/workflows/`)

Two workflows trigger on every PR:

- **ci.yaml** — installs the package, runs `pytest -m "not L5"` on
  Python 3.11 and 3.12. L5 is the nightly cross-product smoke layer
  and is not gated on PRs.

- **method-completeness.yaml** — installs the package, runs
  `tools/check_method_completeness.py`, then smoke-runs every
  illustration script. Fails the PR if any registered method is
  missing its brief, property tests, or illustration; fails if any
  brief is missing one of the nine required sections; fails if any
  illustration script raises in `--smoke` mode.

## Cookbook: adding a method by hand (when not using `/propose-method`)

For trivial additions (a missing test, a brief tweak) the slash command
is overkill. This is the file-by-file checklist for a new method
called `<name>`, of kind `<kind>` ∈ {tilting, statistic, experiment,
diagnostic, model}. The completeness checker will gate every step.

### Step 1: brief skeleton

`docs/methods/<name>.md`. Copy `docs/methods/_template.md`. Fill the
nine required headers (Summary, Motivation, Definition, Derivation,
Predicted behavior, Failure modes, Invariants, Literature, Links).
For stubs leave Derivation as a TODO; the completeness checker
requires the *header* to be present, not its content.

### Step 2: source file

The location depends on `<kind>`:

| kind        | path                                          | template |
|-------------|-----------------------------------------------|----------|
| tilting     | `src/frasian/tilting/<name>.py`               | `tilting/fisher_rao.py` (stub), `tilting/ot.py` (impl, with general 1D path), or `tilting/power_law.py` (impl, conjugate-only) |
| statistic   | `src/frasian/statistics/<name>.py`            | `statistics/lrt.py` (stub) or `statistics/wald.py` (impl) |
| experiment  | `src/frasian/experiments/<name>.py`           | `experiments/coverage.py` |
| diagnostic  | `src/frasian/diagnostics/<name>_table.py`     | `diagnostics/coverage_table.py` |
| model       | `src/frasian/models/<name>.py`                | `models/bernoulli.py` |

The minimum is a class decorated with `@register_<kind>(name=...,
brief="docs/methods/<name>.md", status=...)`. For stubs each method
raises `NotImplementedError` with `"see docs/methods/<name>.md"`. For
implementations, conform to the protocol in `src/frasian/<kind>/base.py`.

If the method is Normal-only by construction, use
`models._dispatch.require_model(model, NormalNormalModel, caller=...)`
for a uniform error message — see `statistics/wald.py:_require_normal_normal`.

### Step 3: register in the bootstrap

`src/frasian/_registry_bootstrap.py`. Add a new line:
```python
from .<kind>s import <name> as _<short>  # noqa: F401
```
Static enumeration depends on this; `mypy` and the completeness
checker walk the bootstrapped imports.

### Step 4: property tests

`tests/properties/test_<name>_invariants.py` (or
`tests/experiments/test_<name>_experiment.py` for experiments). At
least three tests. For stubs, mark them `@pytest.mark.skip(reason="stub
- see docs/methods/<name>.md")`. Each invariant in the brief should
have one corresponding test.

For tiltings/statistics with closed forms, also add a regression test
at `tests/regression/test_<name>.py` pinning the formula at
`atol=1e-12` against either the legacy implementation or an
independent re-derivation.

### Step 5: illustration (only if `status="implemented"`)

`src/frasian/experiments/illustrations/<name>_demo.py`. Mirror the
pattern of `power_law_demo.py`: takes `--smoke` and `--out` flags,
writes a PNG, exits cleanly. The CI smoke-runs every demo.

Stubs are exempt — the completeness checker skips illustration checks
when `status="stub"`.

### Step 6: verify locally

```bash
python -m pytest tests/properties/test_<name>_invariants.py     # green
python tools/check_method_completeness.py                       # 0 errors
python -m scripts.run --list                                    # <name> appears
```

If `status="implemented"`, also:
```bash
python -m frasian.experiments.illustrations.<name>_demo --smoke
```

### Step 7: cross-product runs (optional but recommended)

If the method is a tilting or statistic, see how it scores on the
existing diagnostics:

```bash
python -m scripts.run --fast experiment=smoothness   # smoothness metrics
python -m scripts.run --fast experiment=coverage     # frequentist coverage
python -m scripts.run --fast experiment=width        # mean CI width
```

Compare the (`<name>`, waldo) row to the (power_law, waldo) baseline.
The Step-5 smoothness diagnostic is the gating evidence: if the
Lipschitz / TV / discontinuity numbers don't beat the baseline, the
new scheme isn't a smoothness improvement.

### Step 8: open the PR

```bash
git push -u origin feat/<name>
```

The two GitHub workflows fire automatically. If both are green and
the skeptic verdict is `accept` or `accept-with-caveats`, the PR is
ready for review.

## Anti-patterns

- **Adding a method without `/propose-method`.** Skipping the
  scaffolder is fine for trivial changes (renames, doc fixes), but
  for new mathematical content the discipline catches gaps that an
  ad-hoc commit will not.

- **Editing a brief without updating its property tests.** The
  brief's Invariants section is the contract; tests are the proof.
  If they drift, the skeptic will flag the next time it is run.

- **Marking a stub `status="implemented"` to silence the completeness
  checker.** The status field is informative — flipping it without
  the implementation is a lie that will surface as a test failure
  the moment a downstream cell tries to use the method.

- **Pushing past a `block` skeptic verdict.** The skeptic is hostile
  by design; ignore it once and you will not notice the next bug.
