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
/propose-method ot_normal tilting

  ↓ deriver fills Derivation
  ↓ literature-reviewer fills Literature
  ↓ scaffolds src/, tests/, brief/, illustration/
  ↓ skeptic returns "accept-with-caveats"
  ↓ user fixes caveats

# verify locally
python -m pytest tests/properties/test_ot_normal_invariants.py
python tools/check_method_completeness.py
python -m frasian.experiments.illustrations.ot_normal_demo --smoke

# open PR
git push -u origin feat/ot-normal

  ↓ ci.yaml runs L0-L4 pytest
  ↓ method-completeness.yaml runs the checker + smoke-runs all demos
  ↓ both green → merge
```

A subsequent refinement (e.g. flipping a `pytest.mark.skip` to a
passing assertion when an invariant becomes provable):

```
/critique src/frasian/tilting/ot_normal.py

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
