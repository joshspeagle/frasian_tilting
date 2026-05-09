# Empirical findings & session notes

Dated narratives describing experiments run, problems diagnosed, and
findings that wouldn't otherwise survive a `git log` reading: **what
we tried, why it worked or didn't, and what numbers we saw**.

This is the "results notebook" for the framework. CLAUDE.md is the
orientation document; `docs/methods/<name>.md` are per-method briefs;
this directory is for everything else that someone reading the repo
later might want to know.

## When to add a note

- Diagnosed a regression and identified its cause.
- Ran an audit / experiment whose results you want to cite later.
- Made a non-obvious framework decision (and want the rationale
  preserved in the repo, not just in chat history).
- Noticed a calibration / numerical / scaling property that's worth
  knowing about even though it isn't a method itself.

## When NOT to add a note

- Routine `git log` material — commit messages cover that.
- Per-method theory / derivations — those go in `docs/methods/<name>.md`.
- Implementation plans — those live in `docs/superpowers/plans/`
  (gitignored except the plans dir itself).
- Audit reports from `docs/audit/` — those are dumped from the
  multi-agent audit workflow.

## Naming

`YYYY-MM-DD-<short-slug>.md`. Date is when the work concluded
(not started). Slug is a lowercase-hyphen description that pairs
naturally with the work product (commits, fixtures, notes).

Examples:
- `2026-05-09-phase-g-v4-fix.md`
- `2026-04-12-ot-quantile-mixture-derivation.md`

## Format

Free-form Markdown. The most useful notes have:

1. **TL;DR** (1-3 sentences): the headline finding.
2. **Context**: what we were doing and why it mattered.
3. **What we tried**: the experiments, in chronological-ish order.
4. **What we found**: the numbers, the pattern, the cause.
5. **What changed in the codebase** (commit refs / file paths).
6. **Open questions / follow-ups**.

A reader 6 months from now should be able to skim the TL;DR, decide
whether to read further, and reproduce the key check from the "what
changed" section if they care to.

## Index

- [2026-05-09-phase-g-v4-fix.md](./2026-05-09-phase-g-v4-fix.md) —
  σ-anchored θ training + input normalization unblocks the Phase G
  v4 conditional learned-η selector (which initially collapsed to
  η ≈ 1 = Wald). Includes per-loss audit results and the
  `LearnedDynamicEtaSelector` clamp-instead-of-refuse change.
