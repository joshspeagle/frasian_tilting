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
  σ₀-anchored θ training + input normalization unblocks the Phase G
  v4 conditional learned-η selector (which initially collapsed to
  η ≈ 1 = Wald). Includes per-loss audit results and the
  `LearnedDynamicEtaSelector` clamp-instead-of-refuse change.
- [2026-05-09-mixture-smoothness-and-learned-eta-tails.md](./2026-05-09-mixture-smoothness-and-learned-eta-tails.md) —
  After implementing MixtureTilting, the bare-scheme smoothness
  comparison shows mixture with `lipschitz_eta` ~1/3 of PL — but
  most of that is a scope artifact of mixture's tighter admissible
  η-range. Also flags the learned-η tail-decay puzzle (η drops at
  large |θ| instead of approaching 1) and proposes loss-landscape
  probes to disambiguate before Stage C training. Includes the
  `_spectral_roughness` DC-bin bug-fix.
- [2026-05-10-followup-todo.md](./2026-05-10-followup-todo.md) —
  Three follow-up items after the 2026-05-10 corrected-framing
  investigation: (1) audit the 4 scripts per experimental-design cell
  for correct evaluation framing; (2) re-run wald-audit for PL and
  OT with OOD-θ clamp + σ₀-anchored evaluation; (3) return to
  mixture Stage C training.
- [2026-05-10-eta-conventions-and-loss-derivation.md](./2026-05-10-eta-conventions-and-loss-derivation.md) —
  Critical distinctions that cause confused diagnostic results: η=0
  is WALDO and η=1 is Wald in `power_law_tilted_pvalue_jax` (names
  feel backwards); the integrated_p loss equals "average CI width
  across all α-levels" via Fubini; the σ₀-anchored integration
  domain makes Wald loss appear σ₀-dependent (truncation artifact);
  per-θ_test ≠ per-θ_true argmin η; v4 trained on σ₀-anchored θ_test
  cannot learn far-from-μ₀ Wald-fallback. Reference for future
  diagnostic work.
- [2026-05-10-learned-eta-intervention-design.md](./2026-05-10-learned-eta-intervention-design.md) —
  Deep investigation into the learned-η training failures. Computes
  calibrated benchmarks (real headroom is 14%, not 30% as
  post-selection oracle suggested), systematic eval of all 14 trained
  fixtures, multiple Phase-1/Phase-2 stability tests. **Key finding:**
  the integrated_p loss and the framework's `DynamicNumericalEtaSelector`
  optimize different objectives — the v4 fixtures train against
  integrated_p (Basin A attractor) while the framework's "calibrated
  default" minimizes static_width (which can drive η to extreme
  negatives). Best fixture (no_boundary, lambda_max=0) captures 15%
  of the 14% calibrated headroom — i.e., ~2% absolute improvement
  over Wald. Lists pivot options.
- [2026-05-10-mixture-cd-variance-instability.md](./2026-05-10-mixture-cd-variance-instability.md) —
  Systematic investigation of mixture cd_variance training
  instability, from initial E0 landscape probe through 8-config
  hyperparameter sweep to the architectural sigmoid bound that
  ultimately resolved it. **Resolution:** EtaNet gained an optional
  `output_bounds: tuple[float,float] | None` field; mixture's
  `param_space.training_output_bounds=(0.0, 1.0)` dispatches the
  bound at training time. Pre-bound cd_var val=13.57 / η_valid=0.811
  / coverage 0.22-0.93 → post-bound val=1.25 / η_valid=1.000 /
  coverage 0.955-0.965. No-op for intp/static_w (their optima
  already in [0, 1]).
