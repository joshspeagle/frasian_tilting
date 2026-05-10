# Mixture cd_variance training instability — systematic investigation

**TL;DR.** Mixture NN trains cleanly under `integrated_p` loss
(val=1.61, η_valid=1.000, monotonic) but is unstable under
`cd_variance` (val=11.5+ best, η_valid=0.55-0.73, val explodes).
Same pipeline works for PL and OT cd_variance. This note plans a
systematic investigation to identify the root cause and decide
whether to fix or to ship mixture intp-only.

## Problem statement

Mixture cd_variance training under default hyperparameters
(lr=3e-4, clip=1.0, λ_max=10, warmup_frac=0.3, eta_explore_box=
`[-1.0, 3.0]`, n_epochs=30):

| epoch | val_width | best | η_valid (eval) |
|---|---|---|---|
| 5 | 11.32 | 11.32 | 0.55 |
| 6 | 14.46 | 11.32 | — |
| 9 | 17.20 | 11.32 | — |
| 12 | 19.95 | 11.32 | — |

Lowering lr to 1e-4 marginally improves: η_valid=0.73, val=11.47.
Still below the runtime selector's 80% admissibility threshold.

PL cd_variance and OT cd_variance both train cleanly under the same
defaults (val=1.29, 1.25 respectively; η_valid=1.0 in both cases).

## Why mixture might be different

Three structural differences from PL/OT:

1. **Narrower admissibility window.** Mixture: `η ∈ [0, eta_max(D,
   prior, model)]` with eta_max varying per-sample (typically 1-3, can
   be smaller in high-conflict regimes). PL: `(-w/(1-w), 1/(1-w))` —
   wide for w near 0.5, very wide for w near 1. OT: `[0, 1]`.

2. **m-geodesic vs e-geodesic / W2-geodesic.** Mixture is a
   density-space mixture; tilted distribution is a mixture of normals
   with heavier tails. The CD's variance can grow unboundedly as η
   approaches the boundary (heavy-tail contribution).

3. **Loss surface "punishes wrong" instead of "saturating".** PL's
   tilted_pvalue clamps `denom = max(1 - η(1-w), 1e-6)` — extreme η
   gives bounded p ∈ [0, 1]. Mixture at extreme η gives a
   non-density (negative pdf in some θ region) → CD variance can be
   arbitrarily large. So Adam-momentum drift past the boundary is
   amplified by unbounded loss values rather than saturated.

## Hypotheses to test

Listed in expected priority. Each hypothesis maps to a specific
ablation experiment in the matrix below.

| H | Hypothesis | Test |
|---|---|---|
| H1 | Boundary penalty too weak (λ_max=10 doesn't beat unbounded cd_var) | E3: λ_max=50 |
| H2 | Gradient clip too loose | E2: clip=0.25 |
| H3 | λ warmup too slow (network drifts before λ reaches max) | E4: warmup_frac=0.05 |
| H4 | eta_explore_box too wide on negative side (Head B over-trains on always-invalid samples) | E5: box=[-0.2, 1.5] |
| H5 | Adam momentum compounds drift; SGD or no-momentum more stable | E6: optimizer change (out-of-scope for quick test) |
| H6 | cd_variance has numerical pathology at extreme η for mixture (NOT a generic loss-landscape issue) | E0 (probe): compute fn-min on η-grid |
| H7 | Pretrained init (from intp fixture) escapes the bad basin | E8: pretrained start |
| H8 | Need anti-wald or anti-collapse regularizer | E9: anti_wald_max=1.0 |
| H9 | Network architecture lacks structural admissibility constraint | E10: structural fix (out-of-scope for short term) |

## Probe E0 (run BEFORE training experiments)

**Question:** Does cd_variance for mixture have a stable interior
optimum, or is the optimum genuinely at the admissibility boundary?

**Method:** Take the mixture intp fixture. For a few fixed (μ₀, σ₀,
σ) slices, compute cd_variance loss as a function of constant η on
[0, eta_max]. Plot. If the cd_variance(η) curve has a clear interior
minimum, the network *should* be able to find it; instability is
training-side. If the curve is monotonic in η (driving toward
boundary), no amount of stabilization will help — mixture cd_variance
is intrinsically a boundary-hugging objective and should be skipped.

**Acceptance for E0:** if interior optimum exists at all 4 slices
with cd_var(η_opt) < cd_var(η=eta_max−ε), proceed with E1-E10. If
optimum is always at boundary, skip cd_var and ship intp-only.

## Training experiment matrix

Each experiment runs the exact same YAML
(`canonical_normal_normal_mixture_v4.yaml`) with one hyperparameter
override. Diagnostics enabled. n_epochs=30.

| # | Override | Tests | Estimated runtime |
|---|---|---|---|
| E1 | none (baseline reproduce) | sanity | ~5 min |
| E2 | --grad-clip-max-norm 0.25 | H2 | ~5 min |
| E3 | --lambda-max 50 | H1 | ~5 min |
| E4 | --lambda-warmup-frac 0.05 | H3 | ~5 min |
| E5 | YAML edit eta_box=[-0.2, 1.5] | H4 | ~5 min |
| E6 | --lr-a 1e-4 --grad-clip-max-norm 0.25 | H1+H2 | ~5 min |
| E7 | --lambda-max 50 --grad-clip-max-norm 0.25 | H1+H2 | ~5 min |
| E8 | --pretrained-eta-path \<intp fixture\> | H7 | ~5 min |
| E9 | --anti-wald-max 1.0 (decay over 0.5 frac) | H8 | ~5 min |

Total: ~45 min for the matrix, plus diagnostics review.

## Acceptance gates per experiment

A variant "passes" if:
- η_valid_rate ≥ 0.8 at the saved checkpoint
- best val_width within 5× of mixture intp val (1.61) — i.e., cd_var
  is reasonable (~5-10) since cd_var has different magnitude than intp
- val trajectory monotonic OR has at most one minor regression (no
  >2× explosions like the current 11→77→61 sequence)

If multiple variants pass, the one with smallest val_width × η_valid_rate
ratio wins.

## Diagnostic decision tree

| E0 says | … and best E1-E10 says | Action |
|---|---|---|
| Interior optimum exists | passes all gates | Apply that variant's setting as a per-scheme override; commit and audit |
| Interior optimum exists | none pass all gates | Deeper investigation: structural admissibility transform (H9), or accept and document |
| Optimum at boundary | (E1-E10 don't matter) | Mixture cd_variance is intrinsically boundary-hugging; skip |

## Findings (run 2026-05-10)

E0 probe ran first. Confirmed cd_variance optima are admissible (often
at η=0 boundary, sometimes interior in [0, 1]). Optima fit within
[0, 1] for all four tested slices — so a structural sigmoid bound on
the EtaNet output to (0, 1) would be near-optimal for mixture
cd_variance, while also being structurally admissible.

Hyperparameter sweep (variants of E1-E10):

| # | Config | η_valid | best val |
|---|---|---|---|
| 1 | default lr=3e-4, λ=10, no mask | 0.55 | 11.32 |
| 2 | lr=1e-4 only | 0.73 | 11.47 |
| 3 | λ=50 only | 0.63 | 15.37 |
| 4 | mask (gradient-only) + defaults | 0.595 | 13.07 |
| 5 | **mask + lr=1e-4 + λ=50** | **0.811** | 13.57 |
| 6 | mask + lr=1e-4 + λ=100, no warmup | 0.77 | 13.26 |
| 7 | value-replacement mask + defaults | 0.711 | 19.44 |
| 8 | value-repl mask + lr=1e-4 + λ=50 | 0.800 | 19.06 |

#5 is the best simple variant. **η_valid=0.811** clears the
LearnedDynamicEtaSelector's 0.80 clamp-refusal threshold, so the fixture
loads. **val=13.57** is ~10× the optimum (E0 average var ≈ 0.66) and
~10× PL/OT cd_var (1.29, 1.25) — the saved checkpoint is essentially
early-training state, not a converged cd_var minimizer.

The fixture is shippable (loads at inference, calibrated by Head B's
clamp on out-of-bounds η values), but the inferential quality is
notably worse than mixture intp / static_w which trained cleanly.

## Decision and next step

**Decision (2026-05-10):** initially shipped #5 fixture as
`mx_learned_cd_var` with documented limitations. Coverage audit at
full Config.fast() w_grid revealed catastrophic miscalibration —
0.22 coverage at w=0.35, vs nominal 0.95. The 19% out-of-bounds η
values were producing inadmissible CIs at inference. Fixture not
shippable.

**Resolution (same day, 2026-05-10):** applied the structural
sigmoid bound below. Headline post-fix:

| flavor       | val pre | val post | η_valid pre | η_valid post | coverage post |
|--------------|---------|----------|-------------|--------------|---------------|
| mx intp      | 1.61    | 1.61     | 1.000       | 1.000        | 0.95-0.965    |
| mx cd_var    | 13.57   | **1.25** | **0.811**   | **1.000**    | 0.955-0.965   |
| mx static_w  | 4.07    | 4.06     | 1.000       | 1.000        | 0.96-0.97     |

All three coverages within nominal [0.95, 0.97]. Widths comparable
to WALDO (3.77-4.73 vs WALDO's 3.88-4.59). Bound is a no-op for
intp/static_w (their optima were already in [0, 1]); structural
constraint eliminates the boundary-attractor pathology for cd_var.

**Structural sigmoid bound:** EtaNet output for
mixture. Approach:

1. Add a `bound_eta: tuple[float, float] | None` field to EtaNet (or
   to a wrapper `BoundedEtaNet`). For mixture, set `(0.0, 1.0)`
   (covers all cd_variance optima per E0; extends to ~`eta_max ≈ 3`
   for `static_width` if needed but not required).
2. Apply `eta = lo + sigmoid(raw) * (hi - lo)` at the EtaNet's
   forward output.
3. Thread the same transform through 4 consumers:
   - `_call_normal_normal_pvalue` / `_call_generic_grid_pvalue`
     (training loss).
   - `compose_boundary_penalty` (Head A → Head B signal).
   - `LearnedDynamicEtaSelector` (inference selector).
   - `learned/training/diagnostics.py` (D1-D4 trajectories).
4. Retrain ALL mixture v4 fixtures (intp, cd_var, static_w) with the
   bounded architecture. Existing fixtures become stale.
5. Test agreement: bound is a no-op for the working intp/static_w
   (their η values are already in [0, 1]); for cd_var it bounds the
   network structurally → no Adam-overshoot possible.

Implementation landed 2026-05-10 (this same day). Surface area was
smaller than expected: only EtaNet's `__call__` (sigmoid squash) +
`architecture_kwargs` round-trip + `train.py` per-scheme dispatch
were needed. The other "consumers" (loss / boundary penalty /
selector / diagnostics) all consume `eta_net(...)` outputs, so the
transform propagates automatically. Total: ~50 lines of code.

## Out-of-scope for this investigation

- H9 (architectural admissibility transform): would require modifying
  `EtaNet`'s output to gate through a soft-bounded transform. Real
  work; defer unless E1-E10 all fail.
- Bernoulli + mixture cd_variance (the generic-grid path): same root
  causes likely; same fixes likely apply but separate investigation.

## Linked artifacts

- `scripts/probe_mixture_cd_var_landscape.py` (E0 probe) — to be written.
- `docs/notes/2026-05-10-mixture-cd-variance-instability.md` (this file).
- Findings will be appended to `docs/notes/2026-05-10-followup-todo.md`
  under "mixture cd_variance status".
