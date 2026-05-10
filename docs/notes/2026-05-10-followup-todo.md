# Follow-up TODO after the 2026-05-10 learned-η investigation

After the probe-driven sanity check (see
[`2026-05-10-eta-conventions-and-loss-derivation.md`](2026-05-10-eta-conventions-and-loss-derivation.md))
established that v4 fixtures *are* delivering near-optimal calibrated
CIs on the σ₀-anchored training distribution, three follow-up items
remain. Listed in the order the user expects to address them.

## 1. Audit all 4 scripts in the experimental design matrix

Goal: be confident that **training, validation, evaluation, and NN
output** are all evaluating the right things across the experimental
design matrix (NN × tilting scheme).

For each (model, tilting_scheme) in the matrix — currently
{normal_normal, bernoulli} × {power_law, ot, mixture} — audit:

- Training script (`scripts/train_learned_eta.py` driving
  `learned/training/train.py`): does it use the right θ-distribution,
  loss, and hyperparam ranges?
- Validation/holdout: does the held-out check use the same
  σ₀-anchored frame as training?
- Evaluation scripts (`scripts/run_wald_audit.py`, the per-experiment
  runners): do they match the training assumptions or do they
  extrapolate?
- NN output inspection: does the saved checkpoint's metadata reflect
  what was actually run, and does the OOD-θ clamp fire correctly at
  inference?

Specifically: any script that builds a θ-grid wider than σ₀-anchored
is testing extrapolation, not framework correctness. The OOD-θ clamp
(added 2026-05-10) makes extrapolation safe (returns Wald), but
audits should still verify the right metric is being reported.

## 2. Re-run wald-audit experiments for PL and OT

Once (1) is clean, regenerate the wald-audit results for `power_law`
and `ot` using the corrected understanding:

- v4 fixtures with OOD-θ clamp ON
- σ₀-anchored evaluation domain (matches training)
- Apples-to-apples Wald (η=1) baseline
- 95% CI widths + coverage as the headline metrics (concrete
  inferential numbers; integrated_p remains as the training-loss
  context)

Expected outcome (from current diagnostic probes): PL and OT both
reduce 95% CI width over Wald by ~5-10% at low-w, ~1-3% elsewhere,
with calibrated coverage everywhere. If the audit shows substantially
worse, the audit script itself is using the wrong evaluation frame —
loop back to (1).

## 3. Return to the mixture (m-path) implementation

Stage C of the mixture-tilting work is currently deferred (see
[`2026-05-09-mixture-smoothness-and-learned-eta-tails.md`](2026-05-09-mixture-smoothness-and-learned-eta-tails.md)).
The deferral was based on the (now-corrected) framing that learned-η
training was fundamentally broken. With the framework now confirmed
to be working as designed, mixture Stage C (mixture learned-η
training) can be revisited:

- Train v4 fixtures for mixture using the same σ₀-anchored pipeline
- Run wald-audit on mixture cells (same framing as PL/OT)
- Compare mixture vs PL vs OT smoothness + width on the conflict band

The "input-sensitivity-aware training" hypothesis from the deferral
note may not actually be needed — the symptoms it tried to address
(network outputs ~constant η, anti-correlated with optimum) were
artifacts of evaluating on the wrong distribution, not the training.

## Note on the OOD-θ clamp's reach

The clamp fixes one of **two** mismatches that occur when evaluating v4
outside its training distribution:

- **θ_test outside training box**: clamp overrides η → likelihood-only.
  *Fixed.*
- **Data marginal D from a broader range than training**: the network's
  behavior *inside* the training box is wrong for this D distribution
  too, because the optimal η at any θ_test depends on the data
  distribution. *Not fixed by the θ_test clamp.*

Empirical check (2026-05-10): on a deliberately-stressed
**likelihood-σ-anchored** eval at low-w (σ₀=0.3, σ=1, θ_true ~
U(-5σ, +5σ) — much broader than training's U(-5σ₀, +5σ₀) =
U(-1.5, +1.5)), the clamp reduced the v4 loss-vs-Wald gap from
+0.92 to +0.52 at low-w. The remaining gap comes from the
data-marginal mismatch.

Practical consequence: v4 is calibrated and near-optimal under
σ₀-anchored evaluation (matching training). For likelihood-σ-anchored
or other broader evaluation regimes, training a separate v4 variant
on the target data distribution is the right fix; the OOD-θ clamp is
a calibrated *partial* fallback, not a fix for distributional shift
in D.

## What's already in place

- OOD-θ clamp (`LearnedDynamicEtaSelector.clamp_outside_training`,
  default ON; pinned by
  `tests/regression/test_ood_theta_clamp.py`).
- `eta_likelihood_only` field on `ParamSpec`, populated for power_law,
  ot, mixture; None for identity / fisher_rao stubs.
- Probe scripts that gave the corrected picture:
  - `scripts/probe_function_constrained_min.py` (likelihood-anchored
    fn-min vs Wald)
  - `scripts/probe_optimal_eta_and_ci.py` (σ₀-anchored, with 95% CI
    widths + coverage)
  - `scripts/probe_v4_per_slice_eval.py` (apples-to-apples eval
    with both WALDO and Wald baselines, OOD clamp applied)
  - `scripts/probe_per_theta_test_argmin.py` (per-θ_test argmin η
    oracle, with v4 overlays)
- Training-time diagnostics (D1-D4 in
  `learned/training/diagnostics.py`) with JSON sidecar output —
  available via `--diagnostics-out` on
  `scripts/train_learned_eta.py`. Should be the default for any
  re-training in (2) / (3).

## Acceptance criteria for closing this follow-up

- (1) audit produces a checklist (one item per script per cell)
  showing each script evaluates the right thing, or a fix list with
  PRs.
- (2) wald-audit for PL+OT shows monotonic Wald → trained CI-width
  improvement at every (σ₀, σ) bin, with calibrated coverage.
- (3) mixture wald-audit reaches parity with PL/OT (similar
  improvements + calibrated).

Each step's outputs (audit notes, audit results, mixture trained
checkpoint) become their own dated note in `docs/notes/`.
