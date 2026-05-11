# Fisher-Rao cd_variance training — hyperparameter regime + the row-13b reframe

**TL;DR.** FR cd_variance under default hyperparameters (`--lr-a 3e-4
--grad-clip-max-norm 1.0`) diverges: 22 of 34 steps in the first 10
epochs produce non-finite losses, best val 26.18 at epoch 2, then val
climbs to 57.73 by epoch 6. Retraining with `--lr-a 1e-4
--grad-clip-max-norm 0.5` produces clean training (0 non-finite skips,
47% val reduction to best=14.48 at epoch 14) AND the strongest
per-θ + cross-cell adaptation of any FR head trained so far
(cross-cell spread 1.43 vs integrated_p 0.16, vs static_width 0.08).

This note records (a) the recommended hyperparams, (b) why FR is
structurally more fragile than PL/OT/MX for cd_var, and (c) the
broader implication that the framework's row-13b "near-constant
per-cell" finding is loss-specific, not architecture-specific.

## Problem statement

FR cd_var with default hyperparams on
`experiments/canonical_normal_normal_fisher_rao_v4.yaml`:

| epoch | val_width | best | non-finite step count |
|-------|-----------|------|-----------------------|
| 1 | 30.67 | 30.67 | — |
| 2 | 26.18 | 26.18 | — |
| 3 | 39.36 | 26.18 | accumulating |
| 6 | 57.73 | 26.18 | 22 by ep 10 |
| 9 | 43.66 | 26.18 | — |
| 10 | early stop | 26.18 | total 22 / 34 steps skipped |

For comparison, FR integrated_p and static_width under the same defaults
train cleanly (45% and 67% val reduction, 0 non-finite skips).

## Why FR is structurally fragile for cd_var

Two FR-specific structural factors compound:

1. **Head B (ValidityNet) is class-degenerate for FR.** FR is
   geodesically complete on the Gaussian half-plane (Cartan-Hadamard),
   so every finite η ∈ ℝ is admissible. ValidityNet's training labels
   are uniformly 1.0; BCE on a single-class target trivially
   minimises with P(valid) = 1 and no learning happens. The training
   loop fires the warning
   `[head B] aux validity rate = 1.000 (outside (0.05, 0.95))`
   every epoch.

2. **The boundary penalty is structurally inert.** Head A's loss is
   `cd_variance + λ · (-log P_B(valid))`. Since P_B(valid) ≈ 1
   everywhere for FR, the penalty term ≈ 0 and λ-ramping does
   nothing. The mechanism that keeps PL/OT/MX η inside a sensible
   admissibility-aware range has **no effect** for FR.

cd_variance specifically explodes under unconstrained η: the variance
of the FR-tilted CD can grow arbitrarily as η extrapolates past the
prior or likelihood endpoints (geometrically the half-plane allows
this; statistically the resulting CD has very heavy tails). Without
either an admissibility-derived boundary penalty (PL/OT/MX) or a soft
training-time regulariser (none exists for FR currently), the
optimiser's Adam-momentum drifts past the well-behaved region and
hits regions where the loss is non-finite (skipped) or simply huge
(spike, divergence).

This is the FR analog of mixture row-13c (see
`2026-05-10-mixture-cd-variance-instability.md`), but where mixture's
fix was a structural sigmoid output bound `(0, 1)` matching its
admissibility window, FR has no formal bound to apply. The
recommended interim mitigation is the hyperparameter regime below.

## Recommended hyperparameter regime

```bash
python -m scripts.train_learned_eta \
    --config experiments/canonical_normal_normal_fisher_rao_v4.yaml \
    --loss cd_variance \
    --lr-a 1e-4 \
    --grad-clip-max-norm 0.5 \
    --out artifacts/learned_eta_canonical_normal_normal_fisher_rao_phaseC_cd_variance_v4.eqx
```

Other defaults preserved: `--lr-b 3e-4`, `--lambda-max 10.0`,
`--lambda-warmup-frac 0.3`, `--n-epochs 30`, `--batch-size 256`,
`--patience 8`. (Note: `--lambda-max` is structurally inert for FR per
the above, but harmless to leave at default.)

`integrated_p` and `static_width` trainings for FR can use the
defaults; only `cd_variance` requires the tightened regime.

## Results comparison

All three FR heads, post-fix:

| Head | val descent | non-finite skips | per-cell std | cross-cell spread | η range |
|------|-------------|------------------|--------------|--------------------|---------|
| integrated_p | 2.91 → 1.61 (45%) | 0 | ~5e-4 | 0.16 | [0.71, 0.87] |
| static_width | 12.54 → 4.06 (67%) | 0 | ~3e-3 | 0.08 | [0.76, 0.84] |
| cd_variance (defaults) | 30.67 → 26.18, diverged | 22 | ~5e-2 | 0.24 | [-0.43, -0.01] |
| **cd_variance (lr=1e-4, gc=0.5)** | **27.09 → 14.48 (47%)** | **0** | **~0.13–0.21** | **1.43** | **[-1.98, 0.03]** |

cd_var prefers strong **negative η** for high-σ₀ (strong-prior) cells:
mean η = -1.62 at (μ₀=1, σ₀=2, σ=1.5) and -1.44 at (μ₀=-1, σ₀=1,
σ=0.5). Geometrically this is extrapolation **past the prior
endpoint** of the FR geodesic, allowed by Cartan-Hadamard, that
maximally tightens the CD variance for those cells. For PL this region
would be inadmissible (η < 0 already pushes past WALDO toward
"anti-data"); for FR it is geodesically valid and the cd_variance
objective genuinely prefers it.

## Implication: row-13b is loss-specific, not architecture-specific

CLAUDE.md row 13b documents a known limitation of Phase G v4 training:
"trained networks output range is ~10% of the per-slice optimum range
across the hyperparam grid, with negative correlation to the
analytical optimum on PL/OT". The diagnosis was "input-insensitivity"
treated as an architectural fault of the EtaNet/ValidityNet pipeline.

The FR cd_var result above refutes that diagnosis for at least one
configuration: under cd_variance, FR's network learns strong per-θ
(std 0.13–0.21) AND strong per-cell (spread 1.43) adaptation — orders
of magnitude more than integrated_p (std 5e-4, spread 0.16) on the
same architecture, same scheme, same hyperparam range. The
architecture is **capable** of input-sensitive learning; it is
integrated_p's loss landscape that produces the near-constant
artifact.

Empirically on FR, the order is:
- **cd_variance**: strongest per-θ + per-cell adaptation
- **integrated_p**: weakest per-θ adaptation (~5e-4); moderate
  per-cell adaptation
- **static_width**: between, but closer to integrated_p

Whether this loss-specificity carries over to PL/MX/OT is untested.
A worthwhile follow-up: retrain PL/MX/OT cd_variance heads with the
same probe and check whether the row-13b "negative correlation to
analytical optimum" is also loss-specific. If so, the framework's
production-default would shift away from integrated_p as the
"calibrated and narrower" headline cell.

## Open questions

- **Should FR cd_var ship a soft regulariser** (option D: tanh output
  cap on EtaNet, or option E: replace the dead boundary penalty with
  a direct L2 on η)? The hyperparameter regime above works but is
  brittle to YAML changes. A structural fix would be more robust.
- **Does PL/MX/OT cd_variance show similar adaptation strength
  to FR's** when retrained with a probe? If yes, row-13b's "row" in
  CLAUDE.md needs a rewrite.
- **Does the `--lr-a 1e-4 --grad-clip-max-norm 0.5` regime also
  improve PL/MX/OT cd_var**, or is it only needed for FR? (PL/MX/OT
  have an active boundary penalty so defaults already work, but the
  tighter regime might still help convergence quality.)

## Provenance

- Default-hyperparam pathology observed at commit
  `feat/fisher-rao-tilting` (this branch), training run 2026-05-11
  in the morning; logs at `/tmp/fr_cd_var_training.log` (not
  committed).
- Tightened-hyperparam fix verified at the same commit, training
  run 2026-05-11 afternoon; logs at `/tmp/fr_cd_var_retry_A2.log`.
- Tests pinning the trained artifact: 15 cases in
  `tests/regression/test_fisher_rao_v4_fixture.py`.
