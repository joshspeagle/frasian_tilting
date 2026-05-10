# Mixture vs PL/OT smoothness, and the learned-η tail-decay puzzle

## TL;DR

After implementing `MixtureTilting` (Stage A+B of the mixture-tilting
plan), the bare-scheme smoothness comparison showed the m-geodesic
with a `lipschitz_eta` ~1/3 of PL's e-geodesic and ~55% of OT's W2
geodesic. **Most of this advantage is a scope artifact**: mixture's
admissible η-range is `[0, η_max]` (no oversharpening) versus PL's
~`[-1, ~1]`, so a smaller swing produces a smaller Lipschitz. After
normalizing by η-swing, the schemes are within 20% of each other.

Separately, the learned-η fixtures (PL v4) show a surprising drop in
η at large |θ| — η decreases away from the Wald limit instead of
approaching it. This correlates with proximity to the
`SigmaAnchoredUniformThetaDistribution(K=5)` training boundary, but
is NOT obviously just sparse-data extrapolation; the loss objectives
themselves (`integrated_p`, `cd_variance`, `static_width`) integrate
over (θ, D, prior_hp, lik_hp) and may legitimately disagree with the
per-D `NumericalEtaSelector` analytic optimum. Before training
mixture's learned-η in Stage C, we want a loss-landscape probe to
disambiguate sparse-data extrapolation from a real loss-shape feature.

Also caught + fixed a long-standing bug: `_spectral_roughness`
returned ~1e31 on small (n ≤ 12) η-grids because it included the
DC bin (≡0 after mean-detrending) in its "low frequency" band. Fixed
in [`smoothness_metrics.py`](../../src/frasian/diagnostics/smoothness_metrics.py)
on commit `1d8110a`.

## Context

Stage A+B of `docs/superpowers/plans/2026-05-09-mixture-tilting.md`
promoted MixtureTilting from stub to full Normal-Normal
implementation. We then ran the smoothness experiment on the bare
mixture scheme and compared against the existing PL/OT bare-scheme
smoothness outputs to verify the implementation produces sensible
results before moving on to Stage C (JAX layer + learned-η training).

## Smoothness comparison

Audit config (Config.fast, single w=0.5, n_reps=200, abs_delta_grid
of 11 points from 0 to 5):

| metric                       | power_law | ot      | mixture | mx/pl | mx/ot |
|------------------------------|----------:|--------:|--------:|------:|------:|
| `lipschitz_eta`              |     3.42  |   2.27  |   1.26  |  0.37 |  0.55 |
| `total_variation_eta`        |     2.14  |   1.85  |   1.05  |  0.49 |  0.57 |
| `discontinuity_count_eta`    |        3  |      3  |      3  |  1.00 |  1.00 |
| `spectral_roughness_eta`*    |     1.17  |   0.93  |   1.00  |  0.86 |  1.08 |
| `lipschitz_width`            |     0.65  |   0.65  |   0.64  |  0.99 |  0.99 |
| `total_variation_width`      |     0.61  |   0.61  |   0.62  |  1.01 |  1.01 |

*after the bug-fix on commit `1d8110a`; pre-fix all three were ~1e31
(division-by-near-zero, see "Bug fix" section below).

### The η-curve range explains the bulk of the Lipschitz difference

Per-grid-point η_star arrays:

```
|Δ|     0.0    0.5    1.0    1.5    2.0    2.5    3.0   ...   5.0
PL    -0.85  -1.00  +0.71  +0.90  +0.94  +0.96  +0.97       +0.99
OT    -0.30  -0.58  +0.55  +0.81  +0.89  +0.93  +0.95       +0.98
MX    +0.41  +0.18  +0.81  +0.92  +0.96  +0.98  +0.98       +0.99
```

η-swing (max − min):
- PL: 1.99 (η spans roughly [-1, +1])
- OT: 1.56 (η spans [-0.58, +0.98])
- MX: 0.82 (η spans [+0.18, +0.99] — non-negative everywhere)

Lipschitz / swing:
- PL: 1.72
- OT: 1.45
- MX: 1.55

**Conclusion**: when the metric is normalized by the η-range a scheme
actually traverses, the three are within 20% of each other. Mixture's
"smoothness advantage" on the raw `lipschitz_eta` is largely a
**scope artifact** of its tighter admissible-η set.

### Why mixture has the tighter range

Admissibility on Normal-Normal:
- PL: any finite η < 1/(1-w) is admissible. The dynamic-η scan can
  freely reach η ≈ -1 (oversharpening: tilted distribution
  narrower than bare posterior) at low |Δ|.
- OT: similar; η < 0 is admissible up to a quantile-monotonicity bound.
- MX: η ∈ [0, η_max] only; **η < 0 is never admissible** because the
  prior (Gaussian) has tails going to 0, so subtracting positive
  weight on the likelihood inevitably yields negative density
  somewhere in the tails (derived in `docs/methods/mixture.md`
  §"Admissibility on Normal-Normal" D1c).

The "missing oversharpening regime" of mixture isn't a bug — it's a
fundamental feature of the m-geodesic: linear-density interpolation
can't sharpen below the input components, only mix between them.

### Is mixture's intrinsic geometry meaningfully smoother?

The η-swing-normalized Lipschitz numbers say *probably no*. The three
schemes have similar per-unit-of-η-range slopes; PL's discontinuity
at |Δ| ≈ 0.5 (where it saturates at η = -1 and then jumps to +0.71)
is the main reason its raw Lipschitz looks higher.

On the other hand, the **smoothness comparison still has practical
value**: in the dynamic-η inversion, what we care about is the
empirical Lipschitz of the η-curve we end up using, not its
normalized form. Mixture's empirical η-curve is genuinely less
jagged because it doesn't have the saturate-at-(-1)-then-jump
pattern that PL exhibits at low |Δ|.

So:
- ✅ "Mixture produces smoother η*(|Δ|) curves in absolute terms" — true.
- ❌ "The m-geodesic is intrinsically smoother than the e-/W2-geodesics" — overclaim.
- ✅ "Mixture lacks an oversharpening regime, which removes one source of η-curve jaggedness" — accurate.

### Width-curve smoothness is similar across schemes

`lipschitz_width` ≈ 0.65 and `total_variation_width` ≈ 0.61 across
all three schemes. The dynamic-η selector is calibrated for all
three (coverage at nominal 0.95), so the resulting CI-width vs. |Δ|
envelope is dominated by the calibration target rather than the
geometry.

## Width comparison (audit-config, single w=0.5)

For completeness, here's the width data at w=0.5 across θ:

```
theta    wald     waldo    pl_dyn    mx_dyn   mx-pl    mx/pl
-3.0    3.92     4.29     4.60      4.50     -0.10    97.85%
-2.0    3.92     3.76     3.93      3.90     -0.03    99.19%
-1.0    3.92     3.44     3.51      3.50     -0.00    99.96%
+0.0    3.92     3.36     3.40      3.40     +0.01   100.16%
+1.0    3.92     3.43     3.50      3.50     +0.00   100.04%
+2.0    3.92     3.74     3.90      3.87     -0.03    99.24%
+3.0    3.92     4.29     4.59      4.50     -0.09    97.98%
+4.0    3.92     4.87     5.15      5.02     -0.14    97.36%
```

Mixture is consistently 1-3% narrower than PL at high conflict
(|θ| ≥ 2). At low conflict the two are within MC noise. Both are
calibrated and slightly wider than bare WALDO (which over-covers
without dynamic-η correction).

## The learned-η tail-decay puzzle

Inspecting the v4 learned-η plots
(`results/phaseC_eta_plots/eta_curves_per_loss.png` and
`output/illustrations/v4_eta_curves.png`), all three losses
(`integrated_p`, `cd_variance`, `static_width`) show a striking
pattern at the demo slice (μ₀=0, σ₀=σ=1):

- |θ| ≤ 2: η ∈ [0.85, 1.0] (close to Wald, ignoring conflicting prior).
- |θ| > 2: **η decreases toward ~0.5-0.65 as |θ| → 5**.

The analytic `NumericalEtaSelector` reference (gray dashed) sits at
η = -1 in the central region (oversharpening) and η = +1 in the
tails (Wald). The learned curves match neither.

### Hypothesis 1: Sparse-data extrapolation at training boundary

The training θ-distribution is
[`SigmaAnchoredUniformThetaDistribution(K=5)`](../../src/frasian/learned/training/sampling.py),
which samples θ ∈ [μ₀ - 5σ₀, μ₀ + 5σ₀] per training element. At the
σ₀=1 demo slice, training θ-range is [-5, +5] — exactly the plot
range. θ = ±5 is on the training boundary; θ = ±4 is approaching it.

Cross-check from panel B of `v4_eta_curves.png`: at σ₀=4 (training
θ-range = [-20, +20]), η stays cleanly at ~1.1 across |θ| ≤ 4 — no
drift, because |θ|=4 is well-supported by training samples. At σ₀=1
(training boundary at |θ|=5), the drift appears. **Strong correlation
between drift and proximity to training boundary.**

But this might not be the whole story (see Hypothesis 3).

### Hypothesis 2: Loss-shape mismatch

The three losses optimize different objectives:
- `static_width`: minimize the analytic Wald-like CI width formula.
- `integrated_p`: minimize ∫ p(θ; D, η) dθ over a θ-grid.
- `cd_variance`: minimize the variance of the confidence distribution.

`NumericalEtaSelector(objective="static_width")` works at fixed D
(per-D optimum). The learned losses are integrated/averaged over D,
prior_hp, lik_hp, AND the per-batch θ. So the learned curves
optimize a different functional, even if they used static-width as
their per-batch loss.

It's possible the integrated optima genuinely have intermediate-η
optima at large |θ| that we haven't analytically anticipated.

### Hypothesis 3: Validity head pulling η down

The Phase G architecture adds a `ValidityNet` that learns
`P(valid | θ, η, prior_hp, lik_hp)` from offline samples and
contributes a `−log P(valid)` boundary penalty to Head A's loss.
At large |θ| with σ₀ = 1, the dynamic-η admissibility might
genuinely shrink (η_max declines as conflict grows even for PL,
where the formula `1/(1-w)` is η-independent — but for the
dynamic regime, ValidityNet may have learned to refuse extreme η
even at conflict). Resulting boundary penalty would push η downward.

### What experiments would disambiguate

Pick `(θ_test, w, μ₀, σ₀, σ)` near the boundary regime (e.g., θ=4,
σ₀=1, σ=1, μ₀=0) and:

A. **Loss-landscape probe.** Sweep η ∈ [-0.5, 1.5] and plot each
   loss (integrated_p, cd_variance, static_width) as a function of
   η. Observation: if there's a clear minimum at η ≈ 1 but the
   trained network sits at η ≈ 0.5, the network failed to learn
   the optimum (training pathology). If the loss is genuinely
   minimized at η ≈ 0.5, the network is correct and the surprise
   is our intuition.

B. **Validity-head ablation.** Train a fixture with the boundary
   penalty weight λ = 0 (or very small). If the drift goes away,
   ValidityNet is the cause. If not, the cause is in the main loss.

C. **Heavier-tailed θ-distribution.** Train with `K=10` (training
   θ-range [-10, +10]) and see if the |θ|=4 region stabilizes. If
   yes, sparse-data extrapolation at the boundary is the cause.

D. **Fingerprint the trained model output at training-distribution
   boundary vs. center.** If η is very different at θ = 4.99
   (within training) vs. θ = 5.01 (outside), boundary effect; if
   it's smooth, more likely a real feature of the loss optimum.

These experiments would take ~20-30 minutes per fixture and don't
require new infrastructure (just `scripts.train_learned_eta` with
modified configs).

### Why this matters for Stage C

Stage C trains MixtureTilting's learned-η fixtures for the audit. If
the tail-decay pattern is a generic training pathology, mixture will
inherit it; the audit results will look "good" but with the same
mysterious large-|θ| behavior. If it's a real loss-optimum feature,
the mixture results are interpretable as-is.

**Recommendation:** before training mixture's 6 fixtures, run
experiment A (loss-landscape probe) at one PL and one mixture cell
to see whether the loss-shape disagreement with the analytic optimum
is intrinsic. ~30 minutes; potentially saves us from training 6
fixtures whose interpretation is ambiguous.

## Bug fix: `_spectral_roughness` on small grids

In `src/frasian/diagnostics/smoothness_metrics.py:63`, the spectral
roughness metric was returning ~1e31 on the audit-config 11-point
η-grid. Root cause:

1. The function mean-detrends the input (`y_c = y - y.mean()`), so
   `spectrum[0]` (the DC bin) ≡ 0 by construction.
2. The "low frequencies" band was defined as `spectrum[:cutoff]`.
3. With n=11 input → 6 frequency bins → `cutoff = max(1, 6//4) = 1`.
4. So `low = spectrum[0]² ≈ 1e-31` (float noise) and `high = O(1)`.
5. Ratio = O(1) / O(1e-31) = O(1e31).

Fix: exclude the DC bin entirely from the partition.

```python
non_dc = spectrum[1:]
n_nd = non_dc.size
cutoff = max(1, n_nd // 4)
low  = sum(non_dc[:cutoff] ** 2)
high = sum(non_dc[cutoff:] ** 2)
```

Existing tests at n=100 still pass; new regression test pins the
small-grid case at <100 (was ~1e31 pre-fix).

## What changed in the codebase

- Stage A: `0a17adc` (closed-form NN tilt + WALDO p-value + CI inversion)
- Stage B: `fcb6257` (generic-MC path + 4 selectors + 9 mx_* audit flavors)
- Bug fix + ot_dyn_numerical: `1d8110a` (this commit)

## Loss-landscape probe results

### Per-θ marginal probe (`scripts/loss_landscape_probe.py`)

For each (scheme, θ_test) at fixed (μ₀=0, σ₀=σ=1, D=0), swept
η ∈ [-1, 1.5] and plotted `p(θ_test; D, η)`. The argmin of this
quantity is where the **per-θ marginal contribution** of
`integrated_pvalue_loss` is minimized.

Result: argmin at η ≈ -1 for every θ_test from 0 to 5 (PL and OT),
while trained nets sit at η ≈ +0.7 to +0.9 — the opposite corner of
η-space.

**At |θ| ≥ 3, p(θ;D,η) is essentially zero across the entire η range.**
The loss has no gradient signal in the tail region. So the tail-decay
isn't an "incorrect optimum" — it's the network's smooth interpolation
filling in something where the loss is genuinely uninformative. This
explains the visual "drift toward 0.5-0.65" at large |θ| in
`v4_eta_curves.png`.

### Full-loss probe at constant η (`scripts/full_loss_landscape_probe.py`)

Evaluated `integrated_pvalue_loss(p)` for `p = p(θ_grid; D, η_const)`
over the same θ-grid as training (`[-5, +5]`, K=5 σ-anchored). The
loss is **monotonically increasing in η_const over [-1, 1.5]**:

```
  eta:    -1.00  -0.50   0.00  +0.25  +0.50  +0.70  +0.85  +1.00  +1.10  +1.25  +1.50
  loss:   0.958  0.998  1.064  1.117  1.197  1.297  1.412  1.596  1.795  2.386  5.798
```

Trained net mean η ≈ +0.885 → loss ≈ 1.41. **That is 1.5× the global
optimum** at η = -1 (loss = 0.96).

There is no plateau, no second basin, no flat region anywhere in
[-1, 1.5]. The trained network has stopped at a point where the
loss gradient is clearly nonzero and pointing toward η = -1. This
is a **training optimization failure**, not a "loss legitimately
prefers η ≈ +0.85" answer.

### Why the optimizer gets stuck at η ≈ +0.85

Looking at `_train_loop.py:299-304`, the total training loss is:

```
total = loss_width
      + λ · boundary_penalty           (validity head)
      + λ_anti_wald · anti_wald_penalty (mean relu(η)² — pushes η ≤ 0)
      + λ_anti_collapse · eta_collapse_penalty (1/var(η) — rewards spread)
```

The `anti_wald` and `anti_collapse` penalties are **scheduled to decay
during training** (per the comment in `losses.py:155-179`):

> *Combined with a decay schedule, it perturbs the optimizer out of
> the Wald basin during early training, then releases its bias so the
> underlying width loss owns convergence.*

Empirically: the optimizer escapes from η ≈ +1 (the original Wald
collapse documented in
[2026-05-09-phase-g-v4-fix.md](./2026-05-09-phase-g-v4-fix.md)) but
**only partway**. After anti-wald decays, the optimizer is left at
η ≈ +0.85 — neither at the Wald plateau (+1) nor at the width-loss
optimum (-1). The width loss has positive gradient toward η=-1 but
it's not strong enough to drag the network across η-space in the
remaining training steps.

This explains:
- Why the σ-anchored fix from the prior note "beats v3" without
  reaching the analytic optimum: it moved the basin from +1 to
  +0.85, both still on the wrong side of where the width loss
  alone wants η.
- Why all three losses (`integrated_p`, `cd_variance`, `static_width`)
  produce similar trained-η means (~0.7-0.9, all positive): the
  optimizer's stuck location is dominated by the optimization
  dynamics, not by the loss-function shape itself.
- Why the per-θ shape varies by loss while the means cluster: in
  the central region where signals exist, the losses produce
  different η(θ) responses; in the tails where the loss is flat,
  all three drift smoothly toward whatever the central value
  is.

### Implication for Stage C (mixture training)

Mixture's admissible η-range is `[0, η_max]` — so **the width-loss
optimum at η=-1 is inadmissible**. The network can't be "stuck on
the wrong side of the optimum" the way PL/OT are; the optimum is
**inside** mixture's admissible range or at its boundary.

The trained mixture η-curves will likely settle at η-values
similar in absolute terms (~0.7 to ~+1) to PL/OT — but for mixture
this might actually be **near-optimal** rather than 1.5× off. We'll
need to verify this with a similar full-loss probe on the trained
mixture fixtures after Stage C.

The tail-decay (loss-flat region) will probably appear in mixture
too, for the same reasons. That's universal across schemes.

## What changed in the codebase (probe scripts)

- `scripts/loss_landscape_probe.py` — per-θ_test marginal probe
  (commit incoming).
- `scripts/full_loss_landscape_probe.py` — constant-η full-loss probe
  (commit incoming).
- Probe figures saved to `output/illustrations/loss_landscape_probe.png`
  and `output/illustrations/full_loss_landscape_probe.png`.

## Open questions / follow-ups

1. **Settle the smoothness narrative.** Is the empirical Lipschitz
   advantage of mixture (~1/3 of PL's) the right metric to highlight,
   or should we report swing-normalized Lipschitz (~similar across
   schemes) as the primary measure?

2. **Validity-head ablation.** Train one PL fixture with the
   validity-head boundary penalty weight λ = 0 to see whether the
   network reaches η ≈ -1. If yes, the validity head is what's
   keeping the optimizer at +0.85.

3. **anti_wald-extension experiment.** Try running training with the
   anti-wald penalty kept on for the full schedule (no decay). If
   the trained network reaches η ≈ -1 cleanly, the decay schedule
   is the real culprit — current schedule decays before width-loss
   takes over. Saved as a follow-up because it requires a config
   change.

4. **Audit results re-interpretation.** The existing
   `pl_learned_*` and `ot_learned_*` audit cells reflect learned-η
   solutions that are **1.5× off the per-slice loss optimum**. They
   still beat Wald/WALDO in the headline (per the prior note), but
   the absolute width / coverage numbers underestimate the true
   "best possible learned-η" performance for those schemes.

5. **Fix the η-curve plot script** to also show training-θ-range
   shading.
