# Learned-η intervention design (post-Stage-2)

## TL;DR

The Stage-2 diagnostics revealed: (1) default training collapses to constant
near-Wald regardless of regularizer perturbations; (2) the gradient signal
toward per-slice optimum **does exist** (positive correlation up to +0.17
recoverable); (3) `anti_collapse + small batch + large LR + no-decay`
finds and stabilizes a structured solution; (4) the current `1/var` anti-
collapse formulation is pathological (drives η to ±15). This memo proposes
a focused recipe — switch to a **targeted-variance** anti-collapse + bound
the network output — and tests it through a small experiment grid.

## What we know

From investigation 2026-05-09 and 2026-05-10:

- All 5 v4 ablations (baseline / no_boundary / no_norm / anti_wald=10 /
  stratified) converge to η_mean ≈ 0.85, η_std ≈ 0.05, corr ≈ 0 with
  per-slice argmin. Constant-near-Wald is a strong attractor.
- The constant attractor IS NOT a true loss minimum: oracle constant-η
  loss is 1.16; trained nets sit at 1.44 (envelope 0.90-1.01 between
  optimum and constant-Wald = 1.44).
- Trained nets are equivalent in loss to constant Wald — they reproduce
  Wald, not better.
- Bug-hunter audit: WALDO formulas, JAX kernels, hook timing all
  verified correct. 3 minor bugs found and fixed.
- Optimizer changes alone (batch=32, lr=3e-3, AdamW) do NOT escape the
  Wald attractor — proved by direct test (corr stayed -0.10 to +0.09).
- anti_collapse=1 alone (`1/var` formulation) breaks the attractor but
  produces undirected spread (corr ≈ 0).
- anti_collapse=1 + small batch + large LR + no-decay schedule produces
  structured solution (corr +0.17 sustained) but with pathological
  extreme η values (mean -10, std 12) due to `1/var` rewarding
  unbounded variance.

**Conclusion:** the structured solution exists in the loss landscape
and gradient descent CAN find it; what we need is a non-pathological
pressure that breaks the constant attractor without driving η to
extremes.

## Intervention design

### 1. Anti-collapse formulation: targeted variance

Replace `1/(var + ε)` with:

```python
def eta_collapse_penalty(eta_pred, target_var=1.0):
    var = jnp.var(eta_pred)
    # ReLU-clipped: no penalty above target_var.
    return (jax.nn.relu(target_var - var)) ** 2
```

Properties:
- Quadratic below target → strong push when collapsed.
- Zero above target → no reward for unbounded variance.
- Smooth (no `1/x` blow-up).

Default `target_var = 1.0` matches the natural scale of per-slice argmin
(range -1.5 to +1.5 → std ≈ 1).

### 2. Bound the network output (architectural)

Add a soft bound to the EtaNet output to prevent pathological extremes.
Wrap the final-layer output in `eta_max * tanh(raw)` where `eta_max = 2`
(slightly outside the [-1.5, +1.5] argmin range to allow some margin).

This is a small change to `EtaNet.__call__`: multiply the MLP output
by a tanh and scale.

```python
# Current:
out = self.mlp(x)  # raw real-valued
# Proposed:
out = eta_max * jnp.tanh(self.mlp(x))  # bounded to [-eta_max, +eta_max]
```

### 3. Schedule: constant or near-constant

Keep anti_collapse active throughout training. Use `--anti-decay-frac 5.0`
(decays to 0.8 × max at end) to retain pressure.

### 4. Optimizer / training settings

- `batch_size = 32` (8× smaller than baseline 256)
- `lr_a = 3e-3` (3× larger than baseline 1e-3)
- `n_epochs = 30` (default)
- `AdamW` (already default)
- `anti_collapse_max = 1.0`
- `anti_decay_frac = 5.0`

### 5. Skip anti_wald

Anti-wald pushes η toward 0 unconditionally. The per-slice argmin has
mean +0.27 — many slices want positive η, some want negative. A uniform
push is wrong; per-slice direction info must come from the width-loss
gradient via anti-collapse-induced spread.

## Success criteria

For a fixture to be considered "working learned-η":
- **corr_with_argmin > 0.3** at convergence
- **|η| ≤ 2** for ≥ 95% of probe samples (admissibility)
- **envelope position < 0.5** between optimum (1.16) and Wald (1.44),
  i.e., loss ≤ 1.30
- **Stable**: corr remains positive throughout the latter half of training

If all four criteria are met, the recipe is a candidate for production.

## Experiment grid

5 targeted experiments, run sequentially:

| # | Description | Settings |
|---|---|---|
| E1 | Baseline of new formulation | target_var=1.0, relu form, no bounds |
| E2 | E1 + output bounding | E1 + tanh×2 on EtaNet output |
| E3 | Tighter target | target_var=0.5, otherwise like E2 |
| E4 | Stronger pressure | E2 + anti_collapse_max=10 |
| E5 | Generalization to cd_variance | E2 settings, --loss cd_variance |

E1-E2 are the main candidates. E3-E4 are sensitivity checks. E5 tests
whether the recipe transfers across loss functions.

## Risks

- **Targeted variance might still produce uncorrelated spread** (variance
  = right magnitude but wrong direction). If so, anti_collapse alone is
  fundamentally insufficient and we need a directional signal (anti_wald
  per-slice, or supervised target).
- **Output bounding might mask the real issue** by clipping symptoms
  rather than fixing root cause. We'd want to see corr improve even
  without bounds.
- **target_var=1.0 might be wrong scale**. Sensitivity check via E3/E4.
- **The recipe might not generalize** to cd_variance / static_width / OT.
  E5 is the canary; if it fails we'd need scheme/loss-specific tuning.

## After this experiment grid

If a fixture meets all four success criteria:
- Update `losses.py` with the new formulation as default.
- Update default training settings.
- Document in CLAUDE.md.
- Re-train all production v4 fixtures.
- Re-run audit.

If no fixture meets the criteria:
- Pivot to supervised target (A3 from the May 9 brainstorm) or
  architectural intervention (mixture of experts).

## Implementation notes

- The new `eta_collapse_penalty` formulation goes in `losses.py`
  (replaces or extends the existing one).
- The output bounding goes in `architecture.py` (EtaNet.__call__).
  Need a flag to switch on/off so we can test E1 (no bound) vs E2 (with
  bound) cleanly.
- Cleanup: archive the existing 5 probe fixtures + 2 anti_collapse
  experiments as evidence; the new fixtures land alongside them.

---

# Results, findings, and conclusions

## Calibrated benchmarks (the single most important finding)

Computed three reference loss values against a fixed n=32 probe batch
sampled from the v4 hyperparam distribution (`probe_calibrated_benchmark.py`):

| Benchmark | Loss | Description |
|---|---:|---|
| #1 per-sample post-selection oracle | 0.953 | argmin η per sample using D[i]; NOT calibrated, free-lunch upper bound |
| #2 best constant η (calibrated) | 1.359 | one η for whole batch; argmin = +0.900 |
| #3 per-(θ,prior,lik) D-marginalized oracle | **1.175** | calibrated achievable ceiling |
| Wald (η=1) | 1.363 | constant; calibrated baseline |
| WALDO (η=0) | 1.653 | constant; far worse than Wald |

**Real calibrated headroom: 14% (1.363 → 1.175).** The post-selection
oracle (#1) suggested 30% headroom but that was illusory.

#2 ≈ Wald means: the best CONSTANT-η learner has essentially no
headroom over Wald. Real headroom requires INPUT-conditional structure.

## Systematic evaluation of all fixtures

Evaluated 14 fixtures against the calibrated benchmark
(`probe_systematic_evaluation.py` + ad-hoc evals). Sorted by `env_cal`
(0 = oracle, 1 = Wald, >1 = worse than Wald):

```
fixture                                env_cal   eta_mean   max|η|
no_boundary (default + λ=0)              0.851    +0.825     0.99   ← BEST
baseline (default everything)            0.940    +0.834     0.95
smallb+lr+noboundary                     0.987    +0.845     0.99
smallb+lr+noac                           0.995    +0.829     1.01
basinB init (drift)                      1.036    +0.634     0.92
stratified                               1.084    +0.740     0.99
smallb+lr+ac1 decay=0.8                  1.126    +0.762     1.31
anti_wald_10                             1.140    +0.695     0.96
no_norm                                  1.232    +0.814     1.32
phase2 release (post-sel teacher)        1.313    +0.101     3.82
anti_collapse=1 (1/var)                  2.439    -0.075     1.58
anti_collapse=100 (1/var)                2.441    -0.076     1.58
phase2ana_pretrained (analytic teacher)  2.252    -0.144   ~50.00
smallb+lr+ac1 nodecay (corr=+0.17)       2.502   -15.898    53.05
phase2 pre-trained (post-sel teacher)    2.394    +0.332     2.40
phase2cal_pretrained (D-marg, 20 mc)     3.024    +0.520    >2.00
```

**Best fixture (`no_boundary`) captures 15% of the 14% calibrated
headroom = 2.1% absolute improvement over Wald.** All "structured"
fixtures (anti_collapse, phase2 variants) end up WORSE than Wald.

## Hypotheses tested across the investigation

| # | Hypothesis | Result |
|---|---|---|
| 1 | Optimizer changes (small batch, large LR, AdamW) | Don't help |
| 2 | Anti-wald penalty | Hurts (forces η < 0 uniformly, but argmin varies in sign) |
| 3 | Anti-collapse with `1/var` formulation | Drives to extreme η values; loss worse than Wald |
| 4 | Stratified batching | No effect on env_cal |
| 5 | Removing input normalization | No effect (or slightly worse) |
| 6 | Removing boundary penalty | Modest improvement (best at 0.851) |
| 7 | Longer training (100 epochs) | No improvement; converges by epoch 14 |
| 8 | Basin B initialization | Network drifts back toward A; lands at intermediate worse spot |
| 9 | Phase 2 teacher-forcing (post-selection oracle) | High held-out loss; trajectory toward A |
| 10 | Phase 2 teacher-forcing (D-marg MC oracle, 20 samples) | Overfits noise; held-out loss worse |
| 11 | Phase 2 teacher-forcing (analytic DynamicNumericalEtaSelector) | Targets in [-47, +1]; held-out loss worse |

## The deeper finding (the smoking gun)

**The `integrated_pvalue_loss` and the framework's `DynamicNumericalEtaSelector`
optimize DIFFERENT objectives.**

The `DynamicNumericalEtaSelector` minimizes the analytic Wald-like CI
*width* formula at each θ. For PowerLaw with no lower bound on
admissible η, this drives η to large negatives (we saw -47).

Meanwhile the v4 fixtures train against `integrated_pvalue_loss`, which
integrates p(θ) over a θ-grid. This loss has a Basin A attractor
around constant η ≈ +0.85.

The two objectives don't agree:
- The "calibrated default" (analytic selector) picks per-θ η values
  driven by width minimization that don't track the integrated_p
  optimum.
- Training the network to minimize integrated_p collapses it to
  Basin A, not the analytic-selector solution.

**Implication: the framework's design assumes these two are
equivalent or close. They're not.** The v4 fixtures aren't "broken"
— they're correctly minimizing what we asked them to (integrated_p).
The framework's stated calibrated default is a *different* function.

## Robust observations across the investigation

1. **Basin A is a real local minimum** of the integrated_p loss
   surface. Multiple attempts to escape it (init, regularization,
   teacher-forcing) all drift back to it under width-loss training.

2. **No simple training-config tweak claims more than 2.1% absolute
   over Wald.** anti_collapse can produce structured solutions but
   they have higher loss than Basin A.

3. **The post-selection per-sample oracle (probe.argmin_eta) is NOT
   a useful teacher signal.** Its values are tuned for specific D
   samples; held-out loss is high.

4. **The framework's existing v4 fixtures (PL, OT) are doing
   approximately as well as gradient descent on integrated_p can
   do** — they capture ~15% of the calibrated headroom. Not great,
   but not failure either.

5. **The `corr_with_argmin` metric was misleading.** Many "structured"
   fixtures had positive correlation but worse loss than Basin A.
   Correlation is a directional indicator that doesn't account for
   magnitude pathology.

6. **The 22% loss gap from #1 (post-selection oracle) to Wald was
   illusory.** Real calibrated gap is 14%, and 15% of that is
   already captured by current fixtures.

## What this means for the framework

Three honest interpretations:

### A. The current design works as well as it can.

The integrated_p loss ⇒ Basin A ⇒ ~2% improvement over Wald is the
true ceiling for this loss + this hyperparam range. To get more, the
framework needs to change the training loss (not the network or
training-config).

### B. The framework conflates two different objectives.

The "calibrated default" `DynamicNumericalEtaSelector` is the
documented production answer, but it doesn't minimize integrated_p
loss. The v4 fixtures train against integrated_p but are supposed to
mimic the analytic selector. There's an inconsistency in the
framework's design.

### C. The integrated_p loss is the wrong objective for a calibrated
learner.

The post-selection oracle (lowest loss but not calibrated) and the
analytic selector (calibrated but unbounded η) bracket the design
space. Neither is what we actually want. We want a loss whose
minimum IS the calibrated D-marginalized oracle. That requires
either D-marginalization in the loss itself (slow) or a different
formulation entirely.

## Pivot options

1. **Ship `learned-eta` as it is** — the v4 fixtures capture 2%
   improvement over Wald; document the limitation in CLAUDE.md and
   move on. Mixture-tilting Stage C can proceed without learned-eta
   being "fixed".

2. **Replace learned-eta with a lookup table** — precompute the
   D-marginalized calibrated oracle on a coarse grid offline,
   interpolate at inference. Deterministic, no training pathology,
   captures the full 14% headroom. ~hours of one-time compute.

3. **Reframe the loss as D-marginalized integrated_p** — change the
   training loss to E_D[integrated_p] via per-batch MC. ~10× slower
   training but the loss minimum IS the calibrated oracle.

4. **Investigate the framework's design assumption** — if the v4
   fixtures are training against integrated_p but are SUPPOSED to
   mimic the analytic selector, the framework's design has an
   inconsistency that needs reconciling. This might be a bigger
   research project than fixing the training.

Option 1 is fastest. Option 2 is most likely to actually work.
Option 3 is the most rigorous. Option 4 is the deepest but slowest.

## Files produced during this investigation

Scripts (all in `scripts/`):
- `probe_loss_distance.py` — initial post-selection oracle measure
- `probe_calibrated_benchmark.py` — D-marginalized calibrated oracle
- `probe_systematic_evaluation.py` — all-fixtures comparison
- `probe_eta_curves.py` — η(θ) at demo slice
- `probe_phase2_teacher.py` — Phase 2a (post-selection oracle)
- `probe_phase2_calibrated_teacher.py` — Phase 2 (D-marg MC, 20 MC)
- `probe_phase2_analytic_teacher.py` — Phase 2 (analytic selector)

Fixtures (in `artifacts/`, all gitignored):
- `probe_v4_baseline.eqx` + 4 disambiguation probes
- `probe_v4_anti_collapse_{1,100}.eqx`
- `probe_v4_smallbatch_largelr*.eqx` (3 variants)
- `probe_v4_basinB_init.eqx`
- `probe_v4_phase2_pretrained.eqx`, `_release.eqx`
- `probe_v4_phase2cal_pretrained.eqx`
- `probe_v4_phase2ana_pretrained.eqx`
- `probe_v4_noboundary_100ep.eqx`

All fixtures trained on PowerLaw + integrated_p loss with the v4
hyperparam distribution. Total compute: ~3-4 hours of CPU time.
