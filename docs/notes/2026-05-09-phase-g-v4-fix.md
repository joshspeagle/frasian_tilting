# Phase G v4 fix: σ₀-anchored θ training + input normalization

**Date:** 2026-05-09
**Commits:** `3e12f08`, `8905ac6`, `accf141`, `b69cbeb`
**Affected:** `src/frasian/learned/training/`, `src/frasian/tilting/{eta_selectors,ot}.py`,
`experiments/canonical_normal_normal_*_v4.yaml`, `scripts/run_wald_audit.py`,
`scripts/plot_v4_eta_curves.py`

## TL;DR

The initial Phase G v4 conditional learned-η fixtures collapsed to
η ≈ 1 (= Wald, prior dropped) — wider than Wald on average. Two
architectural changes restored beats-Wald behavior:

1. **σ₀-anchored θ training distribution** — per-batch-element
   `θ_train ~ U(μ₀ − Kσ₀, μ₀ + Kσ₀)` with K=5, replacing
   `Uniform[-10, 10]`. Concentrates training mass where the prior
   is informative.
2. **Per-channel input z-score normalization** on EtaNet/ValidityNet
   (loguniform features in log-space). Keeps θ from being bled out
   of the network by weight decay.

Plus a runtime change: `LearnedDynamicEtaSelector` now clamps
out-of-admissible η instead of refusing past a 20% threshold. The
dynamic-CI calibration guarantee holds for any η, so clamping
preserves coverage; refusing was overly conservative once σ₀-anchored
training surfaced benign extrapolation.

After these fixes: 24/24 random hp triples beat Wald (mean Δ = −0.32
≈ 8% narrower); demo-slice headline matches v3; 240/240 audit cells
across 6 (PL × OT × 3 losses) flavors hold coverage within 3·SE of
nominal.

## Context

Phase G (Steps 1-12 in the migration table) made the learned-η
selector **conditional**: a single checkpoint trained over a *range*
of `(prior_hp, lik_hp)` rather than a per-experiment fixture. v3 had
been trained per-(model, prior) and refused cross-experiment use; v4
generalizes via `EtaNet(θ, prior_hp, lik_hp) → η` and
`ValidityNet(θ, prior_hp, lik_hp, η) → logit`.

The plan completed cleanly through Step 11. We then refreshed the
Wald audit (the `(NN × tilting × statistic)` matrix at
`scripts/run_wald_audit.py`) against the new v4 fixtures. The
non-learned flavors were unchanged (deterministic). The learned
flavors (`pl_learned_intp`, `pl_learned_cd_var`, `pl_learned_static_w`)
needed retraining at the v4 architecture.

## What we tried

Each step that follows was an attempt to fix the η ≈ 1 collapse.

### 1. Initial v4 retrain (failed)

Trained `pl_learned_intp` on the original v4 YAML
(`Uniform[-10, 10]` θ_train + raw input concat + `[64, 64]` hidden +
default budget). Audit showed mean width 4.36 vs Wald 4.12 — **5%
wider**. Probed η(θ) curves: essentially constant η ≈ +1.0 across
all (θ, σ₀) slices.

### 2. Capacity bump (failed)

`[64, 64] → [128, 128, 128]` hidden. Same collapse, marginal width
improvement (4.15). η still constant.

### 3. Anti-Wald regularizer (failed, but kept as opt-in)

Added `losses.anti_wald_penalty(η_pred) = mean(relu(η_pred)²)` and
`losses.eta_collapse_penalty(η_pred) = 1 / (Var_B[η_pred] + ε)` with
a `decay_schedule` mirroring the existing `lambda_schedule` for
boundary penalty (but inverted in time — starts large, decays to 0).
Even at λ_max=10 over 50% of training, the model snapped back to
η ≈ 1 in the post-decay phase. Anti-collapse alone increased η
*spread* but in the wrong direction (η went UP at small σ₀ instead
of down).

These are now plumbed in `losses.py` / `_train_loop.py` /
`scripts/train_learned_eta.py` (`--anti-wald-max`,
`--anti-collapse-max`, `--anti-decay-frac`) but **default to 0** since
they didn't escape the basin without the architectural fix.

### 4. Other losses (failed identically)

Tried `static_width` (directly minimizes α=0.05 width) and
`cd_variance`. Both converged to η ≈ 1 with the same magnitude. The
loss surface around Wald is a wide flat plateau under the broken
training distribution; no smooth loss escapes it.

### 5. Gradient diagnostic (revealing)

Built a numerical check of `∂(integrated_p loss)/∂η` at the failing
slice (μ₀=0, σ₀=0.4, σ=1, w=0.14) using `jax.grad`. Result:

- Loss at η=0 (WALDO): 1.04
- Loss at η=1 (Wald): 1.60
- ∂L/∂η at η=1: **+10.0** (strong signal toward smaller η)

So the gradient was correct; the model was failing to use it.

### 6. Average-loss-across-batch diagnostic (the smoking gun)

Computed mean loss at constant η ∈ {-2, -1, 0, +1} averaged over a
batch of (μ₀, σ₀, σ, θ_train, D) triples sampled from the training
distribution. With θ_train ~ Uniform[-10, 10]:

| η | mean loss | per-sample optimum frequency |
|---|---|---|
| -2.0 | 3.99 | 18% |
| 0.0 | 3.50 | 3% |
| +1.0 | **1.75** | **61%** |

**η = +1 (Wald) is the global optimum on the training distribution.**
Most batch samples have D far from any μ₀ ∈ [-2, 2] (since θ_train
is uniform on [-10, 10], 80%+ of training samples are at θ values
where any plausible prior is wrong). At those samples, η ≈ 1 = Wald
genuinely is optimal — the model is correctly converging.

### 7. Switching to prior-conditional θ_train (the fix)

Verified the loss landscape flips when θ_train is sampled near the
prior. With θ_train ~ N(μ₀, σ₀²):

| η | mean loss | per-sample optimum |
|---|---|---|
| -2.0 | 1.56 | **55%** |
| 0.0 | 1.51 | 10% |
| +1.0 | **1.88** | 10% |

Mean loss(η=0) = 1.51 < mean loss(Wald) = 1.88 — η < 0 is now
optimal for most samples.

### 8. σ₀-anchored implementation

Rather than draw from the prior directly (which would lose audit
coverage at θ far from μ₀), we settled on
`SigmaAnchoredUniformThetaDistribution(K)`:
`θ_train ~ U(μ₀ − Kσ₀, μ₀ + Kσ₀)` per batch element. K=5 covers
the prior ±5σ (matches typical CI evaluation grids).

Plumbing required:
- New `ThetaDistribution` variant with `is_anchored: ClassVar[bool] = True`.
- `anchor_theta_to_prior` helper in `sampling.py`.
- New `theta_grid_lo` / `theta_grid_hi` ExperimentConfig fields
  (absolute integration grid; the relative U(-K, K) sampler doesn't
  define one).
- Conversion sites in `_training_step`, `prepare_held_out_validity`,
  `_validity_data.collect_validity_batch`, `train.py` held-out validation,
  and the input-normalization stats lookup.

### 9. Per-channel input normalization

Independently of the σ₀-anchored fix, EtaNet's raw concat of
`[θ, μ₀, σ₀, σ]` had heterogeneous scales (θ range 20, σ₀ range
~5×, etc.). Added `feature_loc` / `feature_scale` / `feature_log`
static fields to EtaNet/ValidityNet. `feature_log=True` for
loguniform-distributed features (σ₀, σ); the network sees
`(log(x) − log_mean) / log_std` for those channels. Default ON;
opt-out via `--no-normalize-inputs`.

Without σ₀-anchored: normalization alone made the collapse *worse*
(narrow-hp test went from beating Wald slightly to losing). With
σ₀-anchored: normalization is necessary to keep θ in the network at
the wider hp ranges.

### 10. Clamp-instead-of-refuse (commit `b69cbeb`)

After the σ₀-anchored fix, the audit's `pl_learned_cd_var` and
`pl_learned_static_w` flavors hit a *different* failure: at small
σ₀ slices (e.g. w=0.2 → σ₀=0.5) the dynamic CI scan queries
θ ≈ D ± 8σ, reaching θ ≈ ±13 — well outside the per-slice trained
range μ₀ ± Kσ₀ = [-2.5, +2.5]. The cd_variance fixture's η
extrapolation drifts out-of-admissible at those θ values, which the
runtime selector was raising on past a 20% threshold.

The dynamic-CI calibration guarantee — "η at each θ depends only on
θ, not on D, so the WALDO p-value at any fixed η is U[0, 1] under
H_0" — holds for *any* η, including a clamped one. So clamping
preserves coverage; only widths inflate at the boundary. Bumped
`_CLAMP_FAIL_THRESHOLD` from 0.20 to 1.0 (effectively disabling the
hard refusal). The warning still fires for inspection.

## What we found

### Headline (commit `8905ac6` regen)

```
                            θ=0    θ=1    θ=2    θ=3    θ=4
Wald                        3.92   3.92   3.92   3.92   3.92
bare WALDO                  3.33   3.43   3.75   4.23   4.78
power_law[numerical]        3.37   3.49   3.91   4.54   5.24
power_law[learned]          3.61   3.62   3.67   3.76   3.86
```

`power_law[learned]` essentially matches v3's pre-Phase-G headline
(3.63, 3.64, 3.68, 3.75, 3.82). The Phase G v4 conditional
architecture now matches v3 at the demo slice while also
generalizing across the full (μ₀, σ₀, σ) hyperparam range.

### HP-sweep narrowness criterion

24 random hp triples drawn from the trained range, 50 reps each at
θ_true=μ₀:

| fixture | mean Δ vs Wald | beats Wald |
|---|---|---|
| (initial wide-θ) | +0.20 | 42% |
| narrow-θ uniform + norm | -0.04 | 55% |
| **σ₀-anchored U(-5,5) + norm** | **-0.32** | **100% (24/24)** |

### Per-loss audit (commit `accf141` for OT plumbing,
`b69cbeb` for clamp fix)

Mean width across the audit's (8 θ_true × 5 w = 40 cells) grid for
each of the 6 learned audit flavors (Wald = 3.92):

| flavor | mean | Δ vs Wald |
|---|---|---|
| pl_learned_intp | 3.80 | −0.12 |
| pl_learned_cd_var | 3.85 | −0.07 |
| pl_learned_static_w | 3.79 | −0.13 |
| ot_learned_intp | 3.75 | −0.17 |
| ot_learned_cd_var | 3.74 | −0.18 |
| ot_learned_static_w | 3.75 | −0.17 |

Coverage: **240/240 cells** (6 flavors × 40) within 3·SE of nominal
0.95.

Per-loss tradeoff at the demo slice:
- `cd_variance` — sharpest at center (3.45 at θ=0), worst at audit
  edges (5.28 at extreme conflict). Tightest in-trained-range, pays
  on extrapolation.
- `integrated_p` — balanced (3.61 at θ=0, 3.88 at θ=4). Best uniform
  performer, safest extrapolation.
- `static_width` — narrowest at conflict (3.67 at θ=3, 3.73 at θ=4),
  barely better than Wald at center (3.86). Optimized for the
  conflict band.

### η(θ) curves

Plot at `output/illustrations/v4_eta_curves.png`
(via `scripts/plot_v4_eta_curves.py`). Two-panel: Panel A overlays
the 3 learned losses + numerical optimum at the demo slice; Panel B
shows σ₀ adaptation for the cd_variance fixture.

Key qualitative observation: the learned curves are a
**muted inverse-V** relative to numerical's per-D V-shape. This is
the D-marginalized version of the per-D optimum: the conditional
EtaNet picks η based on `E[D | θ_query, hp]`, which favors a milder
prior-use than the per-D optimum (over-aggression at the prior
center hurts when D actually disagrees with the prior).

### OT generic dynamic CI inversion (commit `accf141`)

Power_law had a Phase F enhancement enabling
`dynamic + force_generic` on NN; OT didn't. Added:

- `_generic_tilted_pvalue_ot_vec` — vectorized generic MC tilted
  p-value (currently a Python loop over (θ_i, η_i); a true
  triple-batched version mirroring power_law's chunked layout would
  bring wall time from ~5-6 s/CI down to ~1-2 s/CI).
- `OTTilting.dynamic_tilted_confidence_interval_ot_generic` — full
  dynamic CI inversion via the new vec helper.
- Dispatch in `OTTilting.confidence_regions` for
  `force_generic + dynamic + NN` (still raises for non-NN).

Sanity-checked against closed-form at the demo slice: Δw ≤ 0.04 at
n_mc=2000.

## What changed in the codebase

| File | Change |
|---|---|
| `src/frasian/learned/training/sampling.py` | New `SigmaAnchoredUniformThetaDistribution`, `anchor_theta_to_prior` helper, `theta_grid_lo`/`theta_grid_hi` ExperimentConfig fields. |
| `src/frasian/learned/training/architecture.py` | New `feature_loc`/`feature_scale`/`feature_log` static fields on EtaNet + ValidityNet, applied at forward. |
| `src/frasian/learned/training/hyperparam_distribution.py` | `feature_stats()` method on ScalarDist + HyperparamDistribution (for normalization stats). |
| `src/frasian/learned/training/losses.py` | New `anti_wald_penalty`, `eta_collapse_penalty` (kept opt-in, default off). |
| `src/frasian/learned/training/_losses_compose.py` | New `decay_schedule` (mirror of `lambda_schedule`, inverted). |
| `src/frasian/learned/training/_train_loop.py` | Plumbed σ₀-anchored conversion + anti-* regularizers + normalization through head_a_step. |
| `src/frasian/learned/training/_validity_data.py` | σ₀-anchored conversion at aux + held-out sampling sites; NN-vs-generic dispatch fix in `compute_pvalues_per_sample_with_hp` (Bernoulli was returning all-NaN). |
| `src/frasian/learned/training/train.py` | `normalize_inputs=True` default; plumbed all the above. |
| `src/frasian/tilting/eta_selectors.py` | `_CLAMP_FAIL_THRESHOLD = 1.0` (was 0.20); `_check_scheme` reads `meta["experiment_config"]["scheme"]` with fallback. |
| `src/frasian/tilting/ot.py` | `_generic_tilted_pvalue_ot_vec`, `dynamic_tilted_confidence_interval_ot_generic`, `force_generic + dynamic + NN` dispatch. |
| `experiments/canonical_normal_normal_powerlaw_v4.yaml` | `theta_distribution: sigma_anchored_uniform K=5`, `theta_grid_lo: -30`, `theta_grid_hi: 30`. |
| `experiments/canonical_normal_normal_ot_v4.yaml` | Same. |
| `scripts/run_wald_audit.py` | New `pl_learned_intp_generic` + `ot_learned_*` flavors; `_learned_selector(loss, scheme=...)` generalized. |
| `scripts/plot_v4_eta_curves.py` | New (two-panel η(θ) diagnostic figure). |
| `scripts/train_learned_eta.py` | `--no-normalize-inputs`, `--anti-wald-max`, `--anti-collapse-max`, `--anti-decay-frac` flags. |

## Open questions / follow-ups

1. **OT generic vec is a loop wrapper.** A true triple-batched OT
   implementation (mirroring power_law's chunked `D_3d`/`log_lik_3d`
   layout) would make OT generic-MC audits practical (~1-2 s/CI vs
   the current ~5-6 s/CI).
2. **cd_var extrapolation cliff.** At small σ₀ slices, the cd_variance
   fixture's η drops below admissibility quickly outside the trained
   per-slice θ range. Clamping handles it correctly but width inflates
   at the boundary. A regularizer that penalizes the slope of η(θ)
   in extrapolation regions could keep cd_var's in-range tightness
   without the boundary cost.
3. **Bernoulli + power_law audit not refreshed.** Phase G fixed the
   Bernoulli pvalue dispatch (`compute_pvalues_per_sample_with_hp`
   was missing the NN-vs-generic dispatch and returning all-NaN), and
   the canonical Bernoulli v4 fixture is in place, but no
   `bernoulli_learned_*` audit flavor exists. Worth adding when
   non-NN learned cells become a research priority.
4. **σ₀-anchored extrapolation buffer.** Could either widen K (e.g.
   K=8 to cover more of the dynamic-CI search window) or train an
   explicit "extrapolation tail" by mixing σ₀-anchored with a small
   fraction of uniform θ. Tradeoff: K=5 gives the cleanest in-range
   beats-Wald margin; wider K may dilute that.
5. **Anti-Wald / anti-collapse regularizers.** Plumbed but unused.
   They didn't escape the pre-fix collapse, but they might help in
   regimes where σ₀-anchored isn't a clean fix (e.g. non-conjugate
   models where there's no obvious "informative-prior" sampler).
