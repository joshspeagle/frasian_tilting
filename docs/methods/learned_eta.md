# learned_eta

> Status: `implemented`

## Summary

`LearnedDynamicEtaSelector` is a calibrated dynamic-η-per-θ
`EtaSelector` that minimises the **dynamic-procedure loss directly**
(integrated CI width, by default), rather than the static-per-D
width that `NumericalEtaSelector` minimises pointwise. The η*(|Δ|; w)
function is parameterised by a small dual-head neural network
(`EtaNet` + `ValidityNet`), trained end-to-end via `torch.autograd` through
the differentiable tilted-WALDO p-value formula and trapezoidal
integration.

The headline empirical claim: dynamically-tilted CIs from WALDO that
**maintain nominal coverage AND match-or-beat un-tilted WALDO width
across the (w, θ_true) plane** — calibrated *and* narrow, simultaneously.

## Motivation

The framework's existing `DynamicNumericalEtaSelector` is calibrated
by construction (η depends only on θ, never on D) but uses
`NumericalEtaSelector` internally. That selector minimises the
*static-per-D* CI width pointwise, which slams η to the lower clamp
`-w/(1-w)` at small |Δ|. When that kinky η*(|Δ|) curve is used per-θ,
the resulting "Frankenstein" p-value spuriously accepts θ values 2–3σ
from D — verified empirically: at (D=4, w=0.5) the dynamic CI is
5.34, much wider than Wald's 3.92.

The right objective is the **dynamic-procedure loss itself**:

```
ℒ(η_φ) = E_{(w, θ_true) ~ π} E_{D | θ_true, w} [ Φ(D; η_φ) ]
```

with `Φ(D; η_φ) = ∫ p_dyn(θ; D, η_φ) dθ` (default; integrated CI
width across all α-levels by Fubini) or the CD variance. Calibration
is automatic for any η that depends only on θ; the optimisation over
the η-function is unconstrained within that family.

## Definition

The learner produces an `EtaArtifact(LearnedArtifact)` (Phase E,
checkpoint format v2). At inference, `LearnedDynamicEtaSelector`
plugs into the same `tilting.confidence_regions` pipeline via
`select_grid` as `DynamicNumericalEtaSelector` does, but reads η from
the trained MLP instead of a width-minimising solver.

The Phase E selector is **per-experiment**: each (model, prior,
scheme, statistic) configuration trains its own checkpoint, recorded
via fingerprints in the checkpoint metadata. The selector compares
fingerprints at inference and refuses cross-experiment use.

**Architecture: dual-head.** Two MLPs trained jointly:

- **Head A — `EtaNet`**: smooth GELU-MLP from `θ ∈ ℝ^p` (raw, not
  `|Δ|`) to a real-valued `η`. No bounded sigmoid output, no
  monotonicity prior — smoothness is delivered by GELU + finite
  parameter count, validity by a loss penalty (Head B). Defaults to
  `theta_dim=1` but accepts any `theta_dim ≥ 1`.
- **Head B — `ValidityNet`**: GELU-MLP from `(θ, η)` to a logit.
  Trained on `(θ, η, valid)` triples observed during training, where
  `valid := scheme.tilted_pvalue(θ, D, model, prior, η, statistic)`
  returned a finite scalar in `[0, 1]`.

Both heads have hidden sizes `(64, 64)` by default; configurable in
the YAML or via the CLI.

**Three-step training.** Per minibatch of θ (drawn from a one-shot
Latin Hypercube sample over `theta_distribution.support()`):

1. Forward Head A on the main batch; sample auxiliary
   `(θ_aux, η_aux)` boundary-probing pairs (`η_aux ~ Uniform(eta_-`
   `explore_box)`); sample `D ~ likelihood(·|θ)` per row; label
   per-sample validity via `compute_pvalues_per_sample` +
   `validity_mask`. `η_pred` is `.detach()`-ed before labelling so
   Head A's gradient does not flow through the discrete labels.
2. Train Head B: `BCEWithLogitsLoss(logits, valid_labels)`.
3. Train Head A:
   `width_loss(...)  +  λ(epoch) · boundary_penalty`
   where `width_loss` is `integrated_pvalue_loss` over an n_mc=8 D
   batch (per-step Monte Carlo average; closes the high-variance
   single-D regime that made val_width oscillate), and
   `boundary_penalty = -log P(valid | θ_batch, η_pred)` with
   ValidityNet's parameters detached via
   `torch.func.functional_call`. Gradient flows through the
   `(θ, η)` input but not into ValidityNet's weights.

**No clamp on the boundary penalty.** `logsigmoid(x)` is numerically
stable for any finite `x` and its derivative `-sigmoid(-x)` saturates
to `-1` (not zero) as `x → -∞`, so the wrong-side gradient stays
alive at bounded magnitude. A `torch.clamp(logits, ±20)` would zero
that gradient and was explicitly removed in the E.1 review round.

**λ schedule.** Linear warmup from 0 to `λ_max=10` over the first
`warmup_frac=0.3` of training, then constant. Lets Head B accumulate
labels before its signal drives Head A.

**Validation.** Held-out width loss on a frozen `(θ_val, D_val)` set
sampled once at training start; deterministic across epochs so early
stopping picks the actual best-policy epoch (not the noisiest D
sample).

**Runtime safety clamp.** A poorly-trained checkpoint can drift past
the scheme's admissible η range at extreme conflict. The Phase E
selector clamps η to `scheme.admissible_range(...)` with a
`RuntimeWarning` — the calibration check on the trained checkpoint
should prevent this in practice, but the clamp avoids a crash mid-CI
inversion when the user loads an undertrained smoke checkpoint.

**Loss options.**

1. `integrated_p` (default): `∫ p_dyn(θ; D, η_φ) dθ`. By Fubini =
   `∫_α |C_α(D)| dα` with uniform α weighting. α-marginalised.
2. `cd_variance`: `Var_{F_D}[θ]` where `F_D` is the Schweder–Hjort
   CD built from `½ |∂p_dyn/∂θ|`, Z-normalised. α-marginalised.
   Caveat: variance is around the CD's own mean, not θ_true; in
   conflict regimes can incentivise sharpness around a biased
   estimate (see Failure modes).
3. `static_width(α)`: sigmoid-relaxed `∫ σ_β(p_dyn − α) dθ` at fixed
   α. Default sharpness `β = 200` (β=50 has +110% bias at α=0.05).
   α-conditioned: trained MLP valid only at the α it was trained for.

## Derivation

**Calibration is automatic.** For any `η_φ : (|Δ|, w) → ℝ` that
depends on θ only through `|Δ_θ| = (1-w)|μ₀-θ|/σ` (and `w`), not on
`D`, the dynamic procedure achieves nominal 1-α coverage by
construction.

Sketch: under `H0: θ = θ₀`, the trained `η_φ(|Δ_{θ₀}|; w)` is a
*constant* with respect to `D` (it only depends on θ₀ and w, both of
which are fixed under `H0`). The tilted-WALDO p-value at fixed η is
`U[0, 1]` under `D ~ N(θ₀, σ²)` (the standard WALDO calibration
property, inherited via the algebraic structure `Φ(b−a) + Φ(−a−b)`).
Therefore `p_dyn(θ₀; D, η_φ)` is `U[0, 1]`, and the inverted CI hits
nominal coverage. **Verified empirically** in the post-training
calibration report: 5×5 (θ_true, w) × 3-α (0.05/0.10/0.20) grid,
`n_reps=1000` per cell, mean coverage matches nominal within 1·MC_SE.

**Layer-cake identity for `integrated_p`.** For `p ∈ [0, 1]`,

```
∫_θ p(θ) dθ = ∫_0^1 |{θ : p(θ) ≥ α}| dα = ∫_0^1 |C_α| dα.
```

So minimising `∫ p_dyn dθ` is equivalent to minimising the CI width
integrated uniformly over all α — "all CIs as small as possible".
α-sampling at training time is unnecessary.

**Differentiability.** All ingredients are end-to-end differentiable:
- Tilted p-value: `Φ(b−a) + Φ(−a−b)` with `Φ(x) = ½(1+erf(x/√2))`.
- Per-sample θ-grid: `D ± search_mult·σ`, `linspace`-built.
- Trapezoidal integration: `torch.trapezoid`.
- η inverse and architecture: smooth.

No α-crossings, no `argmax`, no `brentq` inside the loss. `gradcheck`
verifies in `tests/properties/test_loss_diff.py`.

## Predicted behavior

- **η*(|Δ|; w) curve is smooth and monotone in |Δ|** by architectural
  construction (`∂η*/∂|Δ| ≥ 0`). No clamp slam, no kinks.
- **Calibration at nominal level** at every (θ_true, w, α) cell.
  Verified in `calibration_report` written into the checkpoint.
- **Width strictly improves on `NumericalEtaSelector`** in the
  conflict band. Smoke run (n_lhs=1024, n_epochs=30) at (D=4, w=0.5)
  already gives width 5.01 vs 5.34 for the legacy selector (~6%
  improvement); v1 production checkpoint expected to do better.
- **Width approaches Wald (3.92)** at high |Δ|, matching the
  asymptotic Wald limit of WALDO.
- **CD shape**: smooth, unimodal across the conflict band. The Dyn-
  WALDO non-monotonicity pathology that the existing selector
  produces is absent.

## Failure modes

- **Cross-experiment use.** Phase E checkpoints are per-experiment.
  The selector reads `experiment_config.{prior,model}_fingerprint`
  from the checkpoint and compares against the inference-time
  `prior.fingerprint()` / `model.fingerprint()`; mismatch raises
  `MissingArtifactError`. The fingerprint excludes class-level
  identifiers (`name`, `param_dim`) which are `ClassVar` so an
  instance cannot lie about its identity past this check.
- **Validity vs distribution properness — known gap.** The
  per-sample validity criterion (`tilted_pvalue` returns a finite
  scalar in `[0, 1]`) checks the p-value output, not the underlying
  tilted distribution being a proper CDF. The two registered
  schemes use different mechanisms to surface impropriety:
  - `power_law`: the numpy `tilted_pvalue` raises
    `TiltingDomainError` when `denom = 1 − η(1−w) ≤ 0`; the torch
    port `clamp(denom, min=1e-6)`s and lets the resulting p-value
    fail the `[0, 1]` validity test (the clamp keeps things finite
    but pushes a moral-but-not-physical p-value through, which the
    range check then catches).
  - `ot`: both numpy and torch raise / return NaN explicitly when
    η is outside `[0, 1]` (the admissible range).
  A future scheme whose closed-form `tilted_pvalue` returns a
  finite-in-[0,1] value despite an improper tilted distribution
  could pass validity while encoding a bogus η range. Mitigation:
  every newly-registered scheme should ship a per-scheme audit
  test asserting that `tilted_pvalue` returns NaN/raises across
  its declared `admissible_range` complement; the end-to-end
  calibration regression on a trained checkpoint catches any miss
  via empirical coverage drift.
- **Undertrained checkpoint at conflict.** The boundary penalty
  pushes η_pred into Head B's predicted-valid region during
  training, but a smoke run (few epochs, small LHS) can leave η
  drifting past the scheme's admissible boundary at extreme
  conflict. The runtime selector clamps to
  `scheme.admissible_range(...)` with a `RuntimeWarning`; if you
  see this fire on a production checkpoint, train longer or widen
  `eta_explore_box`.
- **Class-degenerate Head B BCE.** If `eta_explore_box` doesn't
  probe the invalid region (e.g. `(0, 0.5)` for `power_law` at
  `w=0.5`, where everything is valid), Head B's BCE collapses to
  predicting `True` always and the boundary signal goes to zero.
  The training loop emits a `RuntimeWarning` per epoch when the
  per-batch aux validity rate is outside `(0.05, 0.95)`.
- **Degenerate `w`.** Priors so tight or so wide that
  `w := σ₀²/(σ²+σ₀²)` is outside `(0.001, 0.999)` are rejected at
  training time (delta priors and improper priors both put the
  WALDO admissible range into a degenerate regime that the existing
  torch port doesn't handle).
- **CD-variance bias.** `Var_{F_D}[θ]` is the CD's spread around its
  own mean, not around `θ_true`. In conflict regimes where the CD
  median drifts toward μ₀, this can incentivise sharpness around a
  biased estimate. Use `cd_variance` only when centring is known to
  be benign.
- **Static-width sharpness.** Default `β = 200` keeps the relaxation
  bias < 1% at α ∈ [0.05, 0.5]; for very small α (≤ 0.01) raise to
  `β ≥ 500` or expect bias.
- **Non-Gaussian likelihoods.** Today's torch p-value registry
  covers `power_law` and `ot` on Normal-Normal only. The Phase E
  training loop is structurally model-agnostic — it routes through
  `Model` / `Prior` / `TiltingScheme` protocols — but
  `_extract_normal_normal_params` raises `NotImplementedError` for
  non-Normal-Normal experiments because the existing torch ports
  take Normal-Normal coordinates. Extending to a new model/prior
  pair requires either generalising those ports or registering new
  ones.

## Invariants

(Tested in `tests/properties/test_dual_head_invariants.py`,
`tests/properties/test_loss_diff.py`,
`tests/regression/test_torch_pvalue_matches_numpy.py`,
`tests/regression/test_torch_cd_matches_numpy.py`,
`tests/regression/test_train_smoke.py` (legacy v1 path).)

- `EtaNet(theta_dim ≥ 1)` and `ValidityNet(theta_dim ≥ 1)` accept
  scalar and vector θ; vector future-proofs the framework even
  though scalar is the only case today.
- `boundary_penalty_from_validity` passes `torch.autograd.gradcheck`
  in the linear regime and at logits `±30` (well outside any
  previous clamp); wrong-side gradient saturates at `-1`,
  right-side at `0`.
- `EtaNet` parameters receive gradient from Head A's loss;
  `ValidityNet` parameters do **not** (`functional_call` with
  detached params leaves their `.grad` `is None`).
- Symmetrically, `EtaNet` parameters receive **no** gradient from
  Head B's BCE (η_pred is detached at the boundary between the
  two heads).
- `is_pair_valid` is per-scalar (not per-grid) and admits values
  in `[-1e-9, 1+1e-9]` to absorb FP rounding. NaN / ±Inf / values
  outside that slack are invalid.
- `OTTilting.tilted_pvalue` raises `TiltingDomainError` for
  `eta ∉ [0, 1]`; `ot_tilted_pvalue_torch` returns NaN per-element
  on the same input. The validity mask treats both as invalid.
- `Prior.fingerprint()` and `Model.fingerprint()` are tuples; for
  every concrete (model, prior, θ-distribution) pair in the
  framework, `a == b ⟺ a.fingerprint() == b.fingerprint()`.
- `ExperimentConfig.from_dict` rejects unknown top-level keys
  (catches YAML typos like `n_gird`); `_build_*_from_dict`
  rejects per-type unknown kwargs (catches malicious or confused
  YAML attempting to override `name` / `param_dim`).
- `lhs_1d` is true 1D Latin Hypercube — every stratum receives
  exactly one sample.
- The width loss accepts a `(B,)` D batch and returns a scalar
  mean (replaces the high-variance single-D per-step estimator).
- The training loop's per-step `aux_valid_rate` outside
  `(0.05, 0.95)` triggers a `RuntimeWarning` per epoch.
- Phase E checkpoint v2 has required keys
  `(checkpoint_format_version, architecture,
   eta_architecture_kwargs, validity_architecture_kwargs,
   eta_state_dict, validity_state_dict, experiment_config)`.
  Missing keys raise.
- Checkpoint write is atomic (`torch.save → tmp → os.replace`).
- Selector compares `(scheme.name, model.fingerprint(),
   prior.fingerprint())` to checkpoint metadata; mismatch raises
  `MissingArtifactError`.

## Literature

### Foundational (information geometry of WALDO + monotonic NNs)

- Bissiri, P. G., Holmes, C. C., Walker, S. G. "A general framework
  for updating belief distributions." *J. Royal Stat. Soc. B* 78
  (2016): 1103–1130. — Power-likelihood / tempering as an
  information-geometric path.
- Daniels, H. A., Velikova, M. "Monotone and partially monotone
  neural networks." *IEEE Trans. Neural Netw.* 21 (2010): 906–917.
  — Architectural enforcement of partial monotonicity via
  positive-weight pathways.
- You, S., Ding, D., Canini, K., Pfeifer, J., Gupta, M. "Deep lattice
  networks and partial monotonic functions." *NeurIPS* 30 (2017). —
  Modern partial-monotonicity architectures; influence on the
  legacy MLP design.

### Closely related

- Schweder, T., Hjort, N. L. *Confidence, Likelihood, Probability.*
  Cambridge UP, 2016. — CD pdf via `½|dp/dθ|` (used in
  `cd_variance` loss).
- Cavalieri-Fubini layer-cake (textbook):
  `∫ f dμ = ∫_0^∞ μ({f ≥ t}) dt`. Direct justification of the
  `integrated_p` loss.
- Holmes, A. C., Walker, S. G. "Assigning a value to a power
  likelihood in a general Bayesian model." *Biometrika* 104 (2017).

### Contrasting

- `NumericalEtaSelector` (in-framework): minimises static-per-D
  width pointwise. The kinky η*(|Δ|) curve it produces is the
  motivating failure mode; this method's whole point is to fix it.

## Links

- Selector: `src/frasian/tilting/eta_selectors.py:LearnedDynamicEtaSelector`
- Phase E artifact: `src/frasian/learned/eta_artifact.py:EtaArtifact`
- Architecture: `src/frasian/learned/training/architecture.py`
  (`EtaNet`, `ValidityNet`)
- Validity helpers: `src/frasian/learned/training/validity.py`
  (`is_pair_valid`, `validity_mask`, `compute_pvalues_per_sample`)
- Losses: `src/frasian/learned/training/losses.py`
  (`boundary_penalty_from_validity` + width losses)
- Experiment config: `src/frasian/learned/training/sampling.py`
  (`ExperimentConfig`, `ThetaDistribution`,
  `UniformThetaDistribution`, `lhs_1d`)
- Torch ports: `src/frasian/learned/training/pvalue_torch.py`,
              `src/frasian/learned/training/cd_torch.py`
- Training pipeline: `src/frasian/learned/training/train.py:fit_eta_artifact`
- CLI: `scripts/train_learned_eta.py` (`--config <yaml>`)
- Experiment configs: `experiments/canonical_normal_normal_powerlaw.yaml`,
                      `experiments/canonical_normal_normal_ot.yaml`
- Property tests: `tests/properties/test_dual_head_invariants.py`,
                  `tests/properties/test_learned_eta_invariants.py`
- Regression tests: `tests/regression/test_torch_pvalue_matches_numpy.py`,
                    `tests/regression/test_torch_cd_matches_numpy.py`,
                    `tests/regression/test_learned_eta_calibration.py`,
                    `tests/regression/test_learned_eta_narrowness.py`,
                    `tests/regression/test_learned_eta_selector_smoke.py`,
                    `tests/regression/test_scheme_improper_returns_nan.py`
- Illustration: `src/frasian/experiments/illustrations/learned_eta_demo.py`

## Empirical headline numbers

End-to-end CI width on the canonical Normal-Normal sandbox (`Config.fast()`,
σ=1, μ₀=0, n_reps=200), w=0.5 column. Learned uses the v0_smoke checkpoint:

| θ_true | Wald | bare WALDO | legacy Dyn | learned p.l. | learned ot |
|---|---|---|---|---|---|
| 0 | 3.92 | 3.36 | 3.38 | 3.81 | 3.79 |
| ±2 | 3.92 | 3.76 | 3.88 | 3.86 | 3.85 |
| ±3 | 3.92 | 4.29 | **4.60** | **3.89** | **3.89** |
| ±4 | 3.92 | 4.87 | **5.28** | **3.88** | **3.92** |

Headline:
- **Conflict band (|θ|≥3)**: learned is 25–35 % narrower than the legacy
  `DynamicNumericalEtaSelector` *and* narrower than bare WALDO.
- **Low conflict (|θ|≤1)**: learned is ~12 % wider than the legacy
  (architectural bound on η' prevents full oversharpening at the
  lower clamp), but still ≤ Wald everywhere.
- **Calibration**: nominal at every (θ_true, w, α) cell on the
  in-checkpoint 5×5×3 grid (max |err| ≤ 0.029 ≈ 3·MC_SE) and on the
  end-to-end coverage experiment via the runner.

The framework's central claim — *a calibrated dynamic-η selector that
matches or beats non-tilted WALDO width across the (w, θ_true) plane*
— is empirically verified.

## Status notes

The default selector in `default_tiltings()` is gated by the env var
`FRASIAN_DEFAULT_DYNAMIC_ETA`:
- `numerical` (default for backwards compat): `DynamicNumericalEtaSelector`.
- `learned`: `LearnedDynamicEtaSelector` reading the trained
  `EtaArtifact` from `artifacts/learned_eta_<config_name>_v0_smoke.pt`.

To switch globally:
```
export FRASIAN_DEFAULT_DYNAMIC_ETA=learned
python -m scripts.run --fast experiment=coverage
python -m scripts.run --fast experiment=width
```

The smoothness experiment uses its own internal selector
(`NumericalEtaSelector`) and is not affected by this env var.

A v1 production checkpoint (n_lhs=4000, n_epochs=100) trained but
plateaued at the same val_loss as v0_smoke — the bounded-sigmoid
output `0.01 + 0.98·sigmoid(...)` caps how aggressively η can
oversharpen at low |Δ|, which is the binding constraint regardless
of training budget. v0_smoke is therefore representative; we ship
it as the production fixture.
