# learned_eta

> Status: `implemented` (post-Phase-F JAX/Equinox/Optax port; see CLAUDE.md
> Phase 10 / Migration Status).

> ⚠️ **Phase G v4 update (2026-05-09 → 2026-05-12)**: this brief was
> originally written for **Phase E** (per-experiment, checkpoint
> format v3, EtaNet input = θ only). The current production
> implementation is **Phase G v4** with a **conditional** architecture
> trained over a *range* of hyperparameters:
>
> - `EtaNet : (θ, prior_hp, lik_hp) → η` — three inputs, not just θ.
> - `ValidityNet : (θ, prior_hp, lik_hp, η) → logit`.
> - Checkpoint format version is **v4** (`CHECKPOINT_FORMAT_VERSION = 4`
>   in `src/frasian/learned/training/_checkpoint.py`). v3 fixtures
>   were deleted in CLAUDE.md migration row 11.
> - Training uses `Sigma0AnchoredUniformThetaDistribution` (σ₀-anchored
>   θ sampling — the prior's σ₀, NOT the likelihood's σ) and
>   per-channel input z-score normalization, both default ON. Without
>   them, training collapses to η ≈ 1 (= Wald) per CLAUDE.md row 12 /
>   `docs/notes/2026-05-09-phase-g-v4-fix.md`.
> - Runtime clamps η to admissible range when EtaNet extrapolates
>   out-of-bounds (with a warning), instead of refusing (per CLAUDE.md
>   row 12).
> - Empirical headline numbers in this brief cite v0_smoke fixtures
>   and are no longer canonical. Current numbers regenerated via
>   `PYTHONHASHSEED=0 python -m scripts.regen_headline`. Post-FR-merge
>   audit headline:
>   [`docs/notes/2026-05-12-cross-scheme-wald-audit.md`](../notes/2026-05-12-cross-scheme-wald-audit.md).
>
> ⚠️ **η convention (Easily-Conflated)**: on `power_law`,
> **η = 0 is bare WALDO** (full prior in) and **η = 1 is Wald**
> (prior cancels). The naming feels backwards; verify which η you
> mean. See CLAUDE.md "Easily-Conflated Distinctions" §1 and
> `docs/notes/2026-05-10-eta-conventions-and-loss-derivation.md`.
>
> ⚠️ **FR cd_variance hyperparam regime**: trained with default
> hyperparams diverges; requires `--lr-a 1e-4 --grad-clip-max-norm
> 0.5`. See `docs/notes/2026-05-11-fisher-rao-cd-var-hyperparams.md`.
> Open follow-up: rewrite this brief end-to-end for Phase G v4 (TODO).

## Summary

`LearnedDynamicEtaSelector` is a calibrated dynamic-η-per-θ
`EtaSelector` that minimises the **dynamic-procedure loss directly**
(integrated CI width, by default), rather than the static-per-D
width that `NumericalEtaSelector` minimises pointwise. The `η(θ)`
function is parameterised by a small dual-head neural network
(`EtaNet` + `ValidityNet`), trained end-to-end via JAX autograd
through the differentiable tilted-WALDO p-value formula and
trapezoidal integration, with Optax driving the parameter updates and
Equinox handling the model serialisation (`.eqx` checkpoint format).

The headline empirical claim: dynamically-tilted CIs from WALDO that
**maintain nominal coverage AND match-or-beat un-tilted WALDO width
across the (w, θ_true) plane** — calibrated *and* narrow,
simultaneously. See [Empirical headline numbers](#empirical-headline-numbers).

## Motivation

The framework's existing `DynamicNumericalEtaSelector` is calibrated
by construction (η depends only on θ, never on D) but uses
`NumericalEtaSelector` internally. That selector minimises the
*static-per-D* CI width pointwise, which slams η to the lower clamp
`-w/(1-w)` at small |Δ|. When that kinky η*(|Δ|) curve is used per-θ,
the resulting "Frankenstein" p-value spuriously accepts θ values 2–3σ
from D — verified empirically: at (D=4, w=0.5) the dynamic CI is
~5.24, much wider than Wald's 3.92.

The right objective is the **dynamic-procedure loss itself**:

```
ℒ(η_φ) = E_{(w, θ_true) ~ π} E_{D | θ_true, w} [ Φ(D; η_φ) ]
```

with `Φ(D; η_φ) = ∫ p_dyn(θ; D, η_φ) dθ` (default; integrated CI
width across all α-levels by Fubini) or the CD variance. Calibration
is automatic for any η that depends only on θ; the optimisation over
the η-function is unconstrained within that family.

## Definition

The learner produces an `EtaArtifact` (Phase E, checkpoint format
v3 — the post-Phase-F Equinox `.eqx` format; v2 was the pre-port
torch `.pt` format and is no longer written, only loadable for
back-compat). It is a thin checkpoint wrapper — not a structural
`LearnedArtifact` Protocol implementation; the dual-head design
exposes `predict_eta(theta)` and `predict_validity(theta, eta)`
rather than the single `predict(x)` that `LearnedArtifact`
specifies. At inference, `LearnedDynamicEtaSelector` plugs into the
same `tilting.confidence_regions` pipeline via `select_grid` as
`DynamicNumericalEtaSelector` does, but reads η from the trained
network instead of a width-minimising solver.

The Phase E selector is **per-experiment**: each (model, prior,
scheme, statistic) configuration trains its own checkpoint, recorded
via fingerprints in the checkpoint metadata. The selector compares
fingerprints at inference and refuses cross-experiment use.

**Architecture: dual-head Equinox modules.** Two MLPs trained
jointly:

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
the YAML or via the CLI. Both modules are `equinox.Module` subclasses
with explicit `architecture_kwargs()` for round-trip checkpoint
metadata.

**Three-step training.** Per minibatch of θ (drawn from a one-shot
Latin Hypercube sample over `theta_distribution.support()`):

1. Forward Head A on the main batch; sample auxiliary
   `(θ_aux, η_aux)` boundary-probing pairs (`η_aux ~ Uniform(eta_-`
   `explore_box)`); sample `D ~ likelihood(·|θ)` per row; label
   per-sample validity via `compute_pvalues_per_sample` +
   `validity_mask`. `η_pred` is wrapped in `jax.lax.stop_gradient`
   before labelling so Head A's gradient does not flow through the
   discrete labels.
2. Train Head B: BCE-with-logits loss on `(logits, valid_labels)`
   (Optax `optax.sigmoid_binary_cross_entropy`).
3. Train Head A:
   `width_loss(...)  +  λ(epoch) · boundary_penalty`
   where `width_loss` is `integrated_pvalue_loss` over an n_mc=8 D
   batch (per-step Monte Carlo average; closes the high-variance
   single-D regime that made val_width oscillate), and
   `boundary_penalty = -log P(valid | θ_batch, η_pred)` evaluated
   under Head B with its parameters held constant via
   `equinox.filter` partition + `jax.lax.stop_gradient`. Gradient
   flows through the `(θ, η)` input but not into ValidityNet's weights.

**No clamp on the boundary penalty.** `jax.nn.log_sigmoid(x)` is
numerically stable for any finite `x` and its derivative
`-sigmoid(-x)` saturates to `-1` (not zero) as `x → -∞`, so the
wrong-side gradient stays alive at bounded magnitude.

**λ schedule.** Linear warmup from 0 to `λ_max=10` over the first
`warmup_frac=0.3` of training, then constant. Lets Head B accumulate
labels before its signal drives Head A.

**Validation.** Held-out width loss on a frozen `(θ_val, D_val)` set
sampled once at training start; deterministic across epochs so early
stopping picks the actual best-policy epoch (not the noisiest D
sample).

**Runtime safety clamp.** A poorly-trained checkpoint can drift past
the scheme's admissible η range at extreme conflict. The Phase E
selector clamps η to the Normal-Normal closed-form admissible range
for `power_law` and the explicit `eta_explore_box` for non-NN models,
emitting a `RuntimeWarning` per call when a sample is clamped.
Cumulative `_clamped_calls` and `_last_clamped_fraction` counters on
the selector instance track how often the clamp fires.

**Loss options.**

1. `integrated_p` (default): `∫ p_dyn(θ; D, η_φ) dθ`. By Fubini =
   `∫_α |C_α(D)| dα` with uniform α weighting. α-marginalised.
   Recorded in the checkpoint as `alpha_mode = "marginalised"`.
2. `cd_variance`: `Var_{F_D}[θ]` where `F_D` is the Schweder–Hjort
   CD built from `½ |∂p_dyn/∂θ|`, Z-normalised. α-marginalised.
   Caveat: variance is around the CD's own mean, not θ_true; in
   conflict regimes can incentivise sharpness around a biased
   estimate (see Failure modes).
3. `static_width(α)`: sigmoid-relaxed `∫ σ_β(p_dyn − α) dθ` at fixed
   α. Default sharpness `β = 200` (β=50 has +110% bias at α=0.05).
   α-conditioned: trained MLP valid only at the α it was trained for.
   Recorded in the checkpoint as `alpha_mode = "fixed"`.

`alpha_mode` is the explicit gate consulted by
`LearnedDynamicEtaSelector._check_alpha` at inference time (audit
P0-16); the previous "alpha is None ⇒ marginalised" overload is
preserved as a legacy fallback that emits a UserWarning.

## Derivation

**Calibration is automatic.** For any `η_φ : θ → ℝ` that depends on
θ only (and on the *model* and *prior*, both fixed across the
training distribution), not on `D`, the dynamic procedure achieves
nominal 1-α coverage by construction.

Sketch: under `H0: θ = θ₀`, the trained `η_φ(θ₀)` is a *constant* with
respect to `D` (it only depends on θ₀, which is fixed under `H0`). The
tilted-WALDO p-value at fixed η is `U[0, 1]` under
`D ~ N(θ₀, σ²)` (the standard WALDO calibration property, inherited
via the algebraic structure `Φ(b−a) + Φ(−a−b)`). Therefore
`p_dyn(θ₀; D, η_φ)` is `U[0, 1]`, and the inverted CI hits nominal
coverage. **Verified empirically** in the post-training calibration
report: 5×3 (θ_true × α∈{0.05, 0.10, 0.20}) grid, `n_reps≥300`,
mean coverage matches nominal within 3·MC_SE
(`tests/regression/test_learned_eta_calibration.py`).

**Layer-cake identity for `integrated_p`.** For `p ∈ [0, 1]`,

```
∫_θ p(θ) dθ = ∫_0^1 |{θ : p(θ) ≥ α}| dα = ∫_0^1 |C_α| dα.
```

So minimising `∫ p_dyn dθ` is equivalent to minimising the CI width
integrated uniformly over all α — "all CIs as small as possible".
α-sampling at training time is unnecessary.

**Differentiability.** All ingredients are end-to-end differentiable
through JAX autograd:
- Tilted p-value: `Φ(b−a) + Φ(−a−b)` with `Φ(x) = ½(1+erf(x/√2))`
  (`pvalue_jax.power_law_tilted_pvalue_jax` for power_law,
  `ot_tilted_pvalue_jax` for OT).
- Per-sample θ-grid: `D ± search_mult·σ`, `jnp.linspace`-built.
- Trapezoidal integration: `jnp.trapezoid`.
- η inverse and architecture: smooth (GELU activation throughout).

No α-crossings, no `argmax`, no `brentq` inside the loss. Verified by
finite-difference gradient agreement in
`tests/properties/test_loss_diff.py`.

**Surrogate-vs-Theorem-8 bias.** The training uses the *generic
grid* surrogate `generic_grid_tilted_pvalue` for non-Normal-Normal
schemes; on Normal-Normal it routes to the closed-form
`power_law_tilted_pvalue_jax` which agrees with Theorem 8 byte-equal.
The grid surrogate has a deliberate documented bias (≤ 0.20 on the
p-value at b=0; argmin-η drift ≤ 1.5 at |Δ|≥1.5) pinned by
`tests/regression/test_grid_surrogate_vs_theorem8.py`.

## Predicted behavior

- **η(θ) curve is smooth** by architectural construction (GELU MLP,
  no kinks). Approximate symmetry about `μ₀` is empirical, not
  enforced — for symmetric Normal-Normal training distributions it
  emerges from the training objective.
- **Calibration at nominal level** at every (θ_true, α) cell. The
  per-checkpoint `final_eta_pred_valid_rate` summarises in-sample
  validity; the empirical L3 calibration regression
  (`tests/regression/test_learned_eta_calibration.py`) MC-checks
  coverage at the trained `w` across θ ∈ [-4, 4] and
  α ∈ {0.05, 0.10, 0.20}.
- **Width strictly improves on `NumericalEtaSelector`** in the
  conflict band; see [headline numbers](#empirical-headline-numbers).
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

  **`select` API contract.** `LearnedDynamicEtaSelector.select` and
  `select_grid` require explicit `model_fingerprint=` and
  `prior_fingerprint=` kwargs at every call site; bare
  `select(scheme, statistic=...)` raises `ValueError` rather than
  falling back to the w-only derived check (which cannot distinguish
  two `(σ, σ₀)` pairs giving the same `w`). All in-framework callers
  pass `model.fingerprint()` / `prior.fingerprint()` from the
  inference call site; third-party callers must do the same.

- **Validity vs distribution properness — known gap.** The per-sample
  validity criterion (`tilted_pvalue` returns a finite scalar in
  `[0, 1]`) checks the p-value output, not the underlying tilted
  distribution being a proper CDF. There is a deliberate numpy/JAX
  split in the registered schemes:
  - **Numpy** `tilted_pvalue` (used by the validity helper to label
    Head B's BCE) **raises** `TiltingDomainError` for invalid η —
    `power_law` when `1 − η(1−w) ≤ 0`, `ot` when
    `eta ≤ -w/(1-w)` (post-audit P0-4: OT no longer hard-clamps to
    `[0, 1]`; the closed-form pvalue is admissible whenever
    `s_t = (w + η(1-w))·σ > 0`).
    `compute_pvalues_per_sample` catches and writes NaN; the
    validity mask drops the slot.
  - **JAX** ports (used by Head A's width loss for autograd) instead
    `clamp(denom or s_t, min=1e-6)` so the loss surface stays smooth
    and gradient-bearing past the boundary. Without this, an EtaNet
    initialised outside the admissible region produces 100 % NaN
    loss and never recovers (verified for OT during training). The
    validity helper independently labels Head B correctly because it
    uses the numpy path.

  A future scheme whose closed-form numpy `tilted_pvalue` returns
  a finite-in-[0,1] value despite an improper tilted distribution
  could pass validity while encoding a bogus η range. Mitigation:
  every newly-registered scheme must ship a per-scheme audit
  test (`tests/regression/test_scheme_improper_returns_nan.py`)
  asserting `compute_pvalues_per_sample` returns NaN across the
  declared admissible-range complement; the end-to-end calibration
  regression on a trained checkpoint catches any miss via empirical
  coverage drift.

- **Undertrained checkpoint at conflict.** The boundary penalty
  pushes η_pred into Head B's predicted-valid region during
  training, but a smoke run (few epochs, small LHS) can leave η
  drifting past the scheme's admissible boundary at extreme
  conflict. The runtime selector clamps with a `RuntimeWarning`; if
  you see this fire on a production checkpoint, train longer or
  widen `eta_explore_box`. The OT v0_smoke is documented as
  undertrained (Head B accuracy ~0.67 vs power_law's ~0.97); v1
  retraining is recommended for production-grade OT.

- **Class-degenerate Head B BCE.** If `eta_explore_box` doesn't
  probe the invalid region (e.g. `(0, 0.5)` for `power_law` at
  `w=0.5`, where everything is valid), Head B's BCE collapses to
  predicting `True` always and the boundary signal goes to zero.
  The training loop emits a `RuntimeWarning` per epoch when the
  per-batch aux validity rate is outside `(0.05, 0.95)`.

- **Degenerate `w`.** Priors so tight or so wide that
  `w := σ₀²/(σ²+σ₀²)` is outside `(0.001, 0.999)` are rejected at
  training time (`extract_normal_normal_params` `_W_EPS = 1e-3` in
  `_losses_compose.py:97`) and at inference time
  (`LearnedDynamicEtaSelector._check_experiment` `_W_EPS = 1e-3` in
  `eta_selectors.py:839`). Delta priors and improper priors both put
  the WALDO admissible range into a degenerate regime: as `w → 0`
  (delta prior) the WALDO denominator `s_t = (w + η(1−w))·σ` becomes
  η-collinear with `(1−w)·σ`, and as `w → 1` (improper prior) the
  prior contribution to the tilted posterior vanishes and the
  problem degenerates to bare Wald. The JAX p-value port
  (`pvalue_jax.py`) clamps the closed-form denominator at `1e-6` to
  avoid `inf` mid-trace; that clamp distorts the gradient near the
  degenerate boundary and would produce a learned-η surface with
  silently biased coverage. The `_W_EPS` guard enforces a clean
  margin away from the clamp on both sides (audit P1 J.2).

- **CD-variance bias.** `Var_{F_D}[θ]` is the CD's spread around its
  own mean, not around `θ_true`. In conflict regimes where the CD
  median drifts toward μ₀, this can incentivise sharpness around a
  biased estimate. Use `cd_variance` only when centring is known to
  be benign.

- **Static-width sharpness.** Default `β = 200` keeps the relaxation
  bias < 1% at α ∈ [0.05, 0.5]; for very small α (≤ 0.01) raise to
  `β ≥ 500` or expect bias.

- **Non-Gaussian likelihoods.** Phase 4 generalised the JAX
  tilted-pvalue registry from `dict[str, ...]` to `dict[tuple[
  scheme_name, model_kind], ...]`. Closed-form NN paths register
  under `("power_law", "normal_normal")` / `("ot", "normal_normal")`
  and stay byte-equal with the pre-Phase-4 surface. The new generic
  grid kernel (`pvalue_jax.generic_grid_tilted_pvalue`) registers
  under `("power_law", "generic")` and serves any `(Model, Prior)`
  with finite `model.support()` and `prior.logpdf` defined on the
  grid. `ExperimentConfig` carries `n_data: int = 1` so non-NN
  experiments (e.g. Bernoulli + Beta(2, 2)) can sample the
  multi-observation likelihood per training θ. Trained non-NN
  checkpoints round-trip through the same selector. The Bernoulli
  smoke fixture
  `artifacts/learned_eta_canonical_bernoulli_powerlaw_v0_smoke.eqx`
  is committed; its YAML is `experiments/canonical_bernoulli_powerlaw.yaml`.

## Invariants

(Tested in `tests/properties/test_dual_head_invariants.py`,
`tests/properties/test_loss_diff.py`,
`tests/properties/test_learned_eta_invariants.py`,
`tests/properties/test_eta_net_jax_invariants.py`,
`tests/regression/test_jax_pvalue_matches_numpy.py`,
`tests/regression/test_jax_cd_matches_numpy.py`,
`tests/regression/test_jax_determinism.py`,
`tests/regression/test_learned_eta_calibration.py`,
`tests/regression/test_learned_eta_narrowness.py`,
`tests/regression/test_learned_eta_selector_smoke.py`,
`tests/regression/test_alpha_mode_gating.py`,
`tests/regression/test_scheme_improper_returns_nan.py`,
`tests/regression/test_checkpoint_metadata_compat.py`.)

- `EtaNet(theta_dim ≥ 1)` and `ValidityNet(theta_dim ≥ 1)` accept
  scalar and vector θ; vector future-proofs the framework even
  though scalar is the only case today.
- `boundary_penalty_from_validity` is finite-difference-gradient-
  agreeing in the linear regime and at logits `±30`; wrong-side
  gradient saturates at `-1`, right-side at `0`.
- `EtaNet` parameters receive gradient from Head A's loss;
  `ValidityNet` parameters do **not** (Equinox `filter` partition +
  `jax.lax.stop_gradient` on Head B's params during Head A's
  gradient pass leaves Head B unchanged).
- Symmetrically, `EtaNet` parameters receive **no** gradient from
  Head B's BCE (η_pred is `stop_gradient`-detached at the boundary
  between the two heads).
- `is_pair_valid` is per-scalar (not per-grid) and admits values
  in `[-1e-9, 1+1e-9]` to absorb FP rounding. NaN / ±Inf / values
  outside that slack are invalid.
- `OTTilting.tilted_pvalue` (numpy) raises `TiltingDomainError`
  for `eta ≤ -w/(1-w)`; `ot_tilted_pvalue_jax` clamps `s_t` to
  keep the loss surface differentiable past the boundary instead
  (the validity helper uses the numpy path so labels are correct
  regardless). Same dual mechanism for `power_law`. Audited per-
  scheme in `tests/regression/test_scheme_improper_returns_nan.py`.
- `Prior.fingerprint()` and `Model.fingerprint()` are tuples; for
  every concrete (model, prior, θ-distribution) pair in the
  framework, `a == b ⟺ a.fingerprint() == b.fingerprint()`.
- `ExperimentConfig.from_dict` rejects unknown top-level keys
  (catches YAML typos like `n_gird`); `_build_*_from_dict`
  rejects per-type unknown kwargs.
- `lhs_1d` is true 1D Latin Hypercube — every stratum receives
  exactly one sample.
- The width loss accepts a `(B,)` D batch and returns a scalar
  mean (replaces the high-variance single-D per-step estimator).
- The training loop's per-step `aux_valid_rate` outside
  `(0.05, 0.95)` triggers a `RuntimeWarning` per epoch.
- Phase E checkpoint v3 (Equinox `.eqx`) has required metadata keys
  including `(checkpoint_format_version=3, architecture,
  arch_sha, eta_architecture_kwargs, validity_architecture_kwargs,
  experiment_config, loss_kind, alpha, alpha_mode,
  equinox_version, jax_version)`. Missing required keys raise.
- `alpha_mode in {"marginalised", "fixed"}` is the inference-time
  gate consulted by `_check_alpha`; the previous `alpha is None`
  overload is preserved as a legacy fallback with a `UserWarning`.
- Checkpoint write is atomic (Equinox `tree_serialise_leaves` →
  tmp file → `os.replace`).
- Selector compares `(scheme.name, model.fingerprint(),
  prior.fingerprint())` to checkpoint metadata; mismatch raises
  `MissingArtifactError`. v2 (legacy torch) checkpoints are
  accepted by the metadata loader but produce a clear error if the
  Equinox loader is asked to reconstruct them — they are
  read-metadata-only.

## Literature

### Foundational (information geometry of WALDO + neural surrogate losses)

- Bissiri, P. G., Holmes, C. C., Walker, S. G. "A general framework
  for updating belief distributions." *J. Royal Stat. Soc. B* 78
  (2016): 1103–1130. — Power-likelihood / tempering as an
  information-geometric path.
- Daniels, H. A., Velikova, M. "Monotone and partially monotone
  neural networks." *IEEE Trans. Neural Netw.* 21 (2010): 906–917.
  — Architectural enforcement of partial monotonicity (rejected
  here in favour of validity-driven smoothness; cited for contrast).
- You, S., Ding, D., Canini, K., Pfeifer, J., Gupta, M. "Deep lattice
  networks and partial monotonic functions." *NeurIPS* 30 (2017). —
  Modern partial-monotonicity architectures; influence on the
  legacy Phase D MLP design that the dual-head approach replaces.

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
  Bears `is_post_selection = True` flag (audit P0-7) to surface
  the calibration-gap explicitly.

## Links

- Selector: `src/frasian/tilting/eta_selectors.py:LearnedDynamicEtaSelector`
- Phase E artifact: `src/frasian/learned/eta_artifact.py:EtaArtifact`
- Architecture: `src/frasian/learned/training/architecture.py`
  (`EtaNet`, `ValidityNet` — Equinox modules)
- Validity helpers: `src/frasian/learned/training/validity.py`
  (`is_pair_valid`, `validity_mask`, `compute_pvalues_per_sample`,
  `_admissibility_mask`)
- Losses: `src/frasian/learned/training/losses.py`
  (`boundary_penalty_from_validity` + width losses);
  `_losses_compose.py` (loss-kind dispatch).
- Experiment config: `src/frasian/learned/training/sampling.py`
  (`ExperimentConfig`, `ThetaDistribution`,
  `UniformThetaDistribution`, `lhs_1d`)
- JAX kernels: `src/frasian/learned/training/pvalue_jax.py`
  (`power_law_tilted_pvalue_jax`, `ot_tilted_pvalue_jax`,
  `generic_grid_tilted_pvalue`),
  `src/frasian/learned/training/cd_jax.py` (CD density via
  `½|∂p/∂θ|`).
- Training pipeline: `src/frasian/learned/training/train.py:fit_eta_artifact`
  (orchestrator); `_train_loop.py` (Optax step loop);
  `_setup.py` (`resolve_device`, `validate_loss_kind`,
  `enable_determinism`); `_checkpoint.py` (Equinox `.eqx` write).
- CLI: `scripts/train_learned_eta.py` (`--config <yaml>`;
  `--force` required to overwrite, audit P0-15).
- Reproduction: `scripts/regen_headline.py`
  (requires `PYTHONHASHSEED=0`, audit P0-12).
- Experiment configs: `experiments/canonical_normal_normal_powerlaw.yaml`,
  `experiments/canonical_normal_normal_ot.yaml`,
  `experiments/canonical_bernoulli_powerlaw.yaml`.
- Property tests: `tests/properties/test_dual_head_invariants.py`,
  `tests/properties/test_learned_eta_invariants.py`,
  `tests/properties/test_eta_net_jax_invariants.py`,
  `tests/properties/test_loss_diff.py`.
- Regression tests: `tests/regression/test_jax_pvalue_matches_numpy.py`,
  `tests/regression/test_jax_cd_matches_numpy.py`,
  `tests/regression/test_jax_determinism.py`,
  `tests/regression/test_learned_eta_calibration.py`,
  `tests/regression/test_learned_eta_narrowness.py`,
  `tests/regression/test_learned_eta_selector_smoke.py`,
  `tests/regression/test_alpha_mode_gating.py`,
  `tests/regression/test_scheme_improper_returns_nan.py`,
  `tests/regression/test_checkpoint_metadata_compat.py`.
- Illustration: `src/frasian/experiments/illustrations/learned_eta_demo.py`
- Committed v0_smoke artifacts:
  `artifacts/learned_eta_canonical_normal_normal_powerlaw_v0_smoke.eqx`,
  `artifacts/learned_eta_canonical_normal_normal_ot_v0_smoke.eqx`,
  `artifacts/learned_eta_canonical_bernoulli_powerlaw_v0_smoke.eqx`.

## Empirical headline numbers

End-to-end CI width on the canonical Normal-Normal sandbox (σ=1,
μ₀=0, σ₀=1 → w=0.5, n_reps=200, α=0.05). Phase E v0_smoke
checkpoint, post-Equinox-port:

| θ_true | Wald | bare WALDO | numerical Dyn | power_law[learned] |
|---|---|---|---|---|
| 0  | 3.92 | 3.33 | 3.36     | 3.63 |
| 1  | 3.92 | 3.43 | 3.49     | 3.64 |
| 2  | 3.92 | 3.75 | 3.91     | 3.68 |
| 3  | 3.92 | 4.23 | **4.54** | **3.75** |
| 4  | 3.92 | 4.78 | **5.24** | **3.82** |

Single-seed v0_smoke checkpoint; standard error ≈ 0.05 across
α=0.05 narrowness MC repeats. v1 production retraining will
produce variability within ~1× this SE. To regenerate, run

```
PYTHONHASHSEED=0 python -m scripts.regen_headline
```

(Requires JAX + Equinox; audit P0-12 made the `PYTHONHASHSEED`
requirement explicit — without it the narrowness test seed derivation
drifts across Python processes.)

These numbers were trained with `antithetic=False` (the pre-Phase-4
default). The current default is `antithetic=True` (only effective
for `loss_kind='static_width'`); re-trained checkpoints will produce
different EtaNet weights — expected within MC noise of these values,
but unverified.

These numbers are **not bit-equal** to the pre-Phase-F torch numbers.
JAX's `random.PRNGKey` differs from torch's RNG even at the same
nominal seed, so retrained weights drift within ~1·MC SE of the
torch-era numbers. The qualitative pattern (calibrated AND ≤ Wald
across θ, narrow at conflict) is preserved.

The headline table is for `power_law` only. `ot[learned]` is wired
into `default_tiltings()` and runs through the coverage/width
experiments, but the OT smoke checkpoint is undertrained relative
to power_law (Head B accuracy ~0.67 on the v0_smoke); train a v1
checkpoint via `scripts/train_learned_eta` for production-grade OT.

Headline:
- **Conflict band (|θ|≥3)**: learned is ~17–28 % narrower than the
  `numerical` Dynamic selector *and* narrower than bare WALDO.
- **Low conflict (|θ|≤1)**: learned is ~8 % wider than the
  `numerical` Dynamic selector. (Both are calibrated by
  construction — η depends only on θ; the static `NumericalEta-`
  `Selector` is the one that undercovers via post-selection
  inference. The learned cell trades width back at low conflict
  in exchange for the smooth η(θ) curve that prevents the legacy
  selector's kinky width inflation at high conflict.)
- **Calibration**: nominal at every θ_true within 3·MC_SE on the
  L3 calibration regression at the trained w (α=0.05).

Production v1 checkpoints (longer training, larger LHS) are
expected to tighten the low-conflict gap further. Audit P0-13
added `test_learned_strict_narrowness_at_v1` (nightly) which fires
only when a v1 checkpoint is on disk and asserts strict
narrowness with rel_tol=0 + n_reps=500.

## Status notes

The default selector in `default_tiltings()` is gated by the env var
`FRASIAN_DEFAULT_DYNAMIC_ETA`:
- `numerical` (default for backwards compat): `DynamicNumericalEtaSelector`.
- `learned`: `LearnedDynamicEtaSelector` reading the trained
  `EtaArtifact` from
  `artifacts/learned_eta_<config_name>_v0_smoke.eqx` (or `_v1.eqx`
  if available; resolution order in
  `_default_cells._make_learned_selector` prefers v1 → v0_smoke).

To switch globally:
```
export FRASIAN_DEFAULT_DYNAMIC_ETA=learned
python -m scripts.run --fast experiment=coverage
python -m scripts.run --fast experiment=width
```

The smoothness experiment uses its own internal selector
(`NumericalEtaSelector`) and is not affected by this env var.

v1 production checkpoints aren't committed (~each ~75 KB, easy to
re-train); the v0_smoke shipped here is sufficient for L2/L3/L4
gates. Run

```
python -m scripts.train_learned_eta \
    --config experiments/<config>.yaml \
    --out artifacts/learned_eta_<config_name>_v1.eqx \
    --version v1
```

for a longer training budget. Audit P0-15 added the refuse-on-exists
default; pass `--force` to overwrite an existing artifact.

Phase E architectural change vs Phase D: `EtaNet` is unbounded and
non-monotonic (raw GELU MLP on θ); validity is enforced by training
(boundary penalty + Head B), not architecture. The Phase D
`MonotonicEtaNet` (legacy `(w, |Δ'|)` input + bounded sigmoid +
positive-weight ReLU pathway) is removed; the framework is
per-experiment from here on.

Phase F architectural change vs the original Phase E: torch →
JAX/Equinox/Optax. The numerical kernels were ported byte-equal
where possible (closed-form Theorem 6/8 paths agree at atol 1e-10),
but trained-weight bit-equality was not preserved across the port
because JAX's PRNG primitive differs from torch's. Re-trained v0_smoke
checkpoints drift within ~1·MC SE of the torch-era headline numbers.
