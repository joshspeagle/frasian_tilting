# learned_eta

> Status: `implemented`

## Summary

`LearnedDynamicEtaSelector` is a calibrated dynamic-η-per-θ
`EtaSelector` that minimises the **dynamic-procedure loss directly**
(integrated CI width, by default), rather than the static-per-D
width that `NumericalEtaSelector` minimises pointwise. The η*(|Δ|; w)
function is parameterised by a small monotonic neural network
(`MonotonicEtaNet`), trained end-to-end via `torch.autograd` through
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

The learner produces a `MonotonicEtaArtifact(LearnedArtifact)`. At
inference, `LearnedDynamicEtaSelector` plugs into the existing
`tilting.confidence_regions` pipeline via `select_grid`, just like
`DynamicNumericalEtaSelector` does, but reads η from the trained
model instead of a width-minimising solver.

**Architecture (`MonotonicEtaNet`).** Partial-monotonicity dual-pathway
NN ported from `legacy/src/frasian/simulations/mlp_monotonic.py`,
with α dropped from inputs (the loss is α-marginalised by Fubini):

```
output = 0.01 + 0.98 · sigmoid( base(w) + softplus(scale(w)) · mono(|Δ'|) )
```

- `base(w), scale(w)`: unconstrained MLPs (GELU activations).
- `mono(|Δ'|)`: positive-weight + ReLU MLP — structurally monotone.
- `softplus` keeps the scale positive.
- `sigmoid` is monotone and bounded.
- Output `∈ [0.01, 0.99]`, transformed to η via per-scheme
  `eta_inverse` (`(η' - w)/(1 - w)` for `power_law`; identity for `ot`).

Inputs are standardised assuming uniform `[0, 1]`. `Δ' = Δ/(1+Δ)`
maps `[0, ∞)` to `[0, 1)`.

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

- **Stale artifact.** A checkpoint trained for one scheme (or one α
  in static_width mode) cannot be used for another. The selector
  verifies this at `load()` and raises `MissingArtifactError`.
- **Architecture-kwargs drift.** If the `architecture_kwargs` in the
  checkpoint don't match the trained weights, `load_state_dict` will
  fail with a cryptic torch error. The training script always writes
  the kwargs alongside the state; downstream loaders read them
  unconditionally. Don't hand-edit checkpoint metadata.
- **CD-variance bias.** `Var_{F_D}[θ]` is the CD's spread around its
  own mean, not around `θ_true`. In conflict regimes where the CD
  median drifts toward μ₀, this can incentivise sharpness around a
  biased estimate. The default `integrated_p` loss does not suffer
  from this. Use `cd_variance` only when the centring is known to be
  benign (e.g. on the Normal-Normal sandbox where Wald-side CD is
  centred at `D`).
- **Static-width sharpness.** Default `β = 200` keeps the relaxation
  bias < 1% at α ∈ [0.05, 0.5]; for very small α (≤ 0.01) raise to
  `β ≥ 500` or expect bias.
- **Non-Gaussian likelihoods.** The torch p-value registry covers
  `power_law` and `ot` only. Adding a new scheme requires registering
  its torch tilted-p-value in
  `src/frasian/learned/training/pvalue_torch.py`.
- **Numerical drift at the boundary.** The bounded sigmoid output
  keeps `η ∈ (η_min(w), 1)` strictly, so `denom = 1 - η(1-w)` stays
  positive in `power_law`. A `denom.clamp(min=1e-6)` in the torch
  port masks any rare drift; for inference precision, the numpy
  `tilted_pvalue` is used (no clamp).

## Invariants

(Tested in `tests/properties/test_eta_transforms.py`,
`tests/properties/test_loss_diff.py`,
`tests/regression/test_torch_pvalue_matches_numpy.py`,
`tests/regression/test_torch_cd_matches_numpy.py`,
`tests/regression/test_train_smoke.py`.)

- `delta_transform`/`inverse`, `eta_transform`/`inverse` round-trip
  to atol 1e-10 (incl. boundary `Δ → ∞`, `w → 1`).
- `MonotonicEtaNet` output `∈ [0.01, 0.99]`; `∂output/∂Δ' ≥ 0` by
  construction.
- Torch tilted p-value matches numpy to atol 1e-10 across (D, σ₀, η)
  for both `power_law` and `ot`, both `wald` and `waldo`.
- Torch CD density matches numpy `build_cd_from_pvalue.pdf_values` to
  atol 5e-4; integrates to 1.
- All three losses pass `torch.autograd.gradcheck`.
- All five `(scheme, loss)` combos train end-to-end at tiny budgets:
  `(power_law, integrated_p)`, `(power_law, cd_variance)`,
  `(power_law, static_width@α=0.05)`, `(ot, integrated_p)`,
  `(ot, cd_variance)`.
- Checkpoint format is self-describing: required keys
  `(checkpoint_format_version, architecture, architecture_kwargs,
   model_state_dict, scheme, loss, alpha_mode)`. Missing keys raise.
- Calibration report written into every checkpoint; `coverage` array
  shape is `(n_θ, n_w, n_α)`, values in `[0, 1]`.

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

- Implementation: `src/frasian/tilting/eta_selectors.py:LearnedDynamicEtaSelector`
- Artifact wrapper: `src/frasian/learned/monotonic_eta.py`
- Architecture: `src/frasian/learned/training/architecture.py`
- Losses: `src/frasian/learned/training/losses.py`
- Training pipeline: `src/frasian/learned/training/train.py`
- CLI: `scripts/train_learned_eta.py`
- Property tests: `tests/properties/test_eta_transforms.py`,
                  `tests/properties/test_loss_diff.py`
- Regression tests: `tests/regression/test_torch_pvalue_matches_numpy.py`,
                    `tests/regression/test_torch_cd_matches_numpy.py`,
                    `tests/regression/test_train_smoke.py`,
                    `tests/regression/test_learned_eta_calibration.py`,
                    `tests/regression/test_learned_eta_narrowness.py`
- Illustration: `src/frasian/experiments/illustrations/learned_eta_demo.py`
- Smoke checkpoint: `artifacts/learned_eta_power_law_v0_smoke.pt`
- Production checkpoint: `artifacts/learned_eta_power_law_v1.pt`

## Status notes

The default selector in `default_tiltings()` is gated by the env var
`FRASIAN_DEFAULT_DYNAMIC_ETA`:
- `numerical` (default until benchmarks pass): `DynamicNumericalEtaSelector`
- `learned`: `LearnedDynamicEtaSelector` with the v1 production checkpoint.

The switch is conservative because the headline narrowness regression
(`test_learned_eta_narrowness.py`) needs to pass empirically before
we change the default for everyone.
