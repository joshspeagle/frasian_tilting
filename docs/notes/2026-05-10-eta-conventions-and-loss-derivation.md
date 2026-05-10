# η conventions, integrated_p loss derivation, and eval-domain distinctions

**TL;DR.** Five easy-to-conflate distinctions caused a half-day of
confused diagnostic results during the learned-η investigation. They
are worth fixing in your head once and writing down so future readers
don't re-trip on them.

**Naming cleanup (2026-05-10).** The previously-confusing class name
`SigmaAnchoredUniformThetaDistribution` was renamed to
`Sigma0AnchoredUniformThetaDistribution` to make explicit that the
"σ" used for anchoring is the **prior's σ₀**, not the likelihood's σ.
The serialization identifier `sigma_anchored_uniform` is frozen for
backwards compat with v4 checkpoints, and the old class name is
retained as an alias. Any docs/comments still saying "σ-anchored"
in this codebase mean σ₀-anchored unless explicitly disambiguated as
"likelihood-σ-anchored".

---

## 1. η = 0 is **WALDO**; η = 1 is **Wald**

In `power_law_tilted_pvalue_jax` (the JAX port of `PowerLawTilting.tilted_pvalue`
under the WALDO statistic), the η parameter controls the tilt:

- **η = 0**: `mu_eta = w·D + (1−w)·μ₀` (posterior mean), `b = (1−w)(μ₀−θ)/(wσ)` ≠ 0.
  This is the **WALDO** p-value with full prior contribution.
- **η = 1**: `denom = w`, so `mu_eta = D` and `b = 0`. The formula collapses to
  `2Φ(−|D−θ|/σ)` — the **Wald** two-sided p-value, data only.

The naming is opposite to what the variable name suggests (η=1 means
"prior turned off"). The intuition: η is the *power* on `Posterior(θ|D) ×
Prior(θ)^η` … but the actual algebra of the WALDO p-value cancels the
prior at η=1 and keeps it at η=0. **Always verify which η value you mean
when you say "Wald".**

This bit me once already: an apples-to-apples eval that computed "Wald"
with η=0 was actually computing WALDO. Numbers looked great relative to
that baseline; the real Wald baseline (η=1) reversed the conclusion.

---

## 2. The `integrated_p` loss IS "average CI width across all α-levels"

Standard derivation. For a continuous test statistic, the (1−α) CI is
`{θ : p(θ) ≥ α}` and its width is

$$W(\alpha) = \int \mathbf{1}[p(\theta) \ge \alpha]\, d\theta.$$

Average over confidence levels α ∈ [0,1]:

$$\int_0^1 W(\alpha)\, d\alpha
= \int_0^1 \int \mathbf{1}[p(\theta) \ge \alpha]\, d\theta\, d\alpha
= \int \int_0^1 \mathbf{1}[p(\theta) \ge \alpha]\, d\alpha\, d\theta
= \int p(\theta)\, d\theta,$$

since `∫_0^1 1[p ≥ α] dα = p(θ)` for `0 ≤ p ≤ 1`. So

$$\boxed{\int p(\theta)\, d\theta \;=\; \int_0^1 W(\alpha)\, d\alpha
\;=\; \text{mean CI width across all α-levels}.}$$

`integrated_p` minimizes "all CIs at once," not just the 95% one. That
was the design intent.

---

## 3. The integration domain is σ₀-anchored — be careful about implications

The framework integrates `p(θ)` over `θ ∈ [μ₀ − 5σ₀, μ₀ + 5σ₀]`. So the
loss is the truncated integral

$$\int_{\mu_0 - 5\sigma_0}^{\mu_0 + 5\sigma_0} p(\theta)\, d\theta
= \int_0^1 |\,\text{CI}(\alpha) \cap \Omega\,|\, d\alpha$$

— the average width of CIs *intersected with the prior's natural
support*. Two consequences that bit us:

- **The "Wald loss" appears to depend on σ₀** even though Wald CI width
  is just 2zσ. At low σ₀ the integration domain is narrower than the
  Wald CI's high-confidence tails; the integral truncates them. At
  high σ₀ the domain is wide enough to capture the full Wald CI and
  the loss saturates at the analytical limit `4/√(2π) ≈ 1.596σ`.
  **The CI width itself does not change.**
- **Training-loss magnitude scales with σ₀** (since `|Ω| ≈ 10σ₀`). A
  batch mixing low- and high-σ₀ samples has gradient dominated by
  high-σ₀ samples. This biases what the network learns.

If you ever want a σ₀-independent reference frame, use a
likelihood-anchored domain `[μ₀ − Kσ, μ₀ + Kσ]`. With that change,
Wald loss is constant across σ₀ and the achievable headroom over Wald
reframes more honestly (3-5% per slice in the conjugate Normal sandbox,
not the 17-29% claimed against WALDO). See
`scripts/probe_function_constrained_min.py` for the implementation.

---

## 4. Per-θ_test argmin η ≠ per-θ_true argmin η

Crucial distinction for diagnostic work.

The loss is

$$L = \mathbb{E}_{\theta_{\text{true}}, D \mid \theta_{\text{true}}}\Big[\int p(\theta_{\text{test}}; D, \eta(\theta_{\text{test}}))\, d\theta_{\text{test}}\Big]
\;=\; \int G(\theta_{\text{test}}, \eta(\theta_{\text{test}}))\, d\theta_{\text{test}}$$

where `G(θ_test, η) = E_{θ_true, D | θ_true}[p(θ_test; D, η)]` is the
data-marginal of the p-value at one (θ_test, η) pair.

This decomposition shows: **the optimal η(·) is independent at each θ_test**.

So:

- **Per-θ_test argmin** η: at each θ_test, find `argmin_η G(θ_test, η)`.
  Sum gives the *function-constrained minimum* — the lowest loss
  achievable by any function η(θ_test). This is the right oracle for
  fn-min.
- **Per-θ_true argmin** η: for each θ_true realization, find the best
  constant η for *that* θ_true (averaging over D | θ_true). A
  different oracle that's neither tight nor easy to relate to fn-min.

I conflated these in `probe_function_constrained_min.py` initially —
the "per-test argmin" there is actually per-θ_TRUE. Use
`probe_per_theta_test_argmin.py` for the correct fn-min oracle.

---

## 5. The framework's hypothesis is conditional on the prior being informative

The intended η(θ_test) shape:

- **η small (toward 0)** when θ_test is close to μ₀ (in prior support).
  Tightens the CI by leveraging prior information — the framework's
  selling point.
- **η → 1 (Wald)** when θ_test is far from μ₀ (outside prior support).
  Falls back to data-only inference — graceful degradation.

This is a U-shape in η(θ_test). The selector should output it explicitly,
or clamp to η=1 outside the training distribution.

The current v4 fixtures are trained on σ₀-anchored θ_test, so they only
see θ_test inside the prior's support. They cannot learn the
"far-from-μ₀" Wald fallback because they never see those values during
training. **Evaluating v4 on a likelihood-anchored grid is an
extrapolation test**, not a test of framework correctness.

To get the U-shape, the network needs to either:
- be trained on a wider domain that covers "far from μ₀", or
- be explicitly clamped to η=1 outside its training-distribution support.

---

## What this changes about the project

- The 14% "calibrated headroom" claim from
  `2026-05-10-learned-eta-intervention-design.md` was relative to a
  σ₀-anchored loss with WALDO-as-baseline confusion. The honest
  picture: 3-5% achievable per-slice over Wald (η=1) at mid/high-w,
  ~0% at low-w under broad θ_true sampling.
- The "training is failing" framing was partially wrong. v4 fixtures
  capture roughly what's achievable *within their σ₀-anchored
  training domain*. They fail outside it because they were never
  trained there.
- The framework's design (η < 1 near μ₀, η = 1 far away) is
  conceptually correct; the implementation needs broader training
  domain or explicit out-of-distribution clamping.

## Files relevant to this note

- `src/frasian/learned/training/pvalue_jax.py:50-91` — η=0/η=1
  semantics for power_law tilting under WALDO statistic.
- `src/frasian/learned/training/losses.py` — `integrated_pvalue_loss`.
- `src/frasian/learned/training/sampling.py` —
  `Sigma0AnchoredUniformThetaDistribution` (canonical name; backwards-
  compat alias `SigmaAnchoredUniformThetaDistribution` retained).
- `scripts/probe_function_constrained_min.py` — fn-min and per-test
  argmin probe, with both σ₀-anchored (matching training) and
  likelihood-σ-anchored (broader stress test) variants used during
  this investigation.
- `scripts/probe_v4_per_slice_eval.py` — apples-to-apples eval of v4
  fixtures with both WALDO and Wald baselines.
