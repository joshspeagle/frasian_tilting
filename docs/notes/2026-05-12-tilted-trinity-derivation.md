# Tilted trinity derivation (PL/OT/FR vs MX)

**TL;DR.** On Normal-Normal+Normal, every Gaussian q(θ) makes
`tau_WALDO == tau_LRTO == tau_SCOREO` exactly — same number, same
H_0-MC reference distribution, same p-value, same CI. PL, OT, and
FR all keep q_η Gaussian for every admissible η, so on those three
schemes `lrto` and `scoreo` route to the closed-form `waldo`
formula. MX is the exception: q_η is a genuine two-component
Gaussian mixture, so the three statistics decouple and need
mixture-aware mode-finding plus score/information formulas. This
note is the math foundation that Tasks 1-13 of the
`tilted-lrt-score-pairs` plan cite.

## 1. Trinity collapse on a Gaussian q

Let `q(θ; params) = N(μ_q, σ_q²)`. Then

```
log q(θ) = -(θ - μ_q)² / (2 σ_q²) + const,                       (G1)
```

so the mode of q is

```
θ_MAP,q = argmax_θ log q(θ) = μ_q.                                (G2)
```

Substituting into the three statistics:

- **WALDO-style** (the Bayesian counterpart to `wald` — square of
  posterior pivot scaled by posterior precision):

  ```
  tau_WALDO(θ) = (μ_q - θ)² / σ_q².                               (G3)
  ```

- **LRTO** (posterior log-density gap relative to MAP — see
  [`docs/methods/lrto.md`](../methods/lrto.md)):

  ```
  tau_LRTO(θ) = -2 [log q(θ) - log q(θ_MAP,q)]
              = -2 [-(θ-μ_q)²/(2σ_q²) - 0]
              = (μ_q - θ)² / σ_q²
              = tau_WALDO(θ).                                     (G4)
  ```

- **SCOREO** (posterior score squared, scaled by posterior
  information — see [`docs/methods/scoreo.md`](../methods/scoreo.md)):

  ```
  U_q(θ)     = ∂ log q / ∂θ = -(θ - μ_q) / σ_q²,
  I_q(θ)     = -∂² log q / ∂θ² = 1 / σ_q²  (constant on Gaussians),
  tau_SCOREO = U_q(θ)² / I_q(θ)
             = [(θ-μ_q)/σ_q²]² · σ_q²
             = (μ_q - θ)² / σ_q²
             = tau_WALDO(θ).                                      (G5)
  ```

So on **any** Gaussian q, the three τ-values are identical for every
θ. Calibration is identical too (item 7 below), so p-values and
inverted CIs match as well.

## 2-5. The tilted q_η under each scheme on NN+Normal

Throughout this section: posterior `N(μ_n, σ_n²)` with `μ_n = wD +
(1-w)μ₀`, `σ_n² = w σ²`; likelihood-as-Gaussian-on-θ `N(D, σ²)`;
weight `w = σ₀²/(σ²+σ₀²)`.

### 2. PL — power-law (e-geodesic)

Theorem 6 in [`docs/methods/power_law.md:44`](../methods/power_law.md)
gives the closed form

```
denom = 1 - η(1 - w),
μ_η   = (wD + (1-η)(1-w)μ₀) / denom,                              (PL1)
σ_η²  = w σ² / denom,
q_η   = N(μ_η, σ_η²).                                              (PL2)
```

Single Gaussian for every admissible `η < 1/(1-w)`. Trinity collapses
to `tau_WALDO_PL(θ) = (μ_η - θ)² / σ_η²`.

### 3. OT — Wasserstein-2 geodesic

[`docs/methods/ot.md`](../methods/ot.md) §"closed form on Gaussians"
(lines 75-90) gives the 1D W2 geodesic between two Gaussians as
another Gaussian:

```
μ_t = (1-η) μ_n + η D,
s_t = (1-η) σ_n + η σ,
q_η = N(μ_t, s_t²).                                                (OT1)
```

(Standard-deviations interpolate linearly, not variances.) Single
Gaussian for every `η ∈ [0, 1]`. Trinity collapses to
`tau_WALDO_OT(θ) = (μ_t - θ)² / s_t²`.

### 4. FR — Fisher-Rao / Levi-Civita geodesic on the half-plane

The Gaussian family `{N(μ, σ²) : σ > 0}` with Fisher metric is the
hyperbolic half-plane (`σ > 0`). The Levi-Civita geodesic between
`N(μ_n, σ_n²)` and `N(D, σ²)` stays in this submanifold — every
point along the path is a Gaussian. So

```
q_η = N(μ_FR(η), σ_FR(η)²)                                         (FR1)
```

for closed-form coefficients `(μ_FR(η), σ_FR(η))` derived in
[`docs/methods/fisher_rao.md`](../methods/fisher_rao.md) (curved
trajectory in the `(μ, log σ)` plane, not linear like OT). Single
Gaussian for every η ∈ ℝ. Trinity collapses to
`tau_WALDO_FR(θ) = (μ_FR(η) - θ)² / σ_FR(η)²`.

### 5. MX — mixture (m-geodesic)

[`docs/methods/mixture.md`](../methods/mixture.md) defines the
m-geodesic (linear interpolation of densities) as

```
q_η,mix(θ) = (1-η) · N(θ; μ_n, σ_n²) + η · N(θ; D, σ²).            (MX1)
```

Set `w₁ = 1-η`, `w₂ = η`, `φ_i = N(·; μ_i, σ_i²)` with
`(μ₁, σ₁) = (μ_n, σ_n)`, `(μ₂, σ₂) = (D, σ)`. **Genuine
two-component Gaussian mixture**, not a single Gaussian. The trinity
no longer collapses.

## 6. Mixture analytic derivatives

For a generic 2-component Gaussian mixture `q(θ) = w₁ φ₁(θ) + w₂
φ₂(θ)`:

```
log q(θ) = logsumexp_i [ log w_i + log φ_i(θ) ].                   (MX2)
```

Define **responsibilities**

```
r_i(θ) = w_i φ_i(θ) / q(θ),     with Σ_i r_i = 1.                  (MX3)
```

Per-component score and information

```
u_i(θ) = ∂ log φ_i / ∂θ = -(θ - μ_i) / σ_i²,                       (MX4)
I_i    = -∂² log φ_i / ∂θ² = 1 / σ_i²    (constant in θ per component).
```

Then by direct differentiation of (MX2):

```
U(θ) = ∂ log q / ∂θ = Σ_i r_i(θ) u_i(θ),                           (MX5)
I(θ) = -∂² log q / ∂θ²
     = Σ_i r_i I_i  -  Var_r[u_i]
     = Σ_i r_i I_i  -  ( Σ_i r_i u_i²  -  ( Σ_i r_i u_i )² ).      (MX6)
```

(MX6) is the standard mixture observed-information identity:
expected-per-component information minus the score variance across
components.

**Mode finding** for `tau_LRTO,mix`: `θ_MAP,q` is `argmax_θ q(θ)`,
i.e. a root of `U(θ) = 0`. Unimodal regime — `brentq` on the
bracketing interval `[min_i μ_i, max_i μ_i]` works (the mode of a
2-Gaussian mixture lies between the component means). Bimodal regime
(when `|μ₁ - μ₂|` is large relative to `σ_i` and weights are
balanced) — `brentq` may converge to a saddle; fallback to a fine
grid over `[min_i μ_i, max_i μ_i]` (or a slightly wider window) and
take `argmax q(θ)` directly.

Once `θ_MAP,q` is in hand, the three τ-statistics on MX:

```
tau_WALDO,mix  = (E_q[θ] - θ_test)² / Var_q[θ],                    (MX7)
tau_LRTO,mix   = -2 [ log q_η(θ_test) - log q_η(θ_MAP,q) ],        (MX8)
tau_SCOREO,mix = U(θ_test)² / I(θ_test).                           (MX9)
```

These are **three different functions of θ_test** in general: the
mean of `q_η,mix` is not its mode, the log-density gap is not
`(μ-θ)²/σ²`, and the score is responsibility-weighted (affine in θ
only inside each component, not across the mixture).

## 7. H_0 reference is identical for tilted-WALDO / LRTO / SCOREO

The Monte Carlo calibration scheme is shared. For each statistic
`tau_X` (X ∈ {WALDO, LRTO, SCOREO}) the p-value at `(θ_0, data)` is

```
p_X(θ_0) = P_{D' ~ likelihood(·|θ_0)} [ tau_X(θ_0; D', prior) ≥ tau_X(θ_0; data, prior) ].   (H1)
```

The reference draws `D'` from `likelihood(·|θ_0)` — **not** from
`q_η`. So the MC scaffold is one and the same:

1. Draw `D' ~ likelihood(·|θ_0)` (model.sample_data; uses CRN seed
   for brentq stability — see
   [`src/frasian/tilting/_generic_pvalue.py`](../../src/frasian/tilting/_generic_pvalue.py)).
2. Rebuild `q_η(·; D', prior)` (closed-form on NN, autodiff/grid
   otherwise).
3. Recompute `tau_X` from `q_η` and `θ_0`.
4. Average `1{ tau_X(D') ≥ tau_X(data) }` over MC draws.

Only step 3 — the τ formula — varies across the three statistics.
The same `model.sample_data`, the same posterior rebuild, the same
`brentq` outer loop, the same CRN seed. **The lrto/scoreo wiring
work is therefore: implement step 3 for each (scheme, statistic)
cell, then reuse the existing tilted-WALDO MC machinery for steps
1/2/4.**

## Why MX is special

Because `q_η,mix` has mass at two distinct centers, its mode ≠ its
mean, the log-density gap is not `(μ-θ)²/σ²`, and the score is
responsibility-weighted, not affine-in-θ. The (G3)-(G5) collapse
identities all use `log q` being quadratic in θ, which fails for
any mixture with separated components. Practically, this means:

- `lrto` on MX must compute `θ_MAP,q` numerically (brentq + grid
  fallback per item 6); on PL/OT/FR it reduces to the closed-form
  `waldo` (item 1).
- `scoreo` on MX must form (MX5)/(MX6); on PL/OT/FR it reduces to
  the closed-form `waldo` (item 1).
- `tau_WALDO` itself on MX uses the mixture's mean and variance
  (closed-form moments of a 2-Gaussian mixture); on PL/OT/FR these
  reduce to `μ_η`, `σ_η²` directly.

In code, this is the dispatch decision: for `(scheme ∈
{PL, OT, FR}, statistic ∈ {lrto, scoreo})` cells the implementation
can route to the existing `tau_WALDO_<scheme>` formula; for
`(MX, lrto)` and `(MX, scoreo)` cells a fresh mixture-aware kernel is
required.

## Links

- [`docs/methods/lrto.md`](../methods/lrto.md) — Bayesian LRT
  on the un-tilted posterior; the `(identity, lrto)` cell.
- [`docs/methods/scoreo.md`](../methods/scoreo.md) — Bayesian score
  test on the un-tilted posterior; the `(identity, scoreo)` cell.
- [`docs/methods/power_law.md`](../methods/power_law.md) §Theorem 6 —
  PL tilt closed form, line 44 onward.
- [`docs/methods/ot.md`](../methods/ot.md) §closed form on Gaussians
  — OT tilt closed form, lines 75-90.
- [`docs/methods/fisher_rao.md`](../methods/fisher_rao.md) — FR tilt
  closed form, Stage A.
- [`docs/methods/mixture.md`](../methods/mixture.md) — MX tilt
  definition and the 2-Gaussian-mixture analytic forms.
- [`src/frasian/tilting/_generic_pvalue.py`](../../src/frasian/tilting/_generic_pvalue.py)
  — shared generic-MC tilted_pvalue scaffold (item 7).
- Plan: `docs/superpowers/plans/2026-05-12-tilted-lrt-score-pairs.md`
  (gitignored; in the local plans dir).
