# ot

> Status: `implemented`

## Summary

Wasserstein-2 (optimal-transport) geodesic tilting between the
posterior and the likelihood-induced Gaussian. The general 1D
implementation uses the **quantile-mixture** representation
`F_t^{-1}(u) = (1-t)·F_p^{-1}(u) + t·F_q^{-1}(u)` (McCann
displacement interpolation), which works for any two endpoints
exposing `quantile`. On the Normal-Normal sandbox this collapses to
linear interpolation in `(mu, sigma)` and the tilted distribution
stays Gaussian — the implementation recognises this Gaussian fast
path. Endpoints follow the framework convention `eta=0` -> posterior,
`eta=1` -> likelihood-as-Gaussian, matching `power_law`.

### Endpoint convention across schemes

The framework uses a uniform endpoint contract: every `TiltingScheme`
anchors `eta=0` at the posterior (the identity / no-tilt endpoint) and
moves the family toward the likelihood as `eta` increases. Source of
truth: `tilting/power_law.py` lines ~91-94 (`eta_default=0.0`,
`eta_identity=0.0`, "eta=0 recovers WALDO; eta=1 recovers Wald").

| Scheme        | `eta=0`              | `eta=1` (or other distinguished) | Notes |
|---------------|----------------------|----------------------------------|-------|
| `identity`    | posterior            | (no other endpoint)              | No-op; the matrix's identity element. |
| `power_law`   | posterior            | `eta=1` → Wald (likelihood-only) | e-geodesic; admissible η ∈ (-w/(1-w), 1/(1-w)) (open). |
| `ot`          | posterior            | `eta=1` → likelihood-as-Gaussian | W2 geodesic; admissible η ∈ [0, 1] (closed). |
| `mixture`     | posterior            | `eta=1` → likelihood-as-Gaussian | m-geodesic, dual partner of `power_law`; planned. |
| `fisher_rao`  | posterior            | `eta=1` → likelihood-as-Gaussian | Levi-Civita / Fisher-Rao geodesic; planned. |

All implemented schemes set `param_space.eta_identity = 0.0`; tests in
`tests/properties/` enforce `tilt(posterior, prior, lik, eta=0) ==
posterior` for each.

## Motivation

`power_law` (the e-geodesic / log-linear path) clamps against
`eta = -w/(1-w)` near the prior boundary; this clamp is a
parameterisation artefact, not a feature of the data, and produces a
sharp transition in the smoothness diagnostic
(`docs/methods/smoothness_experiment.md`). The W2 geodesic is the
**displacement** path between the same endpoints — interpolation in
mass-transport coordinates rather than in log-density coordinates —
and is structurally clamp-free on `[0, 1]`. The framework's central
hypothesis is that this geometry produces a smoother `eta*(|Delta|)`
curve at the same calibration. (Hypothesis specific to this
framework; no prior empirical evidence.)

This scheme is the third of three natural geometries on the space of
distributions: e-geodesic (`power_law`), m-geodesic (`mixture`), and
the Wasserstein geodesic (this method). The first two are the dually
flat pair under the Fisher metric (Amari & Nagaoka 2000); W2 is a
distinct, mass-displacement geometry (Pistone & Malagò 2018; Takatsu
2011).

## Definition

For two 1D distributions `p` and `q` with finite second moments, the
constant-speed W2 geodesic at `t in [0, 1]` is the **quantile-mixture**

```
F_t^{-1}(u) = (1 - t) · F_p^{-1}(u) + t · F_q^{-1}(u),  u in [0, 1].
```

Equivalently, the law of `(1-t)·F_p^{-1}(U) + t·F_q^{-1}(U)` for
`U ~ Uniform[0, 1]` (McCann 1997; Villani 2003 Ch. 2; Santambrogio
2015 Ch. 2). For Borel measures with finite variance the formula is
exact even with atoms; the framework's tilting only needs absolutely-
continuous endpoints, so no caveats apply here.

**Gaussian fast path.** When both endpoints are Gaussian the
geodesic stays in the Gaussian family and admits the closed form

```
mu_t    = (1 - t) · mu_a + t · mu_b
sigma_t = (1 - t) · sigma_a + t · sigma_b      (linear in sigma, NOT sigma^2)
```

(Olkin & Pukelsheim 1982; Dowson & Landau 1982; Takatsu 2011; Bhatia,
Jain & Lim 2019). Note: `sigma_t^2 = ((1-t)·sigma_a + t·sigma_b)^2`,
**not** `(1-t)·sigma_a^2 + t·sigma_b^2`.

**Endpoints.** The framework anchors the path at posterior (eta=0)
and the likelihood-induced Gaussian `N(D, sigma^2)` (eta=1):

```
mu_t    = (1 - eta) · mu_n + eta · D
sigma_t = (1 - eta) · sigma_n + eta · sigma.
```

## Derivation

**1D W2 geodesic = quantile-mixture.** For 1D measures the optimal
transport map under quadratic cost is the monotone rearrangement
`T = F_q^{-1} ∘ F_p` (Brenier 1991; Villani 2003 §2.2). McCann's
displacement interpolation `((1-t)·id + t·T)_# p` evaluated at this
map gives the quantile formula above (Santambrogio 2015 Prop. 2.13).

**Gaussian closed form.** For univariate Gaussians the quantile is
`F^{-1}(u) = mu + sigma · Phi^{-1}(u)`, so the quantile-mixture's
quantile is `((1-t)mu_a + t mu_b) + ((1-t)sigma_a + t sigma_b)·
Phi^{-1}(u)` — the quantile of `N(mu_t, sigma_t^2)` with the linear
form above (multivariate analogue: Bures-Wasserstein, Bhatia-Jain-Lim
2019).

**OT-tilted WALDO p-value (Normal-Normal sandbox).** Substituting the
W2-tilted posterior into the WALDO formula produces a closed-form
p-value. The key identity: `mu_t` is a linear function of `D` with
slope `dmu_t/dD = w + eta·(1 - w)`, so under repeated sampling
`D | theta ~ N(theta, sigma^2)`,

```
s_t := sd(mu_t | theta) = (w + eta · (1 - w)) · sigma,
E[mu_t - theta | theta]  = (1 - eta) · (1 - w) · (mu0 - theta).
```

Substituting into the bare WALDO structure gives

```
a(theta) = |mu_t - theta| / s_t
b(theta) = (1 - eta) · (1 - w) · (mu0 - theta) / s_t
p(theta) = Phi(b - a) + Phi(-a - b).
```

Endpoint sanity (verified symbolically + 2000 random numerical draws,
atol = 0):
- At `eta = 0`: `s_t = w · sigma`, `mu_t = mu_n`, formula collapses
  to bare WALDO (matches `src/frasian/statistics/waldo.py:36-42`).
- At `eta = 1`: `s_t = sigma`, `mu_t = D`, `b = 0`, `a = |D-theta|/sigma`,
  so `p = 2 · Phi(-|D-theta|/sigma)` — bare two-sided Wald.

**`s_t` vs `sigma_t` — a subtle but important distinction.** The
standard error `s_t` appearing in the p-value is **not** the W2-
geodesic posterior std `sigma_t = (1 - eta)·sigma_n + eta·sigma`.
The two coincide only at `eta = 1`:

```
s_t(0)     = w · sigma             sigma_t(0)     = sqrt(w) · sigma
s_t(1)     = sigma                 sigma_t(1)     = sigma
```

`s_t` is the dispersion of `mu_t` as an estimator of `theta` (a
sampling SD); `sigma_t` is the spread of the tilted posterior (a
distributional SD). The bare WALDO denominator is
`sd(mu_n | theta) = w·sigma` — see `src/frasian/statistics/waldo.py:40-41`
— not `sigma_n = sqrt(w)·sigma`; the same logic applies under
tilting, hence `s_t`, not `sigma_t`, in the formula. A future reader
expecting `sigma_t` will see the eta=0 endpoint break by ~0.35 in
p-value error on the canonical sandbox.

## Predicted behavior

- **Smoothness.** `eta*(|Delta|)` curve has no clamp (admissible
  range is `[0, 1]` for all `|Delta|`). Lipschitz value on the
  smoothness diagnostic is expected to be smaller than `power_law`'s
  by an order of magnitude. (Hypothesis specific to this framework.)
- **Coverage.** Calibrated under `DynamicNumericalEtaSelector` (the
  η used at each θ depends only on θ, so the WALDO p-value at fixed
  η is U[0,1] under H0 and the CI achieves nominal 1-α coverage by
  construction — same argument as `power_law[dynamic_numerical]`).
- **Width.** η=0 recovers WALDO width; η=1 recovers Wald width.
  Interior optima depend on `|Delta|`; comparison with `power_law` is
  the empirical question the framework exists to answer.
- **Endpoint identities.** At `eta = 0` the tilted-WALDO p-value is
  exactly bare WALDO. At `eta = 1` it is exactly bare two-sided Wald.

## Failure modes

- **Variance-mode interpolation `sigma_t^2 = (1-t)sigma_a^2 + t·sigma_b^2`
  is wrong.** That curve is not the W2 geodesic; using it would break
  the smoothness comparison. The implementation uses linear-in-sigma.
- **`s_t != sigma_t` confusion.** See Derivation; the WALDO
  denominator is the sampling SD `s_t`, not the geodesic-interpolated
  posterior std `sigma_t`.
- **Non-Gaussian likelihood.** The current `OTTilting.tilt`
  implementation requires `GaussianLikelihood` for the likelihood-to-
  distribution conversion; other `Likelihood` types raise
  `NotImplementedError` (matches `power_law`'s discipline). The
  general path machinery (`QuantileMixturePath`) is, however,
  agnostic to the endpoint family — it runs on any pair of
  `Distribution` instances. The illustration exercises this by
  interpolating between two Beta distributions.

## Invariants

(Property tests in `tests/properties/test_ot_invariants.py`.)

- `tilt(eta=0)` returns a Gaussian numerically equal to the input
  posterior (atol 1e-12).
- `tilt(eta=1)` returns `N(D, sigma^2)` (atol 1e-12).
- `tilt(...).scale > 0` for every `eta in [0, 1]`.
- `tilt` is continuous in `eta` (Lipschitz constant ≤ 10 on the
  Hypothesis test range).
- `admissible_range(...) == (0.0, 1.0)`.
- `eta < 0` or `eta > 1` raises `TiltingDomainError` (never NaN).
- **Quantile-mixture identity**: `QuantileMixturePath.quantile(u) =
  (1-t)·F_p^{-1}(u) + t·F_q^{-1}(u)` to atol 1e-12.
- **Endpoint p-value recoveries**: `tilted_pvalue(eta=0, ..., 'waldo')`
  equals `WaldoStatistic.pvalue` to atol 1e-12; `tilted_pvalue(eta=1,
  ..., 'waldo')` equals `2 · Phi(-|D-theta|/sigma)` to atol 1e-12.

## Literature

### Foundational

- McCann, R. J. "A convexity principle for interacting gases."
  *Advances in Mathematics* 128 (1997): 153–179. — Displacement
  interpolation; the McCann interpolant is the W2 geodesic in 1D.
- Brenier, Y. "Polar factorization and monotone rearrangement of
  vector-valued functions." *Comm. Pure Appl. Math.* 44 (1991):
  375–417. — Uniqueness of the optimal map under quadratic cost.
- Villani, C. *Topics in Optimal Transportation.* GSM 58, AMS, 2003.
  — Standard reference; Ch. 2 covers 1D and the quantile-mixture.
- Santambrogio, F. *Optimal Transport for Applied Mathematicians.*
  Birkhäuser, 2015. — Ch. 2 "One-dimensional issues."
- Olkin, I., Pukelsheim, F. "The distance between two random vectors
  with given dispersion matrices." *Lin. Alg. Appl.* 48 (1982):
  257–263. — Closed-form W2 distance on Gaussians.
- Dowson, D. C., Landau, B. V. "The Fréchet distance between
  multivariate normal distributions." *J. Multivar. Anal.* 12
  (1982): 450–455. — Independent contemporaneous derivation.
- Takatsu, A. "Wasserstein geometry of Gaussian measures." *Osaka J.
  Math.* 48 (2011): 1005–1026. — W2 geometry of the Gaussian
  submanifold; non-negative sectional curvature contrasting with
  Fisher-Rao's negative curvature.
- Bhatia, R., Jain, T., Lim, Y. "On the Bures-Wasserstein distance
  between positive definite matrices." *Expo. Math.* 37 (2019):
  165–191. — Matrix-analytic exposition of the Bures-Wasserstein
  structure.
- Peyré, G., Cuturi, M. "Computational optimal transport." *FnT in
  ML* 11 (2019): 355–607. — Textbook with the Gaussian closed form
  (§2.6) and 1D case (§3.1).

### Closely related (Bayesian / posterior-approximation context)

- Jordan, R., Kinderlehrer, D., Otto, F. "The variational formulation
  of the Fokker-Planck equation." *SIAM J. Math. Anal.* 29 (1998):
  1–17. — JKO scheme; Wasserstein gradient flow of relative entropy.
- El Moselhy, T. A., Marzouk, Y. M. "Bayesian inference with optimal
  maps." *J. Comput. Phys.* 231 (2012): 7815–7850. — Pushes prior to
  posterior via an optimal map; closest direct precedent for OT-as-
  tilting in Bayesian inference.
- Lambert, M., Chewi, S., Bach, F., Bonnabel, S., Rigollet, P.
  "Variational inference via Wasserstein gradient flows." *NeurIPS*
  35 (2022). arXiv:2205.15902. — VI on the Bures-Wasserstein
  submanifold of Gaussians, exactly the geometry here.

### Contrasting (alternative geometries)

- Amari, S., Nagaoka, H. *Methods of Information Geometry.* AMS /
  Oxford, 2000. — e-/m-connections and dually flat structure;
  positions OT as a third, distinct geometry.
- Pistone, G., Malagò, L. "Wasserstein Riemannian geometry of
  Gaussian densities." *Information Geometry* 1 (2018): 137–179. —
  Direct contrast between Bures-Wasserstein and Amari geometries on
  the Gaussian submanifold.

## Links

- Implementation: `src/frasian/tilting/ot.py`
- General-path Distribution: `src/frasian/tilting/quantile_mixture.py`
- Property tests: `tests/properties/test_ot_invariants.py`
- Regression tests: `tests/regression/test_ot_regression.py`
- Illustration: `src/frasian/experiments/illustrations/ot_demo.py`

## Status notes

The general-path machinery (`QuantileMixturePath`) is endpoint-
agnostic — it runs on any pair of `Distribution` instances exposing
`quantile`. The Gaussian fast path skips the wrapper for performance
and to keep the output in `NormalDistribution` so downstream
consumers hit closed-form scipy code.

The framework's calibrated default selector is
`DynamicNumericalEtaSelector` (per-θ varying η), same as `power_law`.
Cell name picks up the selector when non-default, e.g.
`ot[dynamic_numerical]` is the OT-WALDO calibrated cell.
