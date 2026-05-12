# fisher_rao

> Status: `implemented`

## Summary

Fisher-Rao (Levi-Civita) geodesic on the parametric distribution
manifold equipped with the Fisher information metric (Rao 1945). On
the univariate Gaussian family the manifold is the upper half-plane
in `(mu, sigma)`, with constant negative curvature (Atkinson &
Mitchell 1981; Costa et al. 2015), and geodesics between two
Gaussians are circular arcs perpendicular to the boundary `sigma=0`.
Compared to the W2 geodesic (`ot`), Fisher-Rao respects the
information-geometric structure rather than mass displacement; the
two are genuinely distinct geometries on the Gaussian family
(Takatsu 2011; Chizat et al. 2018; Pistone & Malagò 2018).

## Motivation

Fisher-Rao is the **third** affine connection compatible with the
Fisher metric — the Levi-Civita connection (α=0) — distinct from
`power_law`'s e-connection (α=+1) and `mixture`'s m-connection
(α=-1) (Amari & Nagaoka 2000 §3). Where the e- and m-geodesics are
affine straight lines in their respective dual coordinate systems,
the Fisher-Rao geodesic minimises arc-length under the Fisher metric
itself, producing the *intrinsic* shortest path on the parametric
manifold.

The hypothesis (specific to this framework; no prior empirical
evidence) is that Fisher-Rao's intrinsic geometry produces an even
smoother `eta*(|Delta|)` curve than `ot` (W2) — both lack the power-
law clamp, but the Fisher-Rao path is curvature-aware whereas the
W2 path is straight in `(mu, sigma)` regardless of the metric. For
informative priors (small `sigma_n`) the two paths take noticeably
different routes through the half-plane.

By Chentsov's uniqueness theorem (Cencov 1982), the Fisher metric is
the unique Riemannian metric on a statistical manifold invariant
under sufficient statistics — making the Fisher-Rao geodesic the
*canonical* intrinsic geodesic in a way W2 is not.

## Definition

The Fisher information matrix on the Gaussian family in raw
coordinates `(mu, sigma)` is `diag(1/sigma^2, 2/sigma^2)`, giving the
metric

```
ds^2 = (d mu^2 + 2 d sigma^2) / sigma^2.
```

Substituting `tilde mu = mu/sqrt(2)` rescales this to the standard
Poincaré half-plane metric `(d tilde mu^2 + d sigma^2)/sigma^2` with
constant Gaussian curvature `K = -1` (Costa et al. 2015 §3; Pinele,
Strapasson & Costa 2020 §3). In raw coordinates, `K = -1/2`.

**Closed-form geodesic path** between `N(mu_a, sigma_a^2)` and
`N(mu_b, sigma_b^2)`, parameterised by `t in [0, 1]`:

- If `mu_a = mu_b`: vertical line in the half-plane,
  `mu(t) = mu_a`, `sigma(t) = sigma_a^{1-t} · sigma_b^t`.
- Otherwise: semi-circular arc in `(tilde mu, sigma)` coordinates with
  centre on the `sigma = 0` axis. Centre and radius are determined by
  the perpendicular bisector of the chord through the two
  rescaled-coordinate endpoints. The constant-speed (= arc-length)
  parametrisation uses the antiderivative `s(phi) = ln tan(phi/2)`:

      s(t)  = (1 - t) * ln tan(phi_a/2) + t * ln tan(phi_b/2)
      phi(t) = 2 * arctan(exp(s(t)))
      tilde_mu(t) = c + r * cos(phi(t))
      sigma(t)    = r * sin(phi(t))

  (Earlier drafts of this brief stated `phi` linear in `t` gives a
  constant-speed parametrisation — that is incorrect; sympy verifies
  `ds/dphi = 1/sin phi`, not constant.)

The closed-form *distance* is given in Costa et al. 2015 Eqs. 5-6
(= Pinele et al. 2020 Eq. 22; equivalent rewrite of Costa Eqs. 5-6
via `arccosh(z) = ln(z + sqrt(z^2 - 1))`; Eq. 12 is the symmetrised
KL distance and was cited erroneously in earlier drafts.):

```
d_FR(N_a, N_b) = sqrt(2) · arccosh( 1 + ((mu_a-mu_b)^2/2 + (sigma_a-sigma_b)^2) / (2 sigma_a sigma_b) ).
```

The **path** parameterisation follows from elementary hyperbolic
geometry plus the Atkinson & Mitchell 1981 / Calvo & Oller 1991
treatments.

**Endpoints.** Following the framework convention (matches
`power_law`, `ot`, `mixture`): `eta = 0` -> posterior, `eta = 1` ->
likelihood-induced Gaussian `N(D, sigma^2)`.

## Derivation

**Setup.** Let `phi(x; mu, sigma)` denote the `N(mu, sigma^2)` density,
prior `N(mu0, sigma0^2)`, likelihood `D | theta ~ N(theta, sigma^2)`,
and posterior `N(mu_n, sigma_n^2)` with the Frasian primitives at
`src/frasian/models/normal_normal.py`:

  `w = sigma0^2 / (sigma^2 + sigma0^2)`,
  `mu_n = w D + (1 - w) mu0`,
  `sigma_n = sqrt(w) sigma`,
  `Delta = (1 - w)(mu0 - D)/sigma`.

**Step 1 (Fisher metric on the Gaussian family).** Compute the negative
expected Hessian of `log phi(x; mu, sigma)` under `x ~ N(mu, sigma^2)`:

  `-E[d2 log phi / d mu^2]    = 1/sigma^2`,
  `-E[d2 log phi / d sigma^2] = 2/sigma^2`,
  `-E[d2 log phi / d mu d sigma] = 0`.

Hence `I(mu, sigma) = diag(1/sigma^2, 2/sigma^2)` and the Fisher line
element is `ds_F^2 = (d mu^2 + 2 d sigma^2)/sigma^2` (Costa et al. 2015
Eqs. 1-2).
*Verification:* sympy on `log phi(x; mu, sigma)` and exact integration
of the Hessian against `phi(x; mu, sigma)` returns `1/sigma^2`,
`2/sigma^2`, `0` exactly.

**Step 2 (Rescale to unit-curvature half-plane).** Substitute
`tilde_mu = mu/sqrt(2)` so `d mu = sqrt(2) d tilde_mu` and
`ds_F^2 = (2 d tilde_mu^2 + 2 d sigma^2)/sigma^2 = 2 (d tilde_mu^2 + d sigma^2)/sigma^2`.
Therefore `d_F = sqrt(2) d_H` where `d_H` is the Poincaré half-plane
distance on `(tilde_mu, sigma)` with metric `(d tilde_mu^2 + d sigma^2)/sigma^2`
(Costa Eq. 4). Constant Gaussian curvature `K = -1` in `(tilde_mu, sigma)`,
`K = -1/2` in raw `(mu, sigma)`.

**Step 3 (Vertical geodesic, `tilde_mu_a = tilde_mu_b`).** The Poincaré
geodesic ODE `sigma'' - (sigma')^2/sigma = 0` is solved exactly by
`sigma(t) = sigma_a^{1-t} sigma_b^t`. Verification: `sigma'/sigma =
ln(sigma_b/sigma_a)` is constant in `t`, so `sigma'' = sigma'^2/sigma`
identically.
*Verification:* sympy `simplify(diff(sig, t, 2) - diff(sig, t)**2/sig)`
returns `0` exactly. Arc length `ds/dt = sigma'/sigma = ln(sigma_b/sigma_a)`
constant ⇒ this is a constant-speed parametrisation; total Poincaré length
`|ln(sigma_b/sigma_a)|`, hence Fisher length `sqrt(2) |ln(sigma_b/sigma_a)|`
(Costa Eq. 7).

**Step 4 (Generic geodesic, `tilde_mu_a != tilde_mu_b`).** The geodesic
is the unique semicircle in `(tilde_mu, sigma)` perpendicular to the
boundary `sigma = 0`. Its centre `c` lies on the boundary and is
determined by equating chord radii:

  `(tilde_mu_a - c)^2 + sigma_a^2 = (tilde_mu_b - c)^2 + sigma_b^2`

solving to

  `c = ((tilde_mu_a^2 - tilde_mu_b^2) + (sigma_a^2 - sigma_b^2)) / (2(tilde_mu_a - tilde_mu_b))`,
  `r = sqrt((tilde_mu_a - c)^2 + sigma_a^2)`.

Parametrise the arc by polar angle `phi = atan2(sigma, tilde_mu - c)` so
`tilde_mu(phi) = c + r cos(phi)`, `sigma(phi) = r sin(phi)`, and let
`phi_a = atan2(sigma_a, tilde_mu_a - c)`, `phi_b = atan2(sigma_b, tilde_mu_b - c)`.
*Verification:* sympy solves the chord-equality equation and returns the
brief's expression for `c` exactly.

**Step 5 (Constant-speed parametrisation — CORRECTION TO BRIEF).**
Computing `ds^2/dt^2 = ((tilde_mu')^2 + (sigma')^2)/sigma^2` along the
arc gives `ds/dphi = 1/sin(phi)` (by direct substitution and `sin^2 +
cos^2 = 1`). **The brief's earlier "linear in `phi`" claim was
incorrect** — `phi` linear in `t` does NOT give constant `ds/dt`. The
constant-speed (= arc-length) parametrisation uses the antiderivative
`s(phi) = ln tan(phi/2)` (which integrates `ds/dphi = 1/sin(phi)`):

  `s(t) = (1 - t) ln tan(phi_a/2) + t ln tan(phi_b/2)`,
  `phi(t) = 2 arctan(exp(s(t)))`,
  `tilde_mu(t) = c + r cos(phi(t))`,
  `sigma(t) = r sin(phi(t))`.

*Sign of `s_a, s_b`.* Both signs are valid — `s(t)` is real for any
`phi ∈ (0, π)`, and `phi(t) = 2 arctan(exp(s(t)))` is monotone, so
the geodesic is correctly traced regardless of whether `phi_a < phi_b`
or `phi_a > phi_b`. (`atan2` returns `phi ∈ (0, π)` for `sigma > 0`,
so `tan(phi/2) ∈ (0, ∞)` and the logs are real-valued.) Implementation
need not branch on the sign of `phi_b - phi_a`.

Total Poincaré arc length is `L_H = |ln tan(phi_b/2) - ln tan(phi_a/2)|`,
so `d_F = sqrt(2) L_H`.
*Verification:* sympy `diff(log(tan(phi/2)), phi) = 1/sin(phi)` exactly;
trapezoidal integration of `dphi/sin(phi)` matches the closed form to
`atol < 1e-13` on the three settings below.

**Step 6 (Match Costa et al. 2015 Eqs. 5/6 ≡ Pinele et al. 2020 Eq. 22 — CITATION CORRECTION).**
The brief's compact form

  `d_FR = sqrt(2) arccosh( 1 + ((mu_a - mu_b)^2/2 + (sigma_a - sigma_b)^2) / (2 sigma_a sigma_b) )`

is algebraically identical to **Costa et al. 2015 Eqs. 5-6** (NOT Eq. 12,
which is the symmetrised KL distance). With
`z = 1 + ((mu_a - mu_b)^2/2 + (sigma_a - sigma_b)^2)/(2 sigma_a sigma_b)`
and `arccosh(z) = ln(z + sqrt(z^2 - 1))`, direct algebra gives Costa
Eqs. 5-6 identically. Both forms agree with the arc-length integration
of Step 5 and with Costa Eq. 5 (cross-ratio form) to machine precision.

**Step 7 (FR-tilted distribution at η).** Following the framework's
endpoint convention, define the FR-tilted distribution as the half-plane
geodesic from `N(mu_n, sigma_n^2)` (η = 0) to `N(D, sigma^2)` (η = 1),
parameter `t = eta`. By Step 5 this returns `(mu_FR(eta), sigma_FR(eta))`
with `mu_FR(0) = mu_n`, `sigma_FR(0) = sigma_n`, `mu_FR(1) = D`,
`sigma_FR(1) = sigma`.
*Verification:* numerical `(mu_FR, sigma_FR)` agrees with `(mu_n, sigma_n)`
at η = 0 and with `(D, sigma)` at η = 1 to `atol < 1e-15` on all three
settings.

**Step 8 (Tilted-WALDO p-value — STRUCTURAL CORRECTION; NOT CLOSED FORM AT INTERIOR η).**
The WALDO statistic at observed `D` and candidate `theta` evaluated against
the FR-tilted reference is

  `tau_obs(theta) = (mu_FR(eta) - theta)^2 / sigma_FR(eta)^2`.

Under H0 (`theta_true = theta`, `X ~ N(theta, sigma^2)`), the replicate
statistic is

  `tau_rep(theta; X) = (mu_FR(eta; X) - theta)^2 / sigma_FR(eta; X)^2`,

where `(mu_FR(eta; X), sigma_FR(eta; X))` is the FR geodesic at `t = eta`
between `N(mu_n(X), sigma_n^2)` and `N(X, sigma^2)`, with `mu_n(X) = w X + (1 - w) mu0`.
Crucially, `mu_FR(eta; X)` and `sigma_FR(eta; X)` are **non-linear in `X`**
along the curved arc, so unlike OT (whose tilted moments are
linear/constant in `X` and admit the closed form
`Phi(b - a) + Phi(-a - b)` of `src/frasian/tilting/ot.py:_ot_tilted_pvalue_numpy_scalar`),
**FR's tilted-WALDO p-value has no clean closed form in general**. The
p-value is computed by 1D Gaussian quadrature (or MC) over `X`:

  `p_FR(theta) = integral phi((X - theta)/sigma)/sigma * 1{tau_rep(theta; X) >= tau_obs(theta)} dX`.

*Verification (corrects the brief's working hypothesis):* numerical FR
quadrature differs from OT's closed form (with FR's `(mu_t, s_t)`
substituted) by up to ~0.37 in p-value at intermediate η (e.g. setting
2, theta=0.5, eta=0.5: `p_FR=0.981, p_OT=0.607`). The OT structural
identity holds **only at η = 0 and η = 1**, where both geodesic
endpoints linearise exactly in `X`.

**Step 9 (Endpoint reductions of the FR-tilted p-value).** At η = 0,
`mu_FR(eta=0; X) = mu_n(X) = w X + (1 - w) mu0` (linear in X) and
`sigma_FR(eta=0) = sigma_n`. The replicate `mu_n(X) ~ N(w theta + (1 - w) mu0, w^2 sigma^2)`
gives the closed form

  `a = |mu_n - theta| / (w sigma)`,
  `b = (1 - w)(mu0 - theta) / (w sigma)`,
  `p_FR(eta=0) = Phi(b - a) + Phi(-a - b)` = bare WALDO

(matching `src/frasian/statistics/waldo.py::_pvalue_components`). At η = 1,
`mu_FR(eta=1; X) = X` and `sigma_FR(eta=1) = sigma`, so
`tau_rep(X) = ((X - theta)/sigma)^2` with `X - theta ~ N(0, sigma^2)`,
giving `p_FR(eta=1) = 2 Phi(-|D - theta|/sigma)` = bare two-sided Wald.
*Verification:* on all three settings `(D=0.5, sigma=1, sigma0=1)`,
`(D=-1, sigma=2, sigma0=0.5)`, `(D=2, sigma=0.5, sigma0=2)` and three
`theta` values per setting, FR-quadrature matches bare WALDO at η = 0
and bare Wald at η = 1 to `atol < 1e-4` (residual is finite-grid
trapezoidal truncation; see invariants table).

**Step 10 (Negative invariant: FR ≠ OT at interior `t`).** At `t = 0.5`
the OT midpoint is `((mu_n + D)/2, (sigma_n + sigma)/2)`; the FR midpoint
follows the curved geodesic of Step 5. Numerically:

| Setting (D, sigma, sigma0)   | OT midpoint            | FR midpoint            | `\|d_mu\|`  | `\|d_sigma\|` |
|---:|---:|---:|---:|---:|
| (0.5, 1.0, 1.0)              | (0.3750, 0.8536)       | (0.3536, 0.8454)       | 0.0214    | 0.0082      |
| (-1.0, 2.0, 0.5)             | (-0.0588, 1.2425)      | (0.5149, 1.1173)       | 0.5738    | 0.1252      |
| (2.0, 0.5, 2.0)              | (1.9412, 0.4925)       | (1.9403, 0.4942)       | 0.0009    | 0.0017      |

All three regimes confirm FR and OT are genuinely distinct geometries
on the Gaussian family (Takatsu 2011); equality holds only in the
degenerate sub-cases `mu_n = D` (vertical line) or `sigma_n = sigma`
(horizontal — but FR's "horizontal" geodesic is actually a half-ellipse
of eccentricity `1/sqrt(2)`, see Costa Eq. 8, so even this degenerate
case differs).

**Invariants checked numerically (atol indicated):**
- Setting (sigma=1, sigma0=1, mu0=0, D=0.5): FR endpoint reductions
  `|p_FR(eta=0) - bare_WALDO| < 5e-5`, `|p_FR(eta=1) - bare_Wald| < 7e-5`
  across `theta in {-0.5, 0, 0.5}` (residual is `n_grid=8000` trapezoidal
  truncation; symbolic identities are exact).
- Setting (sigma=2, sigma0=0.5, mu0=1, D=-1): same endpoint reductions
  hold; FR midpoint `(0.515, 1.117)` differs from OT midpoint
  `(-0.059, 1.243)` by `Δμ=0.574`, `Δσ=0.125`.
- Setting (sigma=0.5, sigma0=2, mu0=0, D=2): endpoint reductions hold
  to `atol < 4e-5`; arc length `2.554` matches both Costa Eq. 5
  cross-ratio form and the brief's `arccosh` form to `atol < 1e-15`.

## Predicted behavior

- **Smooth `eta*(|Delta|)`, no clamp.** Hyperbolic geodesics are
  smooth on the open half-plane; Lipschitz value on the smoothness
  diagnostic is expected to be small.
- **Different from `ot` when `sigma_n != sigma_likelihood`.** On a
  fixed-variance sub-family the two geometries restrict to the same
  curve (translation), but on the full 2D manifold they differ
  globally (Takatsu 2011). The strict claim "equal otherwise" was in
  the legacy brief and is too strong; equality is restricted to the
  fixed-variance / fixed-mean sub-cases.
- **Coverage at nominal level** under a per-θ varying η selector,
  for the same calibration argument as `power_law` and `ot`.
- **Width similar to `ot`** to leading order in `|Delta|`, with
  divergence in the conflict band where the curvature-aware path
  matters.

## Failure modes

- **Numerical issues near `sigma -> 0`.** The hyperbolic half-plane's
  boundary is a singular limit; numerical implementations must guard
  against arc-radii diverging when `sigma_a` and `sigma_b` collapse
  toward 0.
- **Branch selection.** When the perpendicular bisector through the
  rescaled-coordinate endpoints passes through high-curvature
  regions, careful arc parameterisation is required (Nielsen 2023
  discusses numerical Fisher-Rao computation).
- **Quadrature truncation at near-endpoint η.** The closed-form fast
  paths at exact `η = 0` and `η = 1` are atol 1e-12; the adaptive
  brentq + Gaussian-CDF path at `η ∈ (0, 1)` introduces an O(1/n_grid)
  coarse-grid residual (~1e-4 at default `n_grid = 256`, falling to
  ~1e-7 at `n_grid = 1024`). brentq-inverted CIs near the endpoints
  may show small discontinuities at the `η = 0^±` and `η = 1^∓`
  transitions. At KS-power `n ≥ 10000` the default coarse grid
  occasionally returns `p = 1.0` for `D` values close to the
  τ-minimum (no sign change on the n=256 X-grid even though the
  algorithm is correct in the limit); driving `n_grid = 1024`
  resolves the calibration check (see
  `tests/properties/test_fisher_rao_invariants.py::test_calibration_under_h0`).
- **JAX kernel gradient stability through the vertical case.** The
  JAX kernel uses fine-grain trapezoidal quadrature (~4e-4 precision
  at `n_grid = 8000`). The gradient through the equal-`μ` vertical-case
  branch is fixed via a symbolic double-where substitution with a
  wider JAX-specific threshold (`_VERTICAL_CASE_EPS_JAX = 1e-6`, vs
  the numpy path's `1e-12`): the bare `jnp.where(|denom| < eps, 1.0,
  denom)` pattern catches forward NaNs at `denom = 0` but leaves the
  reverse-mode gradient corrupted in the small-`denom` regime (autograd
  reverses through `c_tilde ~ 1/denom` whose Jacobian explodes as
  `1/denom²`). The fix is pinned by
  `test_jax_geodesic_gradient_through_vertical`.
- **Non-Gaussian likelihood / prior.** Stage A lands Gaussian-only;
  non-Gaussian endpoints raise `NotImplementedError`, matching
  `power_law`'s discipline. A general `ParametricFamily` interface
  (with explicit Fisher metric for non-Gaussian families, e.g. Beta,
  Bernoulli) is a follow-up refactor; Stage B introduces the autodiff
  + diffrax shooting machinery validated against the closed form on NN.

## Invariants

- `tilt(eta=0)` returns the posterior; `tilt(eta=1)` returns
  `N(D, sigma^2)`.
- Output is Gaussian with `sigma > 0` along the path.
- `tilt` is differentiable in `eta` (smoothness Lipschitz expected
  small).
- Reduces to a vertical-line interpolation on `sigma` when
  `mu_a = mu_b`.
- Distance along the path matches the Costa et al. 2015 / Pinele et
  al. 2020 closed form.
- Differs from `ot` whenever `sigma_a != sigma_b` (i.e. for any
  non-degenerate `(prior, likelihood)` triple).

## Literature

### Foundational

- Rao, C. R. "Information and the accuracy attainable in the estimation
  of statistical parameters." *Bulletin of the Calcutta Mathematical
  Society* 37, no. 3 (1945): 81-91. — Originated the Fisher information
  metric and the Rao distance. (No DOI; reprinted in *Breakthroughs in
  Statistics, Vol. I*, Springer 1992,
  doi:10.1007/978-1-4612-0919-5_16.)
- Atkinson, C., Mitchell, A. F. S. "Rao's distance measure." *Sankhya:
  The Indian Journal of Statistics, Series A* 43, no. 3 (1981):
  345-365. https://www.jstor.org/stable/25050283 — Closed-form Rao
  distances for elementary families; gives the half-plane formula for
  univariate normals.
- Skovgaard, L. T. "A Riemannian geometry of the multivariate normal
  model." *Scandinavian Journal of Statistics* 11, no. 4 (1984):
  211-223. https://www.jstor.org/stable/4615960 — Multivariate
  generalisation; widely-cited Riemannian-geometry treatment.
- Calvo, M., Oller, J. M. "An explicit solution of information
  geodesic equations for the multivariate normal model." *Statistics &
  Decisions* 9, no. 1-2 (1991): 119-138.
  doi:10.1524/strm.1991.9.12.119 — Closed-form geodesics for
  multivariate normals; collapses to half-plane arcs in 1D.
- Costa, S. I. R., Santos, S. A., Strapasson, J. E. "Fisher information
  distance: a geometrical reading." *Discrete Applied Mathematics* 197
  (2015): 59-69. doi:10.1016/j.dam.2014.10.004; arXiv:1210.2354. —
  Hyperbolic-geometry reading of Fisher-Rao for univariate normals;
  the closed-form distance in the rescaled half-plane is **Eqs. 5-6**
  (the brief's `arccosh` form is the equivalent rewrite of these
  `sqrt(2)*ln(...)` expressions).
- Pinele, J., Strapasson, J. E., Costa, S. I. R. "The Fisher-Rao
  distance between multivariate normal distributions: special cases,
  bounds and applications." *Entropy* 22, no. 4 (2020): 404.
  doi:10.3390/e22040404 — Section 3 covers same-covariance and
  mirrored-covariance closed forms; Eq. 22 gives the univariate
  `2*log(...)` form.
- Cencov, N. N. *Statistical Decision Rules and Optimal Inference.*
  Translations of Mathematical Monographs Vol. 53. American
  Mathematical Society, 1982. — Uniqueness theorem: Fisher metric is
  the unique Riemannian metric on a statistical manifold invariant
  under sufficient statistics.

### Closely related (information geometry)

- Amari, S., Nagaoka, H. *Methods of Information Geometry.*
  Translations of Mathematical Monographs Vol. 191. American
  Mathematical Society / Oxford University Press, 2000. —
  α-connections including Levi-Civita (α=0); Chapter 2 covers the
  Gaussian-family Fisher metric.
- Amari, S. *Information Geometry and Its Applications.* Applied
  Mathematical Sciences Vol. 194. Springer, 2016.
  doi:10.1007/978-4-431-55978-8 — Modern textbook; Section 2.5 covers
  the e/m/Levi-Civita trichotomy.
- Miyamoto, H. K., Meneghetti, F. C. C., Pinele, J., Costa, S. I. R.
  "On closed-form expressions for the Fisher-Rao distance."
  *Information Geometry* 7 (2024). doi:10.1007/s41884-024-00143-2;
  arXiv:2304.14885. — Recent unified survey of closed-form Fisher-Rao
  distances across families.
- Nielsen, F. "A numerical approximation method for the Fisher-Rao
  distance between multivariate normal distributions." *Entropy* 25,
  no. 4 (2023): 654. doi:10.3390/e25040654; arXiv:2302.08175. —
  Practical numerical methods (curve discretisation +
  Jeffreys-divergence approximation).
- Miolane, N., Guigui, N., Le Brigant, A., Mathe, J., Hou, B.,
  Thanwerdas, Y., et al. "Geomstats: a Python package for Riemannian
  geometry in machine learning." *Journal of Machine Learning
  Research* 21, no. 223 (2020): 1-9.
  https://jmlr.org/papers/v21/19-027.html — Reference Python
  implementation of hyperbolic / Fisher-Rao geodesics; useful as an
  external numerical baseline.

### Contrasting (Wasserstein vs Fisher-Rao)

- Takatsu, A. "Wasserstein geometry of Gaussian measures." *Osaka
  Journal of Mathematics* 48, no. 4 (2011): 1005-1026.
  https://projecteuclid.org/euclid.ojm/1326291215 — W2 geometry on
  Gaussians has *non-negative* sectional curvature, in contrast to
  Fisher-Rao's constant *negative* curvature.
- Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.-X. "An
  interpolating distance between optimal transport and Fisher-Rao
  metrics." *Foundations of Computational Mathematics* 18, no. 1
  (2018): 1-44. doi:10.1007/s10208-016-9331-y — Constructs an explicit
  one-parameter family connecting W2 and Fisher-Rao, demonstrating
  they are genuinely distinct geometries. Note: this is the
  unbalanced-OT "Wasserstein-Fisher-Rao" metric, i.e. a Hellinger-like
  density-space metric, distinct from the parametric-Riemannian
  Fisher-Rao at the heart of this method.
- Olkin, I., Pukelsheim, F. "The distance between two random vectors
  with given dispersion matrices." *Linear Algebra and its
  Applications* 48 (1982): 257-263.
  doi:10.1016/0024-3795(82)90112-4 — W2 closed form on Gaussians;
  counter-reference to Atkinson & Mitchell.
- Malagò, L., Montrucchio, L., Pistone, G. "Wasserstein Riemannian
  geometry of Gaussian densities." *Information Geometry* 1, no. 2
  (2018): 137-179. doi:10.1007/s41884-018-0014-4; arXiv:1801.09269. —
  Direct contrast between Bures-Wasserstein and Amari geometries on
  the Gaussian submanifold.

## Links

- Implementation: `src/frasian/tilting/fisher_rao.py`
- Property tests: `tests/properties/test_fisher_rao_invariants.py`
                  (50 passing)
- Illustration: `src/frasian/experiments/illustrations/fisher_rao_demo.py`

## Status notes

Implemented 2026-05-11 (feat/fisher-rao-tilting). Stage breakdown:

- **Stage A (implemented).** NN closed-form half-plane geodesic
  (constant-speed `s(t) → phi(t)` parametrisation); adaptive
  quadrature p-value via brentq boundary finding + analytical
  Gaussian CDF integration over accept intervals (no closed form
  at interior `eta` — see Derivation Step 8); 4 selectors wired
  (Fixed, Numerical, DynamicNumerical, LearnedDynamic); production
  audit flavor `fr_dyn_numerical`.
- **Stage B (implemented, commits 6e03d0b–bf9d0c9).** General-
  purpose JAX-autodiff Fisher metric + Christoffel via
  `jax.jacrev` + diffrax `Tsit5` shooting BVP. The autodiff/
  diffrax machinery operates on a `g_fn` metric callable; the
  Gaussian metric (`_gaussian_fisher_metric`) is the only metric
  this PR exercises. Audit flavor `fr_dyn_numerical_generic` runs
  the generic-MC machinery (`_generic_tilt_fr` +
  `_generic_tilted_pvalue_fr`) against the Stage A closed-form
  path to validate correctness. NN-validated only (no non-
  Gaussian endpoint paths).
- **Stage C (pending).** NN learned-η v4 fixture + training +
  input-insensitivity diagnostic. The `fr_learned_*` audit
  flavors (`fr_learned_intp`, `fr_learned_cd_var`,
  `fr_learned_static_w`) are commented out of `_FLAVORS` in
  `scripts/run_wald_audit.py` until the v4 artifacts are
  trained; the `_build_cell` branches stay so they're easy to
  re-enable.
- **Stage D (pending).** Smoothness comparison + headline note.

A general `ParametricFamily` interface — required for Fisher-Rao on
non-Gaussian families like Beta / Bernoulli — is deferred to a
separate refactor PR. The current `Distribution` protocol exposes
no Fisher metric; the Stage B machinery is structured so a
non-Gaussian family plugs in by passing a different `g_fn` callable
to `_fr_geodesic_numerical`.

### Generic-path performance (Stage B)

The general-purpose `_fr_geodesic_numerical` + `_generic_tilt_fr`
machinery is **wall-clock-prohibitive for production audit runs**.
Cost decomposition:

- Each FR-tilted reference requires a diffrax shooting BVP. Per-shoot
  cost on CPU: ~100-500 ms (typical ~200 ms; Newton iterates 5-10
  times, each iterate runs ~10-20 diffrax solves of the geodesic ODE).
- The FR-tilted moments `(mu_FR(eta; X), sigma_FR(eta; X))` are
  **non-linear in the candidate replicate X** (Derivation Step 8),
  so each MC replicate within `_generic_tilted_pvalue_fr` requires
  its own shoot. At default `n_mc = 200`: ~200 × ~200 ms ≈ ~40 s per
  p-value evaluation.
- A dynamic-η CI inverts the p-value via brentq + dynamic-η scan over
  a 401-θ grid (default `Config.fast()`); even with brentq pruning
  the order of magnitude is hours per CI.

**Measured wall-clock.** A single `fr_dyn_numerical_generic` CI at
`w=0.5, data=[0.5]` did **not** return within 10 minutes (measurement
attempted with a 600 s timeout; earlier 120 s+ runs on smaller
variants also exceeded their budgets). The cost characterisation is
therefore **"> 10 minutes per single CI"**. Even **post finding-#3
fix** (the `t ∈ {0, 1}` BVP shortcut that skips the shoot entirely at
exact endpoints), production audit cells remain in the multi-minutes-
per-CI regime: the shortcut only fires at endpoint η, and the
dynamic-η scan spends almost all its budget at interior η values
where the BVP is unavoidable.

**Implication.** The `fr_dyn_numerical_generic` audit flavor is
**math-validation infrastructure, not for production audits.** The
load-bearing correctness gate is the atol-1e-7 closed-form-match test
in `TestFisherRaoGenericTilt`
(`tests/properties/test_fisher_rao_invariants.py`), which validates
the Stage B autodiff/diffrax pipeline against the Stage A closed-form
half-plane formula without an MC sampling layer. The slow-marked
coverage-and-width parity test in
`tests/regression/test_fr_generic_matches_closed_form.py` is a sanity
check that runs only with `pytest -m slow`. The closed-form-geodesic
+ adaptive-brentq-quadrature path (`fr_dyn_numerical`, Stage A) is
the production-precision route.
