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
  rescaled-coordinate endpoints; the arc is parameterised by angle
  linear in `t` (constant-speed geodesic in arc-length).

The closed-form *distance* is given in Costa et al. 2015 Eq. 12 (=
Pinele et al. 2020 Eq. 22):

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

Full step-by-step derivation lives at
`audit/tier2/fisher_rao_derivation.md` (438 lines, ODE-cross-checked
at atol 7.6e-14). Summary:

**Step 1 — Fisher metric on `N(mu, sigma^2)`.** The Fisher
information matrix in coordinates `(mu, sigma)` is
`diag(1/sigma^2, 2/sigma^2)`, giving the line element
`ds^2 = dmu^2/sigma^2 + 2 dsigma^2/sigma^2`.

**Step 2 — Conformal rescaling to the Poincaré half-plane.**
Substituting `u = mu/sqrt(2)` gives
`ds^2 = (2/sigma^2) (du^2 + dsigma^2) = 2 ds_P^2`,
where `ds_P^2 := (du^2 + dsigma^2)/sigma^2` is the standard
Poincaré half-plane metric of constant Gaussian curvature `K_P = -1`.
Geodesics in `(u, sigma)` are geodesics in the original `(mu, sigma)`
coordinates (the rescaling is a homothety; geodesic equations are
invariant under constant rescaling of the metric).

**Step 3 — Closed-form geodesic.** Standard result (Beltrami; see
Amari & Nagaoka 2000 §3.4). Two cases:

*Case A — Vertical (mu_p = mu_q).* The geodesic stays at fixed mu
and the sigma path is the geometric mean:

```
mu(eta)    = mu_p
sigma(eta) = sigma_p * (sigma_q / sigma_p) ** eta
```

The Poincaré arc-length is `|log(sigma_q / sigma_p)|`, so the
Fisher distance is `d_FR = sqrt(2) |log(sigma_q / sigma_p)|`.

*Case B — Semicircle (mu_p != mu_q).* The unique geodesic is the
Euclidean semicircle in `(u, sigma)` centred at `(u_c, 0)` with
radius `R`:

```
u_c = (u_q^2 - u_p^2 + sigma_q^2 - sigma_p^2) / (2 (u_q - u_p))
R   = sqrt((u_p - u_c)^2 + sigma_p^2)
```

Parametrise the polar angle `t in (0, pi)`:
`u(t) = u_c + R cos(t)`, `sigma(t) = R sin(t)`. The Poincaré speed
along this curve is `ds_P/dt = 1/sin(t)`, with antiderivative
`log tan(t/2)`. The arc-length-normalised parametrisation in eta is

```
log tan(t(eta)/2) = (1 - eta) log tan(t_p/2) + eta log tan(t_q/2)
t(eta)            = 2 arctan(exp((1-eta) log tan(t_p/2) + eta log tan(t_q/2)))
u(eta)            = u_c + R cos(t(eta))
sigma(eta)        = R sin(t(eta))
mu(eta)           = sqrt(2) u(eta)
```

with `t_p = atan2(sigma_p, u_p - u_c)`, `t_q = atan2(sigma_q, u_q - u_c)`.

**Step 4 — Tilted distribution.** The geodesic stays in the
Gaussian family because the manifold *is* the Gaussian family:
`Q_eta = N(mu(eta), sigma(eta)^2)`. Contrast with `mixture`
(leaves the Gaussian family) and matches `power_law` / `ot`
(stay Gaussian on this sandbox).

**Step 5 — Tilted p-value.**

*Wald.* eta-independent (the MLE-based statistic ignores the
prior). Same closed form as `power_law` / `ot` Wald branches:
`p_Wald(theta) = 2(1 - Phi(|D - theta|/sigma))`.

*WALDO.* `Q_eta` is Gaussian with explicit `(mu(eta), sigma(eta))`,
so the standard Phi-pair formula applies:

```
a_FR(theta) = |mu(eta) - theta| / sigma(eta)
b_FR(theta) = (mu_0 - theta) / sigma_0       (prior z-score)
p(theta; eta) = Phi(b_FR - a_FR) + Phi(-a_FR - b_FR)
```

Note: WALDO's `b` retains its bare-WALDO meaning (the prior is
fixed); the FR tilting only re-parametrises the posterior along
the geodesic. Contrast with `power_law`, where both `a` and `b`
pick up eta-dependent factors because the e-geodesic re-weights
the prior log-density.

**Step 6 — Admissible range.** `eta in [0, 1]` is always valid:
the half-plane is geodesically complete in `sigma > 0` and both
endpoints lie in the open region. No analogue of `power_law`'s
`denom > 0` clamp.

**Branch threshold.** The vertical case is selected for
`|u_p - u_q| < 1e-12` in the implementation. The semicircle branch
divides by `(u_q - u_p)` and so loses precision below this
threshold; the vertical formula is exact in the limit.

**Geometric-mean signature.** The vertical case gives
`sigma(0.5) = sqrt(sigma_p * sigma_q)` (geometric mean) — the
Fisher-Rao fingerprint that distinguishes it from `ot` (which
gives the *arithmetic* mean `(sigma_p + sigma_q)/2`). For the
fixed-variance sub-family (`sigma_p = sigma_q`) FR and OT
coincide; on the full 2D manifold they differ globally
(Takatsu 2011: W2 has *non-negative* sectional curvature, FR has
constant *negative* curvature).

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
- **Non-Gaussian likelihood / prior.** This stub will land Gaussian-
  only first; non-Gaussian endpoints raise `NotImplementedError`,
  matching `power_law`'s discipline. A general `ParametricFamily`
  interface (with explicit Fisher metric for non-Gaussian families,
  e.g. Beta, Bernoulli) is a follow-up refactor.

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

- Rao, C. R. "Information and the accuracy attainable in the
  estimation of statistical parameters." *Bull. Calcutta Math. Soc.*
  37 (1945): 81–91. — Originated the Fisher information metric and
  the Rao distance.
- Atkinson, C., Mitchell, A. F. S. "Rao's distance measure." *Sankhya
  A* 43 (1981): 345–365. — Standard reference for closed-form Rao
  distances in elementary families; gives the half-plane formula.
- Skovgaard, L. T. "A Riemannian geometry of the multivariate normal
  model." *Scand. J. Stat.* 11 (1984): 211–223. — Multivariate
  generalisation; widely cited Riemannian-geometry treatment.
- Calvo, M., Oller, J. M. "An explicit solution of information
  geodesic equations for the multivariate normal model." *Statistics
  & Decisions* 9 (1991): 119–138. — Closed-form geodesics for
  multivariate normals; collapses to half-plane arcs in 1D.
- Costa, S. I. R., Santos, S. A., Strapasson, J. E. "Fisher
  information distance: a geometrical reading." *Discrete Appl.
  Math.* 197 (2015): 59–69. — Hyperbolic-geometry reading of Fisher-
  Rao; Eq. 12 gives the canonical closed-form distance.
- Pinele, J., Strapasson, J. E., Costa, S. I. R. "The Fisher–Rao
  distance between multivariate normal distributions: special cases,
  bounds and applications." *Entropy* 22 (2020): 404. — Consolidates
  known closed-form sub-cases; cross-citation for the formula above.
- Cencov, N. N. *Statistical Decision Rules and Optimal Inference.*
  AMS, 1982. — Uniqueness theorem: Fisher metric is the unique
  Riemannian metric invariant under sufficient statistics.

### Closely related (information geometry)

- Amari, S., Nagaoka, H. *Methods of Information Geometry.* AMS /
  Oxford, 2000. — α-connections including Levi-Civita (α=0).
  Chapter 2 covers Gaussian-family Fisher metric explicitly.
- Amari, S. *Information Geometry and Its Applications.* Springer,
  2016. — Modern textbook; Section 2.5 covers the e/m/Levi-Civita
  trichotomy.

### Contrasting (Wasserstein vs Fisher-Rao)

- Takatsu, A. "Wasserstein geometry of Gaussian measures." *Osaka J.
  Math.* 48 (2011): 1005–1026. — W2 geometry on Gaussians has
  *non-negative* sectional curvature, in stark contrast to Fisher-
  Rao's constant *negative* curvature; cleanest single citation for
  "OT and Fisher-Rao geodesics differ on Gaussians".
- Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.-X. "An
  interpolating distance between optimal transport and Fisher–Rao
  metrics." *Found. Comput. Math.* 18 (2018): 1–44. — Constructs an
  explicit one-parameter family connecting W2 and Fisher-Rao,
  demonstrating they are genuinely distinct geometries.
- Olkin, I., Pukelsheim, F. "The distance between two random vectors
  with given dispersion matrices." *Lin. Alg. Appl.* 48 (1982):
  257–263. — W2 closed form on Gaussians; counter-reference to
  Atkinson & Mitchell.
- Miyamoto, H. K., Meneghetti, F. C., Pinele, J., Costa, S. I. R.
  "On closed-form expressions for the Fisher–Rao distance."
  *Information Geometry* (2024). — Recent survey; useful for
  numerical sub-cases.
- Nielsen, F. "A simple approximation method for the Fisher–Rao
  distance between multivariate normal distributions." *Entropy* 25
  (2023): 654. — Practical numerical methods.

## Links

- Implementation: `src/frasian/tilting/fisher_rao.py`
- Property tests: `tests/properties/test_fisher_rao_invariants.py`
- Illustration: `src/frasian/experiments/illustrations/fisher_rao_demo.py`
- Generated figure: `output/illustrations/fisher_rao_demo.png`
- Derivation: `audit/tier2/fisher_rao_derivation.md`

## Status notes

Implemented (Gaussian-only). Compare empirically against `ot` on
the smoothness diagnostic to see whether the curvature-aware
Fisher-Rao path beats the straight-in-`(mu, sigma)` W2 path on
Lipschitz / TV / discontinuity metrics.

A general `ParametricFamily` interface — required for Fisher-Rao on
non-Gaussian families like Beta / Bernoulli — is deferred to a
separate refactor PR. The current `Distribution` protocol exposes
no Fisher metric.
