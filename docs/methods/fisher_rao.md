# fisher_rao

> Status: `stub`

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

To be filled in by `/derive fisher_rao` once the Gaussian-only
implementation lands. Should cover (a) the Fisher metric on the
Gaussian family and the rescaling to the unit-curvature half-plane,
(b) the closed-form half-plane geodesic, (c) the constant-speed
arc-length parameterisation, and (d) the contrast with the W2
geodesic on the same endpoints.

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

- Implementation: `src/frasian/tilting/fisher_rao.py` (stub)
- Property tests: `tests/properties/test_fisher_rao_invariants.py`
                  (skipped)
- Illustration: TBD

## Status notes

Stub — Gaussian-only implementation lands first via `/propose-method
fisher_rao`. Compare empirically against `ot` on the smoothness
diagnostic to see whether the curvature-aware Fisher-Rao path beats
the straight-in-`(mu, sigma)` W2 path on Lipschitz / TV /
discontinuity metrics.

A general `ParametricFamily` interface — required for Fisher-Rao on
non-Gaussian families like Beta / Bernoulli — is deferred to a
separate refactor PR. The current `Distribution` protocol exposes
no Fisher metric.
