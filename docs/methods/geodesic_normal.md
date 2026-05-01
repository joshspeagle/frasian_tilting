# geodesic_normal

> Status: `stub`

## Summary

Fisher-Rao (information-geometric) geodesic on the univariate Gaussian
manifold. Distinct from `ot_normal` (W2 geodesic) — Fisher-Rao
respects the *information* structure rather than mass displacement;
the two coincide only when sigma_a = sigma_b. Candidate alternative
to power-law tilting on the smoothness diagnostic.

## Motivation

The Gaussian family with the Fisher metric is the upper half-plane in
hyperbolic geometry; geodesics are arcs of circles perpendicular to
the boundary `sigma = 0`. This produces a smooth, naturally
non-clamping path between distributions that respects the geometry
under which "distance between Gaussians" is most often quoted in the
information-geometry literature. The hypothesis is the same as for
`ot_normal`: replacing the power-law clamp with a geodesic interpolant
should drop the smoothness-diagnostic Lipschitz value by an order of
magnitude.

## Definition

For two Gaussians a = N(mu_a, sigma_a^2) and b = N(mu_b, sigma_b^2),
parameterise via `(mu/sqrt(2), sigma)` so the half-plane has unit
Gaussian-curvature -1. The geodesic at `t in [0, 1]` then satisfies
the standard half-plane geodesic equations (vertical line if
`mu_a = mu_b`; circular arc otherwise) — closed-form parameterisation
in Costa et al. (2015) Eq. 12.

The framework maps `eta in [0, 1]` to the geodesic parameter `t`;
identity `eta = 0` is the chosen reference (TBD by `/derive`).

## Derivation

To be filled in by `/derive geodesic_normal`. Should cover (a) the
Fisher metric on the Gaussian family, (b) the closed-form geodesic in
the (mu, sigma) half-plane, (c) how the mapping `eta -> t` relates to
the WALDO identity element.

## Predicted behavior

- Smooth `eta*(|Delta|)`, no clamp.
- Different from `ot_normal` when sigma_n != sigma_likelihood (i.e.
  when the prior is informative); equal otherwise.
- Coverage at nominal level; CI width similar to `ot_normal` to
  leading order in `|Delta|`.

## Failure modes

- Numerical issues near `sigma -> 0` (the boundary of hyperbolic
  half-plane). Implementation must guard.
- The hyperbolic-geodesic closed form requires care with branch
  selection when the path goes "around" a high-curvature region.

## Invariants

- `tilt(eta=eta_identity)` returns the chosen reference.
- Output is Gaussian with `sigma > 0`.
- Path is differentiable in `eta` (`smoothness` Lipschitz < 1 expected).
- Reduces to a vertical-line interpolation when `mu_a = mu_b`.

## Literature

- Costa, S. I. R., Santos, S. A., Strapasson, J. E. "Fisher information
  distance: a geometrical reading." *Discrete Applied Math.* 197
  (2015): 59-69.
- Amari, S. *Information Geometry and Its Applications.* Springer, 2016.
  Chapter 1 covers the Gaussian-family Fisher metric.
- Calvo, M., Oller, J. M. "An explicit solution of information
  geodesic equations for the multivariate normal model." *Statistics
  & Decisions* 9 (1991): 119-138.

## Links

- Implementation: `src/frasian/tilting/geodesic_normal.py` (stub)
- Property tests: `tests/properties/test_geodesic_normal_invariants.py`
                  (skipped)
- Illustration:   TBD

## Status notes

Stub — `/propose-method geodesic_normal` will fill the derivation +
implementation. Compare empirically against `ot_normal` on the
`smoothness` diagnostic to see which (if either) wins on the
Lipschitz / TV / discontinuity metrics.
