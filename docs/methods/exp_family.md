# exp_family

> Status: `stub`

## Summary

Exponential-family interpolation in natural-parameter space. For two
distributions in the same exponential family with natural parameters
`theta_a, theta_b`, the canonical path is `theta_t = (1-t) theta_a + t theta_b`.
On the conjugate-Normal sandbox this yields a Gaussian whose precision
interpolates linearly. Distinct from `power_law` (which interpolates
the prior by tempering) and from the geometric schemes (`ot_normal`,
`geodesic_normal`).

## Motivation

Exponential-family natural-parameter interpolation is the canonical
"smoothest" path in the family-respecting sense — it is what the
information geometry of exponential families *is*, and it generalises
trivially beyond Gaussian. Including it lets the framework compare
"interpolation in the natural manifold" against the
mass-displacement (`ot_normal`) and Fisher-Rao (`geodesic_normal`)
geodesics. In the Gaussian case all three are similar; the stub
documents this so that future non-Gaussian work has a clean entry point.

## Definition

Gaussian natural parameters: `eta = (mu / sigma^2, -1/(2 sigma^2))`.
The interpolation `eta_t = (1 - t) eta_a + t eta_b` maps back to
`(mu_t, sigma_t)` via the inverse of that mapping. The framework's
tilting parameter is identified with `t in [0, 1]`.

## Derivation

To be filled in by `/derive exp_family`. Should establish (a) the
natural parameter map for Gaussians, (b) the back-substitution to
`(mu, sigma)`, (c) the relation to the precision-weighted average that
defines the conjugate posterior — closing the loop with the WALDO
identity element.

## Predicted behavior

- Smooth `eta*(|Delta|)` curve with no clamp.
- Quantitatively similar to `geodesic_normal` on the canonical
  sandbox (both respect the exponential-family geometry); should
  *differ* on non-Gaussian models.
- Coverage at nominal level (no obvious calibration loss).

## Failure modes

- The natural-parameter map has a `sigma -> 0` singularity (precision
  -> infinity). The path must avoid this for any `t in [0, 1]`.
- For non-Gaussian future models, the mapping may not have a closed
  form — a numerical solver may be required.

## Invariants

- `tilt(eta=eta_identity)` matches the chosen reference (TBD by `/derive`).
- Output is in the exponential family (Gaussian on the sandbox).
- Path is smooth: derivative of `(mu_t, sigma_t)` w.r.t. `t` is
  Lipschitz on `[0, 1]`.
- Reduces to a precision-weighted average when `eta_a` and `eta_b` are
  the prior and likelihood natural parameters.

## Literature

- Amari, S. "Differential-Geometrical Methods in Statistics."
  *Lecture Notes in Statistics* 28, Springer, 1985. (Foundational text
  on exp-family geometry.)
- Brown, L. D. *Fundamentals of Statistical Exponential Families.*
  IMS Lecture Notes 9, 1986.
- Diaconis, P., Ylvisaker, D. "Conjugate priors for exponential
  families." *Ann. Statist.* 7 (1979): 269-281.

## Links

- Implementation: `src/frasian/tilting/exp_family.py` (stub)
- Property tests: `tests/properties/test_exp_family_invariants.py`
                  (skipped)
- Illustration:   TBD

## Status notes

Stub — this scheme's advantage over `geodesic_normal` is non-obvious
on Gaussians; the comparison is informative but expected to be a wash.
The real value is as the Gaussian *sanity check* before extending the
framework to non-Gaussian models.
