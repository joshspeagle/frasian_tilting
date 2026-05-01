# ot_normal

> Status: `stub`

## Summary

Wasserstein-2 geodesic interpolation between the prior and the posterior
on the univariate Gaussian family. Candidate replacement for power-law
tilting; the central hypothesis is that this geodesic produces a
smoother `eta*(|Delta|)` curve and avoids the kink at low `|Delta|`
that the `smoothness` diagnostic detects in `power_law`.

## Motivation

Power-law tilting clamps against `eta = -w/(1-w)` at low `|Delta|`
(see `docs/methods/smoothness_experiment.md`). The clamp has nothing
to do with the data — it is an artefact of the tempering
parameterisation. Optimal-transport interpolation parameterises the
path between distributions geometrically (in mass-displacement
coordinates), with no boundary clamp. If the smoothness diagnostic
shows lower Lipschitz / TV / discontinuity-count for `ot_normal` than
for `power_law`, the framework's central hypothesis is supported.

## Definition

For two univariate Gaussians a = N(mu_a, sigma_a^2) and b = N(mu_b, sigma_b^2),
the W2 geodesic at `t in [0, 1]` is the Gaussian

  mu_t    = (1 - t) * mu_a + t * mu_b
  sigma_t = (1 - t) * sigma_a + t * sigma_b.

The framework's tilting parameterisation maps `eta` to a point on this
path. The natural identity is `eta = 0` -> reference posterior; the
mapping of `eta = 1` (Wald limit) requires the path to terminate at the
likelihood-induced Gaussian N(D, sigma^2). Whether the path goes
prior->posterior, posterior->likelihood, or prior->likelihood is a
design decision — `/derive` is expected to settle this.

## Derivation

To be filled in by `/derive ot_normal`. Expected to cover (a) why the
W2 geodesic on Gaussians is linear in `(mu, sigma)`, and (b) how the
chosen identity element relates to WALDO's `(mu_n, sigma_n)`.

## Predicted behavior

- Smooth `eta*(|Delta|)` curve with no boundary clamp; Lipschitz value
  on the smoothness diagnostic should be at least an order of
  magnitude smaller than `power_law`.
- At low `|Delta|`, `eta*` near posterior (eta -> 0); at high `|Delta|`,
  `eta*` toward likelihood-tilted limit.
- Coverage at the nominal level (no obvious reason to break
  calibration; verify in CoverageExperiment).

## Failure modes

- The W2 geodesic between two Gaussians of very different variance can
  produce a sigma_t that crosses zero if sigma_a and sigma_b are
  *signed* — the parameterisation must constrain `sigma > 0`.
- Identity-element ambiguity: depending on which two endpoints we
  parameterise, `eta = 0` may or may not coincide with WALDO. Tests
  must lock the choice.

## Invariants

- `tilt(eta=eta_identity)` returns a distribution numerically equal to
  the chosen reference (TBD).
- `tilt(...)` is continuous in `eta` (no clamps, no NaNs in the
  admissible range).
- Output is a Gaussian N(mu, sigma^2) with `sigma > 0`.
- Smoothness invariant: the `smoothness` diagnostic must report
  `lipschitz_eta < 1.0` on the canonical sandbox (claim).

## Literature

- Cuturi, M. "Sinkhorn distances: lightspeed computation of optimal
  transport." *NeurIPS 2013*.
- Peyre, G., Cuturi, M. "Computational optimal transport." *Foundations
  and Trends in ML* 11 (2019). (Comprehensive reference for
  W2-on-Gaussian closed forms.)
- Olkin, I., Pukelsheim, F. "The distance between two random vectors
  with given dispersion matrices." *Lin. Alg. Appl.* 48 (1982): 257-263.
  (Earliest derivation of the W2-on-Gaussian formula.)

## Links

- Implementation: `src/frasian/tilting/ot_normal.py` (stub)
- Property tests: `tests/properties/test_ot_normal_invariants.py` (skipped)
- Illustration:   `src/frasian/experiments/illustrations/ot_normal_demo.py`
                  (TBD)

## Status notes

Stub — implementation lands via `/propose-method ot_normal`.
