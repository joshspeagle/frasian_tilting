# coverage_experiment

> Status: `implemented`

## Summary

Empirical frequentist-coverage diagnostic: the fraction of CIs (one per
replicate `D ~ N(theta_true, sigma)`) that contain `theta_true`, evaluated
on a 2D grid of `(theta_true, w)` for each cell of the
`(TiltingScheme x TestStatistic)` cross-product.

## Motivation

Coverage is the load-bearing frequentist guarantee that every method in the
framework either preserves or breaks. Wald has exact 95% coverage by
construction; the posterior credible interval breaks coverage at large
`|Delta|`; WALDO restores it. Re-running this experiment for every new
tilting / statistic is the framework's primary calibration check.

## Definition

For a fixed `(sigma, mu0)` and grid points `(theta_i, w_j)`:

  sigma0_j = sqrt(w_j / (1 - w_j)) * sigma
  Generate D_{i,j,k} ~ N(theta_i, sigma) for k = 1..n_reps
  regions_{i,j,k} = tilting.confidence_regions(alpha, [D_{i,j,k}], model, prior_j, statistic)
  hit_{i,j,k}     = any(lo <= theta_i <= hi for (lo, hi) in regions_{i,j,k})
  coverage_{i,j}  = mean_k hit_{i,j,k}
  coverage_se_{i,j} = sqrt( coverage(1 - coverage) / n_reps )    (Wald-binomial SE)

The region computation is dispatched through the **tilting**:
`IdentityTilting` and static-η `PowerLawTilting` cells return a single
region (the conventional CI); dynamic-η `PowerLawTilting` may return
multiple regions at low |Δ| where the dynamic p-value is multimodal.
**Union semantics**: a replicate is counted as covered iff θ_true lies
in any returned region — for single-region cells this matches the
standard CI containment check; for multi-region cells it honours the
actual CI structure rather than the convex hull. The uniform interface
lets `(identity, wald)`, `(identity, waldo)`, and
`(power_law[dynamic_numerical], waldo)` share one cell loop.

## Derivation

A frequentist test at level `alpha` has the property that
`P_{theta} (theta in CI(D)) >= 1 - alpha` for any true `theta`. The
empirical Monte-Carlo estimate is the sample mean over `n_reps`
replicates; the standard error is the binomial SE clamped away from
zero. (Bootstrap intervals are not used — Wald-binomial is sufficient
for this level of estimation given `n_reps >= 1000`.)

## Predicted behavior

- Wald: ~95% coverage for every `(theta_true, w)`.
- WALDO: ~95% coverage by construction, including under conflict.
- Posterior credible interval: drops below 95% as `|theta_true - mu0|`
  grows (not yet implemented as a registered statistic; would expose
  the calibration breakdown if added).

## Failure modes

- Statistics whose `confidence_interval` raises `NotImplementedError`
  produce `NaN` rows; the diagnostic preserves them rather than masking.
- Strong-prior corners (`w → 0`) produce CIs that depend acutely on
  `D`; the local Lipschitz behavior of the WALDO p-value (see the
  `smoothness` experiment) shows up here as variability between replicates.

## Invariants

- `coverage in [0, 1]` everywhere finite.
- `coverage_se >= 0`.
- For the Wald cell, coverage is independent of `w` (Wald ignores prior).
- Under the nominal `1 - alpha` level, the average over the grid is
  within `~3 * coverage_se` of the nominal level for any properly
  calibrated statistic.

## Literature

- Brown, L. D., Cai, T. T., DasGupta, A. "Interval estimation for a
  binomial proportion." *Statistical Science* 16 (2001): 101–133.
  (Coverage-rate methodology and Wald-binomial SE caveats.)

## Links

- Implementation: `src/frasian/experiments/coverage.py`
- Diagnostic:     `src/frasian/diagnostics/coverage_table.py`
- Tests:          `tests/experiments/test_coverage_experiment.py`

## Status notes

The Tilting dimension is load-bearing: each cell's CI is computed via
`tilting.confidence_regions(...)` (the multi-region-aware interface),
so `(identity, waldo)` produces the plain WALDO CI while
`(power_law[dynamic_numerical], waldo)` produces the Dynamic-WALDO CI
(η*(|Δ|) per θ). Coverage uses **union-of-regions** semantics: a
replicate is "covered" iff θ_true lies in *any* of the returned
regions. For single-region cells (Wald, plain WALDO, all static-η
power_law cells) the union check coincides with the standard
single-interval check; for the dynamic-η power_law cell at low |Δ|,
where the dynamic p-value is multimodal, the union check honours the
actual CI structure rather than the convex hull. Cells gated
incompatible by `statistic.accepts_tilting(tilting)` are recorded in
the manifest with `status="incompatible"` and skip their `run_cell`
entirely.
