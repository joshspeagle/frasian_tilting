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
  CI_{i,j,k}      = statistic.confidence_interval(D_{i,j,k}, model, prior_j, alpha)
  coverage_{i,j}  = mean_k [ theta_i in CI_{i,j,k} ]
  coverage_se_{i,j} = sqrt( coverage(1 - coverage) / n_reps )    (Wald-binomial SE)

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
  `D`; the local Lipschitz behavior of the WALDO p-value (Step-5 study)
  shows up here as variability between replicates.

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

The Tilting dimension currently records `eta = scheme.param_space.eta_identity`
for every cell; this makes coverage independent of the tilting scheme. Step
5's smoothness experiment sweeps `eta` and is where Tilting becomes load-bearing.
