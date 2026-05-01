# width_experiment

> Status: `implemented`

## Summary

Mean CI width on a `(theta_true, w)` grid. The frequentist *efficiency*
counterpart to the coverage diagnostic: at fixed coverage, narrower is
better. The substrate the smoothness experiment reads to detect sharp
regime transitions in optimal `eta*(|Delta|)` for the power-law family.

## Motivation

Coverage alone does not distinguish methods that all hit 95%. Width
measures power: WALDO is narrower than Wald in the prior-aligned regime
and wider under conflict. The legacy framework's central numerical
tables (`CLAUDE.md` "CI Widths" section) live in this experiment.

## Definition

For each `(theta_i, w_j)`:

  Generate D_{i,j,k} ~ N(theta_i, sigma) for k = 1..n_reps
  (lo, hi)        = tilting.confidence_interval(alpha, [D_{i,j,k}], model, prior_j, statistic)
  width_{i,j,k}   = hi - lo
  mean_width_{i,j} = mean_k width_{i,j,k}
  width_se_{i,j}   = std_k(width_{i,j,k}) / sqrt(n_reps)

For dynamic-η tiltings the CI is the convex hull of the (possibly
multi-region) crossings; multi-region count is not propagated through
this experiment (the original `dynamic_ci` experiment exposed it
explicitly, but the Phase-4 refactor folded those measurements into
`coverage` / `width` and dropped the region-count surface).

## Derivation

Width is a deterministic function of `D` for any conjugate-Normal
statistic. The Monte-Carlo expectation `E_{D | theta_true}[ width ]`
estimated here approximates the *frequentist average width* — the
quantity used in the "efficiency" comparisons against Wald.

## Predicted behavior

- Wald: width is constant in `D`, so `width_se ≈ 0` (numerical noise);
  the width itself is constant in both `theta_true` and `w`.
- WALDO: width depends on `D` and `w`; under no conflict (small
  `|Delta|`) the width is *less* than Wald, under strong conflict
  (large `|Delta|`) it is *greater*.
- The minimum mean-width over `eta` for each `(theta_true, w)` is what
  the optimal tilting solver picks; the `smoothness` experiment reads
  this surface.

## Failure modes

- Statistics whose `confidence_interval` raises `NotImplementedError`
  yield `NaN` widths; preserved by the diagnostic.
- Edge `w` values make `confidence_interval` slow due to wider
  bracket-doubling; not a correctness bug, but expect longer runtimes.

## Invariants

- `mean_width > 0` everywhere finite.
- `width_se >= 0`.
- Wald `mean_width = 2 * z_{1-alpha/2} * sigma` exactly, independent of
  `(theta_true, w)`.

## Literature

- Carlin, B. P., Louis, T. A. *Bayesian Methods for Data Analysis*.
  CRC Press, 3rd ed., 2008. (CI-width / power comparisons.)

## Links

- Implementation: `src/frasian/experiments/width.py`
- Diagnostic:     `src/frasian/diagnostics/width_table.py`
- Tests:          `tests/experiments/test_width_experiment.py`

## Status notes

Each cell's CI is computed via `tilting.confidence_interval(...)`: the
tilting owns the η-selector. `(identity, waldo)` produces the plain
WALDO width; `(power_law[dynamic_numerical], waldo)` produces the
Dynamic-WALDO width (formerly the central column of `dynamic_ci`).
Cells gated incompatible by `accepts_tilting` are recorded with
`status="incompatible"` and skipped.
