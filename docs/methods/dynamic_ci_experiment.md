# dynamic_ci_experiment

> Status: `implemented`

## Summary

Empirical behavior of dynamic-η confidence intervals. For each cell of
the `(TiltingScheme x TestStatistic)` cross-product and each grid point
`(theta_true, w)`, simulates `n_reps` data values D ~ N(theta_true,
sigma), computes the *dynamic* CI (η chosen as a function of θ via the
numerical eta-selector and interpolated across the θ scan), and
records empirical coverage, mean total width, and mean region count.
The diagnostic emits all three on one figure so the trade-offs of the
adaptive scheme are visible side-by-side.

## Motivation

The `coverage` and `width` experiments use a single fixed η per cell
(currently `eta_identity`). The legacy framework's empirical
contribution was the *dynamic* extension: η = η*(|Δ(θ)|) varies with
the test point, producing tighter CIs at low conflict and recovering
Wald-like behavior at high conflict. Whether this scheme preserves
frequentist coverage is the question this experiment answers; whether
the resulting CIs are sometimes multimodal is the surprise the
region-count metric reports.

## Definition

For each `(theta_true, w)` and each replicate D_k ~ N(theta_true, sigma):

  R_k = dynamic_tilted_confidence_interval(alpha, D_k, model, prior,
                                             statistic_name, eta_selector)
  -> (regions_k, total_width_k, n_regions_k)

  coverage(theta_true, w)     = mean_k [ theta_true in union(regions_k) ]
  mean_width(theta_true, w)   = mean_k total_width_k
  mean_regions(theta_true, w) = mean_k n_regions_k

`dynamic_tilted_confidence_interval` is implemented on the tilting
scheme; for `power_law` the algorithm is:
  1. Build a coarse |Δ| grid spanning the search range.
  2. Compute η*(|Δ|) on the coarse grid via NumericalEtaSelector.
  3. Interpolate η* to a fine θ scan.
  4. Compute the tilted p-value at each θ.
  5. Find α-crossings and refine via brentq.
  6. Stitch crossings into intervals (multiple regions possible if the
     dynamic p-value is non-monotone).

## Derivation

The dynamic p-value is `p(theta) = tilted_pvalue(theta, D, ..., eta*(|Delta(theta)|))`.
At theta = theta_hat (the dynamic mode) it equals 1; elsewhere it
decays through alpha at a rate set by η*. The confidence-region
construction is the standard inversion `{ theta : p(theta) >= alpha }`,
implemented as a grid scan + brentq refine — the same idiom as the
static `WaldoStatistic.confidence_interval` but with the per-θ η
substitution that makes the p-value non-stationary in θ.

A full treatment (including the fixed-point characterisation of the
dynamic mode, theta* = mu_{eta*(theta*)}) lives in the legacy
framework's Theorem 10 docstrings; promoting that derivation here is
an action item for `/derive dynamic_ci`.

## Predicted behavior

- Coverage at the nominal 1-α level (within MC noise). The
  Schweder-Hjort calibration of the inverted p-value preserves
  coverage even with η varying per θ.
- Width *narrower than the static WALDO CI* at low |Δ| (the
  η < 0 oversharpening regime); convergent to Wald at high |Δ|.
- Region count: usually 1; occasionally > 1 when the dynamic p-value
  is multimodal (small sigma0 + extreme D).
- For (Wald row): Wald is η-independent, so the dynamic CI = static
  Wald CI exactly. Acts as the experiment's smoothness floor.

## Failure modes

- The fine θ scan (`n_grid` default 201) may miss narrow secondary
  modes; smaller features need a finer scan.
- The coarse η lookup (`coarse_n` default 15) introduces interpolation
  error in η*(|Δ(θ)|); not a coverage concern (Brent refines
  crossings exactly) but a width-estimate concern.
- When `n_reps` is small the empirical-coverage SE is large (Wald-
  binomial); the L4 tests use small grids for speed and should not
  be read as calibration evidence.

## Invariants

- `coverage in [0, 1]`, `coverage_se >= 0`.
- `mean_width > 0` everywhere finite.
- `mean_regions >= 1.0` everywhere finite.
- For the (power_law, wald) cell, `mean_width = 2 * z * sigma` exactly
  (Wald is η-independent), and `mean_regions = 1` everywhere.
- Stub tilting cells (no `dynamic_tilted_confidence_interval`) produce
  all-NaN; the diagnostic preserves them.

## Literature

- Singh, K., Xie, M., Strawderman, W. E. "Confidence distribution,
  the frequentist distribution estimator of a parameter: a review."
  *Internat. Statist. Rev.* 81 (2013): 3-39. (Schweder-Hjort
  calibration under non-stationary p-values.)
- Cox, D. R., Reid, N. "Parameter orthogonality and approximate
  conditional inference." *J. R. Stat. Soc. B* 49 (1987): 1-39.
- Bissiri, P. G., Holmes, C. C., Walker, S. G. "A general framework
  for updating belief distributions." *J. R. Stat. Soc. B* 78 (2016).
  (Power-likelihood adaptive tempering.)

## Links

- Implementation: `src/frasian/experiments/dynamic_ci.py`
- Tilting bridge:  `src/frasian/tilting/power_law.py`
                   (`dynamic_tilted_confidence_interval`,
                    `dynamic_tilted_pvalue`)
- η-selector:     `src/frasian/tilting/eta_selectors.py`
- Diagnostic:     `src/frasian/diagnostics/dynamic_ci_table.py`
- Tests:          `tests/experiments/test_dynamic_ci_experiment.py`
- Illustration:   `src/frasian/experiments/illustrations/dynamic_ci_demo.py`

## Status notes

The dynamic-CI bridge is currently a *method on PowerLawTilting* that
specialises on `statistic.name` for the (power_law, waldo) and
(power_law, wald) combos. Step 6 stub tilting schemes do not implement
the bridge and produce all-NaN cells. Generalising via multiple
dispatch is deferred until at least one stub tilting becomes
implemented (Phase 3 next stage).
