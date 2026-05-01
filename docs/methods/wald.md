# wald

> Status: `implemented`

## Summary

Pure-likelihood Wald statistic and confidence interval for the
Normal-Normal model — the prior-ignoring frequentist baseline that every
other method in the framework is compared against.

## Motivation

Wald is the cheapest, most-cited interval estimator for a Normal mean and
the natural η = 1 limit of the power-law tilting family. Including it as
a first-class `TestStatistic` rather than an inline reference value lets
the cross-product runner treat it on equal footing with WALDO/LRT/etc.,
which is the precondition for the smoothness diagnostic to compare them
under the same benchmarks.

## Definition

For a single observation `D` of a Normal location parameter `theta`:

  tau_Wald(theta) = ((D - theta) / sigma)^2,
  CI:               D ± z_{1 - alpha/2} * sigma.

Equivalently, the two-sided p-value is `2 * (1 - Phi(|D - theta| / sigma))`.

## Derivation

Under H0: `theta = theta0`, `(D - theta0)/sigma ~ N(0, 1)`, so
`tau_Wald ~ chi^2_1`. Inverting the level-α test gives the standard
`z_{1-alpha/2}` half-width interval. (Standard textbook derivation.)

## Predicted behavior

- p-value at `theta = D` (the MLE) equals 1.
- p-value strictly decreases with `|D - theta| / sigma`.
- Coverage at the nominal level is exact for any `theta_true`.
- CI width is constant in the data: `2 * z_{1-alpha/2} * sigma`.
- Recovered as the η → 1 limit of `power_law` tilting.

## Failure modes

None for the Normal location family with known `sigma`. (Conditions of
applicability — known variance, no nuisance parameters — are invariants
of the model itself.)

## Invariants

- p-value lies in [0, 1] for all inputs.
- p-value at `D` equals 1 (mode property).
- p-value is monotone decreasing in `|theta - D|`.
- Under H0 (data ~ N(theta_true, sigma^2)), p-values are Uniform[0, 1]
  (KS test in `test_statistic_invariants.py::TestWaldUniformPvalueUnderH0`).
- Acceptance region `[lo, hi]` is symmetric about `theta0` with width
  `2 * z_{1-alpha/2} * sigma`.

## Literature

- A. Wald. "Tests of statistical hypotheses concerning several parameters
  when the number of observations is large." *Trans. Amer. Math. Soc.* 54
  (1943): 426–482.
- George Casella and Roger Berger. *Statistical Inference*. 2nd ed., 2002.
  Chapter 8 (hypothesis tests and CIs).

## Links

- Implementation: `src/frasian/statistics/wald.py`
- Regression tests: `tests/regression/test_waldo_pvalue.py::TestWaldStatistic`
- Property tests: `tests/properties/test_statistic_invariants.py::TestWaldInvariants`
- Statistical test (L3): `tests/properties/test_statistic_invariants.py::TestWaldUniformPvalueUnderH0`
- Illustration: `src/frasian/experiments/illustrations/wald_demo.py`

## Status notes

The `acceptance_region` returns the D-space interval (`D = theta0 ± z*sigma`).
LRT / signed-root / Bartlett variants are scheduled stubs (see
`docs/methods/{lrt,signed_root,bartlett}.md`).

`WaldStatistic.accepts_tilting(tilting)` returns `True` only when
`tilting.name == "identity"` — Wald ignores the prior, so non-identity
cells `(power_law, wald)`, `(ot_normal, wald)`, etc. are degenerate
duplicates of `(identity, wald)`. The runner skips them and records
`status="incompatible"` in the manifest.
