# waldo

> Status: `implemented`

## Summary

WALDO (Weighted Accurate Likelihood-free inference via Diagnostic
Orderings) statistic for the Normal-Normal model: a posterior-mean-based
test statistic whose p-value has the closed form
`p(theta) = Phi(b - a) + Phi(-a - b)`. The Bayesian-frequentist hybrid
that the framework's existing experiments centre on.

## Motivation

WALDO replaces the MLE in the Wald statistic with the posterior mean,
borrowing prior information for shorter CIs in the data-poor regime
without sacrificing frequentist coverage at any single `theta_true`. It
is the η = 0 element of the power-law tilting family — the centre of the
research, and the benchmark every alternative tilting/statistic
combination must beat on the smoothness diagnostic in Step 5.

## Definition

For the conjugate Normal-Normal model (see `normal_normal.md`):

  a(theta) = |mu_n - theta| / (w * sigma)
  b(theta) = (1 - w) * (mu0 - theta) / (w * sigma)
  p(theta) = Phi(b - a) + Phi(-a - b),

with the test statistic `tau_WALDO = (mu_n - theta)^2 / sigma_n^2`.

## Derivation

Theorem 3 in the legacy framework (port retained byte-for-byte; see
`tests/regression/test_waldo_pvalue.py::TestWaldoPvalueMatchesLegacy`):
the p-value is the probability under H0 that
`tau_WALDO(theta_true)` exceeds the observed `tau_WALDO(theta)`.
The decomposition into `Phi(b-a) + Phi(-a-b)` follows from the fact that
under H0, `mu_n - theta` is Gaussian with mean
`b_eff = (1-w)(mu0 - theta)` and variance `w^2 * sigma^2`. The two
addends correspond to the upper and lower tails of the squared form.

A full algebraic derivation is scheduled to live in
`docs/derivations/theorem_3_waldo_pvalue.md` once the `deriver` agent is
wired up in the Step-7 workflow integration.

## Predicted behavior

- p-value equals 1 at `theta = mu_n` (the WALDO mode).
- Coverage is exact under any `theta_true` (frequentist calibration).
- CIs are *narrower* than Wald when the prior is informative and there is
  no conflict; *wider* when the prior conflicts strongly with the data.
- The Lipschitz constant of `p(theta)` scales as `1 / (w * sigma)` —
  small `w` (strong prior) implies steep p-values, the basis for the
  Step-5 smoothness diagnostic.

## Failure modes

- Steep transitions when `w` is small (near-singular Lipschitz constant).
  Not a numerical bug, but a behavioural property of interest.
- Non-Gaussian priors raise `NotImplementedError`.

## Invariants

- p-value lies in [0, 1].
- p-value at `theta = mu_n` equals 1 exactly.
- Under H0, p-values are Uniform[0, 1] (statistical L3, scheduled to be
  added once we have a fast simulator at `experiments/coverage`).
- `pvalue` is continuous in `theta`: no infinite jumps on a fine grid.

## Literature

- A. Masserano, T. Dorigo, R. Izbicki, M. Kuusela, A. B. Lee. "Simulator-
  based inference with WALDO." *AISTATS 2023*.
- D. R. Cox and N. Reid. "Parameter orthogonality and approximate
  conditional inference." *J. Royal Stat. Soc. B*, 49 (1987): 1–39.
  (Background on conditional / hybrid inference.)

## Links

- Implementation: `src/frasian/statistics/waldo.py`
- Regression tests: `tests/regression/test_waldo_pvalue.py`
- Property tests: `tests/properties/test_statistic_invariants.py::TestWaldoInvariants`
- Illustration: `src/frasian/experiments/illustrations/waldo_demo.py`

## Status notes

`acceptance_region` raises `NotImplementedError` until Step 4's
CoverageExperiment provides the numerical inversion; the dual problem in
D-space requires solving the WALDO p-value implicit equation, which the
coverage runner already needs.
