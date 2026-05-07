# wald

> Status: `implemented`

## Summary

Pure-likelihood Wald statistic and confidence interval. Closed-form
fast path on the Normal-Normal sandbox; **generic numerical default**
(τ = (mle − θ)² · I(θ), χ²₁ calibration) for any `Model` exposing
`mle(data)` and `fisher_information(theta)` — Bernoulli + Beta is the
existing smoke target. The prior-ignoring frequentist baseline that
every other method in the framework is compared against.

## Motivation

Wald is the cheapest, most-cited interval estimator for a Normal mean and
the natural η = 1 limit of the power-law tilting family. Including it as
a first-class `TestStatistic` rather than an inline reference value lets
the cross-product runner treat it on equal footing with WALDO/LRT/etc.,
which is the precondition for the smoothness diagnostic to compare them
under the same benchmarks.

## Definition

### Closed-form path (Normal-Normal fast path)

For a single observation `D` of a Normal location parameter `theta`
with known `sigma`:

  tau_Wald(theta) = ((D - theta) / sigma)^2,
  CI:               D ± z_{1 - alpha/2} * sigma.

Equivalently, the two-sided p-value is `2 * (1 - Phi(|D - theta| / sigma))`.

### Generic path (any `Model`)

For any `Model` exposing `mle(data)` and `fisher_information(theta)`:

  tau_Wald(theta) = (mle - theta)^2 * I(theta).

Under H_0: `theta = theta_0`, by Wilks' theorem `tau_Wald ~ chi^2_1`
asymptotically (one-parameter regular model, MLE consistent + asymptotically
normal). p-value via `scipy.stats.chi2.sf(tau, df=1)` (rendered with
`jax.scipy.stats.chi2.cdf`); CI by `brentq`-inversion through the model
support, with bracket half-width `4/sqrt(I(mle))` clipped to `support/2`.

The two paths are dispatched by `isinstance(model, NormalNormalModel)`.
On Normal-Normal both produce identical output to ~1e-10 — pinned by
`tests/regression/test_wald_generic_matches_closed_form.py`.

## Derivation

**Closed form:** Under H0: `theta = theta0`, `(D - theta0)/sigma ~ N(0, 1)`,
so `tau_Wald ~ chi^2_1`. Inverting the level-α test gives the standard
`z_{1-alpha/2}` half-width interval. (Standard textbook derivation.)

**Generic:** Wilks' theorem: under regularity, `2 * (l(mle) - l(theta_0))
~ chi^2_1` for the log-likelihood `l`; the second-order Taylor expansion
of `l` at the MLE gives `tau ≈ (mle - theta_0)^2 * I(theta_0)` (Casella
& Berger §10.3). The χ²₁ tail then converts τ to a p-value.

## Predicted behavior

- p-value at `theta = D` (the MLE) equals 1.
- p-value strictly decreases with `|D - theta| / sigma`.
- Coverage at the nominal level is exact for any `theta_true`.
- CI width is constant in the data: `2 * z_{1-alpha/2} * sigma`.
- Recovered as the η → 1 limit of `power_law` tilting.

## Failure modes

**Closed-form path:** None for the Normal location family with known
`sigma`. (Conditions of applicability — known variance, no nuisance
parameters — are invariants of the model itself.)

**Generic path:**
- Asymptotic-only calibration. Finite-sample coverage is approximate;
  on bounded-support models with extreme MLEs (e.g. Bernoulli at
  `mle ∈ {0, 1}`), the χ²₁ approximation visibly under-covers. The
  Bernoulli pairings test
  (`tests/properties/test_bernoulli_invariants.py::TestBernoulliPairingsGenericVsRaise`)
  smokes finite output but does NOT pin coverage at small `n`.
- `fisher_information(mle)` may be singular at the support boundary
  (Bernoulli `1/(p(1-p)) → ∞` at `p ∈ {0, 1}`); the bracket-width
  estimator clips `1/sqrt(I(mle))` and falls back to `support/2`,
  with `brentq` raising `BracketingFailed` and the CI returning the
  support boundary as an honest "open CI" rather than a numerically
  invalid value.

## Invariants

- p-value lies in [0, 1] for all inputs.
- p-value at `D` (closed form) / `mle` (generic) equals 1 (mode property).
- p-value is monotone decreasing in `|theta - mle|`.
- Under H0 (data ~ N(theta_true, sigma^2)), closed-form p-values are
  Uniform[0, 1] (KS test in
  `test_wald_invariants.py::TestWaldUniformPvalueUnderH0`).
- Acceptance region `[lo, hi]` (closed form, data-space) is symmetric
  about `theta0` with width `2 * z_{1-alpha/2} * sigma`. The
  generic path inverts in θ-space only and raises NotImplementedError
  on `acceptance_region`.
- Generic and closed-form paths agree on Normal-Normal to atol 1e-10
  on p-value and 1e-6 on CI bounds
  (`tests/regression/test_wald_generic_matches_closed_form.py`).

## Literature

- A. Wald. "Tests of statistical hypotheses concerning several parameters
  when the number of observations is large." *Trans. Amer. Math. Soc.* 54
  (1943): 426–482.
- George Casella and Roger Berger. *Statistical Inference*. 2nd ed., 2002.
  Chapter 8 (hypothesis tests and CIs).

## Links

- Implementation: `src/frasian/statistics/wald.py`
- Regression tests: `tests/regression/test_waldo_pvalue.py::TestWaldStatistic`
- Property tests: `tests/properties/test_wald_invariants.py::TestWaldInvariants`
- Statistical test (L3): `tests/properties/test_wald_invariants.py::TestWaldUniformPvalueUnderH0`
- Illustration: `src/frasian/experiments/illustrations/wald_demo.py`

## Status notes

The closed-form `acceptance_region` returns the D-space interval
(`D = theta0 ± z*sigma`). The generic path inverts only in θ-space
(`confidence_interval`); calling `acceptance_region` against a non-
Normal model raises NotImplementedError. LRT / signed-root / Bartlett
variants are scheduled stubs (see `docs/methods/{lrt,signed_root,bartlett}.md`).

`WaldStatistic.accepts_tilting(tilting)` returns `True` only when
`tilting.name == "identity"` — Wald ignores the prior, so non-identity
cells `(power_law, wald)`, `(ot, wald)`, etc. are degenerate
duplicates of `(identity, wald)`. The runner skips them and records
`status="incompatible"` in the manifest. The generic path does not
change this — Wald's prior-independence is intrinsic to the statistic,
not the path.
