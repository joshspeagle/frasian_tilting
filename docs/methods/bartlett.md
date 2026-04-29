# bartlett

> Status: `stub`

## Summary

Bartlett correction applied to the LRT statistic. For an LRT statistic
with E[LRT] != df asymptotically, divide by the expected value to make
the corrected statistic chi^2 to higher order:

  tau_BCLRT(theta) = LRT(theta) / E[LRT(theta)].

On the Normal-location sandbox `LRT ~ chi^2_1` *exactly*, so the
correction is trivial. The stub exists for non-canonical models where
Bartlett's higher-order correction tightens calibration meaningfully.

## Motivation

Bartlett's correction is the textbook example of a higher-order
adjustment that improves small-sample coverage of LRT-based CIs. The
framework's cross-product needs the row to differentiate "approximate
chi^2_1 calibration" methods from exact ones — but the implementation
is uninteresting on the Normal-location sandbox.

The cleanest *architectural* design is `BartlettCorrected(LRTStatistic())`
as a *decorator* over the base LRT, rather than a separate class. That
refactor lands when the base LRT does (`/propose-method bartlett` will
likely propose this).

## Definition

  tau_BCLRT(theta0) = LRT(theta0) / E[ LRT(theta0) ]

where `E[LRT]` is computed under the null (or sometimes via empirical
or simulation-based estimation). Asymptotically `tau_BCLRT ~ chi^2_1`
to a higher order than `tau_LRT` itself.

## Derivation

Standard Bartlett expansion (Bartlett 1937; Lawley 1956): expand
`E[LRT]` in inverse-n series, divide. For Normal-location, `E[LRT] = 1`
exactly so `tau_BCLRT == tau_LRT == tau_Wald`.

## Predicted behavior

- On Normal-location sandbox: identical to `lrt`, which is identical to
  `wald`.
- On non-Gaussian models: tightens chi^2_1 calibration at finite n.
- Coverage at nominal level (slightly more accurate than `lrt` for
  small n on non-Gaussian models).

## Failure modes

- Computing `E[LRT]` analytically requires model-specific work; numerical
  Monte Carlo is a fallback but expensive.
- Some pathological models have negative or zero `E[LRT]` corrections —
  guard against division by zero / sign flip.

## Invariants

- p-value in [0, 1].
- On Normal-location: `bartlett.pvalue == lrt.pvalue == wald.pvalue`.
- Under H0, p-values are Uniform[0, 1] (KS test, asymptotic).
- The decorator pattern (`BartlettCorrected(SomeLRT)`) commutes with
  the registry: registering a corrected variant does not require
  re-registering the base.

## Literature

- Bartlett, M. S. "Properties of sufficiency and statistical tests."
  *Proc. R. Soc. London A* 160 (1937): 268-282.
- Lawley, D. N. "A general method for approximating to the distribution
  of likelihood ratio criteria." *Biometrika* 43 (1956): 295-303.
- Cordeiro, G. M., Cribari-Neto, F. *An Introduction to Bartlett
  Correction and Bias Reduction.* SpringerBriefs, 2014.

## Links

- Implementation: `src/frasian/statistics/bartlett.py` (stub)
- Property tests: `tests/properties/test_bartlett_invariants.py`
                  (skipped)
- Illustration:   TBD

## Status notes

Stub — implementation is "trivial on the sandbox" and "subtle in
general". The decorator factoring is the architectural lift; the math
is well-trodden.
