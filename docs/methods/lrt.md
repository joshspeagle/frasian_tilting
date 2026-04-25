# lrt

> Status: `stub`

## Summary

Likelihood-ratio test statistic
`tau_LRT(theta) = -2 log( L(theta) / max_theta L(theta) )`. On the
Normal location family with known variance, `tau_LRT` reduces *exactly*
to the Wald statistic `((D - theta) / sigma)^2`; on non-Gaussian
models the two diverge. Registering LRT here puts the LRT row into the
`(TiltingScheme x TestStatistic)` cross-product so future non-Gaussian
extensions plug in cleanly.

## Motivation

LRT is the canonical likelihood-based pivot. It generalises beyond the
Normal-location family in a way that Wald does not (Wald requires a
quadratic loglikelihood; LRT does not). Including it now establishes
the interface and lets the framework verify that the cross-product
machinery handles statistics whose calibration distribution is exact
(Wilks' theorem) only asymptotically.

## Definition

  tau_LRT(theta0) = -2 [ log L(theta0) - log L(theta_hat) ]
                  = ((D - theta0) / sigma)^2     (Normal-location case)

Asymptotically `tau_LRT ~ chi^2_1` under H0 (Wilks). The p-value is
`1 - F_chi2_1(tau_LRT)`. The two-sided CI inverts this, equivalent to
solving `tau_LRT(theta) <= chi^2_{1, 1-alpha}`.

## Derivation

Standard Wilks asymptotic. The Normal-location reduction is by direct
substitution: `log L(theta)` is quadratic in theta with curvature
`1/sigma^2`, giving the Wald form exactly.

## Predicted behavior

- On the Normal-location sandbox, `pvalue` and `confidence_interval`
  match Wald's exactly (verify with regression test).
- Under H0, `tau_LRT ~ chi^2_1` (verify with KS test).
- Cross-product cells (`*`, `lrt`) reproduce Wald-row results exactly
  on the sandbox.

## Failure modes

- For more general models the asymptotic chi^2_1 calibration is
  inexact at small n; the Bartlett-corrected variant (`bartlett`) is
  the higher-order fix.
- Multimodal likelihoods would invalidate the single-CI assumption;
  not relevant on the Normal-location family.

## Invariants

- p-value in [0, 1].
- p-value equals 1 at theta = MLE.
- Under H0, p-values are Uniform[0, 1] (KS test).
- On Normal-location: `lrt.pvalue == wald.pvalue` and `lrt.CI == wald.CI`
  to numerical precision.

## Literature

- Wilks, S. S. "The large-sample distribution of the likelihood ratio
  for testing composite hypotheses." *Ann. Math. Stat.* 9 (1938): 60-62.
- Casella, G., Berger, R. *Statistical Inference.* 2nd ed., 2002.
  Chapter 8 (LRT, signed-root, Bartlett).
- Severini, T. A. *Likelihood Methods in Statistics.* Oxford, 2000.

## Links

- Implementation: `src/frasian/statistics/lrt.py` (stub)
- Property tests: `tests/properties/test_lrt_invariants.py` (skipped)
- Illustration:   TBD

## Status notes

Stub — value is mostly architectural: extending the cross-product to
non-Gaussian models will require LRT first. On the Normal-location
sandbox it is redundant with Wald; tests will assert that.
