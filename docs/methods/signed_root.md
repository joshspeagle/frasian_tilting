# signed_root

> Status: `stub`

## Summary

Signed-root LRT statistic
`r(theta) = sign(MLE - theta) * sqrt(LRT(theta))`. Asymptotically
`r ~ N(0, 1)` under H0. On the Normal-location family with known
variance, `r(theta) = (D - theta) / sigma` exactly — same CI as Wald
on the sandbox. Differs from Wald and LRT on non-Normal models, where
signed-root has higher-order coverage accuracy.

## Motivation

The signed-root statistic is one of the standard "second-order" pivots
in the higher-order asymptotics literature (Barndorff-Nielsen's
modified r*, Skovgaard's adjustment). Including it here is part of the
framework's mandate to cover Fraser's higher-order likelihood toolkit;
on the Normal-location sandbox it is uninteresting (= Wald), but the
cross-product framework needs the row.

## Definition

  r(theta0) = sign(theta_hat - theta0) * sqrt( -2 [ log L(theta0)
                                                  - log L(theta_hat) ] )

For Normal-location, theta_hat = D and the expression simplifies to
`(D - theta0) / sigma`. The two-sided p-value is
`2 * (1 - Phi(|r|))`, identical to the two-sided Wald p-value on the
sandbox.

## Derivation

Standard. On the Normal-location family substitute the quadratic
loglikelihood; the square root collapses to the absolute z-score, the
sign function reproduces the direction.

## Predicted behavior

- On the Normal-location sandbox: `r.pvalue == wald.pvalue` and
  `r.CI == wald.CI`.
- Under H0, `r` is Uniform on the unit interval after Phi-transform.
- The added value of `signed_root` (over Wald) appears only in
  non-Gaussian models — out of scope for Step 6.

## Failure modes

- The sign convention requires picking a side of the MLE; for theta0
  exactly equal to the MLE, the sign is conventionally 0 and r = 0.
- On non-Gaussian models, sqrt(LRT) can be undefined for negative LRT
  (a numerical artefact); guard with clipping.

## Invariants

- p-value in [0, 1].
- `r(theta_hat) = 0`; `pvalue(theta_hat) = 1`.
- Under H0, p-values are Uniform[0, 1].
- On Normal-location: `signed_root.pvalue == wald.pvalue` to numerical
  precision (regression test).

## Literature

- Barndorff-Nielsen, O. E. "Modified signed log likelihood ratio."
  *Biometrika* 78 (1991): 557-563.
- Brazzale, A. R., Davison, A. C., Reid, N. *Applied Asymptotics: Case
  Studies in Small-Sample Statistics.* Cambridge, 2007.
- Reid, N. "Saddlepoint methods and statistical inference." *Statistical
  Science* 3 (1988): 213-227.

## Links

- Implementation: `src/frasian/statistics/signed_root.py` (stub)
- Property tests: `tests/properties/test_signed_root_invariants.py`
                  (skipped)
- Illustration:   TBD

## Status notes

Stub — like `lrt`, redundant with Wald on the Normal-location sandbox
but architecturally important. The interesting work happens when the
framework extends to non-Gaussian models.
