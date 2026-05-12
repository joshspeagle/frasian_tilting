# scoreo

> Status: `implemented`

## Summary

Bayesian / posterior score statistic — the WALDO-style
counterpart of `score`:

    tau_Scoreo(theta0; data, prior) = U_post(theta0)^2 / I_post(theta0)

where `U_post(theta) = d/dtheta log pi(theta | data)` is the
posterior score function (gradient of the log-posterior at theta)
and `I_post(theta) = -d^2/dtheta^2 log pi(theta | data)` is the
**observed posterior information**. The asymptotic null is the
same MC reference distribution `waldo` and `lrto` use; the
closed-form Normal-Normal+Normal path collapses to WALDO's
`(mu_n - theta)^2 / sigma_n^2` because the Normal log-posterior
is quadratic, so `U_post(theta)^2 / I_post(theta) =
[(theta - mu_n)/sigma_n^2]^2 * sigma_n^2 = (theta - mu_n)^2 /
sigma_n^2`.

## Motivation

`scoreo` is the **Bayesian** half of the score pair, mirroring
`waldo`'s and `lrto`'s role in their respective pairs. The
framework's `(TiltingScheme x TestStatistic)` cross-product is
symmetric in the statistic dimension: every classical frequentist
pivot (Wald, LRT, Score) has a posterior analog (WALDO, LRTO,
SCOREO). On the Normal-Normal sandbox all three Bayesian variants
collapse to the same scalar `(theta - mu_n)^2 / sigma_n^2` — by
the same exact-quadratic argument that makes Wald = LRT = Score
on the frequentist side.

Concretely:

1. `(identity, scoreo)` evaluates against the un-tilted posterior
   — the "posterior score test" of Lindley (1965), Aitkin (1991)
   — and collapses to `waldo` on the Normal-Normal sandbox.
2. `(power_law[*], scoreo)`, `(ot[*], scoreo)`, etc. evaluate
   against the tilted posterior produced by the scheme.
3. The **trinity-on-the-Bayesian-side**: `(waldo, lrto, scoreo)`
   all coincide on Gaussian posteriors but differ off-Gaussian
   — `scoreo` is the natural pivot for future asymmetric /
   skewed posteriors where Wald's normality assumption fails.

`scoreo` shares `waldo`'s and `lrto`'s MC calibration discipline
(CRN seed, `(k+1)/(n+1)` continuity correction, observation-side
hoisting through `obs_state`).

## Definition

Let `pi(. | data)` be the posterior (possibly tilted) and define:

    U_post(theta) = d/dtheta log pi(theta | data)                          (D1)
    I_post(theta) = -d^2/dtheta^2 log pi(theta | data)                     (D2)
    tau_Scoreo(theta0; data, prior) = U_post(theta0)^2 / I_post(theta0)    (D3)

Under H_0 : theta = theta_0 the calibration is **not** chi^2_1 in
general (unlike `score`): under a non-flat prior the posterior is
not a likelihood, and Rao's theorem does not apply (Lindley 1957;
Aitkin 1997). Calibration is by Monte Carlo, mirroring WALDO and
LRTO:

    p_Scoreo(theta0; data, prior)
      = P_{D' ~ likelihood(.|theta_0)} [ tau_Scoreo(theta0; D', prior)
                                       >= tau_Scoreo(theta0; data, prior) ]  (D4)

estimated with `n_mc` MC draws and the `(k+1)/(n+1)` continuity
correction. The CI is the inversion in theta-space:

    CI_{1-alpha} = { theta : p_Scoreo(theta; data, prior) >= alpha }       (D5)

**Normal-Normal+Normal reduction (NN sandbox).** The posterior is
Gaussian: `pi(theta | D) = N(mu_n, sigma_n^2)`. Then

    log pi(theta | D) = -(theta - mu_n)^2 / (2 sigma_n^2) + const,
    U_post(theta)     = -(theta - mu_n) / sigma_n^2,
    I_post(theta)     = 1 / sigma_n^2,
    tau_Scoreo(theta) = [-(theta - mu_n)/sigma_n^2]^2 / (1/sigma_n^2)
                      = (theta - mu_n)^2 / sigma_n^2
                      = tau_WALDO = tau_LRTO.                              (D1-NN)

The closed-form p-value, CI, and acceptance region therefore
coincide with WALDO's `Phi(b - a) + Phi(-a - b)` formula
(`docs/methods/waldo.md`).

**Tilting compatibility.** `scoreo` accepts any tilting
(`accepts_tilting returns True`), matching `waldo` and `lrto`.

## Derivation

TODO — `/derive` (the deriver agent) fills this section.

Required ingredients:
1. (D1)/(D2)/(D3) as a definition; (D4) as the MC calibration;
   (D5) as the theta-space CI inversion.
2. NN+Normal reduction: differentiate the Gaussian log-posterior,
   recover (D1-NN); verify symbolically and numerically that
   `tau_Scoreo == tau_WALDO == tau_LRTO`.
3. Bayesian trinity on Gaussian posteriors: all three of waldo /
   lrto / scoreo coincide because the log-posterior is quadratic
   with constant curvature `1/sigma_n^2`.
4. Why H_0 is not chi^2_1 in general: same as lrto.
5. Flat-prior limit: as `sigma_0 -> infty`, `mu_n -> D`,
   `sigma_n -> sigma`, `tau_Scoreo -> (theta-D)^2/sigma^2 =
   tau_Score`.

## Predicted behavior

- **On NN+Normal sandbox**: `scoreo.pvalue == waldo.pvalue ==
  lrto.pvalue` and likewise for CIs (atol 1e-12 closed form;
  MC tolerance on the generic path).
- **Under flat-prior limit on NN**: `tau_Scoreo -> tau_Score
  -> tau_LRT = tau_Wald`. Same O(1/sigma_0^2) convergence rate
  as `lrto`.
- **Generic path**: any `(Model, Prior)` exposing
  `posterior(data, prior).logpdf` that is JAX-differentiable
  yields a valid CI. Uses `jax.grad` for `U_post`,
  `jax.hessian` for `I_post`. Cost `O(n_mc)` per brentq probe.

## Failure modes

- **Non-differentiable posterior log-density**: `jax.grad` and
  `jax.hessian` require `posterior.logpdf` to be JAX-traceable.
  Discrete posteriors are unsupported.
- **`I_post(theta_0) <= 0`**: the observed posterior information
  is `>= 0` everywhere only at the MAP and its neighbourhood.
  At a posterior local minimum or saddle, the second derivative
  flips sign and `tau_Scoreo` becomes negative or `inf`.
  Calling code's responsibility; the implementation will emit a
  `RuntimeWarning` (same pattern as `score`).
- **Multimodal posterior**: each mode has its own `I_post > 0`
  region; the score is zero at every mode AND at saddle points.
  `tau_Scoreo(theta_0) = 0` can mean "theta_0 is the MAP" OR
  "theta_0 is a non-MAP saddle". The implementation does not
  distinguish.
- **Tail extrapolation under the chain rule + posterior**: the
  generic path computes `jax.hessian(posterior.logpdf)(theta_0)`.
  For Gaussian-class posteriors this is exact; for non-Gaussian
  posteriors with rough log-densities the Hessian may be
  unstable.

## Invariants

TODO — fill from `/derive` output. Tentative list:

- `p_Scoreo in [0, 1]`.
- `tau_Scoreo >= 0` (with equality at `U_post = 0`).
- **NN+Normal equivalence — p-value**:
  `scoreo.pvalue == waldo.pvalue == lrto.pvalue` (atol 1e-12).
- **NN+Normal equivalence — CI**:
  `scoreo.confidence_interval == waldo.confidence_interval ==
  lrto.confidence_interval` (atol 1e-8).
- `accepts_tilting(...) is True` for every concrete TiltingScheme.
- **Flat-prior limit**: `scoreo -> score` as `sigma_0 -> infty`,
  with `O(1/sigma_0^2)` convergence rate.
- **H_0 uniformity on NN closed form**: KS uniformity passes.

## Literature

TODO — `/litreview` (the literature-reviewer agent) fills this
section. Required anchors:

- Lindley (1965) "Introduction to Probability and Statistics from
  a Bayesian Viewpoint" — early Bayesian-pivot framing.
- Aitkin (1991, 2010) — posterior-likelihood-derived pivots.
- Bernardo & Smith (1994) "Bayesian Theory" — reference Bayesian
  decision-theoretic framing.
- Pereira & Stern (1999) FBST — Bayesian alternative.
- Masserano et al. (2023) — WALDO pairing template.
- Rao (1948), Silvey (1959), Engle (1984), Bera & Bilias (2001)
  — frequentist score that scoreo Bayesianises; cross-reference
  `docs/methods/score.md`.

## Links

- Implementation: `src/frasian/statistics/scoreo.py`
- Property tests: `tests/properties/test_scoreo_invariants.py`
- Regression tests: `tests/regression/test_force_generic_dispatch.py::TestScoreoForceGeneric`
- Illustration: `src/frasian/experiments/illustrations/scoreo_demo.py`
- Generated figure: `output/illustrations/scoreo_demo.png`

## Status notes

`scoreo` collapses to `waldo` and `lrto` on the Normal-Normal+
Normal sandbox by construction; the existence of a separate
implementation is justified by (a) the cross-product cell, (b)
the generic-path posterior-score statistic for future non-
Gaussian posteriors where the Bayesian trinity diverges, and (c)
completing the (`score`, `scoreo`) pair as a structural
counterpart of (`wald`, `waldo`) and (`lrt`, `lrto`).
