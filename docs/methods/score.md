# score

> Status: `implemented`

## Summary

Classical Rao score statistic — the third member of the
asymptotic Wald / LRT / Score trinity (Rao 1948):

    tau_Score(theta0; data) = U(theta0)^2 / I(theta0)

where `U(theta0) = d/dtheta log L(theta0; data)` is the score
function (gradient of the log-likelihood at theta0) and
`I(theta0) = model.fisher_information(theta0)` is the Fisher
information. Asymptotic null: `tau_Score -> chi^2_1` under
H_0:theta=theta_0 (Wilks-equivalent; the trinity coincides
asymptotically). On the Normal-Normal sandbox with n=1 it
**collapses exactly** to `(D - theta_0)^2 / sigma^2 = tau_Wald =
tau_LRT`, so the closed-form p-value reduces to
`2(1 - Phi(|D - theta_0|/sigma))`. Accepts only the `identity`
tilting (it ignores the prior, mirroring `wald` and `lrt`).

## Motivation

`score` completes the **classical trinity** of frequentist
pivots (Wald + LRT + Score) in the framework's `TestStatistic`
column. The three are asymptotically equivalent under regularity
but differ at finite samples / off-asymptotic regimes:

- **Wald**: `(MLE - theta_0)^2 * I(MLE)` — uses MLE-side info;
  cheap; sensitive to parameterisation.
- **LRT**: `-2 log[L(theta_0) / L(MLE)]` — symmetric, requires
  evaluating likelihood at two points; invariant under
  reparameterisation.
- **Score**: `U(theta_0)^2 / I(theta_0)` — uses ONLY the null;
  the only one that does not need the MLE; the Lagrange-
  multiplier test in classical econometrics.

On the Normal-Normal n=1 sandbox they collapse identically; the
existence of separate implementations buys (a) cross-product
cells (`(identity, score)`), (b) a generic-path differentiable
test statistic available for any `Model` with a JAX-traceable
log-likelihood, and (c) the natural home for future score-based
diagnostics (signed-root of score, locally most powerful tests).
`score` is the frequentist half of the `(score, scoreo)` pair;
`scoreo` is the corresponding Bayesian / posterior-score variant.

## Definition

For any `Model` with a differentiable log-likelihood:

    U(theta) = d/dtheta [ log L(theta; data) ]                            (D1)
    I(theta) = model.fisher_information(theta)                            (D2)
    tau_Score(theta0; data) = U(theta0)^2 / I(theta0)                     (D3)

Asymptotic null (Rao 1948; Wilks 1938 for the family):

    tau_Score(theta0; data) ~ chi^2_1   under H_0 : theta = theta_0       (D4)

p-value and CI:

    p_Score(theta0; data) = 1 - F_{chi^2_1}(tau_Score(theta0; data))      (D5)
    CI_{1-alpha} = { theta : tau_Score(theta; data) <= chi^2_{1, 1-alpha} } (D6)

**Normal-Normal n=1 reduction (NN sandbox).** The likelihood
`L(theta; D) = (1/sqrt(2 pi sigma^2)) exp(-(D-theta)^2 / (2 sigma^2))`
gives:

    log L(theta; D) = -(D - theta)^2 / (2 sigma^2) + const,
    U(theta) = (D - theta) / sigma^2,
    I(theta) = 1 / sigma^2,
    tau_Score(theta0; D) = [(D - theta0) / sigma^2]^2 / (1 / sigma^2)
                         = (D - theta0)^2 / sigma^2.                       (D1-NN)

The closed-form p-value is therefore the two-sided z-test:

    p_Score(theta0; D) = 2 (1 - Phi(|D - theta0| / sigma)),                (D5-NN)

identical to `wald.pvalue` and `lrt.pvalue`. The closed-form CI is
`D +/- z_{alpha/2} * sigma`, where `z_{alpha/2}` is the standard
Normal upper alpha/2 quantile.

**For n > 1** on Normal-Normal: `U(theta) = n (D_bar - theta) / sigma^2`,
`I(theta) = n / sigma^2`, so `tau_Score = n (D_bar - theta_0)^2 /
sigma^2` — identical to `wald` and `lrt`.

**Tilting compatibility.** `score.accepts_tilting(tilting)` returns
`True` only when `tilting` is `IdentityTilting`. The Rao score is
defined from the LIKELIHOOD (not the posterior), so any tilting
that modifies the prior contribution to a (tilted) posterior is
irrelevant — and the resulting `(non-identity, score)` cells would
be degenerate duplicates of `(identity, score)`. The Bayesian
counterpart `scoreo` does accept all tiltings.

## Derivation

TODO — `/derive` (the deriver agent) fills this section.

Required ingredients:
1. Equation (D1)/(D2)/(D3) as a definition; (D4) as the
   asymptotic-null statement (cite Rao 1948).
2. The NN n=1 reduction: differentiate log L, evaluate I, show
   the closed-form quadratic.
3. Identity with Wald and LRT on Normal-location: same scalar,
   same p-value, same CI.
4. Why `chi^2_1` is the correct H_0 calibration in general
   (regularity assumptions: identifiable, twice-differentiable
   log-likelihood, finite Fisher info).
5. Behaviour at non-MLE theta0: unlike Wald (which evaluates at
   MLE) and LRT (which uses MLE in the denominator), Score is
   evaluated entirely at theta0 — useful when MLE computation is
   expensive (rare in NN, common in nonlinear models).

## Predicted behavior

- **On NN+anything n=1**: `score.pvalue == wald.pvalue == lrt.pvalue`
  to numerical precision for every `(theta, data, model)`. Pinned
  by property tests.
- **Coverage**: nominal at `1 - alpha` (the chi^2_1 calibration is
  exact at n=1 on Normal-location, identical to Wald).
- **Generic path**: any `Model` exposing `likelihood(data).loglik`
  (JAX-traceable) and `fisher_information(theta)` yields a valid
  CI. Cost is `O(1)` per evaluation (one JAX grad call); CI
  inversion is `O(brentq_iters)`.

## Failure modes

- **Non-differentiable log-likelihood**: `jax.grad` requires the
  log-likelihood to be JAX-traceable. Models with discrete /
  non-smooth likelihoods (e.g. future binomial mixtures) need
  either a custom score function or a finite-difference
  fallback.
- **Fisher information at boundary / singular**: when
  `I(theta_0) -> 0` (e.g. flat likelihood at theta_0),
  `tau_Score -> infinity` for any non-zero U; the chi^2_1
  approximation breaks and the test becomes anti-conservative.
  Calling code's responsibility.
- **`NormalNormalModel` subclassing risk**: same risk as Wald
  — the closed-form NN path dispatches on `is_normal_normal(...)`
  fingerprint, so a subclass that overrides `likelihood` but not
  `fingerprint` will silently route through the closed-form
  formula. Mitigation: `force_generic=True` opt-out.
- **`n > 1` data on the closed-form path**: same caveat as Wald
  / LRT — the closed-form on NN n=1 uses `D = data.mean()` and
  assumes a single scalar `D`. For `n > 1` we route through the
  generic path (`jax.grad` on `loglik`), which uses
  `data.size` correctly.

## Invariants

TODO — fill from `/derive` output. Tentative list:

- `p_Score in [0, 1]` for all `(theta0, data, model)`.
- `tau_Score >= 0` (equality only at the unconstrained MLE).
- **NN n=1 equivalence — p-value**:
  `score.pvalue == wald.pvalue == lrt.pvalue` (atol 1e-12 closed
  form; MC tolerance on the generic path).
- **NN n=1 equivalence — CI**:
  `score.confidence_interval == wald.confidence_interval ==
  lrt.confidence_interval` (atol 1e-8 closed form).
- `accepts_tilting(IdentityTilting()) is True`;
  `accepts_tilting(<anything else>) is False`.
- **H_0 calibration**: under data ~ `model.sample_data(theta_0)`,
  `score.pvalue(theta_0; data)` is Uniform[0,1] (KS uniformity,
  L3 statistical tier).

## Literature

TODO — `/litreview` (the literature-reviewer agent) fills this
section. Required anchors:

- Rao (1948) "Large sample tests of statistical hypotheses
  concerning several parameters with applications to problems of
  estimation" — origin of the score / Lagrange-multiplier test.
- Aitchison & Silvey (1958) for the multivariate / constrained
  variant.
- Bera & Bilias (2001) for a modern review of Rao, Wald, LR.
- Engle (1984) "Wald, likelihood ratio, and Lagrange multiplier
  tests in econometrics" — the canonical trinity-comparison
  reference.
- Casella & Berger (2002) §10.3 / van der Vaart (1998) Theorem 16.7
  for the regular-model regularity conditions.
- Cox & Hinkley (1974) §9 for the small-sample correction
  perspective (score-vs-Wald discrepancy).
- Bartlett (1953a, 1953b) for the locally-most-powerful framing
  (score test ≡ LMP one-sided test).

## Links

- Implementation: `src/frasian/statistics/score.py`
- Property tests: `tests/properties/test_score_invariants.py`
- Regression tests: `tests/regression/test_force_generic_dispatch.py::TestScoreForceGeneric`
- Illustration: `src/frasian/experiments/illustrations/score_demo.py`
- Generated figure: `output/illustrations/score_demo.png`

## Status notes

`score` collapses to `wald` and `lrt` on the Normal-Normal n=1
sandbox by construction; the existence of a separate
implementation is justified by (a) the asymptotic-trinity cross-
product cell, (b) the generic-path JAX-grad-based test statistic
available for any differentiable `Model`, and (c) hosting future
score-based diagnostics (signed-root of score, LMP / score
covering tests).
