# bernoulli

> Status: `implemented`

## Summary

The Bernoulli model with a Beta-conjugate prior. The framework's
second concrete `Model` implementation, included to demonstrate that
the protocols in `src/frasian/models/base.py` accommodate non-Normal
sampling distributions and non-Gaussian conjugate priors. Pairings
with the existing tilting schemes / test statistics raise
`NotImplementedError` (by design — see Failure modes).

## Motivation

The first six steps of the framework refactor were entirely on the
1D conjugate Normal-Normal sandbox. Adding Bernoulli closes a loop:
the protocols *are* generic, even if today's implementations are
not. The architectural payoff is that every `isinstance(model,
NormalNormalModel)` check now surfaces a clear
`NotImplementedError` instead of silent failure when paired with
`BernoulliModel`. The research payoff (someday) is being able to
ask whether the smoothness diagnostic generalises beyond Gaussians.

## Definition

  Prior:        theta ~ Beta(alpha_0, beta_0)
  Likelihood:   X_i | theta ~ Bernoulli(theta), iid for i = 1..n
  Posterior:    theta | X ~ Beta(alpha_0 + k, beta_0 + n - k)

with `k = sum(X_i)` the success count.

Auxiliary primitives:

- *MLE*:       theta_hat = k / n.
- *Fisher information*: I(theta) = 1 / (theta (1 - theta)).
- *Support*:   theta in [0, 1].

## Derivation

Standard conjugate update. The Beta prior's density
`B(alpha_0, beta_0)^{-1} theta^{alpha_0 - 1} (1 - theta)^{beta_0 - 1}`
multiplied by the binomial likelihood
`theta^k (1 - theta)^{n - k}` produces a Beta posterior with shape
parameters `(alpha_0 + k, beta_0 + n - k)` (the `B(...)^{-1}`
normaliser absorbs into a Beta normaliser of the new parameters).

Fisher information follows from differentiating the
log-likelihood twice and taking the negative expectation; the
expected Hessian is `1 / (theta (1 - theta))` per observation.

## Predicted behavior

- `posterior(data, Beta(alpha_0, beta_0))` returns a Beta whose mean
  lies between `prior.mean()` and the MLE, weighted by the relative
  pseudo-counts.
- As `n -> infinity` at fixed `theta_true`, `posterior.var() -> 0`
  (asymptotic concentration). Note: posterior variance is **not**
  monotonically below the prior variance for every finite `n` — when
  the prior is strongly opposed to the data, the posterior may
  transiently widen before contracting.
- `mle(sample_data(theta_true, ...))` is consistent: as `n` grows,
  `mle -> theta_true` (LLN).
- `fisher_information(theta)` blows up as `theta -> 0` or `theta -> 1`
  (boundary effect) — clipping is necessary in practice.

## Failure modes

By design, all of the following raise `NotImplementedError` when
paired with a `BernoulliModel`:

- `WaldStatistic` (Normal-location specific p-value formula).
- `WaldoStatistic` (Normal-Normal Theorem 3 closed form).
- `PowerLawTilting.tilt` (Normal-only Theorem 6 closed form).
- `PowerLawTilting.tilted_pvalue` and
  `PowerLawTilting.tilted_confidence_interval`.

The errors all flow through the `models/_dispatch.py` helper so the
message format is uniform: "`<Class>` currently requires
`NormalNormalModel`; got `BernoulliModel`. Generalising the
implementation to other models is tracked as Phase-3 follow-up
work."

Numerical hazards:

- `theta` near 0 or 1: log-likelihood involves `log(1 - theta)` /
  `log(theta)` which underflow. `BernoulliLikelihood.loglik` clips
  to `eps = 1e-300`.
- Beta posterior with tiny shape parameters (e.g. `alpha_0 = 0.5,
  k = 0`) places significant mass at the boundary; downstream CI
  inversions need to be aware.

## Invariants

- `support() == (0.0, 1.0)`.
- `posterior.mean()` is in the closed interval bracketed by
  `prior.mean()` and `mle(data)`.
- `posterior.var() -> 0` as `n -> infinity` (asymptotic; verified
  by the property test on n=50 vs n=500 at fixed `theta_true`).
- `quantile(cdf(x)) == x` round-trip on the Beta posterior
  (atol ~ 1e-9 in the interior).
- `mle(sample_data(theta_true, rng, n))` converges to `theta_true`
  under increasing `n` (statistical L3 test).

## Literature

- Diaconis, P., Ylvisaker, D. "Conjugate priors for exponential
  families." *Ann. Statist.* 7 (1979): 269-281.
- Gelman, A. et al. *Bayesian Data Analysis.* 3rd ed., 2014.
  Chapter 2 (Beta-Binomial conjugate analysis).
- Brown, L. D. *Fundamentals of Statistical Exponential Families.*
  IMS Lecture Notes 9, 1986.

## Links

- Implementation: `src/frasian/models/bernoulli.py`
- Distributions:  `src/frasian/models/distributions.py`
                  (`BetaDistribution`, `BernoulliLikelihood`)
- Dispatch helper: `src/frasian/models/_dispatch.py`
- Property tests: `tests/properties/test_bernoulli_invariants.py`
- Regression tests: `tests/regression/test_bernoulli_model.py`

## Status notes

Implemented as a *proof-of-concept generic Model*; pairings with
existing tilting schemes / test statistics intentionally raise.
Phase-3 follow-up work would generalise WALDO and the power-law
tilting to Bernoulli, but that is mathematical research rather
than refactoring.
