# normal_normal

> Status: `implemented`

## Summary

The 1D conjugate Normal-Normal model: the working sandbox of the framework.
A scalar parameter `theta` has a Gaussian prior, and a single Gaussian
observation `D` has known noise variance. All math primitives shared by
WALDO, Wald, and the power-law tilting scheme assume this model.

## Motivation

Every research question in the framework — "does optimal-transport tilting
remove the discontinuity that power-law tilting creates?", "does the
Bartlett-corrected LRT improve calibration over WALDO?", and so on — is
first asked in this sandbox because every quantity has a closed form.
Generalization to non-Gaussian or multi-parameter settings is a future
extension; the abstractions in `src/frasian/models/base.py` are designed to
accommodate that without rewriting consumers.

## Definition

  Prior:        theta ~ N(mu0, sigma0^2)
  Likelihood:   D | theta ~ N(theta, sigma^2)
  Posterior:    theta | D ~ N(mu_n, sigma_n^2)

with closed-form posterior parameters

  w        = sigma0^2 / (sigma^2 + sigma0^2)
  mu_n     = w * D + (1 - w) * mu0
  sigma_n  = sqrt(w) * sigma.

The Frasian *data weight* w in [0, 1] interpolates between prior-dominant
(w → 0, strong prior) and data-dominant (w → 1, weak prior).

Auxiliary quantities used by downstream methods:

- *Scaled prior-data conflict*: `Delta = (1 - w) * (mu0 - D) / sigma`.
- *Prior residual*: `delta(theta) = (theta - mu0) / sigma0`.
- *Non-centrality*: `lambda(theta) = (1 - w)^2 * (mu0 - theta)^2 / (w^2 * sigma^2)`.

## Derivation

Standard conjugacy: the Gaussian likelihood is conjugate to the Gaussian
prior, so the posterior is Gaussian with precision equal to the sum of
prior and likelihood precisions, and mean equal to the precision-weighted
average. Detailed steps live in the legacy README; promoting a clean
ported derivation here is a follow-up doc task.

## Predicted behavior

- `posterior.var()` is monotone non-increasing in `n` (informative data
  shrinks the posterior).
- `posterior.mean()` lies between `prior.mean()` and the MLE.
- As `sigma0 → infinity`, the posterior collapses to the MLE (Wald limit).
- As `sigma0 → 0`, the posterior collapses to the prior (no data influence).

## Failure modes

- `sigma <= 0` or `sigma0 <= 0` raises `ValueError` from the
  `NormalDistribution` constructor — the only entry point.
- `sigma0 → 0` makes `w → 0` and `sigma_n → 0`; the WALDO p-value's
  Lipschitz constant scales as `1 / (w * sigma)`, producing the very
  steep behavior the framework is designed to study (see the
  `smoothness` experiment).
- Non-conjugate priors raise `NotImplementedError` from `posterior(...)`;
  generic posterior inference is a future extension.

## Invariants

- `posterior(data, prior).mean()` lies in the closed interval
  `[min(prior.mean(), MLE), max(prior.mean(), MLE)]`.
- `posterior(data, prior).var() <= prior.var()` whenever the likelihood is
  informative (`sigma` finite).
- `quantile(cdf(x)) == x` round-trip on the produced distributions
  (`tests/properties/test_normal_distribution.py`).
- `mle(sample_data(theta, ...))` is consistent under increasing `n`
  (sample mean converges to `theta`).

## Literature

- Carl Friedrich Gauss. *Theoria Motus Corporum Coelestium*. 1809.
  (The Gaussian conjugate update.)
- Andrew Gelman et al. *Bayesian Data Analysis*. CRC Press, 3rd ed., 2014.
  (Standard reference; conjugate Normal model in §3.)

## Links

- Implementation: `src/frasian/models/normal_normal.py`
- Distribution helpers: `src/frasian/models/distributions.py`
- Regression tests: `tests/regression/test_normal_normal.py`
- Property tests: `tests/properties/test_normal_distribution.py`
- Illustration: see the per-statistic / per-tilting demo scripts under
  `src/frasian/experiments/illustrations/`.

## Status notes

Generalization beyond 1D / beyond conjugate Gaussian is anticipated but
not scheduled. The `Model` protocol in `src/frasian/models/base.py` is
the integration point.
