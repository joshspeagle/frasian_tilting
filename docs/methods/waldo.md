# waldo

> Status: `implemented`

## Summary

WALDO (Weighted Accurate Likelihood-free inference via Diagnostic
Orderings) statistic. Closed-form fast path on the Normal-Normal+Normal
sandbox (`p(theta) = Phi(b - a) + Phi(-a - b)`); **generic Monte-Carlo
default** for any `Model` exposing `sample_data` and `posterior` —
Bernoulli + Beta is the existing smoke target. The Bayesian-frequentist
hybrid that the framework's existing experiments centre on.

## Motivation

WALDO replaces the MLE in the Wald statistic with the posterior mean,
borrowing prior information for shorter CIs in the data-poor regime
without sacrificing frequentist coverage at any single `theta_true`. It
is the η = 0 element of the power-law tilting family — the centre of the
research, and the benchmark every alternative tilting/statistic
combination must beat on the `smoothness` diagnostic.

## Definition

### Closed-form path (Normal-Normal + Normal fast path)

For the conjugate Normal-Normal model (see `normal_normal.md`):

  a(theta) = |mu_n - theta| / (w * sigma)
  b(theta) = (1 - w) * (mu0 - theta) / (w * sigma)
  p(theta) = Phi(b - a) + Phi(-a - b),

with the test statistic `tau_WALDO = (mu_n - theta)^2 / sigma_n^2`.

### Generic path (any `Model` + any conjugate `Prior`)

  t(D, theta) = (mu_post - theta)^2 / sigma_post^2,
  p(theta; D) = (1 + #{i : t(D'_i, theta) >= t(D, theta)}) / (n_mc + 1).

The MC reference distribution sampled from H_0:
- `D'_i ~ likelihood(. | theta)` for `i = 1..n_mc` via `model.sample_data`,
- recompute `t(D'_i, theta)` per draw via `model.posterior(D'_i, prior)`.

`WaldoStatistic` exposes two dataclass-field knobs: `n_mc=2000` and
`seed=0xC0FFEE`. Override at construction: `WaldoStatistic(n_mc=4000,
seed=12345)`.

The two paths are dispatched by `isinstance(model, NormalNormalModel)
& isinstance(prior, NormalDistribution)`. On the conjugate-Normal
sandbox both produce results agreeing within MC tolerance —
`tests/regression/test_waldo_generic_matches_closed_form.py` (L3) pins
3-σ MC bounds on p-value and CI.

### Reproducibility & MC discipline

The MC seed is derived from `(data, model.fingerprint(), prior.fingerprint(),
alpha, self.seed)` via `hashlib.blake2b` — cross-process stable
(immune to `PYTHONHASHSEED`). Critically the seed is **independent of
the candidate theta**: a single seed is computed at the start of
`confidence_interval` and threaded through every brentq probe, so
`np.random.default_rng(seed)` produces the same internal uniform
stream at each θ. This makes the empirical p-value a deterministic
function of θ (piecewise-constant for Bernoulli, smooth for Normal)
and lets brentq actually converge instead of locking onto a
re-randomised staircase.

The `(k+1)/(n+1)` continuity correction makes the empirical p-value
strictly **conservative**: empirical coverage ≥ nominal `1 - alpha`
to within MC noise (verified by
`tests/regression/test_waldo_generic_reproducibility.py::test_smoothed_pvalue_is_strictly_conservative`).
The bias is O(1/n_mc) and goes away as `n_mc → ∞`.

## Derivation

**Closed form:** Theorem 3 in the legacy framework (port retained
byte-for-byte; see `tests/regression/test_waldo_pvalue.py::
TestWaldoPvalueMatchesLegacy`): the p-value is the probability under
H0 that `tau_WALDO(theta_true)` exceeds the observed
`tau_WALDO(theta)`. The decomposition into `Phi(b-a) + Phi(-a-b)`
follows from the fact that under H0, `mu_n - theta` is Gaussian with
mean `b_eff = (1-w)(mu0 - theta)` and variance `w^2 * sigma^2`. The
two addends correspond to the upper and lower tails of the squared form.

**Generic:** the same definition applied directly — the MC reference
samples the H_0 distribution of `t(D', theta)` and reports an
empirical tail probability. No closed form is invoked; the only
calibration assumption is that `model.sample_data(theta, ...)`
correctly samples `likelihood(.|theta)`.

The algebraic derivation lives inline above; the longer typed-up
form (and any future re-derivations driven by the `deriver` agent)
will land alongside this brief in `docs/methods/`. There is no
separate `docs/derivations/` directory at this point in the
project.

## Predicted behavior

- p-value equals 1 at `theta = mu_n` (the WALDO mode).
- Coverage is exact under any `theta_true` (frequentist calibration).
- CIs are *narrower* than Wald when the prior is informative and there is
  no conflict; *wider* when the prior conflicts strongly with the data.
- The Lipschitz constant of `p(theta)` scales as `1 / (w * sigma)` —
  small `w` (strong prior) implies steep p-values, the basis for the
  `smoothness` diagnostic.

## Failure modes

**Closed-form path:**
- Steep transitions when `w` is small (near-singular Lipschitz constant).
  Not a numerical bug, but a behavioural property of interest.

**Generic path:**
- MC noise scales as `1/sqrt(n_mc)`. Default `n_mc=2000` gives ~0.022
  SE on a p-value near 0.5; lower for tail probabilities. For tight CI
  inversion, bump `n_mc` at construction time.
- `model.posterior(D', prior)` is called once per MC draw. For
  conjugate Beta or Normal, O(1); for hypothetical NUTS / VI
  posteriors, O(seconds × n_mc). The CI inversion cost is
  `O(n_mc × brentq_iterations)`.
- Coverage is biased *upward* by O(1/n_mc) (conservative `+1`
  smoothing). At small `n_mc` and small `alpha`, the bias is
  noticeable (e.g. n_mc=500, alpha=0.05 ⇒ minimum p-value 1/501 ≈
  0.002, biasing the CI wider than the true 95%).
- 1-D data only: `n_obs = data.size` is correct for the framework's
  n=1 sandbox and for n-trials Bernoulli vectors. A future model with
  structured data (`(n_obs, n_dims)`) needs an explicit `n_obs(data)`
  protocol method (see `_generic_pvalue` docstring NotImplementedError).

## Invariants

- p-value lies in (0, 1] (closed form: [0, 1]; generic: strictly > 0
  thanks to `+1` smoothing).
- p-value at `theta = mu_post` equals 1 (closed form exactly; generic
  always — `t_obs = 0` ≤ every `t_ref` ⇒ `(n_mc+1)/(n_mc+1) = 1`).
- Under H0, closed-form p-values are Uniform[0, 1] (statistical L3,
  scheduled to be added once we have a fast simulator at
  `experiments/coverage`).
- Generic p-values are conservative: empirical coverage ≥ nominal
  (`tests/regression/test_waldo_generic_reproducibility.py::
  test_smoothed_pvalue_is_strictly_conservative`).
- Reproducibility: the generic p-value and CI are bit-identical across
  Python processes / `PYTHONHASHSEED` values
  (`test_waldo_generic_reproducibility.py::test_pvalue_reproducible_*`,
  `::test_ci_brentq_uses_common_random_numbers`).
- `pvalue` is continuous (closed form) / piecewise-constant or smooth
  (generic) in `theta`: no random fluctuations across brentq probes.

## Literature

- A. Masserano, T. Dorigo, R. Izbicki, M. Kuusela, A. B. Lee. "Simulator-
  based inference with WALDO." *AISTATS 2023*.
- D. R. Cox and N. Reid. "Parameter orthogonality and approximate
  conditional inference." *J. Royal Stat. Soc. B*, 49 (1987): 1–39.
  (Background on conditional / hybrid inference.)

## Links

- Implementation: `src/frasian/statistics/waldo.py`
- Regression tests: `tests/regression/test_waldo_pvalue.py`
- Property tests: `tests/properties/test_waldo_invariants.py`
- Illustration: `src/frasian/experiments/illustrations/waldo_demo.py`

## Status notes

The closed-form `acceptance_region` numerically inverts the WALDO
p-value via `brentq_with_doubling`, giving the D-space interval at
fixed `theta0`. The generic path inverts only in θ-space
(`confidence_interval`); calling `acceptance_region` with a non-
conjugate-Normal pair raises `NotImplementedError`.

Brentq closures in both paths use a numpy-eager scalar mirror
(`_closed_form_pvalue_scalar` and the `_generic_mc_reference` path)
to avoid ~200 us JAX dispatch per iteration. The bulk-vector
`pvalue(theta_arr, ...)` call still routes through the JAX path for
autodiff compatibility.
