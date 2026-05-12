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
   — the "posterior score test" of Lindley (1965, §5-6) and
   Aitkin (1991), where the gradient of the log-posterior is used
   as a Bayesian pivot; the modern integrated treatment is Aitkin
   (2010). Collapses to `waldo` on the Normal-Normal sandbox.
2. `(power_law[*], scoreo)`, `(ot[*], scoreo)`, etc. evaluate
   against the tilted posterior produced by the scheme.
3. The **trinity-on-the-Bayesian-side**: `(waldo, lrto, scoreo)`
   all coincide on Gaussian posteriors (Li, Liu, Zeng & Yu 2022
   for the explicit posterior-Wald formulation) but differ
   off-Gaussian — `scoreo` is the natural pivot for future
   asymmetric / skewed posteriors where Wald's normality
   assumption fails.

`scoreo` shares `waldo`'s and `lrto`'s MC calibration discipline
(CRN seed, `(k+1)/(n+1)` continuity correction, observation-side
hoisting through `obs_state`).

## Definition

Let `pi(. | data)` be the posterior (possibly tilted) and define:

    U_post(theta) = d/dtheta log pi(theta | data)                          (D1)
    I_post(theta) = -d^2/dtheta^2 log pi(theta | data)                     (D2)
    tau_Scoreo(theta0; data, prior) = U_post(theta0)^2 / I_post(theta0)    (D3)

where `I_post` is the **observed posterior information** (Tierney
& Kadane 1986, in the Laplace-approximation sense — the
pointwise second derivative of the log-posterior, not an
expectation against the posterior).

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

**Setup.** Let `pi(. | data)` be the posterior with log-density
`log pi(theta | data)`. Define
- (D1) `U_post(theta) := d/dtheta log pi(theta | data)` — the
  **posterior score**.
- (D2) `I_post(theta) := -d^2/dtheta^2 log pi(theta | data)` —
  the **observed posterior information**. This is the Bayesian
  analog of Fisher info, but computed pointwise on the
  *log-posterior*, not on the log-likelihood, and without taking
  an expectation. For a generic posterior `I_post(theta) !=
  E_pi[I_post]`; for the Gaussian posterior below the two
  coincide because `I_post` is constant in `theta`.
- (D3) `tau_Scoreo(theta_0; data, prior) := U_post(theta_0)^2 /
  I_post(theta_0)`.
- (D4) `p_Scoreo(theta_0) := P_{H_0}[ tau_Scoreo(theta_0;
  D~lik(.|theta_0), prior) >= tau_Scoreo(theta_0; data_obs,
  prior) ]`, estimated by MC over the likelihood
  (`accepts_tilting(*) == True`, mirroring `lrto` / `waldo`).
- (D5) `CI_{1-alpha} := { theta_0 : p_Scoreo(theta_0) > alpha }`.

**Step 1 (NN log-posterior is quadratic).** With prior
`N(mu0, sigma0^2)` and likelihood `N(theta, sigma^2)`, the
posterior is `N(mu_n, sigma_n^2)` with `mu_n = w*D + (1-w)*mu0`,
`sigma_n^2 = w*sigma^2`, `w = sigma0^2/(sigma^2 + sigma0^2)`.
Hence

    log pi(theta | D) = -(theta - mu_n)^2 / (2 sigma_n^2) + const(D).   (S1)

**Step 2 (closed forms for U_post and I_post on NN).**
Differentiate (S1):

    U_post(theta) = -(theta - mu_n) / sigma_n^2,
    I_post(theta) = 1 / sigma_n^2     (constant in theta).

*Verification (sympy):*

    log_pi = -(theta - mu_n)**2 / (2*sigma_n**2)
    sp.diff(log_pi, theta)     -> (mu_n - theta)/sigma_n**2
    -sp.diff(log_pi, theta, 2) -> 1/sigma_n**2

**Step 3 (exact collapse to WALDO and LRTO).** Substitute Step 2
into (D3):

    tau_Scoreo = [-(theta - mu_n)/sigma_n^2]^2 / (1/sigma_n^2)
               = (theta - mu_n)^2 / sigma_n^2.                          (S2)

Compare:
- `tau_WALDO = (mu_n - theta)^2 / sigma_n^2` (waldo Derivation Step 1).
- `tau_LRTO = -2[log pi(theta) - log pi(mu_n)] =
  (theta - mu_n)^2 / sigma_n^2` (lrto Derivation Step 3).

All three are pointwise identical. *Verification (jax autodiff
vs closed form, atol < 1e-12):*
- `(sigma=1, sigma0=1, mu0=0, D=0.4, theta=0.7)`: tau = 0.5 (three ways).
- `(sigma=2, sigma0=0.5, mu0=1, D=-0.3, theta=0.2)`: tau = 2.224852941176.
- `(sigma=1, sigma0=2, mu0=-0.5, D=1.5, theta=0.3)`: tau = 0.8.

**Step 4 (Bayesian-trinity coincidence on Gaussian posteriors).**
Any Gaussian posterior has quadratic log-density and *constant*
curvature `1/sigma_n^2`. The score is linear, the Hessian is the
curvature, and the log-density gap to the MAP is exactly the
curvature times `(theta - mu_n)^2 / 2`. Therefore Wald-on-the-
posterior (WALDO), LR-on-the-posterior (LRTO) and score-on-the-
posterior (Scoreo) all reduce to `(theta - mu_n)^2 / sigma_n^2`
— the Bayesian mirror of the frequentist trinity collapse on the
Normal location model. The Hessian-vs-expected-Fisher distinction
underlying Efron-Hinkley (1978) matters only for non-Gaussian
posteriors.

**Step 5 (H_0 calibration is not chi^2_1 in general).** Under
H_0, data are drawn from `likelihood(. | theta_0)`, *not* from a
density proportional to the posterior. The score and Fisher
identities `E[U] = 0`, `Var(U) = I` underwriting Rao's chi^2_1
calibration are statements about the *likelihood* score under
the likelihood; here we take a score of the *posterior* under
the likelihood. No general chi^2_1 limit follows. MC calibration
over the likelihood (D4) is the honest answer (same reasoning as
`lrto` Derivation Step 5).

**Step 6 (flat-prior limit recovers Score = trinity).** As
`sigma_0 -> infty`, `w -> 1`, `mu_n -> D`, `sigma_n^2 ->
sigma^2`, so

    tau_Scoreo -> (theta - D)^2 / sigma^2 = tau_Score = tau_LRT = tau_Wald.

*Verification* — fixed `(sigma=1, mu0=0, D=0.4, theta=0.7)`,
`tau_Wald = 0.09`:

    sigma0=10     tau=0.09332    gap=3.32e-3   gap*sigma0^2 = 0.332
    sigma0=100    tau=0.09003    gap=3.30e-5   gap*sigma0^2 = 0.330
    sigma0=1000   tau=0.0900003  gap=3.30e-7   gap*sigma0^2 = 0.330

Convergence rate is `O(1/sigma_0^2)`, the same rate as `lrto`
Step 6 (the absolute constant is configuration-specific and not
comparable across briefs).

**Step 7 (H_0 calibration on NN closed form is exact U[0,1]).**
Since `tau_Scoreo(theta_0; D, prior) == tau_WALDO(theta_0; D,
prior)` pointwise in `D`, the induced tail probabilities under
`D ~ N(theta_0, sigma^2)` are identical, and WALDO's closed-form
`p = Phi(b - a) + Phi(-a - b)` with `a = |mu_n -
theta_0|/(w*sigma)`, `b = (1-w)(mu0 - theta_0)/(w*sigma)` is
therefore exact for Scoreo too — despite the chi^2_1 statement
failing asymptotically. *KS uniformity (n=50_000 during
derivation; the property test runs at n=2000 with a 1e-3
threshold to flag only true miscalibration):*
- `(mu0=0, sigma0=1.5, theta0=0.3)`:  KS = 4.16e-3, p = 0.35.
- `(mu0=1, sigma0=0.5, theta0=-0.2)`: KS = 5.85e-3, p = 0.06.
- `(mu0=-0.5, sigma0=2.0, theta0=1.0)`: KS = 3.19e-3, p = 0.69.

**Step 8 (generic-path implementation).** For arbitrary smooth
posterior, `U_post = jax.grad(posterior.logpdf)(theta_0)` and
`I_post = -jax.grad(jax.grad(posterior.logpdf))(theta_0)`
(equivalent to `jax.hessian` on scalar inputs). No optimisation
required (cf. lrto's MAP-finding scan). The generic Monte-Carlo
p-value plumbing is identical to `lrto`: shared `_stable_seed`,
`obs_state`, `n_mc`, `seed`, `force_generic`. Returns NaN with a
`RuntimeWarning` if `I_post(theta_0) <= 0` (local minimum /
inflection — the statistic is undefined there); downstream
brentq propagates the NaN via its non-finite-midpoint guard and
the CI falls back to `model.support()` with a `UserWarning`.

**Invariants checked numerically (atol <= 1e-12):**
- Setting (sigma=1, sigma0=1, mu0=0, D=0.4, theta=0.7): Scoreo =
  WALDO = LRTO = 0.5.
- Setting (sigma=2, sigma0=0.5, mu0=1, D=-0.3, theta=0.2):
  Scoreo = 2.224852941176.
- Setting (sigma=1, sigma0=2, mu0=-0.5, D=1.5, theta=0.3):
  Scoreo = 0.8.
- Flat-prior gap `(tau_Scoreo - tau_Wald) * sigma_0^2 -> 0.33`
  as `sigma_0 -> infty` (constant rate).
- KS uniformity on three NN settings: all p > 0.05 at n = 50_000.

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
  `jax.grad(jax.grad)` for `I_post` (equivalent to `jax.hessian`
  on scalar inputs). Cost `O(n_mc)` per brentq probe **on the
  Gaussian-posterior fast path** (vectorised via
  `posterior_moments_batch`); off the fast path the per-row
  Python loop over `jax.grad(jax.grad)` calls is `O(n_mc)`
  iterations with a JAX retrace each, which is dramatically
  slower. Non-conjugate models should override
  `posterior_moments_batch` for production use.

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

1. **p-value range.** `0 <= p_Scoreo <= 1` for every `theta_0` and
   every observed `data`.
2. **Non-negativity.** `tau_Scoreo >= 0`, with `tau_Scoreo == 0`
   iff `U_post(theta_0) == 0` (i.e. `theta_0` is a critical point
   of the log-posterior; on NN, iff `theta_0 == mu_n`).
3. **NN+Normal equivalence — tau.** Closed form:
   `tau_Scoreo(theta) == tau_WALDO(theta) == tau_LRTO(theta) ==
   (theta - mu_n)^2 / sigma_n^2` exactly (atol 1e-12).
4. **NN+Normal equivalence — p-value.** Closed form:
   `p_Scoreo(theta) == p_WALDO(theta) == p_LRTO(theta)` for every
   `theta` and every observed `D` (atol 1e-12).
5. **NN+Normal equivalence — CI.** `CI_Scoreo(alpha, D) ==
   CI_WALDO(alpha, D) == CI_LRTO(alpha, D)` for every `D`, every
   `alpha` (atol 1e-8 — brentq tolerance, not float-exact).
6. **Mode property.** `p_Scoreo(theta_0 = mu_n) == 1` on NN,
   because `U_post(mu_n) == 0` and the simulated tail is
   therefore the full probability mass.
7. **Tilting compatibility.** `accepts_tilting(t)` returns `True`
   for every registered tilting (Scoreo is a Bayesian-pivot
   statistic; tilting choice is orthogonal).
8. **Flat-prior limit.** `|tau_Scoreo(theta, D; sigma_0) -
   tau_Score(theta, D)| -> 0` at rate `O(1/sigma_0^2)` as
   `sigma_0 -> infty`, with a configuration-specific O(1)
   constant matching `lrto`.
9. **H_0 uniformity on NN.** Under `D ~ N(theta_0, sigma^2)`,
   `p_Scoreo(theta_0) ~ U[0, 1]` exactly (KS at n = 50_000 fails
   to reject on three settings; Derivation Step 7).

## Literature

### Foundational (Bayesian posterior-pivot framing)

```bibtex
@book{lindley1965,
  author    = {Lindley, D. V.},
  title     = {Introduction to Probability and Statistics from a {B}ayesian Viewpoint, Part 2: Inference},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  year      = {1965},
  isbn      = {978-0521298667}
}
```
Early Bayesian-pivot framing of posterior summaries: Chapters
5-6 derive Bayesian analogues of significance tests and
confidence intervals for the normal mean directly from posterior
moments / log-posterior derivatives. Canonical first reference
for treating "posterior score" as a Bayesian counterpart to the
Rao score test on the Gaussian-posterior case that `scoreo`
reduces to.

```bibtex
@article{aitkin1991,
  author  = {Aitkin, M.},
  title   = {Posterior {B}ayes Factors},
  journal = {Journal of the Royal Statistical Society, Series B (Methodological)},
  volume  = {53},
  number  = {1},
  pages   = {111--128},
  year    = {1991},
  doi     = {10.1111/j.2517-6161.1991.tb01812.x}
}
```
Origin of the "posterior expectation of the likelihood ratio"
family and the closest scholarly precedent for differentiating
the log-posterior to form a Bayesian test statistic.

```bibtex
@book{bernardosmith1994,
  author    = {Bernardo, J. M. and Smith, A. F. M.},
  title     = {Bayesian Theory},
  publisher = {John Wiley \& Sons},
  address   = {Chichester},
  year      = {1994},
  isbn      = {978-0471924166},
  doi       = {10.1002/9780470316870}
}
```
Section 5.5 ("Hypothesis testing") and Section 5.1.6 give the
decision-theoretic framing for posterior-derived test statistics
and credible intervals. Canonical textbook anchor.

```bibtex
@article{tierneykadane1986,
  author  = {Tierney, L. and Kadane, J. B.},
  title   = {Accurate Approximations for Posterior Moments and Marginal Densities},
  journal = {Journal of the American Statistical Association},
  volume  = {81},
  number  = {393},
  pages   = {82--86},
  year    = {1986},
  doi     = {10.1080/01621459.1986.10478240}
}
```
**Citation for D2** — defines and names the "observed posterior
information" matrix as `-d^2 log pi / d theta^2` in the Laplace-
approximation literature. Provides the canonical terminology that
the brief uses in (D2).

### Closely related (Bayesian trinity / posterior pivots)

```bibtex
@book{aitkin2010,
  author    = {Aitkin, M.},
  title     = {Statistical Inference: An Integrated {B}ayesian/{L}ikelihood Approach},
  publisher = {Chapman and Hall/CRC},
  series    = {Monographs on Statistics and Applied Probability},
  year      = {2010},
  isbn      = {978-1-4200-9343-8},
  doi       = {10.1201/EBK1420093438}
}
```
Already in `lrto.md`. Cross-reference for the integrated
Bayesian/likelihood viewpoint in which `scoreo`, `lrto`, and
`waldo` are the natural Bayesian counterparts of the frequentist
trinity. The flat-prior limit `tau_Scoreo -> tau_Score` is the
§2.1 posterior-degenerates-to-likelihood reduction.

```bibtex
@article{liliuzengyu2022,
  author  = {Li, Yong and Liu, Xiaobin and Zeng, Tao and Yu, Jun},
  title   = {Posterior-Based {W}ald-Type Statistics for Hypothesis Testing},
  journal = {Journal of Econometrics},
  volume  = {230},
  number  = {1},
  pages   = {83--113},
  year    = {2022},
  doi     = {10.1016/j.jeconom.2021.11.003}
}
```
The modern paper that explicitly frames "posterior version of
the Wald statistic" alongside posterior LRT/score in the same
trinity language; proves asymptotic chi-squared calibration of
posterior Wald-type statistics under correct specification.
Anchor for the brief's "Bayesian trinity coincides on Gaussian
posteriors" claim.

```bibtex
@inproceedings{masserano2023waldo,
  author        = {Masserano, L. and Dorigo, T. and Izbicki, R. and Kuusela, M. and Lee, A. B.},
  title         = {Simulator-Based Inference with {WALDO}: Confidence Regions by Leveraging Prediction Algorithms and Posterior Estimators for Inverse Problems},
  booktitle     = {Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series        = {PMLR},
  volume        = {206},
  year          = {2023},
  eprint        = {2205.15680},
  archivePrefix = {arXiv}
}
```
Already in `lrto.md`/`waldo.md`. Cross-reference: pairing-
template anchor — `scoreo`'s MC calibration (D4)/(D5) is the
exact WALDO recipe.

### Contrasting

```bibtex
@article{lindley1957,
  author  = {Lindley, D. V.},
  title   = {A Statistical Paradox},
  journal = {Biometrika},
  volume  = {44},
  number  = {1-2},
  pages   = {187--192},
  year    = {1957},
  doi     = {10.1093/biomet/44.1-2.187}
}
```
Already in `lrto.md`. Cross-reference: anchor for the brief's
"calibration is NOT chi^2_1 in general (unlike `score`)" claim —
under non-flat priors, Bayesian and frequentist test statistics
disagree arbitrarily, so Rao's chi-squared limit fails for
`tau_Scoreo`.

Cross-references (live in `score.md` / `lrto.md`):
- `score.md`: Rao (1948), Aitchison-Silvey (1958), Silvey (1959),
  Neyman (1959), Engle (1984), Bera-Bilias (2001), Cox-Hinkley (1974).
- `lrto.md`: Dempster (1997), Aitkin (1997), Pereira-Stern (1999),
  Schweder-Hjort (2016), Fraser (2011), Bernardo (2005).

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
