# lrto

> Status: `implemented`

## Summary

Bayesian / posterior likelihood-ratio test statistic — the WALDO-
style counterpart of `lrt`:

    tau_LRTO(theta0; data, prior) = -2 [ log pi(theta0 | data) - log pi(theta_MAP | data) ]

where `pi(theta | data)` is the posterior produced by
`model.posterior(data, prior)` (or, in a (TiltingScheme x lrto)
cell, the *tilted* posterior produced by the scheme) and
`theta_MAP = argmax_theta pi(theta | data)`. The asymptotic null is
the same MC reference distribution WALDO uses (sample `D' ~
likelihood(.|theta_0)`, recompute the posterior, recompute tau);
the closed-form Normal-Normal+Normal path collapses to WALDO's
`(mu_n - theta)^2 / sigma_n^2` because the Normal log-posterior is
quadratic.

## Motivation

`lrto` is the **Bayesian** half of the LRT pair, mirroring `waldo`'s
role in the Wald pair. The framework's `(TiltingScheme x TestStatistic)`
cross-product is symmetric in the statistic dimension: every
classical frequentist pivot (Wald / LRT / Score) has a posterior
analog. `wald` + `waldo` covers the theta-direction; `lrt` + `lrto`
covers the log-density direction. Concretely:

1. `(identity, lrto)` evaluates against the un-tilted posterior —
   the "posterior-LRT" of Aitkin (1997) — and collapses to `waldo`
   on the Normal-Normal sandbox.
2. `(power_law[*], lrto)`, `(ot[*], lrto)`, etc. evaluate against
   the tilted posterior produced by the scheme; this is the slot
   where the framework's geodesic-vs-power-law experiments compose
   with the log-density-ratio pivot.
3. The FBST e-value (Pereira & Stern 1999) is essentially this
   statistic with `theta_MAP` replaced by an integrated reference;
   `lrto` is the simpler "pointwise" sibling.

`lrto` is the natural home for posterior-LRT diagnostics
(coverage, width, CD construction) the way `waldo` is the natural
home for posterior-Wald diagnostics. Future signed-root / Bartlett
machinery decorates `lrto` to produce CDs in the same way `lrt`
hosts the frequentist signed-root.

## Definition

Let `pi(. | data)` be the posterior (possibly tilted) and
`theta_MAP = argmax_theta pi(theta | data)`. Then:

    tau_LRTO(theta0; data, prior) = -2 [ log pi(theta0 | data) - log pi(theta_MAP | data) ]   (D1)

Under H_0 : theta = theta_0 the calibration is **not** chi^2_1 in
general (unlike `lrt`): under a non-flat prior the posterior is not
a likelihood, and Wilks' theorem does not apply (Aitkin 1997;
Lindley 1957). Calibration is by Monte Carlo, mirroring WALDO:

    p_LRTO(theta0; data, prior)
      = P_{D' ~ likelihood(.|theta_0)} [ tau_LRTO(theta0; D', prior) >= tau_LRTO(theta0; data, prior) ]   (D2)

estimated with `n_mc` MC draws and the `(k+1)/(n+1)` continuity
correction, same as WALDO. The CI is the inversion in theta-space:

    CI_{1-alpha} = { theta : p_LRTO(theta; data, prior) >= alpha }                          (D3)

**Normal-Normal+Normal reduction (NN sandbox).** The posterior is
Gaussian: `pi(theta | D) = N(mu_n, sigma_n^2)`. Then

    log pi(theta | D) = -(theta - mu_n)^2 / (2 sigma_n^2) + const,
    theta_MAP        = mu_n,
    tau_LRTO(theta)  = (theta - mu_n)^2 / sigma_n^2 = tau_WALDO(theta),                     (D1-NN)

identically (not asymptotically). The closed-form p-value, CI, and
acceptance region therefore coincide with WALDO's
`Phi(b - a) + Phi(-a - b)` formula (Masserano et al. 2023; see
`docs/methods/waldo.md` for the derivation).

**Tilting compatibility.** `lrto` accepts any tilting (`accepts_tilting
returns True`), matching `waldo`'s contract. The tilting machinery
calls `lrto.confidence_interval(...)` against the tilted posterior
produced by `tilting.tilt(...)`.

## Derivation

**Setup.** Let `pi(. | data)` be the posterior produced by
`model.posterior(data, prior)` (or, in a (TiltingScheme x lrto) cell, the
tilted posterior produced by the scheme). Write
`theta_MAP = argmax_theta pi(theta | data)`, assumed unique (the
posterior is unimodal on the NN sandbox; see Step 7 for the generic
case). Define `tau_LRTO`, `p_LRTO`, `CI_{1-alpha}` as in (D1)/(D2)/(D3).

**Step 1 (NN log-posterior is quadratic).** On
`NormalNormalModel(sigma) + NormalDistribution(mu0, sigma_0)`,
`posterior_params(D, mu0, sigma, sigma_0)` returns `(mu_n, sigma_n, w)`
with `w = sigma_0^2 / (sigma^2 + sigma_0^2)`, `mu_n = w*D + (1-w)*mu0`,
`sigma_n^2 = w*sigma^2`. The posterior `pi(theta | D) = N(mu_n,
sigma_n^2)` gives

    log pi(theta | D) = -(theta - mu_n)^2 / (2 sigma_n^2) + const.   (S1)

**Step 2 (theta_MAP = mu_n, uniquely).** Differentiate (S1):
`d log pi / d theta = -(theta - mu_n) / sigma_n^2`, which vanishes
only at `theta = mu_n`. The second derivative is `-1/sigma_n^2 < 0`,
so `mu_n` is the unique global maximiser.

*Verification (sympy).* With `w = sigma0^2/(sigma^2 + sigma0^2)`,
`mu_n = w*D + (1-w)*mu0`:

    >>> sp.solve(sp.diff(log_pi, theta), theta)
    [(D*sigma0**2 + mu0*sigma**2)/(sigma**2 + sigma0**2)]
    >>> sp.simplify(_[0] - mu_n)
    0

**Step 3 (exact collapse to WALDO).** Substitute (S1) into (D1):

    tau_LRTO(theta_0)
      = -2 [ log pi(theta_0 | D) - log pi(mu_n | D) ]
      = -2 [ -(theta_0 - mu_n)^2 / (2 sigma_n^2) - 0 ]
      = (theta_0 - mu_n)^2 / sigma_n^2
      = tau_WALDO(theta_0).                                          (S2)

The additive constants cancel and the quadratic expansion is exact
(no remainder), so the equivalence is identity, not asymptotic.

*Verification (sympy).*

    >>> tau_lrto = -2*(log_pi.subs(theta, theta0) - log_pi.subs(theta, mu_n))
    >>> sp.simplify(tau_lrto - (mu_n - theta0)**2 / sigma_n_sq)
    0

**Step 4 (p-value identity, numerical).** Because `tau_LRTO == tau_WALDO`
pointwise AND the MC reference distributions coincide (both sample
`D' ~ likelihood(.|theta_0)` and recompute the same posterior summary
of `pi(.|D')`), `p_LRTO == p_WALDO` for every `(theta_0, D, mu0,
sigma, sigma_0)`. On the closed-form NN+Normal path that means the
identical Masserano-style formula

    a = |mu_n - theta_0| / (w * sigma),
    b = (1 - w) * (mu_0 - theta_0) / (w * sigma),
    p_LRTO(theta_0) = Phi(b - a) + Phi(-a - b).

*Verification (numerical, exact).* With the WALDO closed-form
`_closed_form_pvalue`:

| sigma | sigma_0 | mu_0 |    D  | theta_0 |    tau_LRTO   |     p_LRTO     |  |p_LRTO - p_WALDO| |
|-------|---------|------|-------|---------|---------------|----------------|---------------------|
|  1.0  |   1.0   |  0.0 |  0.5  |   0.3   |  0.0050000000 |  0.9238379678  |  0.00e+00           |
|  2.0  |   0.5   |  1.0 | -1.3  |   2.7   | 14.3152941176 |  0.0227501319  |  0.00e+00           |
|  0.5  |   2.0   | -1.0 |  4.0  |   4.1   |  0.6601470588 |  0.4908465623  |  0.00e+00           |

**Step 5 (why H_0 is not chi^2_1 in general).** Wilks (1938) applies
to `-2 log[L(theta_0)/L(theta_hat)]` under data drawn from
`L(.|theta_0)` — the density in the likelihood ratio matches the
data-generating density. `tau_LRTO` replaces `L` with the posterior
`pi(.|D) ∝ L(.|D) * pi_0(.)`, which carries a `pi_0(theta_0) /
pi_0(theta_MAP)` factor. Data are still drawn from `L(.|theta_0)`,
not from a density proportional to `pi(.|D)`, so the score / Hessian
identities underwriting Wilks' second-order expansion no longer
hold (Aitkin 1997; Lindley 1957). The null distribution of
`tau_LRTO` therefore depends on the prior and on the conflict
`(mu0 - theta_0)/sigma_0` — exactly the quantity the WALDO MC
reference (D2) computes. The chi^2_1 limit is recovered only when
the prior contribution vanishes (Step 6).

**Step 6 (flat-prior limit).** As `sigma_0 -> infty`, `w -> 1`,
`mu_n -> D`, `sigma_n^2 -> sigma^2`, so

    tau_LRTO -> (D - theta_0)^2 / sigma^2 = tau_LRT = tau_Wald.

Equivalently, the WALDO p-value `Phi(b - a) + Phi(-a - b)` has
`b -> 0` and `a -> |D - theta_0|/sigma`, so

    p_LRTO -> Phi(-|D - theta_0|/sigma) + Phi(-|D - theta_0|/sigma)
            = 2 (1 - Phi(|D - theta_0|/sigma))
            = p_LRT.

*Verification (sympy + numerical).*

    >>> sp.limit(w, sigma0, oo);  sp.limit(mu_n, sigma0, oo);
    1;  D;  sigma**2
    >>> sp.simplify(sp.limit(tau_lrto, sigma0, oo) - (D-theta0)**2/sigma**2)
    0

With `(sigma=1, mu0=0.5, D=0.7, theta_0=-0.2)`, `p_LRT = 0.368120251`:

| sigma_0 |     w    |   mu_n   |     p_LRTO     |  |p_LRTO - p_LRT|  |
|---------|----------|----------|----------------|---------------------|
|     1.0 | 0.500000 | 0.600000 |  0.194784235   |       1.73e-01      |
|    10.0 | 0.990099 | 0.698020 |  0.364418548   |       3.70e-03      |
|   100.0 | 0.999900 | 0.699980 |  0.368083001   |       3.72e-05      |
|  1000.0 | 0.999999 | 0.700000 |  0.368119878   |       3.73e-07      |
| 10000.0 | 1.000000 | 0.700000 |  0.368120247   |       3.73e-09      |

Convergence is `O(1/sigma_0^2)` — each decade in `sigma_0` cuts the
error by 100x — confirming the analytical limit.

**Step 7 (generic-path mode finding).** For a generic (possibly
tilted) posterior `pi(. | data)`, `theta_MAP` is computed as
`scipy.optimize.minimize_scalar(lambda th: -posterior.logpdf(th),
bounds=model.support(), method="bounded")` (with a small grid
warm-start when the support is unbounded). The implementation
**assumes the (tilted) posterior is unimodal on `model.support()`**;
multimodality breaks both the uniqueness of `theta_MAP` and the
strict monotonicity of `tau_LRTO(theta)` away from the mode. On the
NN sandbox the (un-tilted and PowerLawTilting-tilted) posterior is
always unimodal so this assumption holds. The generic-path MC
reference and CI inversion reuse WALDO's CRN seed discipline
(`_stable_seed` over `(data, model, prior, alpha, seed)`,
independent of theta) so brentq probes nest cleanly across the
inversion.

**Step 8 (H_0 calibration on NN closed form is exact U[0,1]).**
Even though Step 5 says the chi^2_1 limit fails generically, on the
closed-form NN+Normal path `p_LRTO == p_WALDO` and the latter is
exactly U[0,1] under H_0 (a closed-form fact: `tau_WALDO` is a
weighted non-central chi^2_1 in `D | theta_0`, the WALDO p-value is
its survival function, and a CDF transform of a continuous random
variable is uniform). The closed-form NN+Normal path is therefore
exact-calibrated; the generic path is conservative by the
`(k+1)/(n+1)` continuity correction.

*Verification (numerical, KS uniform).* `(sigma=1.5, sigma_0=0.7,
mu_0=0.5, theta_true=1.2)`, `n=2000` draws:
`KS D=0.0181, p=0.522`; `mean(p)=0.4928` (expected 0.5);
`var(p)=0.0838` (expected `1/12 = 0.0833`).

**Invariants checked numerically (atol <= 1e-12):**
- `(sigma=1.0, sigma_0=1.0, mu_0=0.0, D=0.5, theta_0=0.3)`:
  `tau_LRTO - tau_WALDO == 0`, `p_LRTO - p_WALDO == 0`.
- `(sigma=2.0, sigma_0=0.5, mu_0=1.0, D=-1.3, theta_0=2.7)`:
  `tau_LRTO == 14.31529`, `p_LRTO == p_WALDO` exactly.
- `(sigma=0.5, sigma_0=2.0, mu_0=-1.0, D=4.0, theta_0=4.1)`:
  `p_LRTO == p_WALDO` exactly; `p(mu_n) == 1` (since `a(mu_n) = 0`).
- Flat-prior limit: `|p_LRTO - p_LRT| = O(sigma_0^{-2})`,
  confirmed at `sigma_0 in {10, 100, 1000, 10000}` with diff
  `{3.7e-3, 3.7e-5, 3.7e-7, 3.7e-9}`.
- H_0 uniformity: `KS p = 0.522` on 2000 draws at
  `(sigma=1.5, sigma_0=0.7, theta_true=1.2)`.

## Predicted behavior

- **On NN+Normal sandbox**: `lrto.pvalue == waldo.pvalue` and
  `lrto.confidence_interval == waldo.confidence_interval` to
  numerical precision for every `(alpha, data, prior, model)`.
  Pinned by property tests + regression dispatch.
- **Under flat-prior limit on NN**: as `sigma_0 -> infty`, the
  posterior degenerates to the likelihood (Aitkin 2010, §2.1),
  `mu_n -> D`, `sigma_n -> sigma`, so `tau_LRTO -> tau_LRT =
  tau_Wald` and the p-value recovers
  `2(1 - Phi(|D - theta|/sigma))`. Same limit as `waldo -> wald`.
- **Generic path**: any `(Model, Prior)` exposing `posterior(data,
  prior).logpdf` and a one-dim parameter support yields a valid
  CI. Cost is `O(n_mc)` per brentq probe; CI inversion is `O(n_mc
  * brentq_iters)`. CRN MC discipline mirrors WALDO so brentq
  actually converges instead of locking onto a re-randomised
  staircase.

## Failure modes

- **Multimodal (tilted) posterior**: the `theta_MAP` argmax is
  not unique; the brent-optimise mode-finder converges to one
  basin. Caller's responsibility (no `confidence_regions`
  semantics on `LRTOStatistic` itself; the `TiltingScheme` layer
  owns multi-region CIs). On the conjugate Normal-Normal sandbox
  the posterior is unimodal so this never triggers.
- **Degenerate posterior variance**: `sigma_n -> 0` (data
  perfectly informative) gives `tau_LRTO -> infty` for any
  `theta != mu_n`. Match WALDO's NaN-on-degenerate contract; the
  generic-path MC reference filters NaN draws.
- **Posterior boundary mode**: when `theta_MAP` is on the support
  boundary the Wilks/Wald approximation around the mode breaks
  (parallels LRT's Chernoff 1954 caveat; see `docs/methods/lrt.md`).
  Not relevant for NN (parameter space is R).
- **`n_mc` too small at small alpha**: tail-MC error scales like
  `1/sqrt(n_mc * alpha)`; the default `n_mc=2000` gives ~0.022 SE
  on `p ~ 0.5` but only ~0.005 on `p ~ 0.05`. CI inversion at
  `alpha=0.01` should bump `n_mc` to ~5000.
- **NormalNormalModel subclassing with overridden likelihood/posterior**:
  same risk as WALDO/Wald — the closed-form dispatch reads only the
  fingerprint. Mitigation: same options as `lrt` (avoid subclass,
  override `fingerprint`, or `force_generic=True`).
- **Fat-tailed posteriors with unbounded support (generic path)**:
  `_find_theta_map`'s slow-path bracket is `mean +/- 10 *
  sqrt(var)`. For thick-tailed posteriors where the MAP sits many
  sigmas from the mean (heavy-tailed mixtures, future Cauchy-style
  posteriors with `var()` returning `inf`), the bracket can miss
  the true mode. The `np.maximum(tau, 0.0)` clamp only fires when
  a test theta happens to hit a region with higher logpdf than the
  returned MAP, so an undetected wrong-MAP can propagate silently.
  Mitigation: future work would adaptively double the bracket
  outward until logpdf crests.
- **Near-flat posteriors (generic path)**: when the posterior is
  almost flat (highly uninformative likelihood + diffuse prior),
  `scipy.optimize.minimize_scalar` returns an arbitrary point in
  the flat region. `res.success` is currently not inspected. The
  resulting CI inversion typically fails to bracket and falls back
  to `model.support()` with a `UserWarning`; downstream consumers
  should treat that warning as "the data + prior gave no
  information at this alpha".
- **CI bracket assumes `posterior.var()` is finite**:
  `_generic_confidence_interval` uses `4 * sqrt(posterior.var())`
  as the initial half-width. For posteriors with infinite variance
  (Cauchy, Student-t with df ≤ 2), the bracket is infinite and
  `brentq_with_doubling` cannot make progress. The
  bracket-exhaustion warning fires and the CI is returned as
  `model.support()`.
- **Gaussian fast-path detection is class-tight**: the
  `posterior_moments_batch` fast path runs only when
  `isinstance(posterior, NormalDistribution)` (not just
  `hasattr(.loc) and hasattr(.scale)`) — to prevent a future
  location-scale wrapper (Student-t, lognormal, etc.) from
  silently being treated as Gaussian and producing a miscalibrated
  H_0 MC reference. New Gaussian-class wrappers must explicitly be
  added to `_is_gaussian_posterior`.

## Invariants

1. **p-value range.** `p_LRTO(theta_0; data, prior) in [0, 1]` for
   all `(theta_0, data, prior, model)` — consequence of the MC
   empirical formula and the WALDO CDF identity on the closed-form
   path.
2. **Non-negativity.** `tau_LRTO(theta_0; data, prior) >= 0` for
   all `theta_0`, with equality at `theta_0 = theta_MAP` (by
   definition of the MAP).
3. **Mode property.** `p_LRTO(theta_MAP; data, prior) == 1` on the
   NN+Normal closed-form path (where `theta_MAP = mu_n` so
   `a(mu_n) = 0` and `Phi(b) + Phi(-b) = 1`); MC-tolerant on the
   generic path.
4. **NN equivalence — p-value.** On `NormalNormalModel +
   NormalDistribution`,
   `lrto.pvalue(theta_0, data, model, prior) ==
   waldo.pvalue(theta_0, data, model, prior)` elementwise
   (`atol <= 1e-12` on the closed-form path; MC tolerance with
   shared CRN seed on `force_generic=True`).
5. **NN equivalence — CI.** On `NormalNormalModel +
   NormalDistribution`,
   `lrto.confidence_interval(alpha, data, model, prior) ==
   waldo.confidence_interval(...)` to `atol <= 1e-8` on the closed-
   form path.
6. **Tilting compatibility.** `lrto.accepts_tilting(tilting) is True`
   for every concrete `TiltingScheme` (identity, power_law, ot,
   mixture, fisher_rao) — `lrto` is prior-aware by design and
   mirrors `waldo`'s contract.
7. **Flat-prior limit.** On `NormalNormalModel`, as `sigma_0 ->
   infty`, `lrto.pvalue -> lrt.pvalue` and
   `lrto.confidence_interval -> lrt.confidence_interval`. Pinned
   by property test at `sigma_0 in {100, 1000, 10000}` with
   tolerance scaling as `O(sigma_0^{-2})`.
8. **H_0 uniformity on NN closed form.** Drawing
   `D ~ N(theta_true, sigma^2)` and evaluating
   `lrto.pvalue(theta_true; D, model, prior)` gives exact
   `Uniform[0,1]` samples; KS uniformity test passes at
   `alpha = 0.01` on n=2000 draws (L3, marker `L3`). This holds
   DESPITE the asymptotic chi^2_1 statement failing in general
   (Derivation Step 5 + 8).
9. **Cross-pair equivalence on NN — p-value + CI level.** On the
   NN+Normal closed-form path, `lrto.pvalue == waldo.pvalue` and
   `lrto.confidence_interval == waldo.confidence_interval` for
   every `(theta, alpha, data, model, prior)`. Pinned by
   `test_matches_waldo_pvalue_on_nn` / `test_matches_waldo_ci_on_nn`
   (property tests) and
   `test_pvalue_matches_waldo_on_nn_closed_form` (regression). At
   the runner / RawResult level cell-name and statistic-class
   metadata necessarily differ; the cross-product machinery is
   common discipline shared with WALDO and is not pinned by an
   lrto-specific test.

## Literature

### Foundational (Bayesian-LR framing)

```bibtex
@article{dempster1997,
  author  = {Dempster, A. P.},
  title   = {The Direct Use of Likelihood for Significance Testing},
  journal = {Statistics and Computing},
  volume  = {7},
  number  = {4},
  pages   = {247--252},
  year    = {1997},
  doi     = {10.1023/A:1018598421607}
}
```
Republication of Dempster's 1974 proposal that the posterior
distribution of the likelihood ratio be used directly for
significance testing — origin of the posterior-LR idea `lrto`
instantiates. Shows that in the Gaussian case the posterior LR
coincides with the frequentist p-value, the direct precedent for
`lrto`'s NN+Normal collapse.

```bibtex
@article{aitkin1997,
  author  = {Aitkin, M.},
  title   = {The Calibration of {P}-Values, Posterior {B}ayes Factors and the {AIC} from the Posterior Distribution of the Likelihood},
  journal = {Statistics and Computing},
  volume  = {7},
  number  = {4},
  pages   = {253--272},
  year    = {1997},
  doi     = {10.1023/A:1018550505678}
}
```
Closest match for the "posterior distribution of the likelihood
ratio" framing. Anchors the Definition-section claim that the H_0
calibration is **not** `chi^2_1` in general — Aitkin's whole point
is that recalibration is needed.

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
Textbook treatment of the posterior-likelihood / posterior-LR
family of statistics; the canonical modern reference for the
integrated Bayesian/likelihood approach `lrto` sits inside.

```bibtex
@article{pereirastern1999,
  author  = {Pereira, C. A. B. and Stern, J. M.},
  title   = {Evidence and Credibility: Full {B}ayesian Significance Test for Precise Hypotheses},
  journal = {Entropy},
  volume  = {1},
  number  = {4},
  pages   = {99--110},
  year    = {1999},
  doi     = {10.3390/e1040099}
}
```
FBST e-value — the integrated-tangential-set sibling of `lrto`'s
pointwise posterior-LR. Direct precedent for the Bayesian-
significance-from-posterior-density-ratio idea.

```bibtex
@article{madruga2001,
  author  = {Madruga, M. R. and Esteves, L. G. and Wechsler, S.},
  title   = {On the {B}ayesianity of {P}ereira--{S}tern Tests},
  journal = {Test},
  volume  = {10},
  number  = {2},
  pages   = {291--299},
  year    = {2001},
  doi     = {10.1007/BF02595698}
}
```
Decision-theoretic axiomatization of the FBST: exhibits a loss
function whose Bayes rule is the Pereira--Stern test. Justifies
citing the FBST family as a principled Bayesian alternative.

### Closely related

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
Pairing-pattern template: WALDO's `(D' ~ likelihood | theta_0)` MC
reference under a posterior-summary statistic is exactly `lrto`'s
calibration recipe. The NN+Normal closed form collapses to WALDO's
`Phi(b−a) + Phi(−a−b)`.

```bibtex
@article{smith2007plr,
  author  = {Smith, M. and Ferrari, S. and Heath, W. P.},
  title   = {Bayesian Point Null Hypothesis Testing via the Posterior Likelihood Ratio},
  journal = {Statistics and Computing},
  volume  = {17},
  number  = {1},
  pages   = {59--74},
  year    = {2007},
  doi     = {10.1007/s11222-005-1310-0}
}
```
Direct follow-on to Dempster/Aitkin extending the posterior-LR to
point-null testing with nuisance parameters.

```bibtex
@book{schwederhjort2016,
  author    = {Schweder, T. and Hjort, N. L.},
  title     = {Confidence, Likelihood, Probability: Statistical Inference with Confidence Distributions},
  publisher = {Cambridge University Press},
  year      = {2016},
  doi       = {10.1017/CBO9781139046671}
}
```
CD-from-posterior-LR construction; the Bayesian-confidence /
fiducial-LRT bridge relevant when `lrto` is composed with the
`confidence_distribution` pipeline. Cross-references `lrt.md`.

```bibtex
@article{chernoff1954,
  author  = {Chernoff, H.},
  title   = {On the Distribution of the Likelihood Ratio},
  journal = {The Annals of Mathematical Statistics},
  volume  = {25},
  number  = {3},
  pages   = {573--578},
  year    = {1954},
  doi     = {10.1214/aoms/1177728725}
}
```
Half-chi-squared boundary distribution; anchor for the
boundary-mode Failure mode (parallels `lrt.md`).

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
The original Lindley/Jeffreys paradox: Bayesian and frequentist
hypothesis tests can disagree arbitrarily under prior choice.
Direct anchor for the "asymptotic null is **not** `chi^2_1` for
posterior LR in general" claim.

```bibtex
@article{fraser2011,
  author  = {Fraser, D. A. S.},
  title   = {Is {B}ayes Posterior just Quick and Dirty Confidence?},
  journal = {Statistical Science},
  volume  = {26},
  number  = {3},
  pages   = {299--316},
  year    = {2011},
  doi     = {10.1214/11-STS352}
}
```
Posterior coincides with confidence only under model linearity;
departures cause Bayesian intervals to be poor frequentist
approximations. Counterweight to any reading of `lrto` as
automatically a confidence procedure.

```bibtex
@article{bernardo2005,
  author  = {Bernardo, J. M.},
  title   = {Intrinsic Credible Regions: An Objective {B}ayesian Approach to Interval Estimation},
  journal = {Test},
  volume  = {14},
  number  = {2},
  pages   = {317--384},
  year    = {2005},
  doi     = {10.1007/BF02595408}
}
```
Reference-prior / intrinsic-discrepancy alternative; another
principled Bayesian-confidence pivot that competes with `lrto`.

## Links

- Implementation: `src/frasian/statistics/lrto.py`
- Property tests: `tests/properties/test_lrto_invariants.py`
- Regression tests: `tests/regression/test_force_generic_dispatch.py::TestLRTOForceGeneric`
- Illustration: `src/frasian/experiments/illustrations/lrto_demo.py`
- Generated figure: `output/illustrations/lrto_demo.png`

## Status notes

`lrto` collapses to `waldo` on the Normal-Normal+Normal sandbox by
construction; the existence of a separate implementation is
justified by (a) the cross-product cell, (b) the generic-path
posterior-LRT story for future non-Gaussian models / non-conjugate
posteriors, and (c) the LRT pair (`lrt` + `lrto`) being the
natural home for `signed_root` and `bartlett` once those land.
Tests assert the NN+Normal collapse so any future regression that
breaks the equivalence is caught immediately.
