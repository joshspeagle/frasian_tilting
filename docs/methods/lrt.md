# lrt

> Status: `implemented`

## Summary

Frequentist likelihood-ratio test statistic

    tau_LRT(theta0) = -2 [ log L(theta0) - log L(theta_hat_MLE) ]

with asymptotic null distribution `chi^2_1` (Wilks 1938). It is the
canonical likelihood-based pivot — Wald is its second-order Taylor
expansion around `theta_hat`, the score test is the first-order
expansion at `theta0` (Casella & Berger 2002, §10.3;
van der Vaart 1998, Theorem 16.7). Identity-tilting only: LRT
ignores the prior.

## Motivation

The framework's `(TestStatistic x TiltingScheme)` cross-product
needs LRT to span the three classical asymptotically-equivalent
test families (Wald / LRT / Score). LRT generalises beyond the
Normal-location family in a way Wald does not (Wald demands a
quadratic loglikelihood; LRT does not), which is why this slot
matters once non-Gaussian models land. On the Normal-Normal
sandbox `lrt` is numerically identical to `wald` cell-for-cell;
its differentiating value is

1. the generic path with `chi^2_1` calibration that any future
   non-Gaussian `Model` plugs into without code changes, and
2. exercising the cross-product machinery on a statistic whose
   calibration is exact only asymptotically (vs Wald's exact-on-NN
   closed form).

LRT generalises beyond the Normal-location family in a way Wald
does not — Wald demands a quadratic loglikelihood while LRT does
not (van der Vaart 1998, §16.3).

`lrt` is the **frequentist** half of the LRT pair; `lrto` (the
WALDO-analog on the tilted posterior, accepting any tilting) is
the other half, following the pairing template of Masserano et al.
(2023).

## Definition

For a `Model` with likelihood `L(theta; data)` and MLE
`theta_hat = argmax_theta L(theta; data)`,

    tau_LRT(theta0; data) = -2 [ log L(theta0; data) - log L(theta_hat; data) ]   (D1)

Under H_0 : theta = theta_0, Wilks' theorem (Wilks 1938) gives
`tau_LRT --d--> chi^2_1` as the sample size grows. The p-value is

    p_LRT(theta0; data) = 1 - F_{chi^2_1}(tau_LRT)                                (D2)

and the two-sided (1 - alpha) confidence interval is the level set

    CI_{1-alpha} = { theta : tau_LRT(theta; data) <= q_{chi^2_1, 1-alpha} }       (D3)

where `q_{chi^2_1, 1-alpha}` is the upper alpha-quantile of `chi^2_1`.

**Normal-location reduction (NN sandbox).** For
`Model = NormalNormalModel(sigma)` with sufficient statistic
`D = mean(data)` the loglikelihood is exactly quadratic with
curvature `1/sigma^2`, so (D1) collapses to

    tau_LRT(theta0) = ((D - theta0)/sigma)^2 = tau_Wald(theta0)                   (D1-NN)

and `p_LRT == p_Wald`, `CI_LRT == CI_Wald = D +/- z_{1-alpha/2} sigma`
exactly.

**Tilting compatibility.** `lrt` ignores the prior and only accepts
`identity` tilting (`accepts_tilting(tilting) == (tilting.name == "identity")`),
matching `wald`'s contract. The prior-aware variant is `lrto`.

## Derivation

**Setup.** Let `Model` be a one-dim regular parametric family with
loglikelihood `l(theta; data) = log L(theta; data)`, score
`U(theta) = d l / d theta`, Fisher information `I(theta) > 0`, and
interior MLE `theta_hat = argmax_theta l(theta; data)`. Write
`tau_LRT(theta0) = -2 [ l(theta0) - l(theta_hat) ]` as in (D1).

**Step 1 (Wilks 1938).** Under H_0 : theta = theta_0 and regularity
(third-derivative-bounded `l`; `theta_0` an interior point of an
open parameter space; `I(theta_0)` finite and positive; MLE
consistent and asymptotically efficient), as `n -> infty`

    tau_LRT(theta_0)  --d-->  chi^2_1.

(Wilks 1938; van der Vaart 1998 Thm 16.7; Casella & Berger 2002
§10.3.1.) The p-value (D2) and CI (D3) follow by the standard
test-inversion duality.

**Step 2 (Wald-LRT-Score equivalence).** Taylor-expand `l` at
`theta_hat`:

    l(theta_0) = l(theta_hat) - (1/2)(theta_hat - theta_0)^2 I(theta_hat)
                 + O((theta_hat - theta_0)^3).

Multiplying by -2 gives `tau_LRT = (theta_hat - theta_0)^2 I(theta_hat)
+ O_p(n^{-1/2}) = tau_Wald + o_p(1)`. A second expansion at
`theta_0` using the score yields `tau_Score = U(theta_0)^2 / I(theta_0)
= tau_LRT + o_p(1)`. All three converge to the same `chi^2_1` limit
(van der Vaart 1998 Thm 16.7).

**Step 3 (NN-sandbox: exact collapse to Wald).** For
`NormalNormalModel(sigma)` the sufficient statistic is
`D = mean(observations)` with `D | theta ~ N(theta, sigma^2)`
(framework's n=1 convention; see `NormalNormalModel.likelihood`).
The loglikelihood is

    l(theta; D) = -(1/2) log(2 pi sigma^2) - (1/2) ((D - theta)/sigma)^2,   (S1)

quadratic in `theta` with curvature `1/sigma^2`. The MLE is
`theta_hat = D` (the unique root of `d l / d theta = (D - theta)/sigma^2`),
matching `NormalNormalModel.mle`. Substituting into (D1):

    tau_LRT(theta_0)
      = -2 [ l(theta_0; D) - l(D; D) ]
      = -2 [ -(1/2)((D - theta_0)/sigma)^2  -  0 ]
      = ((D - theta_0)/sigma)^2
      = tau_Wald(theta_0).                                                  (S2)

The constant `-(1/2) log(2 pi sigma^2)` cancels and the quadratic
expansion is exact (no `O(.)` remainder), so the equivalence holds
identically, not just asymptotically.

*Verification (sympy).* With `l(theta) = -log(2 pi sigma^2)/2 -
((D - theta)/sigma)^2 / 2`:

    >>> sp.solve(sp.diff(l(theta), theta), theta)
    [D]
    >>> sp.simplify(-2*(l(theta0) - l(D)) - ((D - theta0)/sigma)**2)
    0

So `tau_LRT - tau_Wald == 0` symbolically. The Fisher info
`I(theta) = 1/sigma^2` then gives `tau_Wald = (mle - theta_0)^2 *
I(theta_0)`, matching the generic-path formula in
`wald._generic_evaluate`.

**Step 4 (p-value identity, numerical).** With `Z = (D - theta_0)/sigma
~ N(0, 1)` under H_0, `tau_LRT = Z^2 ~ chi^2_1` exactly. The
`chi^2_1` survival function satisfies the identity
`1 - F_{chi^2_1}(z^2) = 2 (1 - Phi(|z|))`, so `p_LRT == p_Wald`
elementwise.

*Verification (numerical, atol < 1e-12).*

| sigma | D     | theta0 |       p_LRT       |       p_Wald      |
|-------|-------|--------|-------------------|-------------------|
| 1.0   |  0.0  |  0.5   | 6.17075077e-01    | 6.17075077e-01    |
| 2.0   | -1.3  |  2.7   | 4.55002639e-02    | 4.55002639e-02    |
| 0.5   |  4.0  |  4.1   | 8.41480581e-01    | 8.41480581e-01    |
| 1.5   | -2.0  | -2.0   | 1.0               | 1.0               |
| 1.0   |  0.0  |  100.0 | 0.0               | 0.0               |

Max `|p_LRT - p_Wald|` = 1.11e-16 (one ULP).

**Step 5 (generic-path inversion).** For a generic `Model` the CI
(D3) is `{ theta : tau_LRT(theta) <= q }` with
`q = q_{chi^2_1, 1-alpha}`. Under unimodal `l(.; data)` (so
`theta_hat` is unique), `tau_LRT(theta)` is strictly increasing in
`|theta - theta_hat|`: writing `g(theta) = l(theta_hat) - l(theta) >= 0`,
`g' = -U`, which vanishes only at `theta_hat`, so `g` is
strictly increasing for `theta > theta_hat` and strictly decreasing
for `theta < theta_hat`. The level set is therefore the interval
`[theta_lo, theta_hi]` where `tau_LRT(theta_{lo, hi}) = q` on each
side of `theta_hat`. The implementation mirrors
`wald._generic_confidence_interval`: brentq with a doubling bracket
initialised at `4 / sqrt(I(theta_hat))` (the Wald half-width;
expanded by the solver if `tau_LRT` flattens before crossing q),
clamped to `model.support()`. Multimodal `l` violates the
strict-monotone assumption and must use the
`confidence_regions` semantics (Failure modes).

**Invariants checked numerically (atol <= 1e-12):**
- `(sigma=1.0, D=0.0, theta0=0.5)`: `p_LRT == p_Wald` (1.11e-16).
- `(sigma=2.0, D=-1.3, theta0=2.7)`: `p_LRT == p_Wald` (1.11e-16).
- `(sigma=0.5, D=4.0, theta0=4.1)`: `p_LRT == p_Wald` (0).
- `(sigma=1.5, D=-2.0, theta0=-2.0)`: `tau_LRT == 0`, `p_LRT == 1`.
- Sympy: `tau_LRT(theta_0) - ((D - theta_0)/sigma)**2 == 0` symbolically.

## Predicted behavior

- **On NN sandbox**: `lrt.pvalue == wald.pvalue` and
  `lrt.confidence_interval == wald.confidence_interval` to numerical
  precision. Regression-pinned.
- **Under H_0** on NN: `tau_LRT ~ chi^2_1` exactly (not just
  asymptotically), so the p-value is `Uniform[0,1]` and the CI
  covers at the nominal level.
- **Generic path**: any `Model` exposing `mle(data)`,
  `loglik(theta, data)`, and a one-dim parameter space yields a
  valid CI; calibration is `chi^2_1` asymptotic, so finite-sample
  coverage degrades slightly off-NN until a Bartlett correction
  (future `bartlett` method, after Bartlett 1937) is applied.

## Failure modes

- **Multimodal likelihoods**: `tau_LRT(theta) <= c` is not
  necessarily an interval; on the NN sandbox the loglikelihood is
  strictly unimodal so this cannot occur, but a generic-path CI on
  a future multimodal model must use `confidence_regions` semantics
  (see CLAUDE.md "CI region semantics") rather than
  `confidence_interval`.
- **MLE on the boundary**: when `theta_hat` lies at a support
  boundary the chi^2 calibration breaks — a half-chi-squared
  distribution applies (Chernoff 1954). Not relevant for NN
  (parameter space is the real line) but matters for any future
  bounded-support model.
- **Generic-path bracket failure**: as in `wald._generic_confidence_interval`,
  pathological Fisher information (zero / infinite / NaN) at `theta_hat`
  has to fall back to a support-width-based bracket. Inherit the
  same fallback policy.

## Invariants

1. **p-value range.** `p_LRT(theta_0; data) in [0, 1]` for all
   `theta_0` and all `data` (consequence of `tau >= 0` and the
   `chi^2_1` survival function having range `[0, 1]`).
2. **Non-negativity.** `tau_LRT(theta_0; data) >= 0` for all
   `theta_0`, since `l(theta_hat) >= l(theta)` by definition of the
   MLE. Equality at `theta_0 = theta_hat`.
3. **Mode property.** `p_LRT(theta_hat; data) == 1` exactly
   (`tau == 0` plugged into `1 - F_{chi^2_1}`). Tested on NN with
   `theta_hat = D`.
4. **NN equivalence — p-value.** On `NormalNormalModel`,
   `lrt.pvalue(theta_0, data, model) == wald.pvalue(theta_0, data,
   model)` for any `(theta_0, data, sigma)`, to `atol <= 1e-12`.
   This is exact (not asymptotic): see Derivation Step 3.
5. **NN equivalence — CI.** On `NormalNormalModel`,
   `lrt.confidence_interval(alpha, data, model) ==
   wald.confidence_interval(alpha, data, model)` for any
   `(alpha, data, sigma)`, to `atol <= 1e-8` (CI from numerical
   inversion is tighter than the closed-form 1e-12 of (4)). Pinned
   in `tests/regression/test_lrt_matches_wald.py`.
6. **Exact `chi^2_1` calibration on NN under H_0.** Drawing
   `data ~ N(theta_true, sigma^2)` and evaluating
   `tau_LRT(theta_true; data)` gives an exact `chi^2_1` sample
   (Derivation Step 3 + 4); KS test of `tau_LRT` against `chi^2_1`,
   and of `p_LRT` against `Uniform[0, 1]`, both pass at the nominal
   level (L3, marker `L3`).
7. **Monotonicity on NN.** `p_LRT(theta_0; data)` is strictly
   decreasing in `|theta_0 - D|` (since `tau_LRT` is strictly
   increasing in `|theta_0 - D|` and `1 - F_{chi^2_1}` is strictly
   decreasing on `(0, infty)`).
8. **Tilting compatibility.** `lrt.accepts_tilting(tilting) is True`
   iff `tilting.name == "identity"`; non-identity tiltings
   (`power_law`, `ot`, `mixture`, `fisher_rao`) return `False`.
   Mirrors `wald`'s contract.

## Literature

### Foundational

```bibtex
@article{wilks1938,
  author  = {Wilks, S. S.},
  title   = {The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses},
  journal = {The Annals of Mathematical Statistics},
  volume  = {9},
  number  = {1},
  pages   = {60--62},
  year    = {1938},
  doi     = {10.1214/aoms/1177732360}
}
```
Establishes `-2 log Lambda --d--> chi^2_k` under regularity conditions;
the calibration assertion in (D2).

```bibtex
@article{neymanpearson1928,
  author  = {Neyman, J. and Pearson, E. S.},
  title   = {On the Use and Interpretation of Certain Test Criteria for Purposes of Statistical Inference, Part I},
  journal = {Biometrika},
  volume  = {20A},
  number  = {1-2},
  pages   = {175--240},
  year    = {1928},
  doi     = {10.1093/biomet/20A.1-2.175}
}
```
Introduces the likelihood-ratio criterion
`Lambda = sup_{H_0} L / sup L`.

```bibtex
@article{neymanpearson1933,
  author  = {Neyman, J. and Pearson, E. S.},
  title   = {On the Problem of the Most Efficient Tests of Statistical Hypotheses},
  journal = {Philosophical Transactions of the Royal Society of London, Series A},
  volume  = {231},
  pages   = {289--337},
  year    = {1933},
  doi     = {10.1098/rsta.1933.0009}
}
```
Optimality (Neyman-Pearson lemma) of LR tests; justifies LRT as the
canonical likelihood-based pivot.

```bibtex
@article{wald1943,
  author  = {Wald, A.},
  title   = {Tests of Statistical Hypotheses Concerning Several Parameters When the Number of Observations is Large},
  journal = {Transactions of the American Mathematical Society},
  volume  = {54},
  number  = {3},
  pages   = {426--482},
  year    = {1943},
  doi     = {10.1090/S0002-9947-1943-0012401-3}
}
```
Wald statistic; anchors the Wald/LRT/Score equivalence trio asserted
in the Summary.

```bibtex
@article{rao1948,
  author  = {Rao, C. R.},
  title   = {Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation},
  journal = {Proceedings of the Cambridge Philosophical Society},
  volume  = {44},
  number  = {1},
  pages   = {50--57},
  year    = {1948},
  doi     = {10.1017/S0305004100023987}
}
```
Score test (third leg of the trio); supports the asymptotic-equivalence
claim and orients toward the future `score` statistic.

```bibtex
@book{vandervaart1998,
  author    = {van der Vaart, A. W.},
  title     = {Asymptotic Statistics},
  publisher = {Cambridge University Press},
  series    = {Cambridge Series in Statistical and Probabilistic Mathematics},
  year      = {1998},
  doi       = {10.1017/CBO9780511802256}
}
```
Modern textbook anchor for the regularity conditions (Chapter 16:
likelihood-ratio tests) and the Wald-LRT-Score equivalence theorem
(Theorem 16.7).

```bibtex
@book{casellaberger2002,
  author    = {Casella, G. and Berger, R. L.},
  title     = {Statistical Inference},
  edition   = {2nd},
  publisher = {Duxbury},
  year      = {2002}
}
```
Textbook treatment of LRT; Chapter 10.3 derives the Wald/LRT/Score
Taylor-expansion connection used in the Summary.

### Closely related

```bibtex
@article{bartlett1937,
  author  = {Bartlett, M. S.},
  title   = {Properties of Sufficiency and Statistical Tests},
  journal = {Proceedings of the Royal Society of London, Series A},
  volume  = {160},
  number  = {901},
  pages   = {268--282},
  year    = {1937},
  doi     = {10.1098/rspa.1937.0109}
}
```
The original Bartlett-correction paper; supports the brief's
forward reference to the future `bartlett` method.

```bibtex
@article{bartlett1953,
  author  = {Bartlett, M. S.},
  title   = {Approximate Confidence Intervals. {II}. More than One Unknown Parameter},
  journal = {Biometrika},
  volume  = {40},
  number  = {3/4},
  pages   = {306--317},
  year    = {1953},
  doi     = {10.1093/biomet/40.3-4.306}
}
```
Extends the Bartlett-correction idea to multi-parameter LRT
inversion; relevant for the generic-path story.

```bibtex
@article{barndorffnielsen1986,
  author  = {Barndorff-Nielsen, O. E.},
  title   = {Inference on Full or Partial Parameters Based on the Standardized Signed Log Likelihood Ratio},
  journal = {Biometrika},
  volume  = {73},
  number  = {2},
  pages   = {307--322},
  year    = {1986},
  doi     = {10.1093/biomet/73.2.307}
}
```
The r* modification of the signed-root LRT; direct anchor for the
`signed_root` companion stub.

```bibtex
@article{barndorffnielsen1991,
  author  = {Barndorff-Nielsen, O. E.},
  title   = {Modified Signed Log Likelihood Ratio},
  journal = {Biometrika},
  volume  = {78},
  number  = {3},
  pages   = {557--563},
  year    = {1991},
  doi     = {10.1093/biomet/78.3.557}
}
```
Companion 1991 r* paper.

```bibtex
@book{schwederhjort2016,
  author    = {Schweder, T. and Hjort, N. L.},
  title     = {Confidence, Likelihood, Probability: Statistical Inference with Confidence Distributions},
  publisher = {Cambridge University Press},
  year      = {2016},
  doi       = {10.1017/CBO9781139046671}
}
```
The LRT/signed-root-as-CD viewpoint.

```bibtex
@article{coxreid1987,
  author  = {Cox, D. R. and Reid, N.},
  title   = {Parameter Orthogonality and Approximate Conditional Inference},
  journal = {Journal of the Royal Statistical Society, Series B},
  volume  = {49},
  number  = {1},
  pages   = {1--39},
  year    = {1987},
  doi     = {10.1111/j.2517-6161.1987.tb01422.x}
}
```
Profile-likelihood / orthogonal-parameter machinery used in
nuisance-parameter LRT inversion.

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
Failure-modes claim about MLE on the support boundary.

### Contrasting / Bayesian counterparts

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
Bayesian counterpart: the posterior distribution of the likelihood
ratio. Orients the reader toward the planned `lrto` method.

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
The FBST e-value — a posterior density-ratio test that competes
with `lrto`-style Bayesian LRTs.

```bibtex
@inproceedings{masserano2023waldo,
  author    = {Masserano, L. and Dorigo, T. and Izbicki, R. and Kuusela, M. and Lee, A. B.},
  title     = {Simulator-Based Inference with {WALDO}: Confidence Regions by Leveraging Prediction Algorithms and Posterior Estimators for Inverse Problems},
  booktitle = {Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {PMLR},
  volume    = {206},
  year      = {2023},
  eprint    = {2205.15680},
  archivePrefix = {arXiv}
}
```
WALDO paper; pairing-pattern template for `lrt` + `lrto`.

## Links

- Implementation: `src/frasian/statistics/lrt.py`
- Property tests: `tests/properties/test_lrt_invariants.py`
- Regression tests: `tests/regression/test_lrt_matches_wald.py` (planned)
- Illustration: `src/frasian/experiments/illustrations/lrt_demo.py`
- Generated figure: `output/illustrations/lrt_demo.png`

## Status notes

`lrt` collapses to `wald` on the Normal-Normal sandbox by
construction; the existence of a separate implementation is
justified by (a) the cross-product cell, (b) the generic-path
calibration story for future non-Gaussian models, and (c) the LRT
pair (`lrt` + `lrto`) being the natural home for `signed_root` and
`bartlett` once those land. Tests assert the NN collapse so any
future regression that breaks the equivalence is caught immediately.
