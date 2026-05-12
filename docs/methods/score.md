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

**Setup.** Let `Model` be a one-dim regular parametric family
with log-likelihood `l(theta; data) = log L(theta; data)`. Write
the score `U(theta) = d l / d theta` and the (expected) Fisher
information `I(theta) = E_theta[U(theta)^2] =
-E_theta[d^2 l / d theta^2]`. The framework's
`model.fisher_information(theta)` returns the *expected*
information. Regularity (as for `lrt`): identifiable family,
twice-differentiable `l` in `theta`, finite and strictly positive
`I(theta_0)`, interior `theta_0`, MLE consistent in the interior
of the support (Casella & Berger 2002 §10.6.2; van der Vaart 1998
§7.2). Define

    tau_Score(theta_0; data) = U(theta_0)^2 / I(theta_0)           (D3)

as in the brief.

**Step 1 (Rao 1948 — asymptotic chi^2_1).** Under H_0 : theta =
theta_0, regularity gives `U(theta_0) / sqrt(n) -d-> N(0,
I_1(theta_0))` (CLT applied to the i.i.d. score-summands), hence
`U(theta_0) / sqrt(n I_1(theta_0)) -d-> N(0,1)`, and squaring,

    tau_Score(theta_0)  =  U(theta_0)^2 / I_n(theta_0)  -d->  chi^2_1,

where `I_n = n I_1` is the total expected information at sample
size `n` (Rao 1948; van der Vaart 1998 Thm 16.7). The p-value
(D5) and CI (D6) follow by test-inversion duality.

**Step 2 (NN n=1 — exact collapse to Wald / LRT).** For
`NormalNormalModel(sigma)` with `D | theta ~ N(theta, sigma^2)`,

    l(theta; D) = -(1/2) log(2 pi sigma^2) - (D - theta)^2 / (2 sigma^2).

Differentiating gives `U(theta) = (D - theta) / sigma^2`. The
expected information is `I(theta) = -E[d^2 l / d theta^2] =
1 / sigma^2`, matching `NormalNormalModel.fisher_information`.
Substituting into (D3):

    tau_Score(theta_0; D)
      = [(D - theta_0)/sigma^2]^2  /  (1/sigma^2)
      = (D - theta_0)^2 / sigma^2
      = tau_Wald(theta_0; D) = tau_LRT(theta_0; D).                (S2)

The collapse is *exact*, not asymptotic (the loglikelihood is
exactly quadratic so the Step-1 CLT is replaced by
`Z = (D - theta_0)/sigma ~ N(0,1)` exactly under H_0, and
`Z^2 ~ chi^2_1` exactly).

*Verification (sympy).* With `l(theta) = -(D - theta)^2 / (2 sigma^2)`:

    >>> U = sp.diff(l, theta);            sp.simplify(U)            # (D - theta)/sigma**2
    >>> I = -sp.diff(l, theta, 2);        sp.simplify(I)            # 1/sigma**2
    >>> sp.simplify(U.subs(theta, theta0)**2 / I.subs(theta, theta0)
    ...             - (D - theta0)**2 / sigma**2)                    # 0

So `tau_Score - tau_Wald == 0` symbolically.

*Verification (numerical, atol < 1e-12).* For `(sigma, D,
theta_0)` triples `(1.0, 0.0, 0.5)`, `(2.0, -1.3, 2.7)`,
`(0.5, 4.0, 4.1)`, `(1.5, -2.0, -2.0)`, `(1.0, 0.0, 100.0)`,
compare `1 - F_{chi^2_1}((D-theta_0)^2/sigma^2)` against
`2(1 - Phi(|D-theta_0|/sigma))`. Max `|p_score - p_wald|` =
1.11e-16 (one ULP).

**Step 3 (NN n > 1 — collapse persists).** For
`data = (D_1, ..., D_n)`,

    l(theta; data) = -sum_i (D_i - theta)^2 / (2 sigma^2) + const,
    U(theta)       = sum_i (D_i - theta)/sigma^2  =  n (Dbar - theta)/sigma^2,
    I(theta)       = n / sigma^2,
    tau_Score      = [n(Dbar - theta_0)/sigma^2]^2 / (n/sigma^2)
                   = n (Dbar - theta_0)^2 / sigma^2.

The MLE is `theta_hat = Dbar` and `I(theta_hat) = n/sigma^2`, so
`tau_Wald = (Dbar - theta_0)^2 (n/sigma^2)` matches term-for-term,
and `tau_LRT` agrees by the exact-quadratic argument in
`docs/methods/lrt.md` Step 3.

**Step 4 (chi^2_1 calibration is exact, not asymptotic, on NN).**
Under H_0 the score `U(theta_0) = (D - theta_0)/sigma^2 ~ N(0,
1/sigma^2)` (because `D ~ N(theta_0, sigma^2)`). Normalising,
`U(theta_0)/sqrt(I(theta_0)) = (D - theta_0)/sigma ~ N(0,1)`
exactly, so `tau_Score ~ chi^2_1` exactly — the same situation as
`lrt`. KS uniformity of `p_Score(theta_0; D)` on 10000 draws at
`(sigma=1.7, theta_true=0.4)`: `D = 0.0093, p = 0.35` (passes).

**Step 5 (reparameterisation invariance at the null).** For a
smooth bijection `phi = g(theta)` with `g'(theta_0) != 0`, the
chain rule gives `U_phi(phi_0) = U_theta(theta_0) / g'(theta_0)`
and the Fisher info transforms tensorially as `I_phi(phi_0) =
I_theta(theta_0) / g'(theta_0)^2`. Substituting into (D3):

    tau_Score^{(phi)}
      = [U_theta / g']^2 / [I_theta / g'^2]
      = U_theta^2 / I_theta
      = tau_Score^{(theta)}.

The score statistic is therefore *exactly* reparameterisation-
invariant at `theta_0`. This is the score test's headline
advantage over Wald, whose value at the MLE depends on the
parameterisation (Cox & Hinkley 1974 §9; Dagenais & Dufour 1991).
Numerical check with `phi = 2 theta` at
`(sigma=1, D=0.7, theta_0=0.3)`: `tau_theta = tau_phi = 0.16`
exactly.

**Step 6 (Score is evaluated entirely at theta_0).** Unlike
`tau_Wald` (which needs `model.mle(data)`) and `tau_LRT` (which
needs `loglik` at both `theta_hat` and `theta_0`),
`tau_Score(theta_0)` requires only one likelihood-gradient
evaluation at `theta_0` (Engle 1984 §3.1; Bera & Bilias 2001 §2).
On the NN sandbox this is academic (the MLE is `Dbar`, a single
arithmetic mean), but for any future Model whose MLE requires
iterative optimisation, the score test's CI inversion amortises a
single `model.fisher_information` lookup per `theta_0` brentq
step — no MLE recomputation per node. The generic-path
implementation (`src/frasian/statistics/score.py`) uses
`jax.grad` on `model.likelihood(data).loglik` for `U(theta_0)`
and dispatches `model.fisher_information(theta_0)` for `I`.

**Invariants checked numerically (atol <= 1e-12):**
- `(sigma=1.0, D=0.0, theta_0=0.5)`: `p_Score == p_Wald == p_LRT` (1.11e-16).
- `(sigma=2.0, D=-1.3, theta_0=2.7)`: `p_Score == p_Wald == p_LRT` (1.11e-16).
- `(sigma=0.5, D=4.0, theta_0=4.1)`: `p_Score == p_Wald == p_LRT` (0).
- `(sigma=1.5, D=-2.0, theta_0=-2.0)`: `tau_Score == 0`, `p_Score == 1`.
- Sympy: `tau_Score(theta_0) - (D - theta_0)^2/sigma^2 == 0`
  symbolically (n=1).
- Sympy: `tau_Score(theta_0) - n(Dbar - theta_0)^2/sigma^2 == 0` (n>1).
- KS uniformity (sigma=1.7, theta_true=0.4, N=10000): KS = 0.0093, p = 0.35.
- Reparam invariance (phi = 2 theta): `|tau_theta - tau_phi| = 0`
  at `(sigma=1, D=0.7, theta_0=0.3)`.

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

1. **p-value range.** `p_Score(theta_0; data) in [0, 1]` for all
   `(theta_0, data, model)` — consequence of `tau >= 0` and the
   `chi^2_1` survival function having range `[0, 1]`.
2. **Non-negativity.** `tau_Score(theta_0; data) >= 0`, with
   equality iff `U(theta_0) = 0` (i.e. `theta_0` is a stationary
   point of the loglikelihood — on NN, `theta_0 = Dbar`).
3. **NN n=1 equivalence — p-value.** On `NormalNormalModel`,
   `score.pvalue(theta_0, data, model) ==
   wald.pvalue(theta_0, data, model) ==
   lrt.pvalue(theta_0, data, model)` for any `(theta_0, data,
   sigma)`, to `atol <= 1e-12` (Derivation Step 2; closed form).
4. **NN n=1 equivalence — CI.** On `NormalNormalModel`,
   `score.confidence_interval(alpha, data, model) ==
   wald.confidence_interval(alpha, data, model) ==
   lrt.confidence_interval(alpha, data, model)` to `atol <= 1e-8`
   (closed form on Score; brentq-inverted on LRT/Score-generic).
5. **Mode property.** On NN, `p_Score(Dbar; data) == 1` exactly
   (`U(Dbar) = 0` => `tau = 0`).
6. **Exact `chi^2_1` calibration on NN under H_0.** Drawing
   `data ~ N(theta_true, sigma^2)` and evaluating
   `tau_Score(theta_true; data)` gives an exact `chi^2_1` sample
   (Derivation Step 4); KS test of `p_Score` against
   `Uniform[0, 1]` passes at the nominal level (L3 marker).
7. **Tilting compatibility.** `score.accepts_tilting(tilting) is
   True` iff `tilting.name == "identity"`; non-identity tiltings
   (`power_law`, `ot`, `mixture`, `fisher_rao`) return `False`.
   Mirrors `wald` / `lrt` contracts.
8. **Reparameterisation invariance at theta_0.** For any smooth
   bijection `phi = g(theta)` with `g'(theta_0) != 0`,
   `tau_Score^{(phi)}(g(theta_0); data) ==
   tau_Score^{(theta)}(theta_0; data)` (Derivation Step 5). On NN
   this is testable via the synthetic transformation
   `phi = 2 theta`: `tau_Score` at `phi_0 = 2 theta_0` under the
   reparameterised `(2D, 2 sigma)` model equals `tau_Score` at
   `theta_0` under `(D, sigma)`. The score test's headline
   distinction from Wald (Wald is *not* reparam-invariant at the
   MLE).

## Literature

### Foundational

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
Origin of the score statistic `U(theta0)^2 / I(theta0)` and the
asymptotic chi-squared null calibration; the paper the brief's
name attaches to.

```bibtex
@article{aitchisonsilvey1958,
  author  = {Aitchison, J. and Silvey, S. D.},
  title   = {Maximum-Likelihood Estimation of Parameters Subject to Restraints},
  journal = {The Annals of Mathematical Statistics},
  volume  = {29},
  number  = {3},
  pages   = {813--828},
  year    = {1958},
  doi     = {10.1214/aoms/1177706538}
}
```
Multi-parameter constrained-MLE framework; sets up the Lagrange-
multiplier formulation Silvey (1959) recognised as the score test.

```bibtex
@article{silvey1959,
  author  = {Silvey, S. D.},
  title   = {The Lagrangian Multiplier Test},
  journal = {The Annals of Mathematical Statistics},
  volume  = {30},
  number  = {2},
  pages   = {389--407},
  year    = {1959},
  doi     = {10.1214/aoms/1177706259}
}
```
"Lagrange-multiplier" framing of the score statistic and Wald/LR/
score asymptotic equivalence under constraints — the econometric
name for the same test.

```bibtex
@incollection{neyman1959,
  author    = {Neyman, J.},
  title     = {Optimal Asymptotic Tests of Composite Statistical Hypotheses},
  booktitle = {Probability and Statistics: The Harald {C}ram\'er Volume},
  editor    = {Grenander, U.},
  publisher = {John Wiley \& Sons},
  address   = {New York},
  pages     = {213--234},
  year      = {1959}
}
```
The C(alpha) test — a score-based locally most powerful test with
nuisance parameters; anchor for the LMP framing and the framework's
future score-based diagnostics.

```bibtex
@article{bartlett1953a,
  author  = {Bartlett, M. S.},
  title   = {Approximate Confidence Intervals},
  journal = {Biometrika},
  volume  = {40},
  number  = {1/2},
  pages   = {12--19},
  year    = {1953},
  doi     = {10.1093/biomet/40.1-2.12}
}
```
Locally most powerful framing of the score statistic in the one-
parameter case; the LMP one-sided-test anchor (Part I).

### Closely related

```bibtex
@incollection{engle1984,
  author    = {Engle, R. F.},
  title     = {Wald, Likelihood Ratio, and {L}agrange Multiplier Tests in Econometrics},
  booktitle = {Handbook of Econometrics, Volume II},
  editor    = {Griliches, Z. and Intriligator, M. D.},
  publisher = {North-Holland},
  address   = {Amsterdam},
  chapter   = {13},
  pages     = {775--826},
  year      = {1984},
  doi       = {10.1016/S1573-4412(84)02005-5}
}
```
Canonical trinity-comparison reference; supports "evaluated
entirely at theta0 (vs Wald at MLE)" and asymptotic-equivalence
claims in Definition.

```bibtex
@article{berabilias2001,
  author  = {Bera, A. K. and Bilias, Y.},
  title   = {{R}ao's Score, {N}eyman's {C}($\alpha$) and {S}ilvey's {LM} Tests: An Essay on Historical Developments and Some New Results},
  journal = {Journal of Statistical Planning and Inference},
  volume  = {97},
  number  = {1},
  pages   = {9--44},
  year    = {2001},
  doi     = {10.1016/S0378-3758(00)00343-8}
}
```
Modern historical-and-technical review threading Rao-Neyman-
Silvey-Engle; cites the "rehabilitation" of the score test in
econometrics; supports the trinity-coincides-asymptotically claim.

```bibtex
@book{coxhinkley1974,
  author    = {Cox, D. R. and Hinkley, D. V.},
  title     = {Theoretical Statistics},
  publisher = {Chapman and Hall},
  address   = {London},
  year      = {1974},
  isbn      = {978-0412124204}
}
```
Chapters 9.2/9.3: score statistic, small-sample-corrected variants,
and reparameterisation properties relative to Wald.

```bibtex
@article{dufourdagenais1991,
  author  = {Dagenais, M. G. and Dufour, J.-M.},
  title   = {Invariance, Nonlinear Models, and Asymptotic Tests},
  journal = {Econometrica},
  volume  = {59},
  number  = {6},
  pages   = {1601--1615},
  year    = {1991},
  doi     = {10.2307/2938291}
}
```
Canonical reference for the reparameterisation behaviour of Wald
vs score/LR: score test is invariant under reparameterisation of
the null `theta_0`; Wald is sensitive at the MLE.

### Contrasting

```bibtex
@article{cordeiroferrari1991,
  author  = {Cordeiro, G. M. and Ferrari, S. L. P.},
  title   = {A Modified Score Test Statistic Having Chi-squared Distribution to Order $n^{-1}$},
  journal = {Biometrika},
  volume  = {78},
  number  = {3},
  pages   = {573--582},
  year    = {1991},
  doi     = {10.1093/biomet/78.3.573}
}
```
Bartlett-type correction for the score statistic — small-sample
alternative to bare chi-squared; partner to the planned `bartlett`
decorator. Demonstrates chi-squared is conservative / anti-
conservative off-asymptotically.

```bibtex
@article{chandrajoshi1983,
  author  = {Chandra, T. K. and Joshi, S. N.},
  title   = {Comparison of the Likelihood Ratio, {R}ao's and {W}ald's Tests and a Conjecture of {C}. {R}. {R}ao},
  journal = {Sankhya, Series A},
  volume  = {45},
  number  = {2},
  pages   = {226--246},
  year    = {1983}
}
```
Documents the local-power discrepancy among Wald, LR and Rao's
score at finite samples; supports the "differ at finite samples /
off-asymptotic regimes" claim.

Cross-references: Wilks (1938), Wald (1943), van der Vaart (1998)
Theorem 16.7, Casella & Berger (2002) §10.3, Bartlett (1937) are
all bibliographied in `docs/methods/lrt.md`.

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
