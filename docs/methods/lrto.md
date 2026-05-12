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
a likelihood, and Wilks' theorem does not apply. Calibration is by
Monte Carlo, mirroring WALDO:

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

TODO — `/derive` (the deriver agent) fills this section.

Required ingredients to cover:
1. Equation (D1) as a definition; the MC calibration (D2); the
   theta-space CI inversion (D3).
2. The NN reduction: log-posterior is quadratic with curvature
   `1/sigma_n^2`, so `tau_LRTO = (theta - mu_n)^2 / sigma_n^2`
   exactly.
3. Identity with WALDO on NN+Normal: `tau_LRTO == tau_WALDO`
   pointwise; the MC reference distributions also coincide because
   both use posterior summaries of `pi(.|D')` under `D' ~
   likelihood(.|theta_0)`. The closed-form p-value formulas
   coincide.
4. Why `chi^2_1` is **NOT** the H_0 calibration in general
   (sketch): the posterior-vs-likelihood prior bias breaks the
   Wilks-theorem regularity. On NN with a flat prior `lim_{sigma_0
   -> infty}`, posterior → likelihood and `lrto -> lrt`; the
   `chi^2_1` calibration is recovered in this limit.
5. The mode of a generic tilted posterior: how the
   implementation finds `theta_MAP` (brent-optimise `logpdf` over
   `model.support()`).

## Predicted behavior

- **On NN+Normal sandbox**: `lrto.pvalue == waldo.pvalue` and
  `lrto.confidence_interval == waldo.confidence_interval` to
  numerical precision for every `(alpha, data, prior, model)`.
  Pinned by property tests + regression dispatch.
- **Under flat-prior limit on NN**: as `sigma_0 -> infty`, the
  posterior degenerates to the likelihood, `mu_n -> D`, `sigma_n
  -> sigma`, so `tau_LRTO -> tau_LRT = tau_Wald` and the p-value
  recovers `2(1 - Phi(|D - theta|/sigma))`. Same limit as `waldo
  -> wald`.
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
  (parallel of LRT's Chernoff 1954 caveat). Not relevant for NN
  (parameter space is R).
- **`n_mc` too small at small alpha**: tail-MC error scales like
  `1/sqrt(n_mc * alpha)`; the default `n_mc=2000` gives ~0.022 SE
  on `p ~ 0.5` but only ~0.005 on `p ~ 0.05`. CI inversion at
  `alpha=0.01` should bump `n_mc` to ~5000.
- **NormalNormalModel subclassing with overridden likelihood/posterior**:
  same risk as WALDO/Wald — the closed-form dispatch reads only the
  fingerprint. Mitigation: same options as `lrt` (avoid subclass,
  override `fingerprint`, or `force_generic=True`).

## Invariants

TODO — fill from `/derive` output. Tentative list:

- `p_LRTO in [0, 1]` for all `(theta0, data, prior, model)`.
- `tau_LRTO >= 0` (with equality at `theta0 = theta_MAP`).
- **NN equivalence — p-value**: on `NormalNormalModel + NormalDistribution`,
  `lrto.pvalue == waldo.pvalue` elementwise (atol 1e-12 closed
  form; MC tolerance on the generic path).
- **NN equivalence — CI**: `lrto.confidence_interval ==
  waldo.confidence_interval` (atol 1e-8 closed form).
- Mode property: `lrto.pvalue(theta_MAP, ...) == 1` (closed form
  on NN; MC-tolerant on generic path).
- `accepts_tilting(...) is True` for every concrete
  TiltingScheme — `lrto` is prior-aware by design.
- **Flat-prior limit**: as `sigma_0 -> infty` (NN), `lrto.pvalue
  -> lrt.pvalue` and `lrto.CI -> lrt.CI`.
- **Cross-pair limit on NN**: at `(identity, lrto)`, the cell
  reproduces `(identity, waldo)` outputs cell-for-cell on the
  cross-product runner.

## Literature

TODO — `/litreview` (the literature-reviewer agent) fills this
section. Required anchors:

- Aitkin (1997) for the posterior likelihood ratio framing.
- Pereira & Stern (1999) FBST for the closest precedent — the
  e-value is essentially this statistic with `theta_MAP` replaced
  by an integration.
- Masserano et al. (2023) for the WALDO pairing template (the
  closed-form NN+Normal collapse uses WALDO's formula).
- Schweder & Hjort (2016) for the Bayesian-confidence /
  fiducial-LRT bridge.
- Wilks (1938), Casella & Berger (2002 §10.3), van der Vaart
  (1998 Thm 16.7) for the frequentist baseline that this is the
  Bayesian counterpart of.
- Bernardo (2005), Berger et al. — Bayesian reference posteriors
  / objective-Bayesian LRT analogs, for context on the prior-
  dependence Failure mode.

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
