# mixture

> Status: `stub`

## Summary

Convex mixture: `q(theta; eta) = (1 - eta) * pi(theta) + eta * post(theta)`
for `eta in [0, 1]`. The simplest possible interpolation, included as
a baseline against which the geometrically motivated schemes
(`ot_normal`, `geodesic_normal`) must justify their complexity.

## Motivation

If the smoothness diagnostic ranks `power_law` worst and `ot_normal`
best, we still need to know whether the improvement comes from the
geometry of OT or just from "anything that does not clamp". A naive
mixture of prior and posterior is the obvious null model: smooth in
the parameters but produces *non-Gaussian* outputs (two-component
mixtures), which may or may not break the test statistics.

## Definition

  q(theta; eta) = (1 - eta) pi(theta) + eta post(theta),
  eta in [0, 1].

For Gaussian prior and posterior, q is a two-component Gaussian
mixture — not Gaussian itself. The framework needs a `Distribution`
implementation for this two-component mixture (mean, var, cdf,
quantile via numerical inversion).

## Derivation

Trivial mathematically; the work is in plumbing the mixture through
the `Posterior` protocol and the `(TiltingScheme, TestStatistic)`
cell evaluators. To be filled in by `/derive mixture`.

## Predicted behavior

- Smooth `eta*(|Delta|)` curve (no clamp by construction).
- *Bimodal* mixture density when prior and posterior disagree
  strongly. WALDO p-value formula assumes a Gaussian posterior — its
  application to a bimodal mixture is not well-defined; the cell
  evaluator must either reject or document the workaround.
- Likely worse coverage than the geometric schemes when the mixture is
  heavily bimodal — the test statistic was designed against Gaussians.

## Failure modes

- Bimodality breaks the WALDO acceptance-region inversion (no single
  CI; may need an HPD set instead).
- Mixture is not a member of any tractable exponential family;
  closed-form moments + cdf only because both components are Gaussian.
- The `Posterior` protocol assumes single-mode `quantile`; adapting
  to bimodal mixtures may require an extension.

## Invariants

- `tilt(eta=0)` returns the prior exactly.
- `tilt(eta=1)` returns the posterior exactly.
- `tilt` is continuous in `eta` (linear).
- For every `eta in [0, 1]`, the output integrates to 1.
- Wald-statistic CI is well-defined (Wald ignores prior); WALDO CI may
  be undefined when the mixture is bimodal — record NaN.

## Literature

- McLachlan, G. J., Peel, D. *Finite Mixture Models.* Wiley, 2000.
- Diaconis, P., Ylvisaker, D. "Conjugate priors for exponential
  families." *Ann. Statist.* 7 (1979): 269-281. (For comparing the
  mixture with the natural exp-family path.)
- O'Hagan, A. "Fractional Bayes factors for model comparison." *J. R.
  Stat. Soc. B* 57 (1995): 99-138. (Background on weighted likelihoods,
  related to mixture parameterisations.)

## Links

- Implementation: `src/frasian/tilting/mixture.py` (stub)
- Property tests: `tests/properties/test_mixture_invariants.py` (skipped)
- Illustration:   TBD

## Status notes

Stub — value lies in being the *baseline*. If `mixture` is no smoother
than `ot_normal` on the diagnostic, the OT geometry is not buying us
anything specific; if it *is* smoother (or has dramatically worse
coverage), the comparison clarifies the trade-offs.
