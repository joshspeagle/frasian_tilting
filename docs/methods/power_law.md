# power_law

> Status: `implemented`

## Summary

The legacy η-tilting scheme as a `TiltingScheme` plugin. Tilts the prior
by a power: `q(theta; eta) ∝ L(theta) * pi(theta)^(1 - eta)`. Identity
element is `eta = 0` (recovers WALDO); `eta = 1` recovers Wald.

The class takes an `EtaSelector` constructor argument: a static selector
(`FixedEtaSelector`, `NumericalEtaSelector`) chooses one η for the whole
inversion; a dynamic selector (`DynamicNumericalEtaSelector`) varies η
per θ via the precomputed-coarse-grid + interpolate strategy. The cell
display name picks up the selector's name when it is non-default
(e.g. `power_law[dynamic_numerical]`), so the `(power_law, waldo)`
matrix entry the runner produces is **Dynamic-WALDO** by default.

## Motivation

Power-law tilting is the *baseline* the framework is designed to
critique. The user's central empirical observation is that selecting `eta`
adaptively as a function of `|Delta|` — the scaled prior-data conflict —
produces a sharp transition between posterior-driven and likelihood-driven
behavior, which is undesirable in practice. Replacement schemes (optimal
transport, Fisher–Rao geodesics, mixture paths) will be evaluated against
this baseline using the `smoothness` experiment's diagnostics.

## Definition

For the Normal-Normal sandbox, the closed form (Theorem 6 in the legacy
derivations) is

  denom    = 1 - eta * (1 - w),
  mu_eta   = (w * D + (1 - eta) * (1 - w) * mu0) / denom,
  sigma_eta^2 = w * sigma^2 / denom,
  w_eta    = w / denom.

The associated noncentrality parameter scales as
`lambda_eta = (1 - eta)^2 * lambda_0`.

## Derivation

The unnormalised tilted posterior is the product
  L(theta) * pi(theta)^(1 - eta)
  ∝ exp(-(D - theta)^2 / (2 sigma^2)) * exp(-(1 - eta)(theta - mu0)^2 / (2 sigma0^2)).
Completing the square in `theta` yields a Gaussian with precision
`1/sigma^2 + (1 - eta)/sigma0^2` and mean equal to the precision-weighted
average of `D` and `mu0`, which simplifies to the closed form above.

Full derivation including special cases (eta = 0, 1) lives in legacy
`tilting.py` docstrings; promoting to a standalone derivation file is
a follow-up cleanup task.

## Predicted behavior

- `eta = 0` reproduces the input WALDO posterior (identity element).
- `eta = 1` reproduces the Wald posterior `N(D, sigma^2)`.
- `eta < 0` *oversharpens*: `mu_eta` is pushed past `mu_n` toward the
  prior, with `sigma_eta < sigma_n`. Empirically yields narrower CIs than
  Wald at low `|Delta|` — this is the discovery of the legacy framework.
- `eta` outside `(-w/(1-w), 1/(1-w))` produces a non-positive variance and
  raises `TiltingDomainError`.
- The optimal `eta*(|Delta|)` curve is monotone non-decreasing in `|Delta|`
  but has a sharp inflection — the `smoothness` diagnostic measures this.

## Failure modes

- Near-singular `denom = 1 - eta(1 - w)` when `eta ≈ 1/(1-w)`. The
  admissible-range check rules this out by construction.
- Sharp local Lipschitz behavior when an η-selector flips between
  regimes. The selectors themselves (`NumericalEtaSelector`,
  `LearnedEtaSelector`) live in `tilting/eta_selectors.py`; this scheme
  only implements the *given-η* tilt.

## Invariants

- `tilt(eta=0)` returns the input posterior exactly.
- `tilt(...).pdf` integrates to 1 (numerical, atol≈5e-4 on a 12-sigma grid).
- `tilt` is continuous in `eta` on the admissible range.
- `tilt(eta=1)` produces `N(D, sigma^2)` for any `(mu0, sigma0)`.
- `admissible_range` returns a non-empty open interval containing `eta = 0`.

## Literature

- Holmes, A. C., Walker, S. G. "Assigning a value to a power likelihood
  in a general Bayesian model." *Biometrika* 104 (2017): 497–503.
  (Power likelihoods.)
- Miller, J. W., Dunson, D. B. "Robust Bayesian inference via coarsening."
  *J. Amer. Statist. Assoc.* 114 (2019): 1113–1125. (Tempering.)
- Bissiri, P. G., Holmes, C. C., Walker, S. G. "A general framework for
  updating belief distributions." *J. Royal Stat. Soc. B* 78 (2016).

## Links

- Implementation: `src/frasian/tilting/power_law.py`
- Solver: `src/frasian/tilting/_solvers.py`
- Regression tests: `tests/regression/test_power_law_tilting.py`
- Property tests: `tests/properties/test_power_law_invariants.py`
- Illustration: `src/frasian/experiments/illustrations/power_law_demo.py`

## Status notes

η-selectors live in `src/frasian/tilting/eta_selectors.py`:

- `FixedEtaSelector(eta=0.0)` — identity selector (default).
- `NumericalEtaSelector` — single η minimising tilted CI width at the
  data's |Δ|. Static. **Post-selection: undercovers by ~2 points at
  α=0.05; not a calibrated estimator.** Exposed via
  `post_selection_demo_tiltings()` for studying the trade-off only.
- `DynamicNumericalEtaSelector` — per-θ varying η*(|Δ|) via
  coarse-grid + interpolation. Dynamic. The framework's calibrated
  default; exact 1-α coverage by construction (η at θ does not depend
  on D, so the WALDO p-value at fixed η is U[0,1] under H0).

`(power_law[FixedEtaSelector(0.0)], waldo)` is numerically equal to
`(identity, waldo)`; the runner ships only the latter to keep the
matrix tight.

### The static-vs-dynamic trade-off, in one paragraph

The static η*-opt CI is genuinely narrower than WALDO at every D and
asymptotes to Wald at large |Δ| — your theoretical intuition that
"η=0 is in the search space, so optimisation should never hurt"
holds for the *width* dimension. But the procedure is post-selection:
η is a function of D, then the narrow CI is reported, so coverage
empirically falls to ~0.93 at α=0.05 (regression-pinned in
`tests/regression/test_post_selection_coverage.py`). The dynamic-η-
per-θ procedure recovers exact calibration by making η depend on θ
(not D), but pays for it with a non-monotone width that detours past
Wald in the conflict band. The framework's research question is then:
**can a smoother tilting family give a per-θ η whose width competes
with the static optimum without the conflict-band detour?** That is
what the OT / Fisher-Rao / mixture / exp-family stubs are intended to
explore.

Optimal-transport, Fisher–Rao-geodesic, and mixture tilting
alternatives are scheduled stubs (see `docs/methods/{ot_normal,geodesic_normal,mixture,exp_family}.md`).
