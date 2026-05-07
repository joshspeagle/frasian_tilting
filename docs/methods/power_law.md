# power_law

> Status: `implemented`

## Summary

The legacy η-tilting scheme as a `TiltingScheme` plugin. Tilts the prior
by a power: `q(theta; eta) ∝ L(theta) * pi(theta)^(1 - eta)`. Identity
element is `eta = 0` (recovers WALDO); `eta = 1` recovers Wald.

Geometrically, this is the **e-geodesic** (exponential / log-linear /
geometric-mean path) of Amari's information geometry — affine in the
natural parameters of the conjugate exponential family. On the
Normal-Normal sandbox, "interpolating natural parameters between
posterior and likelihood" is *algebraically identical* to
`q ∝ L · pi^(1-eta)`, so an `exp_family` natural-parameter scheme
would produce numerically the same path as `power_law` and is omitted
as redundant. The dual partner under the Fisher metric is the
**m-geodesic** (linear-density / arithmetic-mean path), implemented
as `mixture`. A third, distinct geometry — Wasserstein-2 — is
`ot`. See Amari & Nagaoka 2000 §3 for the dually-flat structure that
ties these together.

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
- Neal, R. M. "Annealed importance sampling." *Stat. Comput.* 11
  (2001): 125–139. — Geometric-path tempering, explicitly contrasted
  with the linear/m-geodesic path; cite for the e-geodesic framing.
- Friel, N., Pettitt, A. N. "Marginal likelihood estimation via power
  posteriors." *J. R. Stat. Soc. B* 70 (2008): 589–607. — Power
  posteriors as the Bayesian-statistics e-geodesic.
- Amari, S., Nagaoka, H. *Methods of Information Geometry.* AMS /
  Oxford, 2000. — Foundational reference for the e-/m-/Levi-Civita
  α-connections that classify `power_law` (e), `mixture` (m), and
  `fisher_rao` (Levi-Civita).

## Links

- Implementation: `src/frasian/tilting/power_law.py`
- Generic-path GridDistribution: `src/frasian/tilting/_grid_distribution.py`
- Solver: `src/frasian/tilting/_solvers.py`
- Regression tests:
  - `tests/regression/test_power_law_tilting.py` (closed-form NN)
  - `tests/regression/test_power_law_generic_tilt.py` (generic tilt)
  - `tests/regression/test_power_law_generic_pvalue_ci.py` (generic pvalue + CI)
  - `tests/regression/test_generic_grid_pvalue_matches_closed_form.py` (cross-check)
  - `tests/regression/test_bernoulli_coverage.py` (Bernoulli coverage at nominal 1-α)
  - `tests/integration/test_bernoulli_end_to_end.py` (full public API on Bernoulli)
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

### Phase 4 entry point: `_tilted_pvalue_kernel`

The JAX port factored out a private autodiff-clean kernel
(`src/frasian/tilting/power_law.py::_tilted_pvalue_kernel`) wrapped
in `@jax.jit(static_argnames=("statistic_name",))`. It contains only
the JAX arithmetic — no validation, no Python control flow except the
static `statistic_name` dispatch — so Phase 4's learned-η loss can
close over it directly inside `@jax.jit` and `jax.grad`. The public
`PowerLawTilting.tilted_pvalue` runs validation in numpy (JAX cannot
raise mid-trace) and shape-dispatches between this kernel (for bulk
arrays) and a numpy-eager scalar fast path (`_tilted_pvalue_numpy_scalar`,
for brentq inner loops). See `docs/jax_style.md` for the underlying
principle.

### Generic numerical path (Phase 3c, 3d-fix1)

`PowerLawTilting` works on any `(Model, Prior)` pair via a numerical
default that uses only abstract protocol methods. The Normal-Normal
closed-form (Theorem 6 above) is preserved as a fast path; non-Normal
pairings (e.g. `(BernoulliModel, BetaDistribution)`) route through:

- **`_generic_tilt(posterior, prior, likelihood, eta)`** — builds a
  `GridDistribution` (in `src/frasian/tilting/_grid_distribution.py`)
  from `log q(theta; eta) = log L(theta) + (1 - eta) * log pi(theta)`
  on a 1024-point grid spanning the support's quantile window
  (bounded support) or `mean ± 6 * std` of the posterior + prior
  (unbounded). The deriver verified this formula reduces to
  Theorem 6 on Normal-Normal at atol 1e-7 with N=1024
  (`tests/regression/test_grid_distribution.py`).

- **`_generic_tilted_pvalue(theta, data, model, prior, eta, statistic_name)`**
  — Monte Carlo tilted-WALDO p-value. Computes observed tilted moments
  `(mu_tilted, sigma2_tilted)` via grid integration, draws `n_mc=200`
  synthetic `D' ~ likelihood(.|theta)` samples, evaluates
  `t(D', theta) = (mu_tilted_D' - theta)^2 / sigma2_tilted_D'` per
  draw, returns the empirical tail probability with `(k+1)/(n+1)`
  smoothing. CRN-seeded via blake2b stable hash on
  (data, model.fingerprint, prior.fingerprint, eta, alpha) so brentq
  probes share an internal uniform stream and `f(theta)` is a
  deterministic function of theta. For `statistic_name="wald"` (eta-
  independent), delegates to `WaldStatistic._generic_pvalue`.

- **`_generic_tilted_confidence_interval(alpha, data, model, prior,
  eta, statistic_name)`** — inverts `_generic_tilted_pvalue >= alpha`
  via `brentq_with_doubling`. Hoists the observed tilted moments
  outside the brentq loop (skeptic Phase 3 finding #3) for ~10x
  speedup. Detects boundary saturation explicitly to avoid silent
  snap-to-support-edge.

- **`confidence_regions` dispatch** — non-Normal pairings + static
  selectors (e.g. `FixedEtaSelector`) route here. Dynamic selectors
  (`DynamicNumericalEtaSelector`, `LearnedDynamicEtaSelector`) still
  raise `NotImplementedError` for non-Normal pairings: the
  `dynamic_ci_scan` builds its theta-window from
  `D ± search_mult * sigma`, which is Normal-Normal-flavoured.
  Generalising it to theta-only inputs is a separate research item
  (tracked as Phase 5+).

Cross-check tests pin atol-1e-3 agreement between the generic
numerical path and the closed-form Theorem 6 path on Normal-Normal
across (eta, D, sigma0) grids:
`tests/regression/test_generic_grid_pvalue_matches_closed_form.py`
(L2 cross-check on moments) and the L4 integration smoke
`tests/integration/test_bernoulli_end_to_end.py` (full public API
on Bernoulli + Beta).

### Phase 4 learned-η on the generic path

The Phase 4 learned-η loop trains an `EtaNet` against a JAX-
traceable tilted-pvalue. The closed-form NN kernel registers under
`pvalue_jax.JAX_TILTED_PVALUE[("power_law", "normal_normal")]`; the
new **generic-grid kernel** registers under `("power_law",
"generic")` and works on any `(Model, Prior)` with bounded support
and `prior.logpdf` defined on the grid. The generic kernel uses a
deliberate **symmetric normal-approximation p-value** (`2 * (1 -
Phi(|mu - theta| / sigma))`) rather than Theorem 8's asymmetric form
— the symmetric form has cleaner gradients through eta and is
sufficient as a differentiable training surrogate (production CI
inversion uses MC over D' for the exact reference, not this
surrogate). The moments themselves (Theorem 6 reduction) are
deriver-verified to atol 1e-7 against the NN closed form. See
`docs/methods/learned_eta.md` for the full Phase 4 brief and
`tests/regression/test_generic_grid_pvalue_matches_closed_form.py`
for the moment-agreement cross-check.

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

Optimal-transport (`ot`), Fisher–Rao-geodesic (`fisher_rao`), and
mixture (`mixture`) tilting alternatives sit alongside `power_law` in
the framework's geodesic taxonomy:
`power_law` is the e-geodesic; `mixture` is the dual m-geodesic;
`fisher_rao` is the Levi-Civita / intrinsic Riemannian geodesic;
`ot` is the Wasserstein / mass-displacement geodesic. Briefs:
`docs/methods/{ot,fisher_rao,mixture}.md`.
