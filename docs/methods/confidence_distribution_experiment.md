# confidence_distribution_experiment

> Status: `implemented`

## Summary

Distributional analogue of `coverage` and `width`. For each cell of the
`(TiltingScheme × TestStatistic)` matrix and each `(θ_true, w)` on the
standard grid, the experiment builds the cell's confidence distribution
(CD) per replicate D ~ N(θ_true, σ) and aggregates four summaries:
**CD-median**, **CD 95% width**, **W₁ distance to a Wald reference CD**,
and **fraction of replicates with non-monotone signed_confidence**. The
diagnostic produces one heatmap per metric per cell.

## Motivation

Per-α coverage and width tell us about a single confidence level at a
time. They cannot, for example, distinguish a method that produces a
narrow but mis-located CI from one that produces an honest distribution
shifted toward the prior. Confidence distributions package the entire
α-sweep into one object whose location, scale, shape, and *distance to
a reference* can all be measured. For our central comparison
(Wald vs WALDO vs Dyn-WALDO vs future smooth-tilting families) the CD
view is the natural diagnostic: smoother tilting families should
produce CDs that are *closer* to the Wald reference at large |Δ|
(asymptotic recovery) and *narrower* than WALDO under low |Δ|
(efficiency under prior agreement) — both visible in the W₁ heatmap.

## Definition

For each `(θ_i, w_j)`:

  Generate D_{i,j,k} ~ N(θ_i, σ) for k = 1..n_reps
  cd_{i,j,k}     = build_cd_from_pvalue(tilting, statistic, D, model, prior)
  wald_ref_{k}   = wald_cd(D, σ, theta_grid=cd_{i,j,k}.theta_grid)
  median_{i,j,k} = cd_{i,j,k}.median()
  width_{i,j,k}  = cd_{i,j,k}.interval(0.05) → hi − lo
  w1_{i,j,k}     = wasserstein_1(cd_{i,j,k}, wald_ref_{k})
  nm_{i,j,k}     = 1 if not cd_{i,j,k}.is_monotone_inversion() else 0

  cd_median[i, j]            = mean_k median_{i,j,k}
  cd_median_se[i, j]         = std_k(median_{i,j,k}) / sqrt(n_reps)
  cd_width_95[i, j]          = mean_k width_{i,j,k}
  cd_width_95_se[i, j]       = std_k(width_{i,j,k}) / sqrt(n_reps)
  w1_to_wald_cd[i, j]        = mean_k w1_{i,j,k}
  w1_to_wald_cd_se[i, j]     = std_k(w1_{i,j,k}) / sqrt(n_reps)
  nonmonotone_fraction[i, j] = mean_k nm_{i,j,k}

The reference CD is the closed-form Wald N(D, σ²) per replicate.

## Derivation

The CD is built via the Schweder–Hjort density `c(θ) = ½ |dp/dθ|`
where p(θ) = `tilting.pvalue(θ, [D], model, prior, statistic)`. The
unnormalised density `c̃` is then Z-normalised so the cdf integrates to
1 — relevant for multimodal-p cells where total variation of p exceeds
2 (Dyn-WALDO under conflict). The pdf-primary representation makes
distance metrics (W₁, W₂) act on the *real* probability distribution
implied by the density, not a flattened/rearranged version.

For W₁: `∫|F_a(θ) − F_b(θ)| dθ` integrated trapezoidally on the union
of θ-grids. Both CDFs are monotone non-decreasing by construction
(derived from non-negative pdfs).

For multimodal-p cells, `signed_confidence` (the inversion-based C(θ)
curve) becomes non-monotone; this is the smoothness pathology
surfacing directly in the CD. The `nonmonotone_fraction` metric
quantifies how often it appears across replicates.

## Predicted behavior

- `(identity, wald)`: median ≈ θ_true; width ≈ 2·1.96·σ across all cells;
  W₁ to itself = 0; nonmonotone_fraction = 0.
- `(identity, waldo)`: median ≈ posterior mean; width *narrower* under
  prior agreement, *wider* under conflict (the WALDO efficiency-vs-
  conflict trade); W₁ > 0 but small; nonmonotone_fraction = 0.
- `(power_law[dynamic_numerical], waldo)` (Dyn-WALDO): median similar
  to WALDO; width similar or slightly different; W₁ to Wald larger
  than WALDO's at low |Δ| (where Dyn-WALDO's η-tilting deforms the CD
  away from Wald), but reverting back at large |Δ|; nonmonotone_fraction
  > 0 in the bimodal-p regime.

## Failure modes

- Density renormalisation `Z = ∫|dp/dθ|/2` may exceed 1 for highly
  multimodal p. The pdf is renormalised to integrate to 1 either way;
  this is documented behaviour, not a bug.
- `build_cd_from_pvalue` requires `tilting.pvalue(...)` to return a
  finite, in-`[0,1]` array. If the tilting raises (e.g. a stub), the
  cell is skipped (NaN row, like other experiments).
- For very narrow Dyn-WALDO CDs at high α, the FD-derived density can
  underestimate the peak height around the |D−θ| kinks; mitigated by
  the average-of-one-sided-differences scheme in `from_pvalue`.

## Invariants

- `cd_median ∈ [θ_grid.min(), θ_grid.max()]` everywhere finite.
- `cd_width_95 > 0` everywhere finite.
- `w1_to_wald_cd >= 0` everywhere finite; equals 0 on the
  `(identity, wald)` cell exactly.
- `nonmonotone_fraction ∈ [0, 1]`.

## Literature

- Schweder, T. and Hjort, N. L. *Confidence, Likelihood, Probability:
  Statistical Inference with Confidence Distributions*. Cambridge
  University Press, 2016. (Foundational; Ch. 3 + Ch. 4.)
- Singh, K., Xie, M., and Strawderman, W. E. "Combining information
  from independent sources through confidence distributions." *Annals
  of Statistics* 33 (2005): 159–183. (CD-validity property.)
- Singh, K., Xie, M., and Strawderman, W. E. "Confidence distribution
  (CD) — distribution estimator of a parameter." *IMS Lecture Notes —
  Monograph Series* 54 (2007): 132–150. (CD-mean / CD-median /
  CD-mode point estimators.)
- Xie, M. and Singh, K. "Confidence Distribution, the Frequentist
  Distribution Estimator of a Parameter: A Review." *International
  Statistical Review* 81 (2013): 3–39. (Review.)
- Olkin, I. and Pukelsheim, F. "The distance between two random
  vectors with given dispersion matrices." *Linear Algebra and its
  Applications* 48 (1982): 257–263. (Closed-form W₂ between
  Gaussians; used in test fixtures.)
- Bissiri, P. G., Holmes, C. C., and Walker, S. G. "A general
  framework for updating belief distributions." *J. Royal Stat. Soc. B*
  78 (2016): 1103–1130. (Power-likelihood underpinning η-tilting.)
- Friel, N. and Pettitt, A. N. "Marginal likelihood estimation via
  power posteriors." *J. Royal Stat. Soc. B* 70 (2008): 589–607.
  (η-path lineage.)
- Syring, N. and Martin, R. "Calibrating general posterior credible
  regions." *Biometrika* 106 (2019): 479–486. (Data-dependent η for
  calibration — closest in spirit to Dyn-WALDO.)

## Links

- Implementation: `src/frasian/experiments/confidence_distribution.py`
- Constructor:    `src/frasian/cd/from_pvalue.py`
- Diagnostic:     `src/frasian/diagnostics/cd_summary.py`
- CD container:   `src/frasian/cd/grid.py`
- Distances:      `src/frasian/cd/distances.py`
- Tests:          `tests/experiments/test_confidence_distribution_experiment.py`,
                  `tests/regression/test_cd_*.py`,
                  `tests/properties/test_cd_invariants.py`

## Status notes

The constructor `build_cd_from_pvalue` is the single production path;
closed-form Wald/WALDO/tilted-WALDO CDs in `cd.from_closed_form` exist
only as test fixtures. `wasserstein_2_gaussian` (Olkin–Pukelsheim) is
similarly a test-time reference. The dynamic-η non-monotone
`signed_confidence` is reported via `nonmonotone_fraction` rather than
treated as a validity failure — it's the diagnostic surface, not a
bug.
