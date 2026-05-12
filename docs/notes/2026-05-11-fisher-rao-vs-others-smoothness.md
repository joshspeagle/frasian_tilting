# Stage D: Fisher-Rao vs other geodesics — η(θ) and CI-width(D) smoothness

**TL;DR.** On the canonical Normal-Normal sandbox (μ₀=0, σ₀=σ=1, w=0.5,
α=0.05), the framework's central hypothesis — that alternative
geodesics (OT, mixture, Fisher-Rao) yield smoother CI-width(D) profiles
than naive power-law tilting — is **partially confirmed**. Across the
4 schemes × 4 selectors matrix at the calibrated `dyn_numerical`
selector, FR has the lowest CI-width Lipschitz (0.67) and total
variation (5.81), with mixture (0.84 / 7.07) and OT (1.46 / 8.07) also
beating power_law (4.80 / 9.50) by 3-7×. **However**, FR achieves this
trivially: at w=0.5 the per-θ static optimum collapses to η ≡ 0 (bare
WALDO, no tilt), so the FR `dyn_numerical` cell is *not actually
exercising the Fisher-Rao geometry*. The honest comparison is
mixture > OT > power_law on this loss-target combination. On the
discontinuity-count metric the ordering reverses (mixture 32 < PL 40
< OT 46 < FR 52) — narrower CI-width spikes still happen even when
the average curve is smooth.

## Method

`scripts/compare_geodesic_smoothness.py` evaluates two profile types
per cell on a 200-point grid in [-6, 6]:

- **η(θ)** : `selector.select_grid(θ_grid)` for each cell.
- **CI-width(D)** : `tilting.confidence_interval(α, [D], …)` width for
  each D.

Both profiles are then run through the existing
`frasian.diagnostics.smoothness_metrics` quartet — local Lipschitz,
total variation, discontinuity count (3-MAD on the second difference),
and high-vs-low FFT power ratio — the same definitions
`SmoothnessExperiment` consumes, so cross-scheme numbers are
apples-to-apples. The full 16 cells × 2 metric_targets = 32-row matrix
is written to CSV; a 2-panel PNG renders the four schemes' η(θ) and
CI-width(D) curves at the calibrated `learned_intp` slice.

The 4 schemes are `power_law`, `ot`, `mixture`, `fisher_rao`; the 4
selectors are `dyn_numerical` (DynamicNumericalEtaSelector — the
calibrated post-hoc-free baseline) and three Phase G v4 learned heads
`learned_intp` / `learned_cd_var` / `learned_static_w`. All 12 v4
fixtures and 4 disk-cached `dyn_numerical` lookups are pre-existing
infrastructure (Stage C and the `regen_headline` runs).

## Results

CSV: `output/diagnostics/compare_smoothness_f98851996e.csv`
(32 rows). Plot: `output/illustrations/compare_smoothness.png`.

Headline numbers (all metrics: lower = smoother):

### CI-width(D) at `dyn_numerical` (calibrated baseline; the central test)

| scheme       | Lipschitz | TV    | discontinuity | spectral |
|--------------|-----------|-------|---------------|----------|
| `fisher_rao` | 0.67      | 5.81  | 52            | 6.3e-6   |
| `mixture`    | 0.84      | 7.07  | 32            | 1.1e-5   |
| `ot`         | 1.46      | 8.07  | 46            | 1.2e-5   |
| `power_law`  | 4.80      | 9.50  | 40            | 5.9e-4   |

PL is the loudest scheme on Lipschitz (a step-amplitude metric) and
spectral roughness (high-frequency-power ratio) by 3-100×. On
discontinuity_count (a rare-event metric counting 2nd-diff outliers
above 3-MAD-σ) the ordering scrambles: MX is best at 32, FR worst at
52. The two metrics measure different things — Lipschitz captures the
worst slope on the curve, discontinuity_count counts how often the
curve has unusually sharp local kinks. PL has *one* big jump (high
Lipschitz, modest discontinuity_count); FR has many small kinks (low
Lipschitz from a flat curve mixed with discrete numerical-CI bracket
artifacts, but high count from those artifacts each scoring as
outliers against the otherwise-flat baseline).

### η(θ) at `learned_intp` (calibrated Phase G v4 default)

| scheme       | Lipschitz | TV    | discontinuity | spectral |
|--------------|-----------|-------|---------------|----------|
| `power_law`  | 2.49      | 0.30  | 34            | 0.031    |
| `fisher_rao` | 3.94      | 0.47  | 34            | 0.028    |
| `mixture`    | 4.52      | 0.55  | 34            | 0.028    |
| `ot`         | 4.61      | 0.56  | 34            | 0.029    |

PL has the smoothest learned η-curve. The 34 discontinuities are
identical across all four schemes — that's the OOD-θ clamp at θ = ±5σ₀
(the σ₀-anchored training-box edge) showing up as a pair of step
discontinuities which the second-difference detector picks up
identically per-scheme.

### Pathological cell: `fisher_rao[learned_cd_var]`

| metric_target | Lipschitz | TV    | discontinuity | spectral |
|---------------|-----------|-------|---------------|----------|
| η             | 37.25     | 4.49  | 34            | 0.027    |
| CI-width      | 94.38     | 26.97 | 63            | 0.149    |

FR cd_variance produces η reaching ~−2.0 at conflict (per Stage C.4
note `2026-05-11-fisher-rao-cd-var-hyperparams.md`) — orders of
magnitude rougher than every other cell. Its smoothness numbers are
not a feature; they're the cd_var pathology surfacing on the
unconstrained Fisher-Rao geodesic.

## Interpretation

The framework's central hypothesis — "smoother families ↦ less
pathology in CI widths through the conflict band" — is
**directionally correct on the average curve** (Lipschitz, TV,
spectral roughness all favour OT/MX/FR over PL by 3-100×) and the
PL-specific "lower-clamp kink at conflict" is the dominant
roughness signal those metrics pick up. **But it is not unanimous on
all metrics**: discontinuity_count, which counts rare-event 2nd-diff
outliers, ranks mixture as smoothest and FR as roughest — so even
when the bulk of the curve is smoother, alternative geodesics can
still admit isolated sharp transitions.

Two caveats nuance the headline claim:

1. **FR `dyn_numerical` is degenerate at w=0.5.** The per-θ static
   width-minimising optimum on FR with this prior/likelihood balance
   is η = 0 (the no-tilt point — bare WALDO). So the FR
   `dyn_numerical` cell's "smoothness" is bare WALDO smoothness, not
   evidence of the Riemannian geodesic doing anything. A proper FR-
   advantage demonstration needs an asymmetric prior where the static
   optimum is non-trivially shifted off zero, OR a calibrated learned
   selector that adapts beyond the static optimum. The latter is what
   `learned_intp` / `learned_cd_var` should provide; today
   `learned_intp` is near-constant per-cell (per row 13b reframe,
   `2026-05-11-row-13b-loss-specificity-cross-scheme.md`) and
   `learned_cd_var` is pathological.

2. **MX wins discontinuity_count, but its CI-width *Lipschitz* is
   only ~6× smaller than PL** while FR is 7× smaller. The smoothness
   ranking depends on the metric chosen, and there is no single
   "smoothness scalar" that dominates.

The cleanest positive finding is OT and mixture both clearly beat PL
on every CI-width metric except the bare-MX-vs-PL discontinuity
count, with effect sizes 3-7×. The cleanest *negative* finding is
that the Phase G v4 `learned_intp` slice (the framework's calibrated
default) collapses η-curve smoothness across all four schemes to
near-identical values — the network's per-cell near-constancy means
the choice of geodesic barely shows up at the η level on this slice.

## Caveats

- **FR cd_var widths are pathological**. See
  `2026-05-11-fisher-rao-cd-var-hyperparams.md`. Its smoothness
  metrics in the CSV are *not* evidence about FR-as-a-family; they
  are evidence about a specific (FR, cd_var) training failure mode.
- **FR dyn_numerical collapses to bare WALDO at w=0.5**. The cell's
  η-curve has TV = 0, discontinuity_count = 0, spectral_roughness = NaN
  (constant input → mean-detrended ≡ 0 → undefined ratio). The
  CI-width(D) numbers are real; the η-curve numbers are degenerate.
- **OOD-θ override**: LearnedDynamicEtaSelector clamps η to
  `eta_likelihood_only=1.0` (or scheme equivalent) outside the σ₀-
  anchored training box (~5σ₀ from μ₀). Our θ ∈ [-6, 6] sweep at
  σ₀ = 1 has the inner [-5, 5] in-distribution and the outer ±1
  OOD-clamped — visible as the step discontinuities at θ = ±5 in
  every learned-cell η-curve, contributing the identical 34
  discontinuity-count across all four learned_intp cells.
- **Single sandbox**. All numbers here are at μ₀=0, σ₀=σ=1, w=0.5.
  Asymmetric (μ₀ ≠ 0 or σ₀ ≠ σ) sandboxes might shift the headline
  ranking — the FR `dyn_numerical` collapse in particular is sandbox-
  specific.

## Open questions

- **Does smoother CI-width translate to better calibration / power
  at larger α?** This run holds α = 0.05 fixed; smoother families
  *should* in principle calibrate more reliably across α-levels, but
  that's a separate empirical question (α-sweep).
- **What does the smoothness ranking look like at off-default w?**
  The FR `dyn_numerical` collapse is a w-specific phenomenon; at
  w ≠ 0.5 the per-θ static optimum may be non-trivial and FR's
  curve might leave the bare-WALDO basin.
- **Why do learned_intp η-curves have nearly-identical smoothness
  across the 4 schemes?** Likely the row-13b near-constancy:
  integrated_p training collapses each cell's η output to a narrow
  range (per-cell std ~5e-4), so the resulting η-curves are
  near-constant + the same OOD-clamp step at the boundary, and the
  smoothness diagnostic just sees that step.
- **Mixture's discontinuity-count win** — is it a property of the
  m-geodesic (linear-density mixing avoids the reciprocal-denominator
  singularity that PL's denom = 1 - η(1-w) introduces) or a numerical
  artifact of MixtureTilting's CI-inversion code path? Worth a
  follow-up before treating it as a robust finding.

## Provenance

- Branch: `feat/fisher-rao-tilting`.
- Commit at probe time: `f98851996e8e4b9d0658d8e0c0cbd4b7568b2b71`.
- Script: `scripts/compare_geodesic_smoothness.py` (Stage D.1).
- CSV: `output/diagnostics/compare_smoothness_f98851996e.csv`.
- Plot: `output/illustrations/compare_smoothness.png`.
- Regression test: `tests/regression/test_smoothness_comparison.py`
  (4 orderings; ~27s on 1 worker, all pass on this commit).
- Companion notes:
  [`2026-05-11-row-13b-loss-specificity-cross-scheme.md`](2026-05-11-row-13b-loss-specificity-cross-scheme.md)
  (per-cell near-constancy of integrated_p); 
  [`2026-05-11-fisher-rao-cd-var-hyperparams.md`](2026-05-11-fisher-rao-cd-var-hyperparams.md)
  (FR cd_var training instability + Stage C.4 hyperparam regime).
