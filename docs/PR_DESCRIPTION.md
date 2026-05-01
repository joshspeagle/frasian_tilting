# Confidence-distribution module + multi-region CI uplift

Promotes the framework's analysis from per-α coverage/width to the full
**confidence distribution (CD)** — the cumulative integral of the
Schweder–Hjort density. The CD packages the entire α-sweep into one
object whose location, scale, shape, and **W₁ distance to a Wald
reference** are measurable, letting us see Dyn-WALDO's smoothness
pathology directly in distributional form (multimodal p-value →
non-monotone inversion-based confidence curve).

This PR also lands a prerequisite uplift: **multi-region confidence
intervals**. The framework's `dynamic_tilted_confidence_interval`
already returned a region list (Dyn-WALDO's p-value can be multimodal
at low |Δ|), but the uniform `tilting.confidence_interval(...)`
flattened it to the convex hull. Coverage was over-counting the inter-
region gaps; width was over-stating CIs. Both are now fixed via a new
`confidence_regions(...)` protocol method with union semantics in
`coverage` / `width` cells.

## What's new

| Piece | Where |
|---|---|
| `TiltingScheme.confidence_regions` + `pvalue` | `tilting/base.py`, `tilting/identity.py`, `tilting/power_law.py` |
| `GridConfidenceDistribution` (pdf-primary) | `cd/grid.py` |
| `build_cd_from_pvalue` (universal Schweder–Hjort) | `cd/from_pvalue.py` |
| Closed-form fixtures (Wald / WALDO / tilted-WALDO) | `cd/from_closed_form.py` |
| `wasserstein_1` / `wasserstein_2` / `total_variation` | `cd/distances.py` |
| `ConfidenceDistributionExperiment` + `CDSummaryDiagnostic` | `experiments/confidence_distribution.py`, `diagnostics/cd_summary.py` |
| Illustration + brief | `experiments/illustrations/confidence_distribution_demo.py`, `docs/methods/confidence_distribution_experiment.md` |

## Headline findings on `Config.fast()`

| Metric                | Wald   | WALDO  | Dyn-WALDO |
|-----------------------|-------:|-------:|----------:|
| Coverage (target 0.95)| 0.958  | 0.955  | 0.957     |
| Mean CI width         | 3.920  | 3.891  | 4.048     |
| η-curve discontinuities | 0    | 0      | **3** (in `(power_law, waldo)`) |
| W₁ to Wald CD (mean)  | 0.0002 | 0.721  | 0.784     |
| Non-monotone CD fraction | 0   | 0      | **0.58** (max 1.00) |

All three calibrated. Dyn-WALDO's η-curve discontinuities and the
non-monotone CD fraction are the same smoothness pathology surfacing
at two different layers; the CD experiment makes it falsifiable at the
distributional level.

## Design decisions worth flagging

- **pdf-primary CD**: density `c(θ) ≥ 0` is the primitive, cdf derived
  by cumulative integration (always monotone). W₁/W₂ act on the *real*
  probability distribution implied by the density — no rearrangement —
  even when the underlying p-value is multimodal. The non-monotone
  inversion-based confidence curve C(θ) is preserved as auxiliary
  diagnostic data on the CD object.
- **W₂ via Gauss–Hermite on z = Φ⁻¹(u)**: hits Olkin–Pukelsheim closed
  form to ~1e-5 with `n_quad = 64`; robust to σ-mismatch up to 100×.
- **W₁ via CDF-form trapezoidal**: hits the closed form
  `|σ_a − σ_b|·√(2/π)` (zero-mean σ-mismatched Gaussians) to ~1e-7.
  Gauss–Hermite would have been *worse* here — the quantile-axis
  integrand carries a kink and Gauss–Hermite suffers polynomial-fitting
  errors there. CDF-form is principled.
- **Kink-robust |dp/dθ|** in `from_pvalue`: averaged absolute one-sided
  differences avoid the central-diff cancellation at θ = D (Wald) and
  θ = μ_n (WALDO) where p has a kink.
- **No JAX**: grid + finite differences cover all CD operations on the
  1D conjugate-Normal sandbox. Revisit if/when gradient-based selector
  training enters scope.

## Tests added

| File | Tests | What it pins |
|---|---:|---|
| `test_confidence_regions.py` | 11 | Multi-region CI semantics + bimodal regime at α=0.86 |
| `test_tilting_pvalue.py` | 9 | Selector-aware `tilting.pvalue` dispatch (identity/static/dynamic) |
| `test_cd_grid.py` | 26 | Gaussian/skew/bimodal/non-monotone-signed_confidence cases |
| `test_cd_distances.py` | 30 | W₁ + W₂ closed-form agreement (Olkin–Pukelsheim, σ-mismatch) |
| `test_cd_constructors.py` | 22 | `from_pvalue ≈ closed_form` + Dyn-WALDO non-monotone CD |
| `test_cd_invariants.py` | 12 | Hypothesis-property tests for CD protocol |
| `test_confidence_distribution_experiment.py` | 4 | End-to-end + non-monotone-fraction at conflict |

Suite went from 462 to 541 passing (with 32 stub-skips unchanged).

## Verification

```bash
python -m pytest                                   # ~640 passed, 32 skipped
python tools/check_method_completeness.py          # 17 entries OK
python -m scripts.run --fast experiment=coverage             # ~30s
python -m scripts.run --fast experiment=width                # ~30s
python -m scripts.run --fast experiment=smoothness           # ~30s
python -m scripts.run --fast experiment=confidence_distribution  # ~5min
python -m frasian.experiments.illustrations.confidence_distribution_demo --smoke
```

The illustration figure shows the bimodal CDF/cc(θ) at D=3 directly —
the smoothness pathology in one plot.

## What this PR does not do

- Implement the four stub tilting schemes (OT, geodesic, mixture,
  exp-family). The CD experiment's `w1_to_wald_cd` and
  `nonmonotone_fraction` columns are now the falsification target for
  these — a smoother family should sit closer to Wald asymptotically
  than Dyn-WALDO does, *with* a near-zero non-monotone fraction.
- Implement the three stub statistics (LRT, signed-root, Bartlett).
- Run `Config.default()`-resolution baselines. The `Config.fast()`
  numbers in this PR are the structural validation; a higher-resolution
  run is a separate task.
