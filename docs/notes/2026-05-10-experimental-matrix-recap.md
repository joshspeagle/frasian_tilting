# Experimental matrix recap — 11-flavor audit results

**Context.** After the mixture sigmoid bound landed (2026-05-10), the
audit suite covers 2 baselines + 9 learned-η variants on conjugate
Normal-Normal. This note pins the headline numbers so future
diagnostic work has a reference frame.

## Average widths over (θ_true × w) grid

| Flavor                | cov θ=-3 | cov θ=+3 | cov all | width θ=-3 (lo–hi) | width all |
|-----------------------|----------|----------|---------|--------------------|-----------|
| **Wald** (baseline)   | 0.965    | 0.940    | 0.958   | 3.92 – 3.92        | **3.92**  |
| **WALDO** (baseline)  | 0.961    | 0.945    | 0.955   | 3.88 – 4.59        | 3.89      |
| pl learned intp       | 0.964    | 0.944    | 0.956   | 3.77 – 4.50        | 3.85      |
| pl learned cd_var     | 0.964    | 0.944    | 0.956   | 3.77 – 4.40        | 3.83      |
| pl learned static_w   | 0.961    | 0.944    | 0.956   | 3.78 – 4.84        | 3.91      |
| ot learned intp       | 0.964    | 0.941    | 0.955   | 3.77 – 4.31        | 3.82      |
| **ot learned cd_var** | 0.965    | 0.942    | 0.954   | 3.77 – 4.26        | **3.81**  |
| ot learned static_w   | 0.963    | 0.942    | 0.955   | 3.77 – 4.42        | 3.84      |
| mx learned intp       | 0.960    | 0.945    | 0.955   | 3.78 – 4.73        | 3.89      |
| mx learned cd_var     | 0.961    | 0.944    | 0.955   | 3.77 – 4.69        | 3.89      |
| mx learned static_w   | 0.965    | 0.943    | 0.956   | 3.78 – 4.56        | 3.86      |

Raw CSVs: `results/wald_audit/<flavor>/coverage/coverage_rate.csv`
and `…/width/mean_width.csv`. Settings: `Config.fast()` w-grid (5 bins
0.20–0.80), θ-grid {-3, -2, -1, 0, 1, 2, 3, 4}, n_reps=200, α=0.05.

## Headlines

- **All 11 flavors calibrated.** Coverage 0.94–0.97 across the grid
  (within ~1 SE of nominal 0.95 at n_reps=200). The asymmetric θ=+3
  dip (0.94 vs 0.96 at θ=-3) is present in every cell including Wald
  — sampling / grid artifact, not a learned-η bug.
- **Best cell: OT × cd_variance (3.81 avg width vs Wald 3.92).** That's
  ~2.8% narrower at maintained coverage. cd_var is the narrowest
  objective for both PL and OT.
- **PL ≈ OT > MX on width.** OT edges PL by ~0.5%; mixture sits at
  WALDO baseline (3.89 avg) — the m-geodesic gives up the prior-side
  leverage at extreme conflict that e-/W2-geodesics keep.
- **Best loss objective: cd_variance.** Narrowest for PL and OT,
  mid-pack for MX (where the structural sigmoid bound was necessary
  to make it work at all — see
  [`2026-05-10-mixture-cd-variance-instability.md`](./2026-05-10-mixture-cd-variance-instability.md)).
  `integrated_p` consistently close behind.
- **WALDO is already ~1% narrower than Wald on average** (3.89 vs
  3.92). Learned-η on PL/OT extracts another ~2%; total margin over
  Wald is ~3% width reduction at the same coverage.

## Bottom line

The conjugate Normal-Normal sandbox is "tight" — absolute headroom
over Wald is modest (~3%), but the *qualitative* result is clean:
dynamic-η learned selectors hold calibration and recover real-if-small
width gains, with OT cd_var as the canonical winner. Future
non-conjugate models (Bernoulli, future GLMs) should produce larger
margins where the asymptotic Wald approximation is looser.
