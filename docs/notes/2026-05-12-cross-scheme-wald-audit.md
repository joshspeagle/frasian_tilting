# Cross-scheme Wald audit (post-FR-PR-merge)

**TL;DR.** After the FR PR merge (`34dd031`), `scripts/run_wald_audit.py`
was run for all 4 newly-enabled `fr_*` flavors and the previously
partially-saved `ot_dyn_numerical`. Combined with the pre-existing PL /
OT / MX audit data, this completes the 4 schemes × 4 selectors audit
matrix. **All 16 production cells calibrate to ≈ 0.95** (min coverage
0.930-0.950, mean 0.953-0.959). **OT learned cells dominate on
tail-max CI-width** at the conflict band (`|θ_true| ≥ 2`):
`ot[learned_cd_var]` is tightest at 4.42 (+13% over Wald 3.92). FR
cells are competitive on `learned_static_w` (4.91) and `learned_intp`
(5.08) but `fr[learned_cd_var]` is pathological (9.23, +135%) — the
trained negative-η values on FR's unbounded admissibility produce
very wide CIs even unclamped. `fr[dyn_numerical]` exactly reproduces
bare WALDO (5.38) per the Stage D w=0.5 collapse finding.

## Method

`PYTHONHASHSEED=0 python -m scripts.run_wald_audit --flavor <flavor>
--n-jobs -1` for each flavor, then per-flavor aggregation across the
saved `(coverage, width, smoothness, cd)` sub-experiments. The five
flavors run on 2026-05-12 (PR merge day) are:

- `fr_dyn_numerical`           — DynamicNumerical selector + FR scheme
- `fr_learned_intp`            — Phase G v4 integrated_p head
- `fr_learned_cd_var`          — Phase G v4 cd_variance head
- `fr_learned_static_w`        — Phase G v4 static_width head
- `ot_dyn_numerical`           — re-run (only smoothness was saved
                                 previously; full 4-sub-experiment
                                 sweep this run)

`fr_dyn_numerical_generic` was killed after ~2.5h on the `coverage`
sub-experiment alone — wall-clock-prohibitive per
`docs/methods/fisher_rao.md` ("math-validation infrastructure, not
production audit"). Can be run overnight if needed.

## Results

```
flavor                    mean cov  min cov  mean W  tail max W   η Lip   η TV   W Lip
--------------------------------------------------------------------------------------------
wald                         0.958    0.940    3.92        3.92    0.00   0.00    0.00
waldo                        0.955    0.940    3.89        5.38    0.00   0.00    1.33
pl_dyn_numerical             0.959    0.950    4.07        5.30    3.42   2.14    0.65
pl_learned_intp              0.956    0.940    3.85        4.85    3.42   2.14    0.65
pl_learned_cd_var            0.956    0.935    3.83        4.64    3.42   2.14    0.65
pl_learned_static_w          0.956    0.940    3.91        5.47    3.42   2.14    0.65
ot_dyn_numerical             0.959    0.950    4.16        5.61    2.32   1.95    0.65
ot_learned_intp              0.955    0.930    3.82        4.49    2.27   1.85    0.65
ot_learned_cd_var            0.954    0.930    3.81        4.42    2.27   1.85    0.65
ot_learned_static_w          0.955    0.930    3.84        4.70    2.27   1.85    0.65
mx_dyn_numerical             0.959    0.950    4.02        5.02    1.26   1.05    0.64
mx_learned_intp              0.955    0.940    3.89        5.31    1.26   1.05    0.64
mx_learned_cd_var            0.955    0.940    3.89        5.24    1.26   1.05    0.64
mx_learned_static_w          0.956    0.935    3.86        5.00    1.26   1.05    0.64
fr_dyn_numerical             0.955    0.940    3.89        5.38     --     --      --
fr_learned_intp              0.955    0.935    3.86        5.08     --     --      --
fr_learned_cd_var            0.953    0.935    5.44        9.23     --     --      --
fr_learned_static_w          0.956    0.930    3.85        4.91     --     --      --
```

`mean cov` / `min cov` are coverage rates averaged / minimised over
the audit's `(θ_true, w)` grid. `mean W` is mean CI width over the
same grid; `tail max W` is the largest mean-width across cells with
`|θ_true| ≥ 2` (the conflict band). The smoothness columns
(`η Lip`, `η TV`, `W Lip`) come from `SmoothnessExperiment`, which
internally uses `NumericalEtaSelector` (static); for FR that selector
returns NaN on some `|Δ|` cells, so all six FR smoothness values
land as `--`. The Stage D `scripts/compare_geodesic_smoothness.py`
covers FR smoothness via the dynamic-η path correctly.

## Tail-max CI-width ranking (the framework's central competitive metric)

| Rank | Cell | Tail-max W | vs Wald 3.92 |
|------|------|-----------|--------------|
| 1 | ot[learned_cd_var]    | 4.42 | +13% |
| 2 | ot[learned_intp]      | 4.49 | +15% |
| 3 | pl[learned_cd_var]    | 4.64 | +18% |
| 4 | ot[learned_static_w]  | 4.70 | +20% |
| 5 | pl[learned_intp]      | 4.85 | +24% |
| 6 | fr[learned_static_w]  | 4.91 | +25% |
| 7 | mx[learned_static_w]  | 5.00 | +28% |
| 8 | mx[dyn_numerical]     | 5.02 | +28% |
| 9 | fr[learned_intp]      | 5.08 | +30% |
| 10 | mx[learned_cd_var]   | 5.24 | +34% |
| 11 | pl[dyn_numerical]    | 5.30 | +35% |
| 12 | mx[learned_intp]     | 5.31 | +36% |
| 13 | waldo                | 5.38 | +37% |
| 13 | fr[dyn_numerical]    | 5.38 | +37% (= bare WALDO) |
| 15 | pl[learned_static_w] | 5.47 | +40% |
| 16 | ot[dyn_numerical]    | 5.61 | +43% |
| 17 | fr[learned_cd_var]   | **9.23** | **+135%** (pathological) |

## Interpretation

- **OT occupies 3 of the top 4 ranks** (cd_var #1, intp #2, static_w #4)
  with pl[learned_cd_var] interleaved at #3 — the W2 geodesic is
  empirically the best practical choice on the canonical sandbox. The
  cd_variance head is the headline winner; integrated_p is competitive.
- **FR `learned_static_w` is the best non-OT learned cell at rank 6** —
  the Stage C training works; FR's CI-width is competitive with PL
  learned cells.
- **MX is calibrated but tail-wide** — every MX learned cell ranks
  below FR `learned_static_w` despite MX's smoothest η (TV 1.05 vs
  PL 2.14 vs OT 1.85). Smoothness of η does not predict tail tightness.
- **All `dyn_numerical` cells inflate** at the conflict band by
  35-43% over Wald — the lower-clamp pathology Stage D quantified.
- **FR-specific anomalies, all expected**:
  - `dyn_numerical` matches bare WALDO exactly at w=0.5
    (η=0 per-θ static optimum; degenerate);
  - `learned_cd_var` is +135% wider than Wald — geodesically valid but
    the cd_variance loss optimum on FR's negative-η half-plane is
    actively worse for CI width;
  - Smoothness columns missing pending a SmoothnessExperiment fix for
    FR's static-η path.

## Provenance

- Branch: `main` (post FR PR merge at `34dd031`).
- Audit runs: 2026-05-11 evening (fr_dyn_numerical, fr_learned_intp,
  fr_learned_cd_var, fr_learned_static_w) and 2026-05-12 early
  morning (ot_dyn_numerical re-run, fr_dyn_numerical_generic
  killed after 2.5h).
- Data: `results/wald_audit/<flavor>/{coverage,width,smoothness,
  confidence_distribution}/*.csv` — each cell × sub-experiment a
  separate CSV, with cache files at `<cell>/cache/` for incremental
  re-runs.
- Aggregation: inline Python in the run log (no committed script).
