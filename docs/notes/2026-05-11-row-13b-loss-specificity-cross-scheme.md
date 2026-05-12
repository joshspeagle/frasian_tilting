# Row 13b reframe: cross-scheme input-sensitivity probe

**TL;DR.** Hypothesis **partially confirmed**. Across all 4 schemes
(power_law, ot, mixture, fisher_rao), `integrated_p` produces uniformly
near-constant per-cell η (median per-cell std 5.5e-4, max 7.2e-3) and
modest cross-cell spread (0.10-0.20). This pattern is real and
loss-specific. However, the converse — that `cd_variance` and
`static_width` uniformly produce strong adaptation — does NOT hold:
PL/OT/MX `cd_variance` per-cell stds are still small (1e-3 to 4e-3,
~5x integrated_p, not 100x), and only **specific (scheme, loss)
combinations** show dramatic adaptation: FR cd_variance (per-cell std
~0.17, spread 1.43) and PL static_width (spread 0.778, η range 0.21-1.01).
The architecture **can** learn input-sensitive η when the
(scheme, loss, optimizer) configuration permits it; row-13b's
architectural framing is too strong, but its empirical pattern is
loss-specific to integrated_p across all schemes tested.

## Problem statement

CLAUDE.md row 13b documents a known limitation of Phase G v4 training:

> "Phase G v4 learned-η training has fundamental input-insensitivity:
> trained networks output range is ~10% of the per-slice optimum range
> across the hyperparam grid, with negative correlation to the
> analytical optimum on PL/OT. Boundary penalty / input-norm /
> anti-wald don't fix it. Stage C (mixture learned-η training) deferred
> until input-sensitivity-aware training is implemented."

The diagnosis was treated as **architectural**: the EtaNet/ValidityNet
pipeline was assumed to be incapable of input-sensitive learning. The
new note `docs/notes/2026-05-11-fisher-rao-cd-var-hyperparams.md`
(Stage C.4) found that on Fisher-Rao, `cd_variance` produces 100x more
per-θ adaptation than `integrated_p` on the same architecture, refuting
the architectural framing for FR. The hypothesis tested here: **row-13b
is loss-specific to integrated_p, not architecture-specific**, and this
holds across all 4 schemes.

## Method

`tools/probe_input_sensitivity_cross_scheme.py` loads each Phase G v4
fixture and evaluates `predict_eta` on the same grid used in
`tests/regression/test_fisher_rao_v4_fixture.py`:

- θ grid: 51 points in [-3, 3].
- 3 hyperparam cells: `(mu0, sigma0, sigma) in {(0, 0.5, 1.0),
  (1, 2.0, 1.5), (-1, 1.0, 0.5)}`.
- Per-fixture statistics: per-cell `std(eta over theta)`, per-cell
  `mean(eta over theta)`, cross-cell spread of those means, global η
  range across all cells x θ.

12 fixtures probed (4 schemes x 3 losses). A 13th
`fisher_rao_phaseC_integrated_p_lambda0_v4.eqx` fixture exists on disk
(an FR-only `lambda_max=0` ablation) but is outside the 4x3 matrix
and excluded.

Analytical-optimum correlation (the second half of row-13b's claim) is
**not** computed here. PL/OT have closed-form analytical optima but FR
and MX numerical optima are scheme-specific and costly; cross-scheme
comparison is omitted and left for a follow-up.

## Results

```
    scheme           loss | per-cell std (cell0, cell1, cell2)        per-cell mean (cell0, cell1, cell2)         spread          range
--------------------------------------------------------------------------------------------------------------------
 power_law   integrated_p | std=( 9.96e-04,  7.23e-03,  6.31e-03)  mean=(+0.807, +0.868, +0.903)  spread=0.096  range=[+0.805, +0.912]
 power_law    cd_variance | std=( 4.03e-03,  2.97e-03,  4.23e-03)  mean=(+0.862, +0.894, +0.875)  spread=0.032  range=[+0.854, +0.898]
 power_law   static_width | std=( 1.07e-02,  2.95e-03,  3.37e-03)  mean=(+0.233, +0.862, +1.010)  spread=0.778  range=[+0.212, +1.014]
        ot   integrated_p | std=( 5.17e-04,  4.56e-04,  4.62e-04)  mean=(+0.618, +0.764, +0.815)  spread=0.198  range=[+0.616, +0.816]
        ot    cd_variance | std=( 6.73e-04,  2.86e-04,  3.04e-04)  mean=(+0.656, +0.677, +0.934)  spread=0.278  range=[+0.655, +0.934]
        ot   static_width | std=( 1.43e-03,  1.06e-03,  7.55e-04)  mean=(+0.521, +0.780, +0.770)  spread=0.259  range=[+0.517, +0.781]
   mixture   integrated_p | std=( 9.12e-04,  1.17e-03,  2.40e-04)  mean=(+0.659, +0.809, +0.815)  spread=0.156  range=[+0.658, +0.815]
   mixture    cd_variance | std=( 2.01e-03,  1.29e-03,  1.28e-03)  mean=(+0.699, +0.814, +0.793)  spread=0.116  range=[+0.695, +0.817]
   mixture   static_width | std=( 2.29e-03,  1.99e-03,  6.99e-04)  mean=(+0.785, +0.880, +0.935)  spread=0.149  range=[+0.782, +0.936]
fisher_rao   integrated_p | std=( 3.96e-04,  5.13e-04,  5.82e-04)  mean=(+0.710, +0.806, +0.871)  spread=0.161  range=[+0.710, +0.872]
fisher_rao    cd_variance | std=( 1.34e-01,  2.09e-01,  2.01e-01)  mean=(-0.189, -1.620, -1.436)  spread=1.431  range=[-1.984, +0.032]
fisher_rao   static_width | std=( 3.23e-03,  4.95e-04,  2.78e-03)  mean=(+0.759, +0.838, +0.800)  spread=0.079  range=[+0.753, +0.838]
```

Loss-specificity rollup (across all 4 schemes):

| loss          | median per-cell std | min       | max       | n cells |
|---------------|---------------------|-----------|-----------|---------|
| integrated_p  | 5.50e-4             | 2.40e-4   | 7.23e-3   | 12      |
| cd_variance   | 2.49e-3             | 2.86e-4   | 2.09e-1   | 12      |
| static_width  | 2.14e-3             | 4.95e-4   | 1.07e-2   | 12      |

| loss          | median cross-cell spread | min   | max   | n schemes |
|---------------|--------------------------|-------|-------|-----------|
| integrated_p  | 0.158                    | 0.096 | 0.198 | 4         |
| cd_variance   | 0.197                    | 0.032 | 1.431 | 4         |
| static_width  | 0.204                    | 0.079 | 0.778 | 4         |

**The integrated_p pattern is universal across schemes.** All four
integrated_p fixtures produce per-cell std in the 2.4e-4 to 7.2e-3
band (median 5.5e-4) and cross-cell spread in the 0.10-0.20 band. This
is the row-13b "near-constant per-cell" pattern, and the probe confirms
it holds equally for power_law, ot, mixture, and fisher_rao. The
weakest spread (PL at 0.096) and strongest (OT at 0.198) span only ~2x
within the loss; integrated_p is genuinely a loss-level effect.

**The cd_variance + static_width "stronger adaptation" claim is
non-uniform.** Median per-cell stds for the two non-integrated_p losses
are only ~5x larger than integrated_p, well within an order of
magnitude — for most (scheme, loss) cells, training still produces
near-constant per-cell η. The dramatic exceptions are isolated:

- **FR cd_variance**: per-cell std 0.13-0.21 (250-500x integrated_p),
  cross-cell spread 1.43, η extending to -2.0. This is the Stage C.4
  finding and it remains an outlier.
- **PL static_width**: cross-cell spread 0.778 with η range 0.21 to
  1.01 across cells (mean η jumping 0.23 -> 0.86 -> 1.01). Per-cell
  stds remain modest (1e-3 to 1e-2) — the network learned strong
  per-cell adaptation but stays approximately flat in θ within a cell.

PL/OT/MX cd_variance fixtures look more like integrated_p than like FR
cd_variance: per-cell stds 3e-4 to 4e-3, cross-cell spreads 0.03 to 0.28.

## Implication

CLAUDE.md row 13b's framing should be softened from "fundamental
input-insensitivity" (architectural) to "integrated_p produces near-
constant per-cell output across all schemes; some other (scheme, loss)
configurations break the pattern but most do not." The follow-on claim
about negative correlation to analytical optimum on PL/OT is not
re-tested here and may still hold.

Suggested replacement text for row 13b (proposed only — Stage D.4 owns
the actual CLAUDE.md edit):

> 13b | **partial limitation** | Phase G v4 `integrated_p` training
> produces near-constant per-cell η across all four schemes (median
> per-cell std ~5e-4, cross-cell spread 0.10-0.20). Other losses
> (`cd_variance`, `static_width`) can produce stronger adaptation
> on specific scheme combinations (FR cd_variance: spread 1.43; PL
> static_width: spread 0.78) but most non-integrated_p fixtures still
> show modest per-cell stds (1e-3 to 4e-3). The architecture **is**
> capable of input-sensitive learning; the limitation is loss-specific
> to integrated_p, not architectural. See
> [`docs/notes/2026-05-11-row-13b-loss-specificity-cross-scheme.md`](2026-05-11-row-13b-loss-specificity-cross-scheme.md).

## Open questions

- **Analytical-optimum correlation**. Row 13b's "negative correlation
  to the analytical optimum on PL/OT" claim is not re-tested. If
  cd_variance/static_width on PL/OT also show negative correlation,
  the row's broader story (training landscape misaligned with the
  per-slice optimum) survives even with the architectural framing
  removed.
- **Why is PL static_width an outlier?** The η jumping from mean 0.23
  to 1.01 across cells is a dramatic per-cell adaptation pattern not
  seen in OT or MX static_width. Possibly a hyperparam-cell
  collapse to the static optimum (which happens to vary strongly across
  PL cells) rather than learning per-θ structure.
- **Does FR cd_variance's strong adaptation generalise**, or is it an
  artifact of the Cartan-Hadamard "no admissibility bound" that lets
  EtaNet wander freely into negative η? FR is unique in lacking a
  meaningful boundary penalty (Stage C.4 note); the strong adaptation
  may be downstream of that, not an intrinsic FR-cd_variance feature.
- **Should production-default learned-η switch losses?** The framework
  currently defaults to `integrated_p` (CLAUDE.md "calibrated and
  narrower" headline). If integrated_p is the loss that triggers the
  near-constant pattern, the headline cell may benefit from
  reconsidering — but coverage validity is the critical property, not
  per-θ adaptation strength, so this needs separate evaluation.

## Provenance

- Branch: `feat/fisher-rao-tilting`.
- Commit at probe time: `b6656652d8655e68ab513a29ad67aae9ca2b6480`.
- Probe script: `tools/probe_input_sensitivity_cross_scheme.py`.
- Captured output: `/tmp/probe_input_sensitivity_output.txt`.
- Fixtures: 12 Phase G v4 fixtures at
  `artifacts/learned_eta_canonical_normal_normal_<scheme_token>_phaseC_<loss>_v4.eqx`,
  with `<scheme_token>` in {powerlaw, ot, mixture, fisher_rao} and
  `<loss>` in {integrated_p, cd_variance, static_width}. A 13th
  FR-specific `_phaseC_integrated_p_lambda0_v4.eqx` ablation fixture
  is on disk but excluded from the 4x3 matrix.
- Companion notes:
  [`2026-05-11-fisher-rao-cd-var-hyperparams.md`](2026-05-11-fisher-rao-cd-var-hyperparams.md)
  (Stage C.4 FR characterization that motivated this probe);
  [`2026-05-09-mixture-smoothness-and-learned-eta-tails.md`](2026-05-09-mixture-smoothness-and-learned-eta-tails.md)
  (the original row-13b diagnosis).
