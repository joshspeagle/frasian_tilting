# smoothness_experiment

> Status: `implemented`

## Summary

The framework's central diagnostic. Sweeps `|Δ|` (the scaled prior-data
conflict) on a fine grid at fixed `(w, α, σ)`; for each cell of the
`(TiltingScheme x TestStatistic)` cross-product, picks the optimal tilting
parameter `η*(|Δ|)` via the numerical η-selector and records the resulting
CI endpoints. The downstream `SmoothnessDiagnostic` computes Lipschitz,
total-variation, discontinuity-count, and spectral-roughness metrics on
the resulting curves — making "naive power-law tilting produces sharp
transitions" a falsifiable claim that future tilting schemes (OT,
geodesic, mixture) must do better than.

## Motivation

The user's central empirical observation: power-law tilting flips
abruptly between posterior-driven and likelihood-driven behaviour as
`|Δ|` crosses a threshold. The `legacy/src/frasian/tilting.py:744` MLP
was a band-aid — fitting a smooth approximation to a fundamentally
sharp curve. The framework's purpose is to *measure* the sharpness so
that genuinely smoother tilting families (Wasserstein geodesics, OT
interpolants, Fisher-Rao geodesics) can be evaluated against it.

This experiment is the rigorous version of "is my new tilting scheme
better?" — it produces numbers. A scheme that claims smoothness in
its brief must verify it here.

## Definition

For each grid point `|Δ|_i`:

  D_i      = mu0 - |Δ|_i * sigma / (1 - w)        (invert Delta convention)
  η*_i     = argmin_η  W_eta(D_i, w, α)            (numerical via Brent)
  L_i, U_i = tilted_confidence_interval(α, D_i, model, prior, η*_i, statistic)
  width_i  = U_i - L_i

Diagnostic metrics over the grid:

  Lipschitz(η*)   = max_i |η*_{i+1} - η*_i| / |Δ_{i+1} - Δ_i|
  TV(η*)          = Σ_i |η*_{i+1} - η*_i|
  Discontinuities = #{ i : |Δ²η*_i - median(Δ²η*)| > 3 * 1.4826 * MAD(Δ²η*) }
  Spectral(η*)    = sum_{|f|>f_c}|F[η*]|² / sum_{|f|<=f_c}|F[η*]|²
                    where f_c = N/4 (quarter-Nyquist cutoff).

Same metrics computed on `width_i` for completeness.

## Derivation

The "discontinuity at low |Δ|" prediction follows from the closed-form
power-law optimal η*. Solving `dW_η/dη = 0` with the WALDO width
formula yields, asymptotically, `1 - η* ~ c / |Δ|^p` with `p ≈ 1.7`
(legacy fit at `tilting.py:414`). At `|Δ| → 0`, `1 - η* → ∞`, so `η*`
is *clamped* against its admissible boundary `eta_low = -w/(1-w)`.
The clamp produces a kink: η* sits flat at the boundary for small
`|Δ|`, then suddenly tracks the asymptotic curve. The Lipschitz spike
at the kink is the empirical signature.

Smoother tilting families (e.g. OT geodesics) should not require the
clamp because their geometry is gracefully bounded.

## Predicted behavior

For (`power_law`, `waldo`):

- η*(0) ≈ -w/(1-w) + buffer  (clamped at admissible boundary).
- η*(|Δ| → ∞) → 1            (Wald limit).
- A *visible kink* near `|Δ| ≈ 0.3-0.7` where η* leaves the boundary.
- Lipschitz spike at the kink; TV(η*) ~ 1; discontinuity count > 0.

For (`power_law`, `wald`): η* is irrelevant (Wald ignores prior); η*
returns to its identity element 1.0 trivially. Width is constant.
The diagnostic's purpose for the Wald row is as a *zero baseline*.

## Failure modes

- Numerical η-selector failure at extreme `(w, α)`: bracket tightening
  degrades, `np.inf` width returned. Cell row records NaN.
- Statistic without a tilted-CI bridge (future LRT, signed-root): row
  is all NaN; the diagnostic preserves the result and metrics are NaN.

## Invariants

- `eta_star ∈ admissible_range(context)` (or NaN).
- `ci_upper >= ci_lower` (or both NaN).
- `lipschitz_eta >= 0` (or NaN).
- `total_variation_eta >= 0` (or NaN).
- `discontinuity_count_eta >= 0` integer.
- For (power_law, wald): `eta_star` constant in `|Δ|` → `lipschitz_eta = 0`.

## Literature

- Holmes, A. C., Walker, S. G. "Assigning a value to a power likelihood..."
  *Biometrika* 104 (2017): 497–503. (Power-likelihood tempering.)
- Bissiri, P. G., Holmes, C. C., Walker, S. G. "A general framework for
  updating belief distributions." *J. Royal Stat. Soc. B* 78 (2016).
- Cuturi, M. "Sinkhorn distances: Lightspeed computation of optimal
  transport." *NeurIPS* 2013. (OT path interpolation, future tilting.)
- Amari, S. "Information Geometry and Its Applications." Springer, 2016.
  (Fisher-Rao geodesic on the Gaussian family, future tilting.)

## Links

- Implementation: `src/frasian/experiments/smoothness.py`
- η-selector:     `src/frasian/tilting/eta_selectors.py`
- Diagnostic:     `src/frasian/diagnostics/smoothness_metrics.py`
- Tests:          `tests/experiments/test_smoothness_experiment.py`
- Illustration:   `src/frasian/experiments/illustrations/smoothness_demo.py`

## Status notes

The (TiltingScheme, TestStatistic) cell-evaluator is currently a
specialization on `statistic.name` inside `PowerLawTilting`. Step 6
generalizes via multiple dispatch when more cell shapes exist (LRT,
signed-root). The smoothness experiment is *the* gating diagnostic
for any new tilting scheme.
