# Frasian Inference Framework

## Project Overview

This project implements numerical experiments and publication-quality visualizations for the Frasian inference framework, which connects WALDO (Weighted Accurate Likelihood-free inference via Diagnostic Orderings), Fraser's higher-order likelihood inference, and confidence distributions.

**Key components:**
- Test framework: 182 tests across 5 tiers validating Theorems 1-10
- Simulation infrastructure: Three-layer architecture for MC simulations
- Visualization suite: 21 publication figures + 4 tables

## The Conjugate Normal Model

- **Prior**: θ ~ N(μ₀, σ₀²)
- **Likelihood**: D | θ ~ N(θ, σ²)
- **Posterior**: θ | D ~ N(μₙ, σₙ²)
- **Weight**: w = σ₀²/(σ² + σ₀²)
- **Posterior mean**: μₙ = wD + (1-w)μ₀
- **Posterior variance**: σₙ² = wσ²

### Key Quantities
- **Standardized coordinate**: u = (θ - μₙ)/(wσ)
- **Scaled prior-data conflict**: Δ = (1-w)(μ₀ - D)/σ
- **Prior residual**: δ(θ) = (θ - μ₀)/σ₀
- **Non-centrality**: λ(θ) = δ(θ)²/w

## Extended Tilting Framework

The tilting parameter η interpolates between methods:
- **η = -1**: Maximum oversharpening (mode pushed past prior)
- **η = 0**: WALDO (standard Bayesian-frequentist hybrid)
- **η = 1**: Wald (pure frequentist, ignores prior)

**Key discovery**: Optimal η* can be **negative** at low conflict!
- At |Δ| ≈ 0: η* ≈ -0.98 yields CIs 15% narrower than Wald
- Constraint: η > -w/(1-w) to keep variance positive

### Non-centrality with Tilting
λ_η = (1-η)² · λ₀

For η < 0: λ_η > λ₀ (oversharpening increases non-centrality but narrows CIs)

## Project Structure

```
src/frasian/
├── __init__.py          # Package exports
├── core.py              # Posterior params, coordinates (10 functions)
├── waldo.py             # WALDO statistic, p-value, CIs (15 functions)
├── tilting.py           # Tilted posterior framework (14 functions)
├── confidence.py        # Confidence distribution (12 functions)
├── figure_style.py      # Publication styling, colors, utilities
├── plotting.py          # Basic visualization utilities
└── simulations/         # Three-layer simulation infrastructure
    ├── __init__.py
    ├── raw.py           # Layer 0: Generate raw D samples
    ├── processing.py    # Layer 1: Compute CIs, coverage, widths
    ├── cache.py         # Layer 1.5: Processed results caching
    ├── storage.py       # HDF5 I/O utilities
    └── runner.py        # Orchestration

scripts/
├── run_simulations.py   # Generate raw D samples + optimal eta
├── plot_coverage.py     # Figures 3.1-3.3 (coverage)
├── plot_ci_widths.py    # Figures 4.1-4.3 (CI widths)
├── plot_tilting.py      # Figures 5.1-5.5 (tilting framework)
├── plot_theory.py       # Figures 1.1-1.3 (core theory)
├── plot_estimators.py   # Figures 2.1-2.2 (estimators)
├── plot_regimes.py      # Figures 6.1-6.2 (three regimes)
├── plot_summary.py      # Figures 7.1-7.3 (summary)
├── generate_tables.py   # Tables 1-4
└── generate_all.py      # Master script

tests/
├── conftest.py              # Fixtures: ModelParams, TestConfig
├── tier1_foundations/       # 45 tests: Theorems 1-2
├── tier2_pvalue/            # 64 tests: Theorems 3-5
├── tier3_coverage/          # 19 tests: Coverage, CI widths
├── tier4_tilting/           # 43 tests: Theorems 6-8
└── tier5_integration/       # 11 tests: Theorems 9-10

output/
├── simulations/
│   ├── raw/                 # Raw D samples (permanent)
│   │   ├── coverage_raw.h5
│   │   ├── distribution_raw.h5
│   │   ├── width_raw.h5
│   │   └── optimal_eta.h5   # Precomputed η* grid
│   └── processed/           # Cached results (regenerable)
├── figures/                 # Generated figures by category
└── tables/                  # CSV + LaTeX tables
```

## Simulation Infrastructure

### Three-Layer Architecture

**Layer 0 (Raw)**: Only D ~ N(θ_true, σ) samples are stored. CI endpoints depend on α, so we don't cache them.

**Layer 1 (Processing)**: Compute CIs, coverage indicators, widths from raw D samples on demand.

**Layer 1.5 (Cache)**: Optional caching of processed results with auto-invalidation if raw data is newer.

### Key Functions

```python
# Raw simulation generation
from frasian.simulations import (
    generate_coverage_D_samples,    # D samples for coverage grid
    generate_distribution_D_samples, # D samples for distribution validation
    generate_width_D_samples,       # D samples for width experiments
    generate_optimal_eta_grid,      # Precompute η*(|Δ|) numerically
)

# Optimal eta (USE THIS for plots)
from frasian.simulations import optimal_eta_empirical
eta_star = optimal_eta_empirical(abs_delta=1.5, fast=False)

# Processing
from frasian.simulations import (
    compute_ci,                     # Unified CI for all 5 methods
    compute_coverage_indicators,    # Boolean coverage array
    bootstrap_mean,                 # Mean with SE and CI
    bootstrap_proportion,           # Proportion with SE and CI
)
```

### Running Simulations

```bash
# Generate all raw data (including optimal eta grid)
python scripts/run_simulations.py

# Fast mode for testing
python scripts/run_simulations.py --fast

# Force regeneration
python scripts/run_simulations.py --force

# Show cache status
python scripts/run_simulations.py --list

# Clear caches
python scripts/run_simulations.py --clear-cache  # Keep raw
python scripts/run_simulations.py --clear-all    # Clear everything
```

### Optimal Eta Computation

The optimal η* is computed numerically using Brent optimization:
- Grid: 251 |Δ| points from 0 to 5 (0.02 spacing)
- n_sims: 500 simulations per optimization step
- Parallelized with joblib (n_jobs=-1)
- ~11 minutes on 12-core machine

**DO NOT use `optimal_eta_approximation()`** - this is the old power-law formula that assumes η ≥ 0.

## Running Tests

```bash
pytest                    # All tests
pytest -m tier1           # Specific tier
pytest -m "not slow"      # Skip slow tests
pytest -n auto            # Parallel execution
pytest -v                 # Verbose output
```

## Generating Figures

```bash
# All figures
python scripts/generate_all.py

# Fast mode (reduced MC samples)
python scripts/generate_all.py --fast

# Specific category
python scripts/plot_tilting.py --fast
python scripts/plot_coverage.py --fast --figure 3.1

# Common flags
--fast      Use reduced MC samples
--show      Display figures interactively
--no-save   Don't save to disk
--figure X  Generate only figure X
```

## Key Numerical Results

### Optimal Tilting (Numerically Computed)

| |Δ| | η* | E[W]/W_Wald |
|------|--------|-------------|
| 0.0 | -0.985 | 0.849 |
| 0.5 | -0.113 | 0.878 |
| 1.0 | 0.777 | 0.930 |
| 2.0 | 0.941 | 0.974 |
| 5.0 | 0.991 | 0.995 |

**Key finding**: E[W]/W_Wald < 1 always - optimal tilting never wider than Wald on average.

### Coverage Table

| θ_true | Wald | Posterior | WALDO |
|--------|------|-----------|-------|
| -3 | 95% | 40% | 95% |
| 0 | 95% | 99% | 95% |
| 3 | 95% | 40% | 95% |
| 5 | 95% | 1% | 95% |

### CI Widths

| Δ | W_Wald | W_Post | W_WALDO |
|------|--------|--------|---------|
| 0 | 3.92 | 2.77 | 3.29 |
| -1 | 3.92 | 2.77 | 3.63 |
| -2.5 | 3.92 | 2.77 | 5.53 |

## Core Library Functions

### core.py
- `posterior_params(D, mu0, sigma, sigma0)` → (μₙ, σₙ, w)
- `weight(sigma, sigma0)` → w
- `scaled_conflict(D, mu0, w, sigma)` → Δ
- `prior_residual(theta, mu0, sigma0)` → δ(θ)

### waldo.py
- `waldo_statistic(mu_n, sigma_n, theta)` → τ
- `noncentrality(theta, mu0, w, sigma, sigma0)` → λ(θ)
- `pvalue(theta, mu_n, mu0, w, sigma)` → p
- `confidence_interval(D, mu0, sigma, sigma0, alpha)` → (lower, upper)
- `wald_ci(D, sigma, alpha)` → (lower, upper)
- `posterior_ci(D, mu0, sigma, sigma0, alpha)` → (lower, upper)

### tilting.py
- `tilted_params(D, mu0, sigma, sigma0, eta)` → (μ_η, σ_η, w_η)
- `tilted_noncentrality(lambda0, eta)` → λ_η = (1-η)²λ₀
- `tilted_ci(D, mu0, sigma, sigma0, eta, alpha)` → (lower, upper)
- `tilted_ci_width(D, mu0, sigma, sigma0, eta, alpha)` → width

### confidence.py
- `pvalue_mode(D, mu0, sigma, sigma0)` → θ_mode = μₙ
- `pvalue_mean(D, mu0, sigma, sigma0, method)` → E[θ]

## Figure Style

Colors (colorblind-friendly):
- WALDO: `#2E86AB` (blue)
- Posterior: `#28A745` (green)
- Wald: `#DC3545` (red)
- Tilted: `#6F42C1` (purple)
- MLE: `#FD7E14` (orange)

## Development Notes

- Use canonical coordinates: μ₀=0, σ=1, vary w (or σ₀)
- `tilted_params()` returns 3 values: (mu_eta, sigma_eta, w_eta)
- `posterior_ci_width(sigma, sigma0)` - NOT (D, mu0, sigma, sigma0)
- Bootstrap for proportions near 0 or 1 (more robust than normal approx)
- Savitzky-Golay smoothing available for noisy η* curve: `smooth_optimal_eta()`
