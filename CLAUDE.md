# Frasian Inference Testing Framework

## Project Overview

This project implements numerical experiments to validate the theoretical results from the Frasian inference framework, which connects WALDO (Weighted Accurate Likelihood-free inference via Diagnostic Orderings), Fraser's higher-order likelihood inference, and confidence distributions.

## Key Concepts

### The Conjugate Normal Model
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

### Core Theorems Being Validated

1. **Theorem 1**: μₙ - θ ~ N(b(θ), v) where b(θ) = (1-w)(μ₀ - θ), v = w²σ²
2. **Theorem 2**: τ_WALDO ~ w·χ²₁(λ(θ))
3. **Theorem 3**: p(θ) = Φ(b-a) + Φ(-a-b) where a = |μₙ - θ|/(wσ), b = (1-w)(μ₀ - θ)/(wσ)
4. **Theorem 4**: Mode of p-value function is μₙ (posterior mean)
5. **Theorem 5**: Closed-form mean of confidence distribution
6. **Theorem 6**: Tilted posterior parameters μ_η, σ_η²
7. **Theorem 7**: λ_η = (1-η)²·λ₀ (non-centrality reduction)
8. **Theorem 8**: Tilted p-value formula
9. **Theorem 9**: Three-regime structure of dynamically-tilted CIs
10. **Theorem 10**: Fixed-point equation for dynamically-tilted mode

## Project Structure

```
src/frasian/
├── __init__.py      # Package exports
├── core.py          # Posterior params, coordinates, fundamental computations
├── waldo.py         # WALDO statistic, p-value, confidence intervals
├── tilting.py       # Tilted posterior framework
├── confidence.py    # Confidence distribution functions (mode, mean, sampling)
└── plotting.py      # Visualization utilities

tests/
├── conftest.py              # Shared fixtures
├── tier1_foundations/       # Basic distribution tests (Thm 1-2)
├── tier2_pvalue/            # P-value and estimators (Thm 3-5)
├── tier3_coverage/          # Coverage and CI properties
├── tier4_tilting/           # Tilted framework (Thm 6-8)
└── tier5_integration/       # Advanced tests (Thm 9-10)
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific tier
pytest -m tier1
pytest -m tier2

# Run with parallel execution
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# Run with verbose output
pytest -v
```

## Test Configuration

- **Monte Carlo samples**: 10,000 (default), 5,000 for coverage tests
- **Random seed**: 42 (base)
- **KS test threshold**: p > 0.01
- **Coverage tolerance**: ±1.5%
- **Moment tolerance**: 5% relative error

## Parameter Grids

Standard test parameters:
- w ∈ {0.2, 0.5, 0.8}
- σ ∈ {0.5, 1.0, 2.0}
- |Δ| ∈ {0, 0.5, 1, 1.5, 2, 3, 5}
- η ∈ {0, 0.25, 0.5, 0.75, 1.0}

## Key Results to Reproduce

### Coverage Table (Section 7)
| θ_true | Wald | Posterior | WALDO |
|--------|------|-----------|-------|
| -3     | 95%  | 40%       | 95%   |
| 0      | 95%  | 99%       | 95%   |
| 3      | 95%  | 40%       | 95%   |
| 5      | 95%  | 1%        | 95%   |

### CI Width Table (Section 6)
| Δ    | W_Wald | W_Post | W_WALDO |
|------|--------|--------|---------|
| 0    | 3.92   | 2.77   | 3.29    |
| -1   | 3.92   | 2.77   | 3.63    |
| -2.5 | 3.92   | 2.77   | 5.53    |

### Optimal Tilting (Section 10)
| |Δ| | η*(θ)  | E[W]/W_Wald |
|-----|--------|-------------|
| 0.0 | 0.23   | 0.85        |
| 1.0 | 0.77   | 0.93        |
| 3.0 | 0.97   | 0.99        |

## Development Notes

- All functions should handle both scalar and array inputs via numpy broadcasting
- Use scipy.stats for distribution functions (norm, ncx2)
- scipy.optimize for root-finding and optimization
- Prefer vectorized operations over loops
