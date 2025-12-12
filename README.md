# Frasian Inference Framework

Numerical experiments and visualizations connecting WALDO, Fraser's higher-order likelihood inference, and confidence distributions.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from frasian import posterior_params, confidence_interval, pvalue

# Conjugate normal model: Prior N(0, 1), Likelihood N(θ, 1), observed D=2
D, mu0, sigma, sigma0 = 2.0, 0.0, 1.0, 1.0
mu_n, sigma_n, w = posterior_params(D, mu0, sigma, sigma0)

# WALDO confidence interval (95%)
lower, upper = confidence_interval(D, mu0, sigma, sigma0, alpha=0.05)

# P-value at a specific θ
p = pvalue(theta=1.5, mu_n=mu_n, mu0=mu0, w=w, sigma=sigma)
```

## Generate Figures

```bash
python scripts/generate_all.py        # All figures
python scripts/generate_all.py --fast # Fast mode (fewer MC samples)
```

## Run Tests

```bash
pytest                # All tests
pytest -m "not slow"  # Skip slow tests
```

## Key Results

- **WALDO**: Bayesian-frequentist hybrid with exact 95% coverage for all θ
- **Optimal tilting**: η* can be negative at low conflict, yielding ~15% narrower CIs
- **Confidence distributions**: WALDO CD is a 50-50 Gaussian mixture with mode = μₙ

See `CLAUDE.md` for detailed documentation.
