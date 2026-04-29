"""
Tests for coverage properties (Section 7).

Key coverage results from the document:
- WALDO maintains ~95% coverage for all theta values
- Wald/likelihood CI also maintains 95% coverage
- Posterior credible intervals can have catastrophically wrong coverage
  when theta != mu0 (e.g., 40% at theta=3, 1% at theta=5)
"""

import pytest
import numpy as np
from scipy import stats

from frasian.core import posterior_params
from frasian.waldo import (
    confidence_interval,
    wald_ci,
    posterior_ci,
    pvalue,
)

from conftest import (
    TestConfig,
    ModelParams,
    get_model_from_w,
    simulate_data,
)


def compute_coverage(
    theta_true: float,
    mu0: float,
    sigma: float,
    sigma0: float,
    n_replicates: int,
    rng: np.random.Generator,
    method: str = 'waldo',
    alpha: float = 0.05,
) -> float:
    """
    Compute empirical coverage rate.

    Parameters
    ----------
    theta_true : float
        True parameter value
    mu0 : float
        Prior mean
    sigma : float
        Likelihood standard deviation
    sigma0 : float
        Prior standard deviation
    n_replicates : int
        Number of simulation replicates
    rng : Generator
        Random number generator
    method : str
        'waldo', 'wald', or 'posterior'
    alpha : float
        Significance level

    Returns
    -------
    coverage : float
        Fraction of times theta_true was in the CI
    """
    D_samples = rng.normal(theta_true, sigma, n_replicates)
    covered = 0

    for D in D_samples:
        if method == 'waldo':
            lower, upper = confidence_interval(D, mu0, sigma, sigma0, alpha)
        elif method == 'wald':
            lower, upper = wald_ci(D, sigma, alpha)
        elif method == 'posterior':
            lower, upper = posterior_ci(D, mu0, sigma, sigma0, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        if lower <= theta_true <= upper:
            covered += 1

    return covered / n_replicates


@pytest.mark.tier3
class TestWaldoCoverage:
    """Tests for WALDO coverage properties."""

    @pytest.mark.slow
    def test_waldo_coverage_at_prior_mean(self, balanced_model, config, rng):
        """Test WALDO achieves ~95% coverage at theta = mu0."""
        theta_true = balanced_model.mu0
        model = balanced_model

        coverage = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='waldo'
        )

        assert abs(coverage - 0.95) < config.coverage_tol, (
            f"WALDO coverage at theta=mu0: {coverage:.3f}, expected ~0.95"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("theta_true", [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    def test_waldo_coverage_at_various_theta(self, balanced_model, config, rng, theta_true):
        """Test WALDO maintains ~95% coverage at various theta values."""
        model = balanced_model

        coverage = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='waldo'
        )

        assert abs(coverage - 0.95) < config.coverage_tol, (
            f"WALDO coverage at theta={theta_true}: {coverage:.3f}, expected ~0.95"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("w", [0.2, 0.5, 0.8])
    def test_waldo_coverage_at_various_w(self, config, rng, w):
        """Test WALDO maintains coverage at various weight values."""
        theta_true = 2.0
        model = get_model_from_w(w)

        coverage = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='waldo'
        )

        assert abs(coverage - 0.95) < config.coverage_tol, (
            f"WALDO coverage at w={w}, theta={theta_true}: {coverage:.3f}"
        )


@pytest.mark.tier3
class TestWaldCoverage:
    """Tests for Wald/likelihood CI coverage."""

    @pytest.mark.slow
    @pytest.mark.parametrize("theta_true", [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    def test_wald_coverage_at_various_theta(self, balanced_model, config, rng, theta_true):
        """Test Wald CI maintains ~95% coverage at various theta values."""
        model = balanced_model

        coverage = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='wald'
        )

        assert abs(coverage - 0.95) < config.coverage_tol, (
            f"Wald coverage at theta={theta_true}: {coverage:.3f}, expected ~0.95"
        )


@pytest.mark.tier3
class TestPosteriorUndercoverage:
    """Tests demonstrating posterior CI undercoverage when theta != mu0."""

    @pytest.mark.slow
    def test_posterior_correct_coverage_at_mu0(self, balanced_model, config, rng):
        """Test posterior CI has correct (or overcoverage) at theta = mu0."""
        theta_true = balanced_model.mu0
        model = balanced_model

        coverage = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='posterior'
        )

        # At theta = mu0, posterior should have >= nominal coverage
        assert coverage >= 0.95 - config.coverage_tol, (
            f"Posterior coverage at theta=mu0: {coverage:.3f}, expected >=0.95"
        )

    @pytest.mark.slow
    def test_posterior_undercoverage_away_from_mu0(self, balanced_model, config, rng):
        """Test posterior CI has undercoverage when theta is far from mu0."""
        theta_true = 3.0  # 3 sigma away from mu0
        model = balanced_model

        coverage = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='posterior'
        )

        # Coverage should be well below 95% - document shows ~40% at theta=3
        assert coverage < 0.70, (
            f"Posterior coverage at theta={theta_true}: {coverage:.3f}, "
            f"expected significant undercoverage"
        )

    @pytest.mark.slow
    def test_posterior_severe_undercoverage_far_from_mu0(self, balanced_model, config, rng):
        """Test posterior CI has severe undercoverage at theta = 5."""
        theta_true = 5.0  # 5 sigma away from mu0
        model = balanced_model

        coverage = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='posterior'
        )

        # Document shows ~1.2% coverage at theta=5
        assert coverage < 0.20, (
            f"Posterior coverage at theta={theta_true}: {coverage:.3f}, "
            f"expected severe undercoverage (<20%)"
        )

    @pytest.mark.slow
    def test_posterior_coverage_decreases_with_distance(self, balanced_model, config, rng):
        """Test that posterior coverage decreases as theta moves away from mu0."""
        model = balanced_model
        thetas = [0.0, 1.0, 2.0, 3.0]

        coverages = []
        for theta in thetas:
            cov = compute_coverage(
                theta, model.mu0, model.sigma, model.sigma0,
                n_replicates=config.n_coverage, rng=rng, method='posterior'
            )
            coverages.append(cov)

        # Coverage should generally decrease (may not be monotonic due to noise)
        assert coverages[0] > coverages[-1], (
            f"Coverage should decrease with distance from mu0: {coverages}"
        )


@pytest.mark.tier3
class TestCoverageComparison:
    """Tests comparing coverage across methods."""

    @pytest.mark.slow
    def test_waldo_vs_posterior_at_prior_mean(self, balanced_model, config, rng):
        """Compare WALDO and posterior coverage at theta = mu0."""
        theta_true = balanced_model.mu0
        model = balanced_model

        waldo_cov = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='waldo'
        )
        posterior_cov = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='posterior'
        )

        # Both should be close to nominal at mu0
        assert abs(waldo_cov - 0.95) < config.coverage_tol
        # Posterior may have overcoverage at mu0
        assert posterior_cov >= 0.95 - config.coverage_tol

    @pytest.mark.slow
    def test_waldo_outperforms_posterior_away_from_mu0(self, balanced_model, config, rng):
        """Test WALDO maintains coverage where posterior fails."""
        theta_true = 3.0
        model = balanced_model

        waldo_cov = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='waldo'
        )
        posterior_cov = compute_coverage(
            theta_true, model.mu0, model.sigma, model.sigma0,
            n_replicates=config.n_coverage, rng=rng, method='posterior'
        )

        # WALDO should maintain coverage
        assert abs(waldo_cov - 0.95) < config.coverage_tol
        # Posterior should undercover significantly
        assert posterior_cov < 0.70
        # WALDO should be much better
        assert waldo_cov > posterior_cov + 0.20


@pytest.mark.tier3
class TestCoverageTable:
    """Reproduce coverage table from Section 7."""

    @pytest.mark.slow
    def test_reproduce_coverage_table(self, config, rng):
        """Reproduce the full coverage table from the document.

        Expected results (σ = σ₀ = 1, μ₀ = 0, α = 0.05):
        | θ_true | Wald  | Posterior | WALDO |
        |--------|-------|-----------|-------|
        | -3     | 95.0% | 40.4%     | 95.1% |
        | -1     | 94.8% | 96.1%     | 94.9% |
        |  0     | 95.0% | 99.3%     | 95.0% |
        |  1     | 95.0% | 96.3%     | 95.2% |
        |  3     | 95.6% | 40.1%     | 95.3% |
        |  5     | 95.1% |  1.2%     | 95.1% |
        """
        model = get_model_from_w(0.5)  # sigma = sigma0 = 1 gives w = 0.5
        thetas = [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0]

        results = {}
        for theta in thetas:
            results[theta] = {}
            for method in ['wald', 'posterior', 'waldo']:
                cov = compute_coverage(
                    theta, model.mu0, model.sigma, model.sigma0,
                    n_replicates=config.n_coverage, rng=rng, method=method
                )
                results[theta][method] = cov

        # Verify key results
        # WALDO should have ~95% everywhere
        for theta in thetas:
            assert abs(results[theta]['waldo'] - 0.95) < 2 * config.coverage_tol, (
                f"WALDO coverage at {theta}: {results[theta]['waldo']:.3f}"
            )

        # Wald should have ~95% everywhere
        for theta in thetas:
            assert abs(results[theta]['wald'] - 0.95) < 2 * config.coverage_tol, (
                f"Wald coverage at {theta}: {results[theta]['wald']:.3f}"
            )

        # Posterior should undercover at |theta| >= 3
        assert results[-3.0]['posterior'] < 0.60
        assert results[3.0]['posterior'] < 0.60
        assert results[5.0]['posterior'] < 0.10
