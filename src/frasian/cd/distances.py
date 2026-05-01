"""Distance / divergence functions on confidence distributions.

All metrics here act on the *probability CDF* of a CD — the cumulative
integral of `pdf_values` — which is always monotone non-decreasing in
the framework's pdf-primary design. No rearrangement, no flattening:
W₁ and W₂ on a multimodal Dyn-WALDO CD compare the actual (possibly
bimodal) probability distribution against a Wald reference, since both
sides have valid CDFs.

Two test fixtures:
  - `wasserstein_2_gaussian(μ_a, σ_a, μ_b, σ_b)`: closed-form W₂ between
    two univariate Gaussians (Olkin–Pukelsheim 1982). Used to validate
    `wasserstein_2`.
  - `wasserstein_1_gaussian_shift(μ_a, μ_b)`: closed form for W₁ between
    two Gaussians of equal scale (= |μ_a − μ_b|). Used as a sanity
    check in `wasserstein_1`'s tests.

References
----------
Olkin, I. & Pukelsheim, F. (1982). "The distance between two random
vectors with given dispersion matrices." *Linear Algebra and its
Applications* 48: 257–263. — closed-form `W₂² = (μ_a−μ_b)² + (σ_a−σ_b)²`
for univariate Gaussians.

Villani, C. (2009). *Optimal Transport: Old and New*. Springer. — the
1D quantile-integral form `Wₚ^p = ∫₀¹ |F⁻¹(u) − G⁻¹(u)|^p du`.
"""

from __future__ import annotations

import numpy as np

from .grid import GridConfidenceDistribution


def _merge_grids(a: GridConfidenceDistribution,
                 b: GridConfidenceDistribution) -> np.ndarray:
    """Union of the two θ-grids, sorted and unique. Used for W₁."""
    merged = np.concatenate([a.theta_grid, b.theta_grid])
    merged.sort()
    return np.unique(merged)


def wasserstein_1(a: GridConfidenceDistribution,
                  b: GridConfidenceDistribution) -> float:
    """W₁(F_a, F_b) = ∫|F_a(θ) − F_b(θ)| dθ.

    Both CDFs are evaluated by interpolation onto the union of the two
    θ-grids (each CDF is monotone by construction since pdf ≥ 0), then
    integrated trapezoidally.
    """
    grid = _merge_grids(a, b)
    fa = a.cdf(grid)
    fb = b.cdf(grid)
    return float(np.trapezoid(np.abs(fa - fb), grid))


def wasserstein_2(a: GridConfidenceDistribution,
                  b: GridConfidenceDistribution,
                  *, n_quantile: int = 4001,
                  u_eps: float = 1e-5) -> float:
    """W₂(F_a, F_b) = sqrt(∫₀¹ (F_a^{-1}(u) − F_b^{-1}(u))² du).

    Quantile-integral form, valid for any pair of valid probability CDFs
    (i.e. monotone non-decreasing). The framework's pdf-primary CDFs
    always satisfy this — distance metrics act on the *real* probability
    distributions even when the underlying p-value is multimodal.

    Parameters
    ----------
    n_quantile : int
        Number of u-grid points for the trapezoidal integral.
    u_eps : float
        Small clip on the u-grid endpoints. The integrand
        `(Q_a(u) − Q_b(u))²` may diverge as u → 0, 1 when one CD's tails
        are heavier than the other (e.g. comparing `N(0, 1)` to
        `N(0, 2)`); trapezoidal integration is sensitive to that. The
        default `u_eps = 1e-5` matches the closed-form Olkin–Pukelsheim
        W₂ to within 1e-3 on a 4001-point Gaussian grid, well below the
        framework's tolerance for distance comparisons.
    """
    u = np.linspace(u_eps, 1.0 - u_eps, n_quantile)
    qa = a.quantile(u)
    qb = b.quantile(u)
    sq_diff = (qa - qb) ** 2
    integral = float(np.trapezoid(sq_diff, u))
    # We integrate over (u_eps, 1−u_eps) of length 1−2·u_eps; rescale to
    # the [0, 1] integral so the result is on the same scale as the
    # closed-form value.
    return float(np.sqrt(integral / (1.0 - 2.0 * u_eps)))


def total_variation(a: GridConfidenceDistribution,
                    b: GridConfidenceDistribution) -> float:
    """Kolmogorov-Smirnov distance: sup_θ |F_a(θ) − F_b(θ)|.

    Light add for completeness; not used as a headline metric in the
    framework's diagnostics. Useful as a sanity check that two CDs
    differ by at most the expected MC noise.
    """
    grid = _merge_grids(a, b)
    return float(np.max(np.abs(a.cdf(grid) - b.cdf(grid))))


# ----- Closed-form references for testing -----

def wasserstein_2_gaussian(mu_a: float, sigma_a: float,
                            mu_b: float, sigma_b: float) -> float:
    """Closed-form W₂ between two univariate Gaussians.

    `W₂² = (μ_a − μ_b)² + (σ_a − σ_b)²` (Olkin–Pukelsheim 1982).
    """
    return float(np.sqrt((mu_a - mu_b) ** 2 + (sigma_a - sigma_b) ** 2))


def wasserstein_1_gaussian_shift(mu_a: float, mu_b: float) -> float:
    """Closed-form W₁ between two Gaussians of identical scale.

    For Gaussians of equal σ, W₁ = |μ_a − μ_b| (the optimal coupling
    is the deterministic shift; the cost is the absolute mean shift).
    """
    return float(abs(mu_a - mu_b))
