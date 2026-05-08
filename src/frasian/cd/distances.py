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
from scipy.special import ndtr as _ndtr

from .. import _jax_setup as _x64  # noqa: F401  — ensure float64 active
from .grid import GridConfidenceDistribution

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def _merge_grids(a: GridConfidenceDistribution, b: GridConfidenceDistribution) -> np.ndarray:
    """Union of the two θ-grids, sorted and unique. Used for W₁."""
    merged = np.concatenate([a.theta_grid, b.theta_grid])
    merged.sort()
    return np.unique(merged)


def wasserstein_1(a: GridConfidenceDistribution, b: GridConfidenceDistribution) -> float:
    """W₁(F_a, F_b) = ∫|F_a(θ) − F_b(θ)| dθ.

    Both CDFs are evaluated by interpolation onto the union of the two
    θ-grids (each CDF is monotone by construction since pdf ≥ 0), then
    integrated trapezoidally on the merged grid.

    Why CDF-form rather than Gauss–Hermite (parallel to `wasserstein_2`):
    the equivalent quantile-axis integrand `|Q_a(u) − Q_b(u)|` carries a
    kink wherever the two quantile functions cross (always present when
    `σ_a ≠ σ_b`). Gauss–Hermite quadrature suffers from polynomial-fitting
    errors at the kink — empirically ~5e-3 on a (μ=0, σ_a=1, σ_b=2) pair
    even at n_quad = 64. The CDF integrand `|F_a − F_b|` on the θ-axis
    is smooth in pieces (each F is a smooth Gaussian-like CDF), so
    trapezoidal integration on the framework's 4001-point grid hits the
    closed form `W₁ = |σ_a − σ_b|·√(2/π)` (zero-mean Gaussian σ-mismatch)
    to ~1e-7. CDF-form is the right method here.
    """
    grid = _merge_grids(a, b)
    fa = a.cdf(grid)
    fb = b.cdf(grid)
    return float(np.trapezoid(np.abs(fa - fb), grid))


def wasserstein_2(
    a: GridConfidenceDistribution, b: GridConfidenceDistribution, *, n_quad: int = 64
) -> float:
    """W₂(F_a, F_b) = sqrt(∫₀¹ (F_a^{-1}(u) − F_b^{-1}(u))² du).

    Computed via the change of variables `z = Φ⁻¹(u)`:
        W₂² = ∫_{-∞}^{∞} (Q_a(Φ(z)) − Q_b(Φ(z)))² φ(z) dz
    and integrated by Gauss–Hermite quadrature with `n_quad` nodes.

    Why the change of variables: on the raw u-grid, the integrand
    `(Q_a(u) − Q_b(u))²` blows up at u → 0, 1 whenever the two CDs
    have different tail behaviour (e.g. σ-mismatched Gaussians), and
    trapezoidal integration is sensitive to that. Mapping to the
    z-axis weights tail contributions by φ(z) — the natural weight
    inherited from the standard normal — and Gauss–Hermite nodes are
    placed precisely where the integrand is largest. For Gaussian
    pairs this evaluates the Olkin–Pukelsheim closed form
    `W₂² = (μ_a − μ_b)² + (σ_a − σ_b)²` to machine precision with
    n_quad ≈ 32; n_quad = 64 leaves headroom for non-Gaussian CDs
    (bimodal Dyn-WALDO etc.).

    Both CDs must have monotone non-decreasing cdf — guaranteed by
    the framework's pdf-primary design. No rearrangement, no
    boundary clipping, no rescaling: the integral is exact in the
    quadrature limit.

    Parameters
    ----------
    n_quad : int
        Number of Gauss–Hermite nodes. Default 64 is comfortably
        more than needed for Gaussian agreement at 1e-12; trim to 32
        if profiling shows this is the bottleneck.
    """
    # scipy: numpy.polynomial.hermite_e.hermegauss has no JAX equivalent;
    # the nodes/weights are computed once per call and not on a
    # differentiable path.
    from numpy.polynomial.hermite_e import hermegauss

    # `hermegauss(n)` returns nodes z_i and weights w_i such that
    # `∫ f(z) e^{-z²/2} dz ≈ Σ w_i f(z_i)`. Dividing by √(2π) converts
    # to expectation under the standard normal density φ(z).
    z, w = hermegauss(n_quad)
    # Probabilist's normalisation: divide weights by √(2π).
    w = w / np.sqrt(2.0 * np.pi)
    u = _ndtr(z)
    qa = a.quantile(u)
    qb = b.quantile(u)
    sq_diff = (qa - qb) ** 2
    integral = float(np.sum(w * sq_diff))
    return float(np.sqrt(max(integral, 0.0)))


def total_variation(a: GridConfidenceDistribution, b: GridConfidenceDistribution) -> float:
    """Kolmogorov-Smirnov distance: sup_θ |F_a(θ) − F_b(θ)|.

    Light add for completeness; not used as a headline metric in the
    framework's diagnostics. Useful as a sanity check that two CDs
    differ by at most the expected MC noise.
    """
    grid = _merge_grids(a, b)
    return float(np.max(np.abs(a.cdf(grid) - b.cdf(grid))))


# ----- Closed-form references for testing -----


def wasserstein_2_gaussian(mu_a: float, sigma_a: float, mu_b: float, sigma_b: float) -> float:
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


def wasserstein_1_gaussian_zero_mean_scale(sigma_a: float, sigma_b: float) -> float:
    """Closed-form W₁ between two zero-mean Gaussians of differing scale.

    Derived from the quantile-axis form:

        W₁ = ∫₀¹ |Q_a(u) − Q_b(u)| du
           = ∫₀¹ |σ_a − σ_b|·|Φ⁻¹(u)| du
           = |σ_a − σ_b| · E_{Z~N(0,1)}[|Z|]
           = |σ_a − σ_b| · √(2/π).

    Used as a tight test fixture for the σ-mismatched-Gaussian case
    where the equal-scale closed form `|μ_a − μ_b|` does not apply.
    """
    return float(abs(sigma_a - sigma_b) * np.sqrt(2.0 / np.pi))
