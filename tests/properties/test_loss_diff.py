"""Differentiability: gradcheck on integrated_p / cd_variance / static_width losses."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from frasian.learned.training.losses import (cd_variance_loss,
                                                integrated_pvalue_loss,
                                                static_width_loss)


def _random_p_theta(B: int = 2, N: int = 17, seed: int = 0):
    """Generate a random p-value tensor (B, N) ∈ [0, 1] with grad."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(B, N, generator=g, dtype=torch.float64)
    # Pass through sigmoid to get values in (0, 1) with gradient.
    p = torch.sigmoid(raw)
    p.requires_grad_(True)
    return p


@pytest.mark.L1
@pytest.mark.properties
def test_integrated_pvalue_loss_gradcheck():
    """∫p dθ is differentiable w.r.t. p."""
    p = _random_p_theta()
    theta = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64)

    def f(pp: torch.Tensor) -> torch.Tensor:
        return integrated_pvalue_loss(pp, theta)

    assert torch.autograd.gradcheck(f, (p,), eps=1e-6, atol=1e-4)


@pytest.mark.L1
@pytest.mark.properties
def test_cd_variance_loss_gradcheck():
    """CD variance loss is differentiable w.r.t. p."""
    p = _random_p_theta(N=21)
    theta = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64)

    def f(pp: torch.Tensor) -> torch.Tensor:
        return cd_variance_loss(pp, theta)

    assert torch.autograd.gradcheck(f, (p,), eps=1e-6, atol=1e-3)


@pytest.mark.L1
@pytest.mark.properties
def test_static_width_loss_gradcheck():
    """Sigmoid-relaxed static-width loss is differentiable w.r.t. p."""
    p = _random_p_theta(N=21)
    theta = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64)

    def f(pp: torch.Tensor) -> torch.Tensor:
        return static_width_loss(pp, theta, alpha=0.1, sharpness=10.0)

    assert torch.autograd.gradcheck(f, (p,), eps=1e-6, atol=1e-4)


@pytest.mark.L1
def test_static_width_loss_invalid_alpha():
    """alpha out of (0, 1) raises."""
    p = _random_p_theta()
    theta = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64)
    with pytest.raises(ValueError):
        static_width_loss(p, theta, alpha=0.0)
    with pytest.raises(ValueError):
        static_width_loss(p, theta, alpha=1.0)
