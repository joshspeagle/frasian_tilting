"""Regression: β-annealing on static_width_loss reduces relaxation bias.

Closes audit findings 1.2-NN2 and 1.4-S2.

The loss
    |C_α| ≈ ∫_θ σ_β( p_dyn(θ) − α ) dθ
relaxes the discrete indicator ``1{p_dyn ≥ α}`` so the width is
differentiable. As ``β → ∞`` the relaxation converges. The audit pins
the empirical bias schedule:

    β=50,  α=0.05 → +110 % bias  (relaxed integral catches tail mass)
    β=200, α=0.05 → +0.4 % bias
    β=500, α=0.05 → +0.1 % bias

This test pins ``relaxed_bias(α, β=500) < 0.01`` at α ∈ {0.05, 0.10,
0.20, 0.50}, validated against a closed-form Wald p-value over a wide
θ grid. It also pins the warm-start endpoint (``β=50``) at the same α
range so we know the lower endpoint is *not* sharp enough — the
anneal is doing real work.

Companion test: ``beta_schedule(epoch, n_epochs)`` returns the
expected linear ramp (``50 → 500`` over the first 50 % of epochs).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.learned.training._losses_compose import (
    BETA_MAX,
    BETA_MIN,
    BETA_WARMUP_FRAC,
    beta_schedule,
)
from frasian.learned.training.losses import static_width_loss


def _wald_pvalue_grid(theta_grid: np.ndarray, D: float, sigma: float) -> np.ndarray:
    """Closed-form Wald p-value: p(θ) = 2(1 − Φ(|D − θ|/σ))."""
    z = np.abs(D - theta_grid) / sigma
    return 2.0 * (1.0 - 0.5 * (1.0 + np.array([math.erf(zi / math.sqrt(2.0)) for zi in z])))


def _true_width(p: np.ndarray, theta_grid: np.ndarray, alpha: float) -> float:
    """True ``|C_α|`` = ∫_θ 1{p(θ) ≥ α} dθ via trapezoid."""
    indicator = (p >= alpha).astype(np.float64)
    return float(np.trapezoid(indicator, theta_grid))


def _relaxed_width_torch(
    p: np.ndarray, theta_grid: np.ndarray, alpha: float, beta: float
) -> float:
    """``static_width_loss`` evaluated on a single-sample (1, N) batch."""
    p_t = torch.as_tensor(p, dtype=torch.float64).unsqueeze(0)
    theta_t = torch.as_tensor(theta_grid, dtype=torch.float64)
    with torch.no_grad():
        loss = static_width_loss(p_t, theta_t, alpha=alpha, sharpness=beta)
    return float(loss.item())


@pytest.mark.L2
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 0.50])
def test_relaxed_bias_below_one_percent_at_beta_500(alpha):
    """At β=500 the relaxation bias is < 1 % at α ∈ {0.05, 0.10, 0.20, 0.50}.

    Closed-form Wald p-value, wide θ grid, single D. The bias is
    ``|relaxed − true| / true``; a tight grid is required because the
    indicator's edge cells dominate.
    """
    sigma = 1.0
    D = 0.3
    theta_grid = np.linspace(-8.0, 8.0, 4001, dtype=np.float64)
    p = _wald_pvalue_grid(theta_grid, D, sigma)

    truth = _true_width(p, theta_grid, alpha)
    relaxed = _relaxed_width_torch(p, theta_grid, alpha, beta=500.0)
    bias = abs(relaxed - truth) / max(truth, 1e-12)
    assert bias < 0.01, (
        f"β=500 relaxed-width bias {bias:.4f} ≥ 0.01 at α={alpha}; "
        f"truth={truth:.4f}, relaxed={relaxed:.4f}."
    )


@pytest.mark.L2
def test_beta_50_is_meaningfully_more_biased_than_beta_500():
    """β=50 must over-count CI width relative to β=500 — the anneal does work."""
    sigma = 1.0
    D = 0.0
    alpha = 0.05
    theta_grid = np.linspace(-8.0, 8.0, 4001, dtype=np.float64)
    p = _wald_pvalue_grid(theta_grid, D, sigma)

    truth = _true_width(p, theta_grid, alpha)
    rel_50 = _relaxed_width_torch(p, theta_grid, alpha, beta=50.0)
    rel_500 = _relaxed_width_torch(p, theta_grid, alpha, beta=500.0)

    bias_50 = abs(rel_50 - truth) / truth
    bias_500 = abs(rel_500 - truth) / truth
    # The anneal only saves us if the lower endpoint is ≥ 10× more biased.
    assert bias_50 > 10.0 * bias_500, (
        f"β=50 bias {bias_50:.3f} not ≥ 10× β=500 bias {bias_500:.4f}; "
        f"the anneal is not delivering a real bias reduction."
    )


def test_beta_schedule_endpoints_and_ramp():
    """``beta_schedule`` is a linear ramp from ``BETA_MIN`` to ``BETA_MAX``.

    Pure-numpy unit test (no torch); doubles as a contract for the
    schedule shape so a future refactor can't change endpoints
    without updating this test.
    """
    n_epochs = 100
    # Endpoints.
    assert beta_schedule(0, n_epochs) == pytest.approx(BETA_MIN)
    # At warmup_frac · n_epochs we hit BETA_MAX.
    warmup_epoch = int(BETA_WARMUP_FRAC * n_epochs)
    assert beta_schedule(warmup_epoch, n_epochs) == pytest.approx(BETA_MAX)
    # Past warmup: still BETA_MAX.
    assert beta_schedule(n_epochs, n_epochs) == pytest.approx(BETA_MAX)

    # Midpoint of the ramp: half the gap.
    mid = warmup_epoch // 2
    midpoint = beta_schedule(mid, n_epochs)
    expected = BETA_MIN + 0.5 * (BETA_MAX - BETA_MIN)
    assert midpoint == pytest.approx(expected, rel=0.05)

    # warmup_frac=0 short-circuit: returns BETA_MAX immediately.
    assert beta_schedule(0, n_epochs, warmup_frac=0.0) == pytest.approx(BETA_MAX)

    # n_epochs=0 corner: returns BETA_MAX (defensive).
    assert beta_schedule(0, 0) == pytest.approx(BETA_MAX)
