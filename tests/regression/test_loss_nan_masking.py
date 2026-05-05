"""Regression: training losses mask NaN/Inf samples instead of propagating.

When the torch p-value formula drifts numerically — e.g. at extreme
`(w, η)` combinations or for future non-Normal-Normal models — a
single contaminated sample can poison the entire batch's gradient.
The losses now mask non-finite per-sample contributions and average
over the valid samples only.

If the entire batch is invalid, the loss raises `RuntimeError` so the
training loop fails loudly instead of silently producing NaN
gradients.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from frasian.learned.training.losses import (
    cd_variance_loss,
    integrated_pvalue_loss,
    static_width_loss,
)


def _make_batch(B: int = 4, N: int = 21, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(B, N, generator=g, dtype=torch.float64)
    p = torch.sigmoid(raw)
    theta = torch.linspace(-2.0, 2.0, N, dtype=torch.float64)
    return p, theta


def _inject_nan(p: torch.Tensor, sample_idx: int) -> torch.Tensor:
    """Replace one row of `p` with NaN entries."""
    p_bad = p.clone()
    p_bad[sample_idx] = float("nan")
    return p_bad


@pytest.mark.L1
def test_integrated_p_masks_nan_samples():
    """A single NaN row should not poison the loss; mean is over valid rows."""
    p, theta = _make_batch(B=4)
    loss_clean = float(integrated_pvalue_loss(p, theta).item())

    p_one_nan = _inject_nan(p, sample_idx=0)
    loss_masked = float(integrated_pvalue_loss(p_one_nan, theta).item())
    # Hand-compute the mean over rows 1..3:
    widths = torch.trapezoid(p[1:], theta, dim=-1)
    expected = float(widths.mean().item())
    assert (
        abs(loss_masked - expected) < 1e-12
    ), f"masked loss {loss_masked} ≠ mean of valid rows {expected}"
    # Non-finite per-sample loss is dropped (loss should differ from
    # full-batch mean if the NaN row would have contributed).
    assert loss_masked != loss_clean


@pytest.mark.L1
def test_cd_variance_masks_nan_samples():
    p, theta = _make_batch(B=4, N=31)
    p_bad = _inject_nan(p, sample_idx=2)
    loss = cd_variance_loss(p_bad, theta)
    assert torch.isfinite(loss), "cd_variance loss is NaN/Inf with masked NaN row"


@pytest.mark.L1
def test_static_width_masks_nan_samples():
    p, theta = _make_batch(B=4)
    p_bad = _inject_nan(p, sample_idx=1)
    loss = static_width_loss(p_bad, theta, alpha=0.1)
    assert torch.isfinite(loss)


@pytest.mark.L1
def test_all_nan_raises():
    """If the entire batch is non-finite, the loss raises a helpful error."""
    p, theta = _make_batch(B=3)
    p_all_nan = torch.full_like(p, float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        integrated_pvalue_loss(p_all_nan, theta)


@pytest.mark.L1
def test_masked_mean_preserves_gradient():
    """Gradient through the loss flows through the valid samples.

    Construct `p_pretend` so that row 0 is a constant NaN tensor (no
    autograd source) and rows 1..3 track `p`. The masked mean should
    yield a finite loss whose gradient propagates only through rows
    1..3, leaving row 0 ungraded.
    """
    p, theta = _make_batch(B=3, N=21)  # leaf with grad
    p.requires_grad_(True)

    # Build a row of constant NaN (no grad source).
    nan_row = torch.full((1, p.shape[-1]), float("nan"), dtype=p.dtype)
    p_pretend = torch.cat([nan_row, p], dim=0)  # (4, N), row 0 = NaN

    loss = integrated_pvalue_loss(p_pretend, theta)
    assert torch.isfinite(loss), f"masked loss is not finite: {loss}"
    loss.backward()
    # Gradient of `p` (the rows 1..3 in p_pretend) should be finite.
    assert torch.all(torch.isfinite(p.grad)), f"gradient on valid rows contains NaN: {p.grad}"
