"""Regression: training losses mask NaN/Inf samples instead of propagating.

When the JAX p-value formula drifts numerically — e.g. at extreme
`(w, η)` combinations or for future non-Normal-Normal models — a
single contaminated sample can poison the entire batch's gradient.
The losses mask non-finite per-sample contributions and average
over the valid samples only.

Phase F port commit 2 (JAX kernels): if every sample in the batch is
non-finite, ``_masked_mean`` returns NaN (a sentinel that propagates
without breaking ``jax.jit`` tracing). The training loop is responsible
for raising ``RuntimeError`` on a non-finite scalar loss; that
boundary check moves out of the loss kernel and into the orchestrator
in commit 3.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from frasian.learned.training.losses import (
    cd_variance_loss,
    integrated_pvalue_loss,
    static_width_loss,
)


def _make_batch(B: int = 4, N: int = 21, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((B, N)).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-raw))
    p = jnp.asarray(p)
    theta = jnp.linspace(-2.0, 2.0, N)
    return p, theta


def _inject_nan(p: jnp.ndarray, sample_idx: int) -> jnp.ndarray:
    """Replace one row of `p` with NaN entries."""
    return p.at[sample_idx].set(jnp.nan)


@pytest.mark.L1
def test_integrated_p_masks_nan_samples():
    """A single NaN row should not poison the loss; mean is over valid rows."""
    p, theta = _make_batch(B=4)
    loss_clean = float(integrated_pvalue_loss(p, theta))

    p_one_nan = _inject_nan(p, sample_idx=0)
    loss_masked = float(integrated_pvalue_loss(p_one_nan, theta))
    # Hand-compute the mean over rows 1..3:
    widths = jnp.trapezoid(p[1:], theta, axis=-1)
    expected = float(widths.mean())
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
    assert bool(jnp.isfinite(loss)), "cd_variance loss is NaN/Inf with masked NaN row"


@pytest.mark.L1
def test_static_width_masks_nan_samples():
    p, theta = _make_batch(B=4)
    p_bad = _inject_nan(p, sample_idx=1)
    loss = static_width_loss(p_bad, theta, alpha=0.1)
    assert bool(jnp.isfinite(loss))


@pytest.mark.L1
def test_all_nan_returns_sentinel():
    """If the entire batch is non-finite, the loss returns NaN.

    The legacy torch path raised ``RuntimeError`` from inside the
    loss; the JAX port returns NaN and lets the caller (training
    loop in commit 3) decide whether to raise. Keeping the loss
    jittable matters more than the inline raise.
    """
    p, theta = _make_batch(B=3)
    p_all_nan = jnp.full_like(p, jnp.nan)
    out = integrated_pvalue_loss(p_all_nan, theta)
    assert not bool(jnp.isfinite(out))


@pytest.mark.L1
def test_masked_mean_preserves_gradient():
    """Gradient through the loss flows through the valid samples.

    Construct `p_pretend` so that row 0 is a constant NaN tensor (no
    autograd source) and rows 1..3 track `p`. The masked mean should
    yield a finite loss whose gradient propagates only through rows
    1..3, leaving row 0 ungraded.
    """
    import jax

    p, theta = _make_batch(B=3, N=21)
    nan_row = jnp.full((1, p.shape[-1]), jnp.nan)

    def loss_fn(p_in):
        p_pretend = jnp.concatenate([nan_row, p_in], axis=0)  # (4, N), row 0 = NaN
        return integrated_pvalue_loss(p_pretend, theta)

    loss_val = loss_fn(p)
    assert bool(jnp.isfinite(loss_val)), f"masked loss is not finite: {loss_val}"
    g = jax.grad(loss_fn)(p)
    assert bool(jnp.isfinite(g).all()), f"gradient on valid rows contains NaN: {g}"
