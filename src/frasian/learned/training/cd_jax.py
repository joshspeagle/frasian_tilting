"""JAX port of `frasian.cd.from_pvalue` — Schweder–Hjort CD pdf only.

Reference numpy implementation: `src/frasian/cd/from_pvalue.py`.

The full numpy `build_cd_from_pvalue` includes a `signed_confidence`
curve based on `np.argmax(pvalues)`. ``argmax`` is non-differentiable
— we **skip ``signed_confidence`` during training** and keep only the
``pdf`` path.

Density formula (averaged-one-sided-difference, kink-robust):

    forward[i]  = |p[i+1] - p[i]| / (theta[i+1] - theta[i])     (i < N-1)
    backward[i] = forward[i-1]                                  (i > 0)
    abs_dp = 0.5 (forward + backward)
    pdf_unnorm = 0.5 * abs_dp
    pdf = pdf_unnorm / Z,  Z = ∫ pdf_unnorm dtheta

The averaged-one-sided-diff is what makes the density correct at
kink points (e.g. theta = D for Wald, theta = mu_n for WALDO) where
central differences cancel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active

_FORCE_X64 = _x64  # keep static-analysis from stripping the import


def cd_density_jax(
    p_theta: jax.Array,
    theta_grid: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Schweder-Hjort CD pdf from a p-value curve.

    Accepts either a shared 1D ``theta_grid`` (``(N,)``, used for all
    batch rows) or a per-sample 2D grid (``(B, N)``, one row per batch
    element — the training pipeline uses this since each sample has
    its own ``D ± search_mult * sigma`` window).

    Parameters
    ----------
    p_theta : (B, N) jax.Array
        p-value evaluated on ``theta_grid`` for each batch element.
    theta_grid : (N,) or (B, N) jax.Array
        Strictly-increasing theta grid. If 1D, broadcast to all rows;
        if 2D, must match ``p_theta.shape``.
    eps : float
        Floor for the normalisation constant ``Z`` to avoid div-by-zero
        when p is constant on the grid.

    Returns
    -------
    pdf : (B, N) jax.Array
        Normalised pdf integrating to 1 along ``theta_grid`` row-wise.
    """
    if p_theta.ndim != 2:
        raise ValueError(f"p_theta must be (B, N); got shape {tuple(p_theta.shape)}")
    if theta_grid.ndim == 1:
        if theta_grid.shape[0] != p_theta.shape[1]:
            raise ValueError(
                f"theta_grid (1D) must have N={p_theta.shape[1]} elements; "
                f"got shape {tuple(theta_grid.shape)}"
            )
        # Forward differences along the shared grid.
        dtheta = (theta_grid[1:] - theta_grid[:-1])[None, :]  # (1, N-1)
    elif theta_grid.ndim == 2:
        if theta_grid.shape != p_theta.shape:
            raise ValueError(
                f"theta_grid (2D) must match p_theta shape "
                f"{tuple(p_theta.shape)}; got {tuple(theta_grid.shape)}"
            )
        dtheta = theta_grid[..., 1:] - theta_grid[..., :-1]  # (B, N-1)
    else:
        raise ValueError(
            f"theta_grid must be 1D or 2D; got shape {tuple(theta_grid.shape)}"
        )

    dp = jnp.abs(p_theta[..., 1:] - p_theta[..., :-1])  # (B, N-1)
    forward_inner = dp / dtheta  # (B, N-1)
    forward = jnp.concatenate([forward_inner, forward_inner[..., -1:]], axis=-1)
    backward = jnp.concatenate([forward_inner[..., 0:1], forward_inner], axis=-1)
    abs_dp_dtheta = 0.5 * (forward + backward)
    pdf_unnorm = 0.5 * abs_dp_dtheta  # (B, N)

    # jnp.trapezoid requires the same shape for x and y, so broadcast
    # theta_grid to match pdf_unnorm if it's 1D.
    if theta_grid.ndim == 1:
        x_grid = jnp.broadcast_to(theta_grid, pdf_unnorm.shape)
    else:
        x_grid = theta_grid
    Z = jnp.trapezoid(pdf_unnorm, x_grid, axis=-1)  # (B,)
    # Audit P1 J.1: emit NaN pdf when Z is below the floor (essentially
    # constant p — no CD support on this grid). Pre-fix the code
    # `jnp.maximum(Z, eps)` returned a finite-but-huge pdf
    # (`pdf_unnorm / 1e-12`), which silently contaminated downstream
    # `cd_variance_loss` (var ~ 1e12 × spread²) without tripping the
    # `_masked_mean` non-finite filter. Now those samples become NaN
    # and `_masked_mean` masks them out. Callers who deliberately
    # want the saturating behaviour can pre-floor `Z` themselves.
    z_too_small = Z < eps
    Z_safe = jnp.where(z_too_small, jnp.asarray(1.0, Z.dtype), Z)
    pdf_normalised = pdf_unnorm / Z_safe[..., None]
    return jnp.where(
        z_too_small[..., None], jnp.full_like(pdf_normalised, jnp.nan), pdf_normalised
    )
