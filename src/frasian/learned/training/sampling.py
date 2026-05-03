"""Training distribution `π` over `(w, θ_true)`, plus LHS sampling.

The objective averages over a chosen distribution:

    ℒ(φ) = E_{(w, θ_true) ~ π} E_{D | θ_true, w} [ Φ(D; η_φ) ]

The default `π` for the Normal-Normal sandbox:

- `θ_true ~ Uniform(μ₀ - 10σ, μ₀ + 10σ)` — covers up to |Δ| ≈ 5 at
  w=0.5, plus the asymptotic regime.
- `w ~ Uniform(0.05, 0.95)` — avoids w-boundary singularities.
- `D | θ_true, w ~ N(θ_true, σ²)` (`σ = 1` fixed; the framework is
  scale-invariant).

The LHS sampler takes the (w, θ_true) marginals and draws an evenly
spread set of points; for each point `n_mc` D draws are taken to
form a mini-batch.

`TrainingDistribution` is a dataclass with classmethods for ready-made
defaults; users can construct custom distributions by passing
explicit ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import qmc


@dataclass(frozen=True)
class TrainingDistribution:
    """Configurable π over (w, θ_true) for the Normal-Normal sandbox.

    Attributes
    ----------
    w_range
        `(w_min, w_max)` — uniform sampling interval.
    theta_true_half_width
        Half-width of `θ_true ~ Uniform(μ₀ - X·σ, μ₀ + X·σ)`.
        Default 10 covers the conflict band and asymptotic regime.
    mu0
        Prior mean (default 0.0).
    sigma
        Likelihood std (default 1.0).
    """

    w_range: Tuple[float, float] = (0.05, 0.95)
    theta_true_half_width: float = 10.0
    mu0: float = 0.0
    sigma: float = 1.0

    @classmethod
    def normal_normal_default(cls) -> "TrainingDistribution":
        """Canonical default: `Uniform(0.05, 0.95)` × `Uniform(±10σ)`."""
        return cls()

    def to_dict(self) -> dict:
        """Serialise for embedding in a checkpoint."""
        return {
            "w_range": list(self.w_range),
            "theta_true_half_width": self.theta_true_half_width,
            "mu0": self.mu0,
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingDistribution":
        return cls(
            w_range=tuple(d["w_range"]),
            theta_true_half_width=d["theta_true_half_width"],
            mu0=d["mu0"],
            sigma=d["sigma"],
        )


def lhs_sample(
    distribution: TrainingDistribution,
    n_lhs: int,
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Latin-Hypercube sample (w, θ_true) from `distribution`.

    Returns `(w_arr, theta_true_arr)` each of shape `(n_lhs,)`.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n=n_lhs)                                # (n_lhs, 2) in [0, 1)

    w_lo, w_hi = distribution.w_range
    w = w_lo + u[:, 0] * (w_hi - w_lo)

    half = distribution.theta_true_half_width * distribution.sigma
    theta_true = (distribution.mu0 - half) + u[:, 1] * 2.0 * half

    return w.astype(np.float64), theta_true.astype(np.float64)


def draw_data_batch(
    distribution: TrainingDistribution,
    w_batch: NDArray[np.float64],
    theta_true_batch: NDArray[np.float64],
    n_mc: int,
    rng: np.random.Generator,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Generate `(D, w, theta_true)` torch tensors for a mini-batch.

    For each `(w_i, θ_i)` in the input batch, draws `n_mc` independent
    `D_ij ~ N(θ_i, σ²)`. The returned tensors are flattened so each
    row of the batch is one `(D, w, θ_true)` tuple — simpler for the
    training loop's loss computation.

    Returns a dict with keys:
      D:          (B*n_mc,) tensor
      w:          (B*n_mc,) tensor
      theta_true: (B*n_mc,) tensor
      mu0, sigma: scalar tensors

    where B = len(w_batch).
    """
    B = len(w_batch)
    assert len(theta_true_batch) == B, "w and theta_true batch sizes must match"
    sigma = distribution.sigma
    mu0 = distribution.mu0

    # Each (w_i, theta_i) gets n_mc D draws.
    D = rng.normal(
        loc=theta_true_batch.repeat(n_mc),
        scale=sigma,
        size=B * n_mc,
    )
    w_full = np.tile(w_batch, n_mc)
    theta_full = np.tile(theta_true_batch, n_mc)

    return {
        "D": torch.tensor(D, dtype=dtype, device=device),
        "w": torch.tensor(w_full, dtype=dtype, device=device),
        "theta_true": torch.tensor(theta_full, dtype=dtype, device=device),
        "mu0": torch.tensor(mu0, dtype=dtype, device=device),
        "sigma": torch.tensor(sigma, dtype=dtype, device=device),
    }
