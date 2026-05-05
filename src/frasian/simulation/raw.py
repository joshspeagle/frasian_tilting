"""Layer 0: raw D-sample generation for the Normal-Normal sandbox.

The legacy `simulations/raw.py` had three experiment-specific generators
(coverage / distribution / width) hardcoding their own grids. Here we
expose one generic builder that takes any (theta_grid, model, n_reps)
and yields a typed `RawSamples`. Experiment-specific factories — which
choose the grid — live with the experiments in `frasian/experiments/`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ..models.normal_normal import NormalNormalModel


@dataclass(frozen=True)
class RawSamples:
    """Typed container for a 2D grid of D ~ N(theta, sigma) samples.

    Shape: D[i, j] is the j-th replicate at theta_grid[i].
    """

    name: str
    D: NDArray[np.float64]  # shape (n_theta, n_reps)
    theta_grid: NDArray[np.float64]  # shape (n_theta,)
    sigma: float
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable hash used by the cache to invalidate when raw data changes."""
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(str(self.sigma).encode())
        h.update(str(self.seed).encode())
        h.update(self.theta_grid.tobytes())
        h.update(self.D.tobytes())
        h.update(json.dumps(dict(self.metadata), sort_keys=True).encode())
        return h.hexdigest()[:24]

    @property
    def n_theta(self) -> int:
        return self.D.shape[0]

    @property
    def n_reps(self) -> int:
        return self.D.shape[1]


def generate_normal_D_samples(
    *,
    name: str,
    model: NormalNormalModel,
    theta_grid: NDArray[np.float64],
    n_reps: int,
    rng: Generator,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> RawSamples:
    """Generate D ~ N(theta, sigma) samples on a theta-grid.

    Parameters
    ----------
    name :
        Descriptive label persisted with the samples (e.g. "coverage_v1").
    model :
        Provides the noise scale `sigma` and the sampling primitive.
    theta_grid :
        1D array of true parameter values; one row of replicates per entry.
    n_reps :
        Replicates per theta.
    rng :
        Seeded `numpy.random.Generator` used for sampling.
    seed :
        Integer seed that was used to initialise `rng`. Recorded in the
        result for reproducibility checks and fingerprinting. The caller
        is responsible for keeping `rng` and `seed` consistent.
    """
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    if theta_grid.ndim != 1:
        raise ValueError(f"theta_grid must be 1D, got shape {theta_grid.shape}")
    if n_reps <= 0:
        raise ValueError(f"n_reps must be positive, got {n_reps}")

    D = np.empty((theta_grid.size, n_reps), dtype=np.float64)
    for i, theta in enumerate(theta_grid):
        D[i] = model.sample_data(float(theta), rng, n=n_reps)

    return RawSamples(
        name=name,
        D=D,
        theta_grid=theta_grid,
        sigma=model.sigma,
        seed=seed,
        metadata=dict(metadata or {}),
    )
