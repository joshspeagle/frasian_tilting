"""Layer 1: deterministic transforms from `RawSamples` to `ProcessedResult`.

Concrete processors (coverage rates, CI widths, calibration tables) ship
with the experiments that need them in Step 4. Step 3 only defines the
type and a smoke processor used by the cache tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ProcessedResult:
    """Output of any Layer-1 transform; shape mirrors `simulation.raw.RawSamples`."""

    name: str
    arrays: Mapping[str, NDArray]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def smoke_process(name: str, raw_D: NDArray[np.float64],
                  theta_grid: NDArray[np.float64]) -> ProcessedResult:
    """Trivial processor used by tests of the cache pipeline.

    Computes per-theta sample mean and variance — enough to verify that the
    raw → processed → cached flow is wired correctly without depending on
    any tilting-specific math.
    """
    return ProcessedResult(
        name=name,
        arrays={
            "theta_grid": theta_grid,
            "mean_D": raw_D.mean(axis=1),
            "var_D": raw_D.var(axis=1, ddof=1),
        },
        metadata={"processor": "smoke"},
    )
