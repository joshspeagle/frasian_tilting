"""Layer 1: deterministic transforms from `RawSamples` to `ProcessedResult`.

Concrete processors (coverage rates, CI widths, calibration tables) live
with the experiments that need them under `frasian/experiments/`. This
module defines the type and a smoke processor used by the cache tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

# Audit P2 (Cluster I): schema version for the `ProcessedResult` type
# is mirrored here so static consumers (and the cache invalidation
# story documented in `simulation/storage.py`) can read the version
# from the type itself instead of relying on the on-disk
# `_schema_version` metadata field. Bump in lockstep with
# `storage.SCHEMA_VERSION` whenever the public ProcessedResult shape
# changes (adding/removing array keys, changing dtype contracts, etc.).
PROCESSED_RESULT_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class ProcessedResult:
    """Output of any Layer-1 transform; shape mirrors `simulation.raw.RawSamples`."""

    SCHEMA_VERSION: ClassVar[int] = PROCESSED_RESULT_SCHEMA_VERSION

    name: str
    arrays: Mapping[str, NDArray]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def smoke_process(
    name: str, raw_D: NDArray[np.float64], theta_grid: NDArray[np.float64]
) -> ProcessedResult:
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
