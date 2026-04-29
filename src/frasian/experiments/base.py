"""Experiment protocol and supporting dataclasses.

An `Experiment` defines a parameter sweep grid and a recipe for what gets
computed in each cell. The runner iterates the cartesian product of
(TiltingScheme × TestStatistic) for the registered methods, calls
`run_cell` for each cell, persists `RawResult`s through the three-layer
simulation stack, and emits the diagnostics declared by the experiment.

Versioned analysis configs live in `experiments/*.yaml` (grid + seed +
n_reps); the Python `Experiment` class describes *what* to compute. Same
`Experiment` definition can serve many published runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..config import Config
from ..diagnostics.base import Diagnostic
from ..statistics.base import TestStatistic
from ..tilting.base import TiltingScheme


@dataclass
class ExperimentContext:
    """Runtime context passed into `run_cell`."""

    config: Config
    grid: Mapping[str, NDArray[np.float64]]
    rng_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawResult:
    """Per-cell raw output of `run_cell`. Persisted by simulation.cache."""

    experiment: str
    tilting: str
    statistic: str
    arrays: Mapping[str, NDArray[np.float64]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Experiment(Protocol):
    """Declarative experiment that the runner executes for every cell.

    Invariants:
        - `setup(config)` is deterministic given `config`.
        - `run_cell(ctx, tilting, statistic)` is deterministic given the same
          inputs and `ctx.rng_seed`.
        - `diagnostics()` returns at least one `Diagnostic`.
    """

    name: str

    def setup(self, config: Config) -> ExperimentContext: ...

    def run_cell(self, ctx: ExperimentContext, tilting: TiltingScheme,
                 statistic: TestStatistic) -> RawResult: ...

    def diagnostics(self) -> list[Diagnostic]: ...
