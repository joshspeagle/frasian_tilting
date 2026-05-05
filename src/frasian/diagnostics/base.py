"""Diagnostic protocol and DiagnosticTable.

Diagnostics are pure functions of `RawResult`. The decoupling between the
experiment that produced the data and the diagnostic that summarizes it means
a diagnostic is reusable across experiments and trivially CI-checkable.

`compute` returns a tidy DataFrame; `render` produces a figure file and
returns its path. Diagnostics are expected to be cheap to recompute from
`RawResult`s; expensive aggregation belongs inside `Experiment.run_cell`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Avoid circular import: forward declare RawResult via TYPE_CHECKING.
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from ..experiments.base import RawResult


@dataclass
class DiagnosticTable:
    """Tidy DataFrame plus metadata, the canonical diagnostic output."""

    name: str
    table: pd.DataFrame
    units: dict[str, str]
    metadata: dict[str, object]


@runtime_checkable
class Diagnostic(Protocol):
    """Pure function from RawResult to DiagnosticTable + figure."""

    @property
    def name(self) -> str: ...

    def compute(self, raw: RawResult) -> DiagnosticTable: ...
    def render(self, table: DiagnosticTable, fig_dir: Path) -> Path: ...
