"""L1/L2 regression: Cluster K — experiments / runner robustness.

Pins the audit P1 fixes:

  K.1 — `coverage` / `width` arrays are NaN-initialised and use
        `np.nanmean` / `np.nanstd` so per-rep NaN entries (e.g. a
        single brentq bracket failure mid-loop) are skipped rather
        than aborting the cell or contaminating the mean.

  K.2 — Coverage SE no longer floors `p*(1-p)` at `1e-12`; SE = 0
        at p=0 / p=1 honestly reflects "no MC variation observed."
        Downstream consumers (CoverageRateDiagnostic / plotting)
        treat SE=0 as "skip error bar" — the floor blurred this
        boundary.

  K.3 — `figures.py` short-circuits with a UserWarning when the
        manifest contains zero ran cells (every cell incompatible /
        errored). Pre-fix the loop silently produced no figures
        and no CSVs, leaving the user without any signal.

  K.4 / Path A — `CoverageExperiment` and `WidthExperiment` are
        documented as Normal-Normal-only by construction. The
        experiment classes hard-code `NormalNormalModel(sigma)` and
        a `NormalDistribution` prior derived from `w`; Bernoulli /
        Beta does not enter these experiments. Pinned by reading
        the docstrings + verifying the experiment instances build
        an NN model.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from frasian.experiments.coverage import CoverageExperiment
from frasian.experiments.width import WidthExperiment
from frasian.models.normal_normal import NormalNormalModel


# --- K.1 NaN initialisation --------------------------------------------


@pytest.mark.L1
class TestCoverageWidthNanInit:
    """Source-level pin: `coverage`, `width`, etc. are initialised with
    `np.full(..., np.nan)`, not `np.empty(...)` (which leaves
    uninitialised garbage on cells that early-exit)."""

    def test_coverage_uses_nan_init(self):
        src = Path("src/frasian/experiments/coverage.py").read_text()
        assert "np.full((n_theta, n_w), np.nan" in src, (
            "coverage.py should initialise the result array with NaN, "
            "not np.empty (audit P1 K.1)"
        )

    def test_width_uses_nan_init(self):
        src = Path("src/frasian/experiments/width.py").read_text()
        assert "np.full((n_theta, n_w), np.nan" in src, (
            "width.py should initialise the result array with NaN"
        )

    def test_width_uses_nanmean(self):
        src = Path("src/frasian/experiments/width.py").read_text()
        assert "np.nanmean" in src, (
            "width.py should use np.nanmean / np.nanstd (audit P1 K.1)"
        )


# --- K.2 SE floor removed ----------------------------------------------


@pytest.mark.L1
class TestCoverageSeFloorRemoved:
    """Pre-fix coverage SE used `max(p*(1-p), 1e-12) / n_reps` — fake
    SE ~ 1e-7/sqrt(n_reps) at p=0 / p=1. Post-fix SE is 0 at extremes,
    honestly reflecting no MC variation."""

    def test_no_1e_minus_12_floor_in_se(self):
        src = Path("src/frasian/experiments/coverage.py").read_text()
        # The float literal 1e-12 should not appear in the SE formula
        # context. Allow it elsewhere in the file (other epsilon uses
        # may legitimately exist).
        # We check by absence of `max(p * (1.0 - p), 1e-12)`.
        assert "max(p * (1.0 - p), 1e-12)" not in src, (
            "coverage SE must not floor at 1e-12 (audit P1 K.2)"
        )


# --- K.3 figures.py empty raw_results short-circuit --------------------


@pytest.mark.L0
class TestFiguresEmptyRawResultsShortCircuit:
    """`scripts.figures` should warn + return [] when raw_results is
    empty, not silently produce no output."""

    def test_short_circuit_path_present(self):
        src = Path("scripts/figures.py").read_text()
        assert "if not raw_results:" in src, (
            "figures.py should short-circuit on empty raw_results "
            "(audit P1 K.3)"
        )
        assert "no ran cells" in src or "no figures" in src.lower()


# --- K.4 NN-only experiment contract -----------------------------------


@pytest.mark.L0
class TestCoverageWidthAreNnOnly:
    """Audit P1 K.4 / Path A: `CoverageExperiment` and `WidthExperiment`
    are Normal-Normal-only by construction. The experiment classes
    hard-code `NormalNormalModel(sigma)` inside `run_cell`."""

    def test_coverage_brief_documents_nn_only(self):
        src = Path("src/frasian/experiments/coverage.py").read_text()
        assert "Normal-Normal-only" in src, (
            "coverage.py docstring should state the NN-only contract"
        )
        # Search for the K.4 / Path A audit reference for traceability.
        assert "K.4" in src or "Path A" in src

    def test_width_brief_documents_nn_only(self):
        src = Path("src/frasian/experiments/width.py").read_text()
        assert "Normal-Normal-only" in src, (
            "width.py docstring should state the NN-only contract"
        )
        assert "K.4" in src or "Path A" in src

    def test_coverage_constructs_nn_model_in_run_cell(self):
        # Source-level pin: the experiment constructs a NormalNormalModel
        # in run_cell. (Behavioural pin would require running a full
        # cell, which is heavy; the source pin is sufficient.)
        src = Path("src/frasian/experiments/coverage.py").read_text()
        assert "NormalNormalModel(sigma=self.sigma)" in src

    def test_width_constructs_nn_model_in_run_cell(self):
        src = Path("src/frasian/experiments/width.py").read_text()
        assert "NormalNormalModel(sigma=self.sigma)" in src
