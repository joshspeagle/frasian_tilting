"""Step-1 verification: framework is sound on an empty registry.

These tests cover the core promise of Step 1: importing `frasian` does not
trigger any side effects, the registry starts empty, and `run_experiment`
fails cleanly rather than silently doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from frasian import Config, EmptyRegistryError, list_methods, registry, run_experiment
from frasian.experiments.base import Experiment, ExperimentContext, RawResult


@dataclass
class _NullExperiment:
    """Minimal Experiment that never gets called (registry is empty)."""

    name: str = "null"

    def setup(self, config: Config) -> ExperimentContext:
        return ExperimentContext(config=config, grid={}, rng_seed=0)

    def run_cell(self, ctx, tilting, statistic) -> RawResult:  # pragma: no cover
        raise AssertionError("must not be called when registry is empty")

    def diagnostics(self):  # pragma: no cover
        return []


class TestEmptyRegistry:
    """Verify Step-1 invariants."""

    def test_import_is_side_effect_free(self):
        """`import frasian` must not register any concrete methods."""
        # The conftest fixture clears+restores; here we observe the
        # post-import-only state.
        assert len(registry.models) == 0
        assert len(registry.tiltings) == 0
        assert len(registry.statistics) == 0
        assert len(registry.experiments) == 0
        assert len(registry.diagnostics) == 0

    def test_run_experiment_raises_on_empty_registry(self):
        """`run_experiment` must fail loudly, not silently no-op."""
        with pytest.raises(EmptyRegistryError):
            run_experiment(
                experiment=_NullExperiment(),
                tiltings=[],
                statistics=[],
            )

    def test_run_experiment_raises_with_only_one_dimension(self):
        """A non-empty cartesian product needs both dimensions populated."""
        # Even if we had a tilting, an empty statistics list still yields zero cells.
        with pytest.raises(EmptyRegistryError):
            run_experiment(
                experiment=_NullExperiment(),
                tiltings=[object],  # placeholder; never instantiated
                statistics=[],
            )

    def test_list_methods_returns_empty_groups(self):
        groups = list_methods()
        assert set(groups) == {"models", "tiltings", "statistics", "experiments", "diagnostics"}
        assert all(len(v) == 0 for v in groups.values())

    def test_protocols_are_importable_without_concrete_impls(self):
        """Importing the protocol modules must not require any registered class."""
        from frasian.cd.base import ConfidenceDistribution
        from frasian.diagnostics.base import Diagnostic
        from frasian.experiments.base import Experiment as ExperimentProto
        from frasian.learned.base import LearnedArtifact
        from frasian.models.base import Model
        from frasian.statistics.base import TestStatistic
        from frasian.tilting.base import TiltingScheme

        # Just confirm they are usable as runtime-checkable Protocols.
        for proto in (
            ConfidenceDistribution,
            Diagnostic,
            ExperimentProto,
            LearnedArtifact,
            Model,
            TestStatistic,
            TiltingScheme,
        ):
            assert hasattr(proto, "_is_runtime_protocol") or hasattr(proto, "_is_protocol")
