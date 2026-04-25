"""Property tests for smoothness metric helpers.

These pin the metric functions' behavior on synthetic curves where the
right answer is known by construction. They are unit-style L0 tests
(not hypothesis-based), kept under tests/properties/ alongside the
other invariant tests for discoverability.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.diagnostics.smoothness_metrics import (
    _discontinuity_count,
    _local_lipschitz,
    _spectral_roughness,
    _total_variation,
)


@pytest.mark.L0
@pytest.mark.properties
class TestSmoothnessMetricHelpers:
    def test_lipschitz_zero_on_constant(self):
        x = np.linspace(0, 1, 11)
        y = np.full_like(x, 3.0)
        assert _local_lipschitz(x, y) == pytest.approx(0.0)

    def test_lipschitz_recovers_slope_on_linear(self):
        x = np.linspace(0, 1, 11)
        y = 2.5 * x + 1.0
        assert _local_lipschitz(x, y) == pytest.approx(2.5, rel=1e-9)

    def test_total_variation_zero_on_constant(self):
        assert _total_variation(np.full(20, 1.0)) == 0.0

    def test_total_variation_sums_oscillations(self):
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        assert _total_variation(y) == pytest.approx(4.0)

    def test_discontinuity_count_zero_on_smooth(self):
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        assert _discontinuity_count(y) == 0

    def test_discontinuity_count_nonzero_on_step(self):
        y = np.concatenate([np.zeros(20), np.ones(20)])
        assert _discontinuity_count(y) >= 1

    def test_spectral_roughness_low_on_smooth(self):
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)  # purely low-frequency
        assert _spectral_roughness(y) < 0.05

    def test_spectral_roughness_high_on_jagged(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=100)  # white noise: high HF power
        assert _spectral_roughness(y) > 1.0

    def test_metrics_handle_nan(self):
        y = np.array([0.0, np.nan, 1.0, 1.0, 1.0])
        assert _total_variation(y) == 1.0  # ignores NaN
        # discontinuity_count uses finite mask
        assert _discontinuity_count(y) >= 0
