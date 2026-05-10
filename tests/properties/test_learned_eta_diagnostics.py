"""Property tests for learned-eta training diagnostics."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.L1
@pytest.mark.properties
class TestProbeBatch:
    def test_probe_batch_size_and_shapes(self):
        from frasian.learned.training.diagnostics import build_probe_batch
        rng = np.random.default_rng(0xCAFE)
        pb = build_probe_batch(scheme_name="power_law", n=64, rng=rng)
        assert pb.theta.shape == (64,)
        assert pb.D.shape == (64,)
        assert pb.prior_hp.shape == (64, 2)  # NormalDistribution: (loc, scale)
        assert pb.lik_hp.shape == (64, 1)    # NormalNormalModel: (sigma,)
        assert pb.argmin_eta.shape == (64,)
        assert pb.w.shape == (64,)
        # All eta in [-1.5, +1.5] (the per-slice argmin search range)
        assert np.all(np.abs(pb.argmin_eta) <= 1.5 + 1e-9)
        # All w in (0, 1)
        assert np.all((pb.w > 0.0) & (pb.w < 1.0))

    def test_probe_batch_w_bins_assigned(self):
        from frasian.learned.training.diagnostics import build_probe_batch, w_bin
        rng = np.random.default_rng(0xCAFE)
        pb = build_probe_batch(scheme_name="power_law", n=128, rng=rng)
        bins = np.array([w_bin(float(w)) for w in pb.w])
        # All three bins should be represented in a sample of 128
        assert set(np.unique(bins).tolist()) == {"lowW", "midW", "highW"}

    def test_probe_batch_reproducible(self):
        from frasian.learned.training.diagnostics import build_probe_batch
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        pb_a = build_probe_batch(scheme_name="power_law", n=32, rng=rng_a)
        pb_b = build_probe_batch(scheme_name="power_law", n=32, rng=rng_b)
        np.testing.assert_array_equal(pb_a.theta, pb_b.theta)
        np.testing.assert_array_equal(pb_a.argmin_eta, pb_b.argmin_eta)
