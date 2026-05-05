"""Regression tests for `simulation.raw.generate_normal_D_samples`."""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.normal_normal import NormalNormalModel
from frasian.simulation.raw import RawSamples, generate_normal_D_samples


def _gen(seed=0, n_reps=50) -> RawSamples:
    return generate_normal_D_samples(
        name="demo",
        model=NormalNormalModel(sigma=1.0),
        theta_grid=np.array([-1.0, 0.0, 1.0]),
        n_reps=n_reps,
        rng=np.random.default_rng(seed),
        seed=seed,
    )


@pytest.mark.L0
class TestRawSamplesShape:
    def test_shape_is_n_theta_by_n_reps(self):
        raw = _gen()
        assert raw.D.shape == (3, 50)
        assert raw.n_theta == 3
        assert raw.n_reps == 50

    def test_per_theta_mean_concentrates(self):
        raw = _gen(seed=7, n_reps=10_000)
        means = raw.D.mean(axis=1)
        np.testing.assert_allclose(means, raw.theta_grid, atol=0.05)

    def test_per_theta_var_concentrates(self):
        raw = _gen(seed=7, n_reps=10_000)
        vars_ = raw.D.var(axis=1, ddof=1)
        np.testing.assert_allclose(vars_, np.full(3, 1.0), rtol=0.1)


@pytest.mark.L0
class TestRawSamplesDeterminism:
    def test_same_seed_same_output(self):
        a = _gen(seed=42).D
        b = _gen(seed=42).D
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_output(self):
        a = _gen(seed=1).D
        b = _gen(seed=2).D
        assert not np.array_equal(a, b)


@pytest.mark.L0
class TestRawSamplesFingerprint:
    def test_stable_under_repeat(self):
        a = _gen(seed=0).fingerprint()
        b = _gen(seed=0).fingerprint()
        assert a == b

    def test_changes_with_data(self):
        a = _gen(seed=0).fingerprint()
        b = _gen(seed=1).fingerprint()
        assert a != b

    def test_changes_with_metadata(self):
        m1 = generate_normal_D_samples(
            name="demo",
            model=NormalNormalModel(sigma=1.0),
            theta_grid=np.array([0.0]),
            n_reps=2,
            rng=np.random.default_rng(0),
            seed=0,
            metadata={"k": 1},
        ).fingerprint()
        m2 = generate_normal_D_samples(
            name="demo",
            model=NormalNormalModel(sigma=1.0),
            theta_grid=np.array([0.0]),
            n_reps=2,
            rng=np.random.default_rng(0),
            seed=0,
            metadata={"k": 2},
        ).fingerprint()
        assert m1 != m2


@pytest.mark.L0
class TestRawSamplesValidation:
    def test_2d_theta_grid_rejected(self):
        with pytest.raises(ValueError):
            generate_normal_D_samples(
                name="x",
                model=NormalNormalModel(sigma=1.0),
                theta_grid=np.zeros((2, 2)),
                n_reps=3,
                rng=np.random.default_rng(0),
                seed=0,
            )

    def test_zero_n_reps_rejected(self):
        with pytest.raises(ValueError):
            generate_normal_D_samples(
                name="x",
                model=NormalNormalModel(sigma=1.0),
                theta_grid=np.array([0.0]),
                n_reps=0,
                rng=np.random.default_rng(0),
                seed=0,
            )
