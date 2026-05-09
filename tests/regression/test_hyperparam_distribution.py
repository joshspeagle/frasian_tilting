"""L2 regression: HyperparamDistribution sampler + in-range guard.

Pins the contracts of the new training-time sampler that drives
per-batch (prior_hp, lik_hp) draws for the conditional learned-η.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.learned.training.hyperparam_distribution import (
    HyperparamDistribution,
    ScalarDist,
)


@pytest.mark.L2
class TestScalarDist:
    def test_uniform_sample_in_range(self):
        d = ScalarDist(kind="uniform", low=-1.0, high=2.0)
        rng = np.random.default_rng(0)
        x = d.sample(1000, rng)
        assert x.shape == (1000,)
        assert x.min() >= -1.0 and x.max() <= 2.0

    def test_loguniform_sample_in_range_and_skewed(self):
        d = ScalarDist(kind="loguniform", low=0.1, high=10.0)
        rng = np.random.default_rng(0)
        x = d.sample(10000, rng)
        assert x.min() >= 0.1 and x.max() <= 10.0
        frac_below_1 = (x < 1.0).mean()
        assert 0.45 < frac_below_1 < 0.55

    def test_in_range_endpoints_inclusive(self):
        d = ScalarDist(kind="uniform", low=0.0, high=1.0)
        assert d.in_range(0.0)
        assert d.in_range(0.5)
        assert d.in_range(1.0)
        assert not d.in_range(-0.001)
        assert not d.in_range(1.001)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            ScalarDist(kind="exponential", low=0, high=1)


@pytest.mark.L2
class TestHyperparamDistribution:
    def _example(self) -> HyperparamDistribution:
        return HyperparamDistribution(
            prior_specs={
                "loc":   ScalarDist("uniform",    -2.0, 2.0),
                "scale": ScalarDist("loguniform",  0.2, 5.0),
            },
            lik_specs={
                "sigma": ScalarDist("loguniform", 0.5, 2.0),
            },
        )

    def test_sample_shapes(self):
        d = self._example()
        rng = np.random.default_rng(0)
        prior_b, lik_b = d.sample(100, rng,
                                   prior_names=("loc", "scale"),
                                   lik_names=("sigma",))
        assert prior_b.shape == (100, 2)
        assert lik_b.shape == (100, 1)

    def test_sample_columns_match_names_order(self):
        d = self._example()
        rng = np.random.default_rng(0)
        prior_b, _ = d.sample(50, rng,
                               prior_names=("loc", "scale"),
                               lik_names=("sigma",))
        assert prior_b[:, 0].min() >= -2.0 and prior_b[:, 0].max() <= 2.0
        assert prior_b[:, 1].min() >= 0.2 and prior_b[:, 1].max() <= 5.0

    def test_in_range_all_inside_returns_true(self):
        d = self._example()
        ok = d.in_range(
            prior_hp=np.array([0.0, 1.0]),
            lik_hp=np.array([1.0]),
            prior_names=("loc", "scale"),
            lik_names=("sigma",),
        )
        assert ok

    def test_in_range_one_outside_returns_false(self):
        d = self._example()
        ok = d.in_range(
            prior_hp=np.array([0.0, 100.0]),
            lik_hp=np.array([1.0]),
            prior_names=("loc", "scale"),
            lik_names=("sigma",),
        )
        assert not ok

    def test_first_out_of_range_names_offender(self):
        d = self._example()
        bad = d.first_out_of_range(
            prior_hp=np.array([0.0, 100.0]),
            lik_hp=np.array([1.0]),
            prior_names=("loc", "scale"),
            lik_names=("sigma",),
        )
        assert bad is not None
        assert bad.name == "scale"
        assert bad.value == 100.0

    def test_to_dict_from_dict_roundtrip(self):
        d = self._example()
        spec = d.to_dict()
        d2 = HyperparamDistribution.from_dict(spec)
        assert d2.prior_specs == d.prior_specs
        assert d2.lik_specs == d.lik_specs

    def test_fingerprint_stable_across_construction(self):
        d1 = self._example()
        d2 = self._example()
        assert d1.fingerprint() == d2.fingerprint()
