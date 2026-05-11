"""L2 regression: Phase G hyperparam protocol contracts.

Pins the new optional `Prior` / `Model` protocol additions:
  * `hyperparam_dim` — int, length of the hyperparam vector.
  * `hyperparam_names()` — classmethod returning canonical-order names.
  * `hyperparams()` — instance method returning the vector.
  * `from_hyperparams(arr)` — classmethod reconstructing the instance.

Round-trip: `cls.from_hyperparams(inst.hyperparams())` reproduces `inst`.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution


@pytest.mark.L2
class TestNormalDistributionHyperparams:
    def test_hyperparam_dim(self):
        assert NormalDistribution.hyperparam_dim == 2

    def test_hyperparam_names(self):
        assert NormalDistribution.hyperparam_names() == ("loc", "scale")

    def test_hyperparams_returns_vector(self):
        d = NormalDistribution(loc=2.5, scale=0.7)
        hp = d.hyperparams()
        assert isinstance(hp, np.ndarray)
        assert hp.shape == (2,)
        assert hp[0] == pytest.approx(2.5)
        assert hp[1] == pytest.approx(0.7)

    def test_from_hyperparams_roundtrip(self):
        d = NormalDistribution(loc=-0.3, scale=1.4)
        d2 = NormalDistribution.from_hyperparams(d.hyperparams())
        assert d2.loc == pytest.approx(d.loc)
        assert d2.scale == pytest.approx(d.scale)

    def test_from_hyperparams_validates_shape(self):
        with pytest.raises(ValueError, match="length 2"):
            NormalDistribution.from_hyperparams(np.array([1.0]))


from frasian.models.normal_normal import NormalNormalModel


@pytest.mark.L2
class TestNormalNormalModelHyperparams:
    def test_hyperparam_dim(self):
        assert NormalNormalModel.hyperparam_dim == 1

    def test_hyperparam_names(self):
        assert NormalNormalModel.hyperparam_names() == ("sigma",)

    def test_roundtrip(self):
        m = NormalNormalModel(sigma=2.5)
        m2 = NormalNormalModel.from_hyperparams(m.hyperparams())
        assert m2.sigma == pytest.approx(m.sigma)


@pytest.mark.L2
class TestSampleDataBatchWithHp:
    def test_normal_normal_shape_and_distribution(self):
        m_cls = NormalNormalModel
        rng = np.random.default_rng(0)
        theta = np.array([0.0, 1.0, -1.0, 2.5])
        hp = np.array([[1.0], [2.0], [0.5], [1.5]])  # σ per element
        out = m_cls.sample_data_batch_with_hp(theta, hp, rng, n_data=3)
        assert out.shape == (4, 3)
        per_row_mean = out.mean(axis=1)
        per_row_sigma = hp[:, 0]
        per_row_se = per_row_sigma / np.sqrt(3)
        assert np.all(np.abs(per_row_mean - theta) < 6 * per_row_se), (
            f"row means {per_row_mean} differ from theta {theta} by more "
            f"than 6 SE {6 * per_row_se}"
        )

    def test_normal_normal_validates_hp_shape(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="hp shape"):
            NormalNormalModel.sample_data_batch_with_hp(
                np.array([0.0, 1.0]), np.array([1.0, 2.0]), rng, n_data=1,
            )
