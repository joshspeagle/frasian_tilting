"""Property tests for learned-eta training diagnostics."""

from __future__ import annotations

import numpy as np
import pytest


def _v4_hyperparam_distribution():
    """The canonical v4 NormalNormal + power_law hyperparam distribution.

    Mirrors `experiments/canonical_normal_normal_powerlaw_v4.yaml` so
    probe-batch tests exercise the same ranges the training loop sees.
    """
    from frasian.learned.training.hyperparam_distribution import (
        HyperparamDistribution, ScalarDist,
    )
    return HyperparamDistribution(
        prior_specs={
            "loc":   ScalarDist(kind="uniform",    low=-2.0, high=2.0),
            "scale": ScalarDist(kind="loguniform", low=0.2,  high=5.0),
        },
        lik_specs={
            "sigma": ScalarDist(kind="loguniform", low=0.5,  high=2.0),
        },
    )


@pytest.mark.L1
@pytest.mark.properties
class TestProbeBatch:
    def test_probe_batch_size_and_shapes(self):
        from frasian.learned.training.diagnostics import build_probe_batch
        rng = np.random.default_rng(0xCAFE)
        pb = build_probe_batch(
            scheme_name="power_law", n=64, rng=rng,
            hyperparam_distribution=_v4_hyperparam_distribution(),
        )
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
        pb = build_probe_batch(
            scheme_name="power_law", n=128, rng=rng,
            hyperparam_distribution=_v4_hyperparam_distribution(),
        )
        bins = np.array([w_bin(float(w)) for w in pb.w])
        # All three bins should be represented in a sample of 128
        assert set(np.unique(bins).tolist()) == {"lowW", "midW", "highW"}

    def test_probe_batch_reproducible(self):
        from frasian.learned.training.diagnostics import build_probe_batch
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        hd = _v4_hyperparam_distribution()
        pb_a = build_probe_batch(
            scheme_name="power_law", n=32, rng=rng_a, hyperparam_distribution=hd,
        )
        pb_b = build_probe_batch(
            scheme_name="power_law", n=32, rng=rng_b, hyperparam_distribution=hd,
        )
        np.testing.assert_array_equal(pb_a.theta, pb_b.theta)
        np.testing.assert_array_equal(pb_a.argmin_eta, pb_b.argmin_eta)


@pytest.mark.L1
@pytest.mark.properties
class TestD1OutputStats:
    def _make_random_eta_net(self):
        import jax
        from frasian.learned.training.architecture import EtaNet
        return EtaNet(
            theta_dim=1, prior_dim=2, lik_dim=1,
            hidden_sizes=(16, 16), key=jax.random.PRNGKey(0),
        )

    def test_d1_output_stats_returns_expected_keys(self):
        from frasian.learned.training.diagnostics import (
            build_probe_batch, compute_d1_output_stats,
        )
        rng = np.random.default_rng(0xCAFE)
        pb = build_probe_batch(
            scheme_name="power_law", n=32, rng=rng,
            hyperparam_distribution=_v4_hyperparam_distribution(),
        )
        net = self._make_random_eta_net()
        stats = compute_d1_output_stats(net, pb)
        for k in ("eta_mean", "eta_std", "eta_range",
                  "corr_with_argmin", "residual_mean"):
            assert k in stats, f"missing {k}"
            assert np.isfinite(stats[k]), f"{k} = {stats[k]}"

    def test_d1_corr_with_constant_argmin_is_nan_or_zero(self):
        """If all argmin values are equal, correlation is undefined (NaN/0)."""
        from dataclasses import replace
        from frasian.learned.training.diagnostics import (
            build_probe_batch, compute_d1_output_stats,
        )
        rng = np.random.default_rng(0xCAFE)
        pb = build_probe_batch(
            scheme_name="power_law", n=32, rng=rng,
            hyperparam_distribution=_v4_hyperparam_distribution(),
        )
        # Force all argmins to be 0.5
        pb_const = replace(pb, argmin_eta=np.full_like(pb.argmin_eta, 0.5))
        net = self._make_random_eta_net()
        stats = compute_d1_output_stats(net, pb_const)
        assert np.isnan(stats["corr_with_argmin"]) or stats["corr_with_argmin"] == 0.0


@pytest.mark.L1
@pytest.mark.properties
class TestD3ActivationStats:
    def test_d3_returns_expected_keys(self):
        import jax
        from frasian.learned.training.architecture import EtaNet
        from frasian.learned.training.diagnostics import (
            build_probe_batch, compute_d3_activation_stats,
        )
        rng = np.random.default_rng(0xCAFE)
        pb = build_probe_batch(
            scheme_name="power_law", n=32, rng=rng,
            hyperparam_distribution=_v4_hyperparam_distribution(),
        )
        net = EtaNet(theta_dim=1, prior_dim=2, lik_dim=1,
                     hidden_sizes=(16, 16), key=jax.random.PRNGKey(0))
        stats = compute_d3_activation_stats(net, pb)
        for k in ("penult_std_mean", "penult_std_min", "n_dead_neurons"):
            assert k in stats, f"missing {k}"
        assert isinstance(stats["n_dead_neurons"], int)
        assert stats["n_dead_neurons"] >= 0


@pytest.mark.L1
@pytest.mark.properties
class TestD2GradientNorms:
    def test_d2_returns_expected_keys(self):
        import jax
        from frasian.learned.training.architecture import EtaNet
        from frasian.learned.training.diagnostics import (
            build_probe_batch, compute_d2_gradient_norms,
        )
        rng = np.random.default_rng(0xCAFE)
        pb = build_probe_batch(
            scheme_name="power_law", n=32, rng=rng,
            hyperparam_distribution=_v4_hyperparam_distribution(),
        )
        net = EtaNet(theta_dim=1, prior_dim=2, lik_dim=1,
                     hidden_sizes=(16, 16), key=jax.random.PRNGKey(0))
        norms = compute_d2_gradient_norms(
            net, pb, scheme_name="power_law", statistic_name="waldo",
        )
        # Per-layer (input weight, output weight, output bias)
        for k in ("grad_norm_input_w", "grad_norm_output_w", "grad_norm_output_b"):
            assert k in norms, f"missing {k}"
            assert np.isfinite(norms[k]) and norms[k] >= 0.0
        # Per-w-bin
        for k in ("grad_norm_lowW", "grad_norm_midW", "grad_norm_highW"):
            assert k in norms, f"missing {k}"
            assert np.isfinite(norms[k]) and norms[k] >= 0.0


@pytest.mark.L1
@pytest.mark.properties
class TestD4LossByBin:
    def test_d4_returns_expected_keys(self):
        import jax
        from frasian.learned.training.architecture import EtaNet
        from frasian.learned.training.diagnostics import (
            build_probe_batch, compute_d4_loss_by_bin,
        )
        rng = np.random.default_rng(0xCAFE)
        pb = build_probe_batch(
            scheme_name="power_law", n=64, rng=rng,
            hyperparam_distribution=_v4_hyperparam_distribution(),
        )
        net = EtaNet(theta_dim=1, prior_dim=2, lik_dim=1,
                     hidden_sizes=(16, 16), key=jax.random.PRNGKey(0))
        result = compute_d4_loss_by_bin(
            net, pb, scheme_name="power_law", statistic_name="waldo",
        )
        for k in ("loss_lowW", "loss_midW", "loss_highW"):
            assert k in result, f"missing {k}"
            assert np.isfinite(result[k]) and result[k] >= 0.0


@pytest.mark.L1
@pytest.mark.properties
class TestStratifiedBatch:
    def test_stratified_batches_span_w_bins(self):
        from frasian.learned.training.hyperparam_distribution import (
            HyperparamDistribution,
            ScalarDist,
            StratifiedBatchHyperparamDistribution,
        )
        from frasian.learned.training.diagnostics import w_bin
        base = HyperparamDistribution(
            prior_specs={
                "loc":   ScalarDist(kind="uniform",    low=-2.0, high=2.0),
                "scale": ScalarDist(kind="loguniform", low=0.2,  high=5.0),
            },
            lik_specs={
                "sigma": ScalarDist(kind="loguniform", low=0.5,  high=2.0),
            },
        )
        strat = StratifiedBatchHyperparamDistribution(base=base, n_buckets=4)
        rng = np.random.default_rng(0xCAFE)
        # Sample a batch of 32. Should have all 3 w-bins represented.
        prior_names = ("loc", "scale")
        lik_names = ("sigma",)
        prior_hp, lik_hp = strat.sample(
            32, rng, prior_names=prior_names, lik_names=lik_names,
        )
        assert prior_hp.shape == (32, 2)
        assert lik_hp.shape == (32, 1)
        sigma0 = prior_hp[:, 1]
        sigma = lik_hp[:, 0]
        w = sigma0 ** 2 / (sigma ** 2 + sigma0 ** 2)
        bins = np.array([w_bin(float(wi)) for wi in w])
        assert set(np.unique(bins).tolist()) == {"lowW", "midW", "highW"}, (
            f"expected all 3 bins, got {np.unique(bins)}"
        )
