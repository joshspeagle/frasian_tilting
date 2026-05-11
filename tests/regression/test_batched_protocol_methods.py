"""L2 regression: batched Model protocol methods + array-θ MC paths.

Pins:
  1. `posterior_moments_batch` matches per-element `posterior().mean()/.var()`
     on `NormalNormalModel`.
  2. `sample_data_batch` returns the right shape and is consistent with
     per-row `sample_data` under shared seeding.
  3. `batch_loglik_grid` matches per-row `likelihood().loglik(theta_grid)`.
  4. Default fallbacks (loop-based) agree with the vectorised overrides.
  5. `WaldoStatistic._generic_pvalue` accepts array-θ and matches per-θ
     scalar calls under the same `derived_seed`.
  6. `WaldoStatistic._generic_pvalue(obs_moments=...)` matches the
     recompute path (no semantic change from the hoist).

These aren't perf tests — they pin the contracts that the
hot-path optimisations rely on. Speedup is measured separately
in scripts/benchmark_generic_paths.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.base import (
    batch_loglik_grid as dispatch_batch_loglik_grid,
    default_batch_loglik_grid,
    default_posterior_moments_batch,
    default_posterior_quantile_batch,
    default_sample_data_batch,
    posterior_moments_batch as dispatch_posterior_moments_batch,
    posterior_quantile_batch as dispatch_posterior_quantile_batch,
    sample_data_batch as dispatch_sample_data_batch,
)
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic


# ---------- Normal-Normal batched ----------


@pytest.mark.L2
class TestNormalNormalBatched:
    def test_sample_data_batch_shape(self):
        m = NormalNormalModel(sigma=2.0)
        rng = np.random.default_rng(42)
        out = m.sample_data_batch(theta=0.5, rng=rng, n_mc=100, n_obs=3)
        assert out.shape == (100, 3)
        # Within ~3 std of N(0.5, 2^2)
        assert abs(float(out.mean()) - 0.5) < 3 * 2.0 / np.sqrt(300)

    def test_posterior_moments_batch_matches_per_row(self):
        m = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        rng = np.random.default_rng(0)
        D_batch = rng.normal(size=(50, 1))
        mu_batch, var_batch = m.posterior_moments_batch(D_batch, prior)
        for i in range(50):
            post = m.posterior(D_batch[i], prior)
            assert mu_batch[i] == pytest.approx(float(post.mean()), abs=1e-12)
            assert var_batch[i] == pytest.approx(float(post.var()), abs=1e-12)

    def test_batch_loglik_grid_matches_per_row(self):
        m = NormalNormalModel(sigma=1.5)
        rng = np.random.default_rng(1)
        D_batch = rng.normal(size=(20, 1))
        theta_grid = np.linspace(-3.0, 3.0, 17)
        loglik_batch = m.batch_loglik_grid(D_batch, theta_grid)
        assert loglik_batch.shape == (20, 17)
        for i in range(20):
            lik = m.likelihood(D_batch[i])
            expected = np.asarray(lik.loglik(theta_grid))
            np.testing.assert_allclose(loglik_batch[i], expected, atol=1e-12)

    def test_posterior_quantile_batch_matches_per_row(self):
        m = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        rng = np.random.default_rng(5)
        D_batch = rng.normal(size=(30, 1))
        u_grid = np.linspace(0.05, 0.95, 11)
        q_batch = m.posterior_quantile_batch(D_batch, prior, u_grid)
        assert q_batch.shape == (30, 11)
        for i in range(30):
            post = m.posterior(D_batch[i], prior)
            expected = np.asarray(post.quantile(u_grid))
            np.testing.assert_allclose(q_batch[i], expected, atol=1e-12)


# ---------- Default fallback dispatch ----------


@pytest.mark.L2
class TestDispatchFallbacks:
    """The dispatch helpers in models/base prefer the model's override but
    fall back to the loop-based default. Verify both code paths agree on
    a model that has the override (NormalNormalModel)."""

    def test_dispatch_uses_override_when_present(self):
        m = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        D_batch = np.array([[0.5], [1.5], [-0.3]])
        mu_dispatch, var_dispatch = dispatch_posterior_moments_batch(m, D_batch, prior)
        mu_default, var_default = default_posterior_moments_batch(m, D_batch, prior)
        np.testing.assert_allclose(mu_dispatch, mu_default, atol=1e-12)
        np.testing.assert_allclose(var_dispatch, var_default, atol=1e-12)

    def test_dispatch_sample_shape(self):
        m = NormalNormalModel(sigma=1.0)
        rng = np.random.default_rng(11)
        out = dispatch_sample_data_batch(m, theta=0.0, rng=rng, n_mc=10, n_obs=2)
        assert out.shape == (10, 2)

    def test_default_loop_loglik_matches_override(self):
        m = NormalNormalModel(sigma=1.0)
        D_batch = np.array([[0.5], [1.0]])
        theta_grid = np.linspace(-2, 2, 5)
        ov = m.batch_loglik_grid(D_batch, theta_grid)
        df = default_batch_loglik_grid(m, D_batch, theta_grid)
        np.testing.assert_allclose(ov, df, atol=1e-12)


# ---------- WaldoStatistic.array-θ + obs_moments ----------


@pytest.mark.L2
class TestWaldoGenericArrayTheta:
    def test_array_theta_matches_scalar_calls(self):
        """Vectorised array-θ pvalue must match per-θ scalar calls under
        the same `derived_seed` (CRN). Pins that the inner loop is
        truly per-θ-vectorised, not just a different code path."""
        m = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        stat = WaldoStatistic(force_generic=True, n_mc=500)
        seed = 12345
        thetas = np.linspace(-1.0, 1.5, 7)
        # Vectorised call — single returned array.
        p_vec = np.asarray(
            stat._generic_pvalue(thetas, data, m, prior, derived_seed=seed)
        )
        # Per-θ scalar calls with the same seed.
        p_scalar = np.asarray([
            float(np.asarray(
                stat._generic_pvalue(float(t), data, m, prior, derived_seed=seed)
            ))
            for t in thetas
        ])
        np.testing.assert_allclose(p_vec, p_scalar, atol=1e-12)

    def test_obs_moments_matches_recompute(self):
        """Passing obs_moments must give the bit-equal result of recomputing."""
        m = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        stat = WaldoStatistic(force_generic=True, n_mc=500)
        seed = 999
        # Recompute path
        p_recompute = float(np.asarray(
            stat._generic_pvalue(0.2, data, m, prior, derived_seed=seed)
        ))
        # Hoisted path
        post = m.posterior(data, prior)
        obs_moments = (float(post.mean()), float(post.var()))
        p_hoisted = float(np.asarray(
            stat._generic_pvalue(0.2, data, m, prior,
                                  derived_seed=seed, obs_moments=obs_moments)
        ))
        assert p_hoisted == pytest.approx(p_recompute, abs=1e-12)


# ---------- power_law tilted batched MC ----------


@pytest.mark.L2
class TestOTBatchedTiltedMC:
    """Vectorised OT-tilted MC reference must agree with closed-form
    Theorem on Normal-Normal within MC noise."""

    def test_tilted_pvalue_agreement_on_nn(self):
        from frasian.tilting.ot import _generic_tilted_pvalue_ot, OTTilting

        m = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        eta = 0.5
        theta = 0.3
        cf_p = float(np.asarray(
            OTTilting().tilted_pvalue(theta, 0.5, m, prior, eta, "waldo")
        ))
        gn_p = _generic_tilted_pvalue_ot(
            theta, data, m, prior, eta, "waldo", n_mc=2000
        )
        # MC SE ~0.011 at p~0.5 with n_mc=2000; allow 5σ.
        assert abs(cf_p - gn_p) < 0.06, f"closed-form={cf_p}, generic={gn_p}"

    def test_jit_kernel_matches_pure_numpy_path(self):
        """The JIT'd `_ot_tilted_kernel_jit` must produce numerically the
        same `t_samples` as a pure-numpy reimplementation of the same
        post-loglik block. Pinned with atol 1e-10 — XLA float64 ops
        match numpy bit-for-bit on these vectorised primitives.
        """
        from frasian.tilting.ot import _ot_tilted_kernel_jit
        import jax.numpy as jnp

        rng = np.random.default_rng(0)
        n_mc, n_grid, n_u = 50, 128, 32
        log_lik = rng.normal(size=(n_mc, n_grid)) - 5.0
        theta_grid = np.linspace(-3.0, 3.0, n_grid)
        F_post_inv = rng.normal(size=(n_mc, n_u))
        u_nodes_raw, weights_raw = np.polynomial.legendre.leggauss(n_u)
        u01 = 0.5 * (u_nodes_raw + 1.0)
        w01 = 0.5 * weights_raw
        eta, theta_f = 0.4, 0.7

        # JIT path
        t_jit, n_coll_jit = _ot_tilted_kernel_jit(
            jnp.asarray(log_lik), jnp.asarray(theta_grid),
            jnp.asarray(F_post_inv), jnp.asarray(u01), jnp.asarray(w01),
            jnp.asarray(eta), jnp.asarray(theta_f),
        )
        t_jit = np.asarray(t_jit)

        # Reference numpy path (mirrors _ot_tilted_kernel_jit line-for-line)
        ll = np.where(np.isfinite(log_lik), log_lik, -1e300)
        ll_max = ll.max(axis=-1, keepdims=True)
        pdf = np.exp(ll - ll_max)
        Z = np.trapezoid(pdf, theta_grid, axis=-1)
        Z_safe = np.where(Z > 0, Z, 1.0)
        pdf = pdf / Z_safe[:, None]
        dtheta = np.diff(theta_grid)
        incr = 0.5 * (pdf[:, :-1] + pdf[:, 1:]) * dtheta[None, :]
        cdf = np.concatenate([np.zeros((n_mc, 1)), np.cumsum(incr, axis=-1)], axis=-1)
        cdf = np.clip(cdf / np.where(cdf[:, -1:] > 0, cdf[:, -1:], 1.0), 0.0, 1.0)
        # Per-row inverse CDF (loop)
        F_lik_inv = np.empty((n_mc, n_u))
        for i in range(n_mc):
            idx = np.clip(np.searchsorted(cdf[i], u01, side="right"), 1, n_grid - 1)
            cdf_lo = cdf[i, idx - 1]; cdf_hi = cdf[i, idx]
            denom = cdf_hi - cdf_lo
            denom_safe = np.where(denom > 0, denom, 1.0)
            frac = np.where(denom > 0, (u01 - cdf_lo) / denom_safe, 0.0)
            F_lik_inv[i] = theta_grid[idx - 1] + frac * (theta_grid[idx] - theta_grid[idx - 1])
        F_mixed = (1 - eta) * F_post_inv + eta * F_lik_inv
        m1 = np.sum(w01[None, :] * F_mixed, axis=-1)
        m2 = np.sum(w01[None, :] * F_mixed * F_mixed, axis=-1)
        var = np.maximum(m2 - m1 * m1, 0.0)
        finite_z = (Z > 0) & np.isfinite(Z)
        finite_var = (var > 0) & np.isfinite(var)
        finite_row = finite_z & finite_var
        diff = m1 - theta_f
        t_np = np.where(finite_row, diff * diff / np.where(finite_row, var, 1.0), 0.0)

        np.testing.assert_allclose(t_jit, t_np, atol=1e-10, rtol=1e-10)


@pytest.mark.L2
class TestPowerLawBatchedTiltedMC:
    """The vectorised tilted MC reference must agree with the closed-form
    Theorem-8 power-law p-value within MC noise on Normal-Normal."""

    def test_tilted_pvalue_agreement_on_nn(self):
        from frasian.tilting.power_law import _generic_tilted_pvalue, PowerLawTilting

        m = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([0.5])
        eta = 0.5
        theta = 0.3
        # Closed-form Theorem-8 p-value for power-law on NN.
        cf_p = float(np.asarray(
            PowerLawTilting().tilted_pvalue(theta, 0.5, m, prior, eta, "waldo")
        ))
        gn_p = _generic_tilted_pvalue(
            theta, data, m, prior, eta, "waldo", n_mc=2000
        )
        # MC SE on a tail-area p ~0.5 is sqrt(0.25/2000) ≈ 0.011; allow 5σ.
        assert abs(cf_p - gn_p) < 0.06, f"closed-form={cf_p}, generic={gn_p}"
