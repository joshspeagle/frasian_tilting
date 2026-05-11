"""Property tests for FisherRaoTilting (Stage A — NN closed-form geodesic + quadrature p-value).

Mirrors invariants in `docs/methods/fisher_rao.md`. These are all *active*
now (the stub raise-checks under TestFisherRaoStubActuallyRaises are
removed because the stub is being replaced). Rev 1 corrections applied:
- Test atols 1e-10 → 1e-3 for near-endpoint quadrature (1e-12 for exact-endpoint closed-form fast paths)
- KS calibration becomes load-bearing
- New tests: test_pvalue_near_endpoints_quadrature_residual, test_quadrature_converges_with_grid
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from frasian.models.distributions import GaussianLikelihood, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.fisher_rao import FisherRaoTilting
from frasian.tilting.ot import OTTilting


def _nn_fixtures(sigma=1.0, sigma0=1.0, mu0=0.0, D=0.5):
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    lik = GaussianLikelihood(D=D, sigma=sigma)
    post = model.posterior(np.asarray([D]), prior)
    return model, prior, lik, post


@pytest.mark.L1
@pytest.mark.properties
class TestFisherRaoInvariants:
    def test_identity_at_eta_zero(self):
        """tilt(eta=0) returns the posterior N(mu_n, sigma_n^2) to atol 1e-12."""
        model, prior, lik, post = _nn_fixtures(sigma=1.0, sigma0=1.0, mu0=0.0, D=0.5)
        scheme = FisherRaoTilting()
        out = scheme.tilt(post, prior, lik, 0.0)
        assert isinstance(out, NormalDistribution)
        assert np.isclose(out.loc, post.loc, atol=1e-12)
        assert np.isclose(out.scale, post.scale, atol=1e-12)

    def test_endpoint_at_eta_one(self):
        """tilt(eta=1) returns N(D, sigma^2)."""
        model, prior, lik, post = _nn_fixtures()
        scheme = FisherRaoTilting()
        out = scheme.tilt(post, prior, lik, 1.0)
        assert np.isclose(out.loc, lik.D, atol=1e-12)
        assert np.isclose(out.scale, lik.sigma, atol=1e-12)

    @given(
        sigma=st.floats(min_value=0.1, max_value=5.0),
        sigma0=st.floats(min_value=0.1, max_value=5.0),
        mu0=st.floats(min_value=-3.0, max_value=3.0),
        D=st.floats(min_value=-3.0, max_value=3.0),
        eta=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_sigma_positive_along_path(self, sigma, sigma0, mu0, D, eta):
        """sigma(t) > 0 for all t in [0, 1] and reasonable inputs."""
        _, prior, lik, post = _nn_fixtures(sigma=sigma, sigma0=sigma0, mu0=mu0, D=D)
        out = FisherRaoTilting().tilt(post, prior, lik, eta)
        assert out.scale > 0.0

    def test_continuous_in_eta(self):
        """W1 distance between tilt(eta) and tilt(eta+h) is O(h)."""
        _, prior, lik, post = _nn_fixtures()
        scheme = FisherRaoTilting()
        a = scheme.tilt(post, prior, lik, 0.4)
        b = scheme.tilt(post, prior, lik, 0.401)
        w1 = abs(a.loc - b.loc) + abs(a.scale - b.scale)
        assert w1 < 0.05

    @pytest.mark.parametrize(
        "sigma, sigma0, mu0, D",
        [
            (1.0, 1.0, 0.0, 0.5),
            (2.0, 0.5, 1.0, -1.0),
            (0.5, 2.0, 0.0, 2.0),
        ],
    )
    def test_arc_length_matches_costa_2015(self, sigma, sigma0, mu0, D):
        """Path arc-length matches Costa et al. 2015 Eqs. 5-6 closed form."""
        from frasian.tilting.fisher_rao import (
            _fr_arc_length_costa,
            _fr_geodesic_arc_length_numerical,
        )
        _, prior, lik, post = _nn_fixtures(sigma, sigma0, mu0, D)
        mu_a, s_a = post.loc, post.scale
        mu_b, s_b = lik.D, lik.sigma
        d_costa = _fr_arc_length_costa(mu_a, s_a, mu_b, s_b)
        d_numerical = _fr_geodesic_arc_length_numerical(mu_a, s_a, mu_b, s_b, n_steps=10000)
        assert np.isclose(d_numerical, d_costa, atol=1e-9)

    def test_vertical_case_reduces_to_geometric_mean(self):
        """When mu_a = mu_b, sigma(t) = sigma_a^(1-t) * sigma_b^t."""
        from frasian.tilting.fisher_rao import _fr_geodesic_gaussian_scalar
        mu_a, s_a = 0.0, 1.0
        mu_b, s_b = 0.0, 2.0
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mu_t, s_t = _fr_geodesic_gaussian_scalar(mu_a, s_a, mu_b, s_b, t)
            assert np.isclose(mu_t, 0.0, atol=1e-12)
            assert np.isclose(s_t, s_a ** (1 - t) * s_b ** t, atol=1e-12)

    def test_tilted_pvalue_scalar_returns_python_float(self):
        """Scalar fast path returns float, not jax.Array."""
        from frasian.tilting.fisher_rao import _fr_tilted_pvalue_numpy_scalar
        p = _fr_tilted_pvalue_numpy_scalar(
            theta_f=0.0, eta_f=0.5, D_f=0.5, w=0.5, mu0=0.0, sigma=1.0,
            statistic_name="waldo",
        )
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_pvalue_reduces_to_bare_waldo_at_eta_zero(self):
        """At exact eta=0 the closed-form bare-WALDO fast path activates; atol 1e-12."""
        model, prior, lik, post = _nn_fixtures()
        scheme = FisherRaoTilting()
        bare_waldo = WaldoStatistic()
        theta_grid = np.linspace(-2.0, 2.0, 11)
        for theta in theta_grid:
            p_fr = scheme.tilted_pvalue(
                theta=theta, D=lik.D, model=model, prior=prior, eta=0.0,
                statistic_name="waldo",
            )
            p_bare = bare_waldo.pvalue(np.array([theta]), np.array([lik.D]),
                                        model=model, prior=prior)[0]
            assert np.isclose(float(p_fr), float(p_bare), atol=1e-12)

    def test_pvalue_reduces_to_bare_wald_at_eta_one(self):
        """At exact eta=1 the closed-form bare-Wald fast path activates; atol 1e-12."""
        model, prior, lik, post = _nn_fixtures()
        scheme = FisherRaoTilting()
        theta_grid = np.linspace(-2.0, 2.0, 11)
        for theta in theta_grid:
            p_fr = scheme.tilted_pvalue(
                theta=theta, D=lik.D, model=model, prior=prior, eta=1.0,
                statistic_name="waldo",
            )
            from scipy.stats import norm
            p_wald = 2 * norm.sf(abs(lik.D - theta) / lik.sigma)
            assert np.isclose(float(p_fr), float(p_wald), atol=1e-12)

    def test_pvalue_near_endpoints_quadrature_residual(self):
        """At eta=1e-6 (just inside quadrature path), p_FR matches bare WALDO to atol 1e-3.

        Trapezoidal quadrature truncation residual at n_grid=256.
        """
        model, prior, lik, post = _nn_fixtures()
        scheme = FisherRaoTilting()
        bare_waldo = WaldoStatistic()
        theta_grid = np.linspace(-1.5, 1.5, 7)
        for theta in theta_grid:
            p_fr = scheme.tilted_pvalue(
                theta=theta, D=lik.D, model=model, prior=prior, eta=1e-6,
                statistic_name="waldo",
            )
            p_bare = bare_waldo.pvalue(np.array([theta]), np.array([lik.D]),
                                        model=model, prior=prior)[0]
            assert np.isclose(float(p_fr), float(p_bare), atol=1e-3)

    def test_quadrature_converges_with_grid(self):
        """Adaptive quadrature with brentq boundary finding is essentially
        machine-precision regardless of n_grid (coarse grid only used for
        sign-change discovery; brentq xtol=1e-12 refines roots to machine
        precision). So coarse and fine grids should agree to ~1e-10.
        """
        from frasian.tilting.fisher_rao import _fr_tilted_pvalue_numpy_scalar
        kwargs = dict(theta_f=0.3, eta_f=0.5, D_f=0.5, w=0.5, mu0=0.0,
                      sigma=1.0, statistic_name="waldo")
        p_default = _fr_tilted_pvalue_numpy_scalar(**kwargs)  # default n_grid=64
        p_dense = _fr_tilted_pvalue_numpy_scalar(**kwargs, n_grid=512)
        # Should agree to brentq xtol precision
        assert abs(p_default - p_dense) < 1e-10

    @pytest.mark.L3
    def test_calibration_under_h0(self):
        """KS uniformity of FR tilted-WALDO p-values under H0 (theta=theta_true).

        Rev 1 finding: this is the LOAD-BEARING correctness check for FR
        since algebraic-equality cross-checks have intrinsic quadrature noise.
        """
        rng = np.random.default_rng(42)
        n_replicates = 1000
        theta_true = 0.5
        sigma = 1.0
        sigma0 = 1.0
        mu0 = 0.0
        scheme = FisherRaoTilting()
        pvals = np.empty(n_replicates)
        for i in range(n_replicates):
            D_obs = theta_true + sigma * rng.standard_normal()
            model = NormalNormalModel(sigma=sigma)
            prior = NormalDistribution(loc=mu0, scale=sigma0)
            p = scheme.tilted_pvalue(
                theta=theta_true, D=D_obs, model=model, prior=prior, eta=0.5,
                statistic_name="waldo",
            )
            pvals[i] = float(p)
        from scipy.stats import kstest
        ks_stat, ks_p = kstest(pvals, "uniform")
        assert ks_p > 0.01, f"KS test rejected uniformity (ks_stat={ks_stat:.4f}, p={ks_p:.4f})"

    @pytest.mark.parametrize(
        "sigma, sigma0, mu0, D, eta",
        [
            (1.0, 0.3, 0.0, 1.5, 0.5),
            (2.0, 0.5, 1.0, -1.0, 0.3),
            (0.5, 2.0, 0.0, 2.0, 0.7),
        ],
    )
    def test_differs_from_ot_when_sigma_a_neq_sigma_b(self, sigma, sigma0, mu0, D, eta):
        """When prior and likelihood scales differ, FR (mu_t, sigma_t) != OT linear interp."""
        _, prior, lik, post = _nn_fixtures(sigma, sigma0, mu0, D)
        if np.isclose(post.scale, lik.sigma):
            pytest.skip("equal-sigma case -- FR and OT coincide here")
        fr = FisherRaoTilting().tilt(post, prior, lik, eta)
        ot = OTTilting().tilt(post, prior, lik, eta)
        delta = abs(fr.loc - ot.loc) + abs(fr.scale - ot.scale)
        assert delta > 1e-3


@pytest.mark.L2
@pytest.mark.regression
class TestFisherRaoJaxKernel:
    @pytest.mark.parametrize("eta", [0.1, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("theta", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_jax_kernel_matches_numpy_scalar_waldo(self, eta, theta):
        """JAX kernel (trap rule, n=8000) agrees with numpy scalar (adaptive
        brentq, machine precision) to atol 1e-3. The asymmetry is intentional:
        - Numpy scalar uses brentq boundary-finding + analytical Gaussian
          CDF integration — essentially machine-precision.
        - JAX kernel uses trap-rule quadrature on the discontinuous indicator;
          O(1/n) convergence ⇒ ~4e-4 precision at n=8000. Suitable for
          learned-eta gradient signal but not for production CI inversion.
        """
        from frasian.tilting.fisher_rao import (
            _fr_tilted_pvalue_kernel, _fr_tilted_pvalue_numpy_scalar,
        )
        import jax.numpy as jnp
        D, w, mu0, sigma = 0.5, 0.5, 0.0, 1.0
        p_np = _fr_tilted_pvalue_numpy_scalar(
            theta_f=theta, eta_f=eta, D_f=D, w=w, mu0=mu0, sigma=sigma,
            statistic_name="waldo",
        )
        p_jax = float(_fr_tilted_pvalue_kernel(
            jnp.array(theta), jnp.array(eta), jnp.array(D),
            jnp.array(w), jnp.array(mu0), jnp.array(sigma),
            "waldo",
        ))
        assert np.isclose(p_np, p_jax, atol=1e-3)
