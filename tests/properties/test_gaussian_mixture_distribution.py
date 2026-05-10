"""Property tests for GaussianMixtureDistribution.

Covers Distribution-protocol invariants. All math is for the 2-component
case (the only shape used by the framework: posterior + likelihood).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from frasian.models.distributions import GaussianMixtureDistribution


@pytest.mark.L1
@pytest.mark.properties
class TestGaussianMixtureDistribution:
    def test_two_component_construction(self):
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        assert gm.n_components == 2

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="weights"):
            GaussianMixtureDistribution(
                weights=(0.4, 0.5), means=(0.0, 2.0), scales=(1.0, 0.5)
            )

    def test_weights_must_be_non_negative(self):
        with pytest.raises(ValueError, match="weights"):
            GaussianMixtureDistribution(
                weights=(-0.1, 1.1), means=(0.0, 2.0), scales=(1.0, 0.5)
            )

    def test_scales_must_be_positive(self):
        with pytest.raises(ValueError, match="scale"):
            GaussianMixtureDistribution(
                weights=(0.5, 0.5), means=(0.0, 2.0), scales=(0.0, 0.5)
            )

    def test_mean_formula(self):
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        # E[X] = 0.3*0 + 0.7*2 = 1.4
        assert gm.mean() == pytest.approx(1.4, abs=1e-12)

    def test_var_formula(self):
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        # Var = E[Var|Z] + Var[E[X|Z]]
        #     = 0.3*1 + 0.7*0.25 + (0.3*(0-1.4)^2 + 0.7*(2-1.4)^2)
        #     = 0.475 + (0.588 + 0.252) = 0.475 + 0.84 = 1.315
        assert gm.var() == pytest.approx(1.315, abs=1e-12)

    def test_pdf_integrates_to_one(self):
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        x = np.linspace(-10.0, 12.0, 4001)
        pdf = np.asarray(gm.pdf(x), dtype=np.float64)
        Z = float(np.trapezoid(pdf, x))
        assert Z == pytest.approx(1.0, abs=1e-4)

    def test_cdf_at_minus_inf_is_zero(self):
        gm = GaussianMixtureDistribution(
            weights=(0.5, 0.5), means=(0.0, 0.0), scales=(1.0, 1.0)
        )
        assert float(gm.cdf(np.asarray(-50.0))) == pytest.approx(0.0, abs=1e-12)

    def test_cdf_at_plus_inf_is_one(self):
        gm = GaussianMixtureDistribution(
            weights=(0.5, 0.5), means=(0.0, 0.0), scales=(1.0, 1.0)
        )
        assert float(gm.cdf(np.asarray(50.0))) == pytest.approx(1.0, abs=1e-12)

    def test_cdf_monotone_increasing(self):
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        x = np.linspace(-5.0, 7.0, 200)
        cdf = np.asarray(gm.cdf(x), dtype=np.float64)
        assert np.all(np.diff(cdf) >= -1e-15)

    def test_cdf_quantile_round_trip(self):
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        u = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        x = np.asarray(gm.quantile(u), dtype=np.float64)
        cdf_back = np.asarray(gm.cdf(x), dtype=np.float64)
        np.testing.assert_allclose(cdf_back, u, atol=1e-8)

    def test_sample_empirical_matches_analytical(self):
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        rng = np.random.default_rng(0xBEEF)
        x = gm.sample(rng, 100000)
        # SE of empirical mean is sqrt(var/n) = sqrt(1.315/1e5) ~ 0.0036.
        # Allow 5 SE tolerance.
        assert abs(float(np.mean(x)) - gm.mean()) < 5 * 0.0036

    def test_pdf_logpdf_consistency(self):
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        x = np.linspace(-3.0, 5.0, 50)
        pdf = np.asarray(gm.pdf(x), dtype=np.float64)
        logpdf = np.asarray(gm.logpdf(x), dtype=np.float64)
        np.testing.assert_allclose(np.exp(logpdf), pdf, atol=1e-12)

    def test_unimodal_when_means_close(self):
        # |mu1 - mu2| = 1 < 2*min(sigma1, sigma2) = 2 -> unimodal (Behboodian 1970).
        gm = GaussianMixtureDistribution(
            weights=(0.5, 0.5), means=(0.0, 1.0), scales=(1.0, 1.0)
        )
        x = np.linspace(-5.0, 6.0, 1001)
        pdf = np.asarray(gm.pdf(x), dtype=np.float64)
        # Count local maxima.
        diffs = np.diff(pdf)
        sign_changes = np.diff(np.sign(diffs))
        local_maxima = np.sum(sign_changes < 0)
        assert local_maxima == 1, f"expected 1 mode, got {local_maxima}"

    def test_bimodal_when_means_far(self):
        # |mu1 - mu2| = 6 > 2*min(sigma) = 2 -> bimodal.
        gm = GaussianMixtureDistribution(
            weights=(0.5, 0.5), means=(0.0, 6.0), scales=(1.0, 1.0)
        )
        x = np.linspace(-5.0, 11.0, 2001)
        pdf = np.asarray(gm.pdf(x), dtype=np.float64)
        diffs = np.diff(pdf)
        sign_changes = np.diff(np.sign(diffs))
        local_maxima = np.sum(sign_changes < 0)
        assert local_maxima == 2, f"expected 2 modes, got {local_maxima}"

    @given(
        w=st.floats(min_value=0.05, max_value=0.95),
        m0=st.floats(min_value=-3.0, max_value=3.0),
        m1=st.floats(min_value=-3.0, max_value=3.0),
        s0=st.floats(min_value=0.2, max_value=3.0),
        s1=st.floats(min_value=0.2, max_value=3.0),
    )
    @settings(deadline=None, max_examples=30)
    def test_cdf_quantile_roundtrip_property(self, w, m0, m1, s0, s1):
        gm = GaussianMixtureDistribution(
            weights=(w, 1.0 - w), means=(m0, m1), scales=(s0, s1)
        )
        u = np.array([0.1, 0.5, 0.9])
        x = np.asarray(gm.quantile(u), dtype=np.float64)
        cdf_back = np.asarray(gm.cdf(x), dtype=np.float64)
        np.testing.assert_allclose(cdf_back, u, atol=1e-7)


@pytest.mark.L1
@pytest.mark.properties
class TestMixtureDistributionGeneric:
    """MixtureDistribution accepts any 2 Distribution-protocol endpoints."""

    def test_two_normals_match_gaussian_mixture(self):
        from frasian.models.distributions import (
            MixtureDistribution,
            NormalDistribution,
        )
        p = NormalDistribution(loc=0.0, scale=1.0)
        q = NormalDistribution(loc=2.0, scale=0.5)
        gm = GaussianMixtureDistribution(
            weights=(0.3, 0.7), means=(0.0, 2.0), scales=(1.0, 0.5)
        )
        md = MixtureDistribution(weights=(0.3, 0.7), components=(p, q))
        x = np.linspace(-3.0, 5.0, 50)
        np.testing.assert_allclose(
            np.asarray(md.pdf(x)), np.asarray(gm.pdf(x)), atol=1e-12
        )
        assert md.mean() == pytest.approx(gm.mean(), abs=1e-12)
        # Variance via 64-pt Gauss-Legendre (numerical) vs closed-form GM.
        assert md.var() == pytest.approx(gm.var(), abs=1e-3)

    def test_cdf_monotone(self):
        from frasian.models.distributions import (
            MixtureDistribution,
            NormalDistribution,
        )
        p = NormalDistribution(loc=0.0, scale=1.0)
        q = NormalDistribution(loc=2.0, scale=0.5)
        md = MixtureDistribution(weights=(0.4, 0.6), components=(p, q))
        x = np.linspace(-5.0, 7.0, 200)
        cdf = np.asarray(md.cdf(x), dtype=np.float64)
        assert np.all(np.diff(cdf) >= -1e-12)

    def test_quantile_round_trip(self):
        from frasian.models.distributions import (
            MixtureDistribution,
            NormalDistribution,
        )
        p = NormalDistribution(loc=0.0, scale=1.0)
        q = NormalDistribution(loc=2.0, scale=0.5)
        md = MixtureDistribution(weights=(0.4, 0.6), components=(p, q))
        u = np.array([0.1, 0.5, 0.9])
        x = np.asarray(md.quantile(u), dtype=np.float64)
        cdf_back = np.asarray(md.cdf(x), dtype=np.float64)
        np.testing.assert_allclose(cdf_back, u, atol=1e-7)

    def test_weights_must_sum_to_one(self):
        from frasian.models.distributions import (
            MixtureDistribution,
            NormalDistribution,
        )
        p = NormalDistribution(loc=0.0, scale=1.0)
        q = NormalDistribution(loc=2.0, scale=0.5)
        with pytest.raises(ValueError, match="weights"):
            MixtureDistribution(weights=(0.3, 0.5), components=(p, q))
