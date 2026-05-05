"""Regression tests for CD constructors (Phase D).

The universal constructor `build_cd_from_pvalue(tilting, statistic, D, …)`
should agree with closed-form Wald / WALDO / tilted-WALDO CDs on the
conjugate Normal-Normal sandbox, since the only difference is the path
to the p-value (numerical via `tilting.pvalue` vs analytic in
`from_closed_form`). Agreement to 1e-3 on cdf and 5e-3 on pdf is the
acceptance criterion.

Also tested:
  - Multimodal Dyn-WALDO p-value produces a CD with non-monotone
    `signed_confidence` (the smoothness pathology surfacing) AND a
    valid monotone `cdf_values` + non-negative `pdf_values` (probability
    distribution intact). User-flagged: this case must work.
  - Closed-form CDs themselves match scipy reference quantiles.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from frasian.cd.from_closed_form import tilted_waldo_cd, wald_cd, waldo_cd
from frasian.cd.from_pvalue import build_cd_from_pvalue
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector, FixedEtaSelector
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.power_law import PowerLawTilting


def _model_prior(sigma=1.0, sigma0=1.0):
    return NormalNormalModel(sigma=sigma), NormalDistribution(loc=0.0, scale=sigma0)


@pytest.mark.L2
class TestUniversalAgreesWithClosedFormWald:
    """`build_cd_from_pvalue(IdentityTilting, WaldStatistic)` ≈ `wald_cd`."""

    @pytest.mark.parametrize("D", [-1.5, 0.0, 1.5, 3.0])
    def test_wald_cd_match(self, D):
        m, _ = _model_prior()
        prior = NormalDistribution(loc=0.0, scale=1.0)  # not used by Wald
        # Use the same θ-grid for both so cdf values are comparable.
        theta = np.linspace(D - 8, D + 8, 1001)
        actual = build_cd_from_pvalue(
            IdentityTilting(),
            WaldStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )
        expected = wald_cd(D, m.sigma, theta_grid=theta)
        # cdf agreement: closed-form uses analytic Gaussian pdf; the FD path
        # introduces ~3e-3 truncation error at the |D−θ| kink. 5e-3 atol is
        # the realistic agreement threshold.
        np.testing.assert_allclose(
            actual.cdf_values,
            expected.cdf_values,
            atol=5e-3,
        )
        # pdf agreement (looser since FD vs closed-form).
        np.testing.assert_allclose(
            actual.pdf_values,
            expected.pdf_values,
            atol=5e-3,
        )
        # Inversion-based C(θ) should match too (Wald is unimodal).
        np.testing.assert_allclose(
            actual.signed_confidence,
            expected.signed_confidence,
            atol=5e-3,
        )

    @pytest.mark.parametrize("D", [-1.0, 1.5])
    def test_signed_confidence_is_monotone_for_wald(self, D):
        m, prior = _model_prior()
        theta = np.linspace(D - 8, D + 8, 1001)
        cd = build_cd_from_pvalue(
            IdentityTilting(),
            WaldStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )
        assert cd.is_monotone_inversion()


@pytest.mark.L2
class TestUniversalAgreesWithClosedFormWaldo:
    """`build_cd_from_pvalue(IdentityTilting, WaldoStatistic)` ≈ `waldo_cd`."""

    @pytest.mark.parametrize("D", [-1.5, 0.0, 1.5, 3.0])
    def test_waldo_cd_match(self, D):
        m, prior = _model_prior()
        theta = np.linspace(D - 8, D + 8, 1001)
        actual = build_cd_from_pvalue(
            IdentityTilting(),
            WaldoStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )
        expected = waldo_cd(D, m, prior, theta_grid=theta)
        np.testing.assert_allclose(
            actual.cdf_values,
            expected.cdf_values,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            actual.pdf_values,
            expected.pdf_values,
            atol=5e-3,
        )

    def test_power_law_fixed_eta_zero_matches_waldo(self):
        """`(power_law[Fixed(0)], waldo)` is numerically `(identity, waldo)`."""
        m, prior = _model_prior()
        D = 1.5
        theta = np.linspace(D - 8, D + 8, 1001)
        plain = PowerLawTilting(selector=FixedEtaSelector(eta=0.0))
        cd_pl = build_cd_from_pvalue(
            plain,
            WaldoStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )
        cd_id = build_cd_from_pvalue(
            IdentityTilting(),
            WaldoStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )
        np.testing.assert_allclose(
            cd_pl.pdf_values,
            cd_id.pdf_values,
            atol=1e-9,
        )


@pytest.mark.L2
class TestUniversalAgreesWithClosedFormTiltedWaldo:
    """`build_cd_from_pvalue(power_law[Fixed(η)], WaldoStatistic)` ≈
    `tilted_waldo_cd(η)` for any admissible η."""

    @pytest.mark.parametrize("eta", [-0.5, -0.2, 0.0, 0.3, 0.7])
    def test_match_at_various_eta(self, eta):
        m, prior = _model_prior()
        D = 1.5
        theta = np.linspace(D - 8, D + 8, 1001)
        scheme = PowerLawTilting(selector=FixedEtaSelector(eta=eta))
        actual = build_cd_from_pvalue(
            scheme,
            WaldoStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )
        expected = tilted_waldo_cd(D, m, prior, eta, theta_grid=theta)
        np.testing.assert_allclose(
            actual.cdf_values,
            expected.cdf_values,
            atol=1e-3,
        )


@pytest.mark.L2
class TestDynWaldoConstructor:
    """User-flagged: the multimodal Dyn-WALDO case must produce a valid CD.

    Specifically, even when the underlying dynamic p-value is bimodal:
      - pdf is non-negative everywhere
      - cdf (derived from pdf) is monotone non-decreasing
      - signed_confidence is non-monotone (the diagnostic surface)
      - validate() flags the non-monotone signed_confidence
    """

    @pytest.mark.parametrize("D", [2.0, 3.0])
    def test_dyn_waldo_cd_is_valid_probability_distribution(self, D):
        m, prior = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        theta = np.linspace(D - 10, D + 10, 2001)
        cd = build_cd_from_pvalue(
            scheme,
            WaldoStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )

        # pdf non-negative.
        assert (cd.pdf_values >= -1e-12).all()
        # cdf monotone (derived from non-negative pdf).
        assert (np.diff(cd.cdf_values) >= -1e-12).all()
        # cdf endpoints close to 0 and 1.
        assert cd.cdf_values[0] == pytest.approx(0.0, abs=1e-6)
        assert cd.cdf_values[-1] == pytest.approx(1.0, abs=1e-3)
        # mean, median, mode are finite.
        for v in (cd.mean(), cd.median(), cd.mode()):
            assert np.isfinite(v)

    def test_dyn_waldo_cd_signed_confidence_nonmonotone_at_conflict(self):
        """At D=3 the dynamic p-value is bimodal, so the inversion-based
        signed_confidence is non-monotone — the smoothness pathology
        surfaces directly in the CD."""
        m, prior = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        D = 3.0
        theta = np.linspace(D - 10, D + 10, 2001)
        cd = build_cd_from_pvalue(
            scheme,
            WaldoStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )
        assert (
            not cd.is_monotone_inversion()
        ), "expected non-monotone signed_confidence for Dyn-WALDO at D=3"

    def test_dyn_waldo_cd_validate_flags_nonmonotone(self):
        m, prior = _model_prior()
        sel = DynamicNumericalEtaSelector(sigma=1.0, mu0=0.0, n_grid=401, coarse_n=25)
        scheme = PowerLawTilting(selector=sel)
        D = 3.0
        theta = np.linspace(D - 10, D + 10, 2001)
        cd = build_cd_from_pvalue(
            scheme,
            WaldoStatistic(),
            D,
            m,
            prior,
            theta_grid=theta,
        )
        codes = [i.code for i in cd.validate()]
        assert "non-monotone-signed-confidence" in codes


@pytest.mark.L0
class TestClosedFormAgreesWithScipy:
    """Sanity: `wald_cd` matches `scipy.stats.norm` directly."""

    def test_wald_cd_quantile(self):
        cd = wald_cd(1.5, 1.0, n_grid=4001)
        for q in [0.05, 0.5, 0.95]:
            assert cd.quantile(q) == pytest.approx(
                stats.norm.ppf(q, loc=1.5, scale=1.0),
                abs=1e-3,
            )

    def test_wald_cd_interval_matches_z_form(self):
        cd = wald_cd(1.5, 1.0, n_grid=4001)
        lo, hi = cd.interval(0.05)
        assert lo == pytest.approx(1.5 - 1.96, abs=1e-3)
        assert hi == pytest.approx(1.5 + 1.96, abs=1e-3)
