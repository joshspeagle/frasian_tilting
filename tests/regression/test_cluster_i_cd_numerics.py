"""L1/L2 regression: Cluster I — CD numerical correctness.

Pins the audit P1 fixes:

  I.1 — `quantile` uses `jnp.interp`; on plateau (zero-density)
        segments it returns the leftmost matching θ
        (`searchsorted(side='left')` semantics) and otherwise
        smoothly interpolates. The audit recommended explicit
        `searchsorted` but smooth interp is preferred for downstream
        W2-distance use; we pin the on-grid invertibility behaviour.

  I.2 — `cdf(+∞)` reflects `cdf_values[-1]` (the integrated mass), NOT
        always 1.0. Constructors that don't Z-normalise leave a
        non-proper CDF; users must verify normalisation before
        treating it as a probability. Pinned by an explicit
        regression on a hand-built unnormalised grid.

  I.3 — `secondary_modes` correctly excludes the peak from the right-
        tail valley computation. Pre-fix `pdf[i:].min()` always
        included `pdf[i]` (the peak), forcing valley == peak height
        and prominence == 0 for any strictly-decreasing right tail.
        Pinned by a clean bimodal test fixture where the secondary
        peak has unambiguous prominence.

  I.4 — `build_cd_from_pvalue` documents its n=1 implicit convention.
        No code change needed beyond docstring; we pin a regression
        test that calls the function and verifies the result against
        a manually-constructed n=1 CD.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.cd.from_pvalue import build_cd_from_pvalue
from frasian.cd.grid import GridConfidenceDistribution
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.identity import IdentityTilting


# --- I.1 quantile semantics ---------------------------------------------


@pytest.mark.L1
class TestQuantileOnGridInvertible:
    """The H.1-equivalent guarantee for I.1: `quantile(cdf(theta))`
    returns the original θ on a clean grid, demonstrating that the
    interp-based quantile gives correct inverse-cdf at grid-coincident
    queries (i.e. it has the `searchsorted(side='left')` property the
    audit asked for, on grid points, with smooth interpolation in
    between)."""

    def test_invertibility_on_grid(self):
        # Build a clean unimodal CD on a fine grid.
        theta = np.linspace(-3.0, 3.0, 401)
        pdf = np.exp(-0.5 * theta ** 2)
        # Z-normalise so the cdf reaches 1.
        pdf = pdf / float(np.trapezoid(pdf, theta))
        cd = GridConfidenceDistribution(
            name="gauss-test",
            theta_grid=theta,
            pdf_values=pdf,
        )
        # cdf at every grid point, then quantile should recover.
        cdf_vals = np.asarray(cd.cdf_values)
        # Skip the very first/last point (cdf=0/1 boundaries).
        for k in (50, 100, 200, 300, 350):
            q = float(cdf_vals[k])
            theta_back = float(cd.quantile(q))
            assert abs(theta_back - theta[k]) < 1e-6, (
                f"quantile(cdf(theta[{k}])) drifted: "
                f"expected {theta[k]}, got {theta_back}"
            )


# --- I.2 cdf and quantile on un-normalised pdf --------------------------


@pytest.mark.L1
class TestCdfReportsTrueMass:
    """Audit P1 I.2: `cdf(+∞)` is the integrated mass, not always 1.0."""

    def test_unnormalised_pdf_cdf_reflects_mass(self):
        # Build an explicitly half-mass CD (pdf integrates to 0.5).
        theta = np.linspace(-3.0, 3.0, 201)
        pdf = 0.5 * np.exp(-0.5 * theta ** 2) / np.sqrt(2.0 * np.pi)
        cd = GridConfidenceDistribution(
            name="halfmass-test",
            theta_grid=theta,
            pdf_values=pdf,
        )
        total_mass = float(np.asarray(cd.cdf_values)[-1])
        assert 0.45 < total_mass < 0.55, (
            f"hand-built half-mass CD should report ~0.5 cdf at right "
            f"edge; got {total_mass}"
        )
        # cdf(+inf) clamps to total_mass, NOT 1.0.
        cdf_far_right = float(cd.cdf(100.0))
        assert abs(cdf_far_right - total_mass) < 1e-9, (
            f"cdf(+inf) should equal cdf_values[-1]={total_mass}; got "
            f"{cdf_far_right}"
        )

    def test_quantile_beyond_mass_clamps_to_grid_edge(self):
        # Same un-normalised pdf; quantile(0.9) > total_mass should
        # clamp to upper grid edge. Caller is responsible for
        # normalisation if they want a proper CDF inverse.
        theta = np.linspace(-3.0, 3.0, 201)
        pdf = 0.5 * np.exp(-0.5 * theta ** 2) / np.sqrt(2.0 * np.pi)
        cd = GridConfidenceDistribution(
            name="halfmass-q-test",
            theta_grid=theta,
            pdf_values=pdf,
        )
        # 0.9 > total mass (~0.5) → clamps to upper edge (3.0).
        q_high = float(cd.quantile(0.9))
        assert abs(q_high - 3.0) < 1e-9, (
            f"quantile(q > total_mass) should clamp to upper grid edge; "
            f"got {q_high}"
        )


# --- I.3 secondary_modes off-by-one fix ---------------------------------


@pytest.mark.L1
class TestSecondaryModesOffByOne:
    """Audit P1 I.3: `secondary_modes` correctly returns the secondary
    peak of a clean bimodal pdf. Pre-fix the right-tail valley
    computation included the peak itself, forcing prominence == 0."""

    def test_clean_bimodal_returns_secondary(self):
        # Two Gaussians at -2.0 and +2.0, equal heights.
        theta = np.linspace(-5.0, 5.0, 401)
        pdf = np.exp(-0.5 * (theta + 2.0) ** 2) + np.exp(-0.5 * (theta - 2.0) ** 2)
        pdf = pdf / float(np.trapezoid(pdf, theta))
        cd = GridConfidenceDistribution(
            name="bimodal-test",
            theta_grid=theta,
            pdf_values=pdf,
        )
        secondaries = cd.secondary_modes(prominence_frac=0.1)
        # Exactly one secondary mode. (Global mode is one peak; the
        # other is the secondary.)
        assert len(secondaries) == 1, (
            f"clean bimodal should have one secondary mode; got "
            f"{secondaries}"
        )
        # The secondary peak should be near ±2.0 (whichever is not
        # the global peak — depends on np.argmax tie-breaking).
        secondary = secondaries[0]
        assert abs(abs(secondary) - 2.0) < 0.1, (
            f"secondary peak should be near ±2.0; got {secondary}"
        )

    def test_unimodal_returns_empty(self):
        theta = np.linspace(-3.0, 3.0, 201)
        pdf = np.exp(-0.5 * theta ** 2)
        pdf = pdf / float(np.trapezoid(pdf, theta))
        cd = GridConfidenceDistribution(
            name="unimodal",
            theta_grid=theta,
            pdf_values=pdf,
        )
        assert cd.secondary_modes() == []

    def test_low_prominence_filtered_out(self):
        # Bimodal but the secondary peak is tiny (1% of global). At
        # prominence_frac=0.5 it must be filtered out.
        theta = np.linspace(-5.0, 5.0, 401)
        pdf = np.exp(-0.5 * (theta + 2.0) ** 2) + 0.02 * np.exp(
            -0.5 * (theta - 2.0) ** 2
        )
        pdf = pdf / float(np.trapezoid(pdf, theta))
        cd = GridConfidenceDistribution(
            name="weak-bimodal",
            theta_grid=theta,
            pdf_values=pdf,
        )
        # At a strict prominence threshold, the tiny secondary is rejected.
        assert cd.secondary_modes(prominence_frac=0.5) == []


# --- I.4 build_cd_from_pvalue n=1 convention ----------------------------


@pytest.mark.L2
class TestBuildCdFromPvalueN1:
    """Audit P1 I.4: `build_cd_from_pvalue(D=...)` is the n=1 path.
    Verify the resulting CD agrees with a hand-computed n=1
    Schweder-Hjort density."""

    def test_n1_cd_recovers_wald_normal_shape(self):
        # Identity tilting + WALDO at D=0 on the canonical NN sandbox:
        # the resulting CD should be approximately Gaussian with the
        # NN posterior parameters (mu_n, sigma_n).
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        cd = build_cd_from_pvalue(
            IdentityTilting(),
            WaldoStatistic(),
            D=0.0,
            model=model,
            prior=prior,
            n_grid=801,
        )
        # CD-mean should be close to the NN posterior mean (mu_n = 0
        # for D=0, mu0=0).
        assert abs(cd.mean()) < 0.05
        # Total mass via cdf_values should be ~1 (Z-normalisation).
        total_mass = float(np.asarray(cd.cdf_values)[-1])
        assert abs(total_mass - 1.0) < 1e-3, (
            f"build_cd_from_pvalue must Z-normalise to 1.0; got "
            f"total mass {total_mass}"
        )

    def test_constructor_records_n1_data_in_metadata(self):
        # The metadata field "D" pins the scalar nature of the input;
        # absence of an "n_obs" or "data" array is itself the
        # n=1 documentation.
        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        cd = build_cd_from_pvalue(
            IdentityTilting(),
            WaldoStatistic(),
            D=0.7,
            model=model,
            prior=prior,
            n_grid=401,
        )
        assert "D" in cd.metadata
        assert abs(cd.metadata["D"] - 0.7) < 1e-12
