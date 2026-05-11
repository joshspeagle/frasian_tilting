"""Regression: fr_dyn_numerical_generic matches fr_dyn_numerical on NN.

Validates the Stage B autodiff/diffrax machinery (_generic_tilt_fr +
_generic_tilted_pvalue_fr + _generic_tilted_confidence_interval_fr,
B.5-B.6) against the Stage A closed-form path. The generic path uses:
- _fr_geodesic_numerical: diffrax Tsit5 + Newton shooting BVP
- generic-MC sampling for the WALDO p-value (matches PL/MX/OT pattern)

The closed-form path uses:
- _fr_geodesic_gaussian_scalar: closed-form half-plane arc
- adaptive brentq quadrature for the WALDO p-value (Stage A.5)

Both routes should produce the same CI on Normal-Normal up to:
- ~10-12% width tolerance (MC noise + diffrax integration error)
- ~8pp coverage tolerance (1-sigma at n=15)

## Performance note

The generic path's per-CI cost is dominated by the diffrax shooting BVP
(~100-500 ms per geodesic) × the brentq inner loop (~20 iterations) ≈
2-10 s per CI. The closed-form path is ~100 µs per CI. So each parity
cell at n=15 replicates is ~30-150 s of CPU work, and the full
3-cell matrix is ~2-8 min.

Marked with `@pytest.mark.slow` so it does not run in default pytest;
use `pytest -m slow` to invoke explicitly. The math validations in
tests/properties/test_fisher_rao_invariants.py (B.5/B.6 closed-form-
match tests at atol 1e-7) are the fast, default-run correctness
gates.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from scripts.run_wald_audit import _build_cell


@pytest.mark.L4
@pytest.mark.regression
class TestFrGenericMatchesClosedForm:
    """Coverage + width parity between fr_dyn_numerical and fr_dyn_numerical_generic.

    Smaller test matrix than originally specified (3 cells × n=15 replicates
    instead of 9 × 50) because each diffrax shooting BVP is ~2-10 s.
    The reduced design keeps the full test under ~10 min wall-clock at the
    cost of ~7-8pp coverage SE — appropriate for a sanity-check, not a
    tight calibration regression.

    Three representative cells span the (w) spectrum at theta_true=0:
        w=0.3 (weak prior), w=0.5 (balanced), w=0.8 (strong prior).
    The conflict axis (theta_true) is sampled at θ=0 only; widening to
    multiple theta_true values is a follow-up if the audit surfaces
    regime-specific issues.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("w", [0.3, 0.5, 0.8])
    def test_coverage_and_width_parity_at_theta_zero(self, w):
        """At each w ∈ {0.3, 0.5, 0.8} with theta_true=0, both paths
        produce CIs that agree on coverage within +/- 0.08 and width
        within +/- 12%.

        Tolerances are looser than the 5pp / 10% in the original B.7
        spec because n=15 replicates (vs n=50) inflates MC noise by
        sqrt(50/15) ≈ 1.8x. With 8pp / 12% bounds the test has good
        power to catch genuine mismatch (~20-30%+ width difference or
        ~15pp+ coverage difference) but tolerates ordinary MC variance.
        """
        sigma = 1.0
        sigma0 = sigma * np.sqrt(w / (1.0 - w))
        theta_true = 0.0
        n_replicates = 15

        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=sigma0)
        rng = np.random.default_rng(42)

        sch_cf, stat_cf, _ = _build_cell("fr_dyn_numerical")
        sch_gn, stat_gn, _ = _build_cell("fr_dyn_numerical_generic")

        cf_widths = []
        gn_widths = []
        cf_covered = 0
        gn_covered = 0
        for _ in range(n_replicates):
            D = theta_true + sigma * rng.standard_normal()
            data = np.array([D])
            ci_cf = sch_cf.confidence_interval(0.05, data, model, prior, stat_cf)
            ci_gn = sch_gn.confidence_interval(0.05, data, model, prior, stat_gn)
            cf_widths.append(ci_cf[1] - ci_cf[0])
            gn_widths.append(ci_gn[1] - ci_gn[0])
            cf_covered += int(ci_cf[0] <= theta_true <= ci_cf[1])
            gn_covered += int(ci_gn[0] <= theta_true <= ci_gn[1])

        coverage_cf = cf_covered / n_replicates
        coverage_gn = gn_covered / n_replicates
        width_cf = float(np.mean(cf_widths))
        width_gn = float(np.mean(gn_widths))

        assert abs(coverage_gn - coverage_cf) < 0.08, (
            f"Coverage mismatch at (theta=0, w={w}, n={n_replicates}): "
            f"cf={coverage_cf:.3f} vs gn={coverage_gn:.3f}"
        )
        assert abs(width_gn - width_cf) / width_cf < 0.12, (
            f"Width mismatch at (theta=0, w={w}, n={n_replicates}): "
            f"cf={width_cf:.4f} vs gn={width_gn:.4f}"
        )

    def test_force_generic_dispatch_constructible(self):
        """Default-run gate: the `fr_dyn_numerical_generic` audit flavor
        constructs without error and routes through `force_generic=True`.

        Does NOT execute a CI — even one CI through the generic path
        takes minutes due to the FR-tilted reference being non-linear
        in replicate X (every MC replicate requires its own diffrax
        shooting BVP; deriver finding from Stage A). For the slow
        coverage-and-width parity matrix, see
        `test_coverage_and_width_parity_at_theta_zero` (marked slow).
        For the math validation (atol 1e-7 closed-form match), see
        tests/properties/test_fisher_rao_invariants.py::TestFisherRaoGenericTilt
        from B.6.

        This test confirms the wiring (cell construction, statistic
        flag, scheme selector) — not numerical agreement. The
        numerical agreement IS validated, but in the B.6 tests at
        atol 1e-7 *without* the MC sampling layer, which is the
        right place because the shooting BVP is deterministic and
        the MC is just noise.
        """
        # Audit-flavor construction
        sch_cf, stat_cf, bare_cf = _build_cell("fr_dyn_numerical")
        sch_gn, stat_gn, bare_gn = _build_cell("fr_dyn_numerical_generic")

        # Closed-form path: force_generic=False
        assert stat_cf.force_generic is False, (
            f"fr_dyn_numerical should have force_generic=False; got {stat_cf.force_generic!r}"
        )
        # Generic path: force_generic=True
        assert stat_gn.force_generic is True, (
            f"fr_dyn_numerical_generic should have force_generic=True; got {stat_gn.force_generic!r}"
        )
        # Both use the same scheme class with the same selector type
        assert type(sch_cf).__name__ == "FisherRaoTilting"
        assert type(sch_gn).__name__ == "FisherRaoTilting"
        assert type(sch_cf.selector).__name__ == "DynamicNumericalEtaSelector"
        assert type(sch_gn.selector).__name__ == "DynamicNumericalEtaSelector"
        # Bare smoothness override is a FisherRaoTilting() (no selector arg)
        assert type(bare_cf).__name__ == "FisherRaoTilting"
        assert type(bare_gn).__name__ == "FisherRaoTilting"
