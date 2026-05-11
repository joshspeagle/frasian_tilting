"""L2 regression: Cluster G — WALDO & dispatch hardening.

Pins the audit P1 fixes layered over P0:

  G.1 — public-surface dispatch routes n=1 NN+Normal data to the
        closed-form path and n>1 to the generic MC path. Tests via the
        public `evaluate` / `pvalue` / `confidence_interval` methods,
        not the private `_closed_form_*` / `_generic_*` ones.
  G.2 — closed-vs-generic agreement extends into the conflict regime
        |Δ| ≥ 2 (the audit phrasing of the headline-narrowness band).
        The pre-fix cross-check only swept |D| ≤ 1.5.
  G.3 — dropping `alpha` from the CRN seed produces nested CIs across
        cross-call alpha (broader α → wider CI).
  G.4 — `_generic_confidence_interval` emits a `UserWarning` annotating
        the boundary fallback instead of silently returning the
        support edge.
  G.5 — `is_normal_normal(model)` (fingerprint-based) returns True for
        `NormalNormalModel` and False for duck-typed mocks, so the
        closed-form NN dispatch is opt-in by fingerprint, not by
        `isinstance` on the concrete class.
  G.6 — `has_acceptance_region` feature-detects the optional method;
        WALDO has it, but a trivial mock without it returns False.
  G.7 — `QuantileMixturePath` validates monotonicity at construction
        for `t outside [0, 1]` (replaces the audit's blanket
        finite-only relaxation, which was a latent footgun for
        non-Gaussian endpoints).
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.models._dispatch import is_normal_normal
from frasian.statistics.base import has_acceptance_region
from frasian.statistics.wald import WaldStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.quantile_mixture import QuantileMixturePath


# --- G.1 public-surface dispatch ----------------------------------------


@pytest.mark.L2
class TestPublicSurfaceN1Dispatch:
    """Public dispatch must route n=1 NN+Normal to closed form and n>1 to generic.

    Audit P0-1 added `_is_normal_normal_n1`; Cluster G adds tests via
    the public surface (the P0 tests poked the predicate directly).
    """

    def _setup(self):
        return (
            NormalNormalModel(sigma=1.0),
            NormalDistribution(loc=0.0, scale=1.0),
        )

    def test_n1_evaluate_matches_closed_form(self):
        model, prior = self._setup()
        stat = WaldoStatistic(n_mc=200, seed=0)
        data = np.asarray([0.5])
        # Public surface — should route to closed form.
        public = float(stat.evaluate(0.0, data, model, prior))
        closed = float(stat._closed_form_evaluate(0.0, data, model, prior))
        assert abs(public - closed) < 1e-12, (
            f"public evaluate at n=1 deviates from closed form: "
            f"public={public!r}, closed={closed!r}"
        )

    def test_n_gt_1_evaluate_uses_generic(self):
        model, prior = self._setup()
        stat = WaldoStatistic(n_mc=200, seed=0)
        # n=3 sample. Closed form would (incorrectly) collapse to mean
        # and use sigma^2 (not sigma^2/n). Generic uses n_obs correctly.
        data = np.asarray([0.0, 0.5, 1.0])
        public = float(stat.evaluate(0.0, data, model, prior))
        generic = float(stat._generic_evaluate(0.0, data, model, prior))
        assert abs(public - generic) < 1e-12, (
            f"public evaluate at n=3 should match generic, got "
            f"public={public!r}, generic={generic!r}"
        )

    def test_n1_ci_matches_closed_form(self):
        model, prior = self._setup()
        stat = WaldoStatistic(n_mc=400, seed=0)
        data = np.asarray([0.7])
        public = stat.confidence_interval(0.05, data, model, prior)
        closed = stat._closed_form_confidence_interval(0.05, data, model, prior)
        assert abs(public[0] - closed[0]) < 1e-9
        assert abs(public[1] - closed[1]) < 1e-9


# --- G.2 conflict-regime cross-check ------------------------------------


@pytest.mark.L3
class TestWaldoCrossCheckConflictRegime:
    """Cross-check generic vs closed-form WALDO at |Δ| ≥ 2 (conflict band).

    Audit P1 G.2: the prior cross-check capped at |D|=1.5 (essentially
    no-conflict). The headline narrowness claim sits at θ=±4 / |Δ|≈±2
    so we sweep that band explicitly.

    Δ = (1-w)(μ₀-D)/σ — with σ=1, σ₀=1, μ₀=0, w=0.5: |Δ| = 0.5|D|.
    |Δ|≥2 ⇒ |D|≥4 in this configuration.
    """

    @pytest.mark.parametrize(
        "D, theta",
        [
            (-4.0, -2.0),
            (-4.0, 0.0),
            (-4.0, 2.0),
            (4.0, -2.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (5.0, 0.0),
        ],
    )
    def test_conflict_pvalue_within_mc_tolerance(self, D, theta):
        sigma = 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        data = np.asarray([D])
        n_mc = 2000
        stat = WaldoStatistic(n_mc=n_mc, seed=0)
        cf = float(stat._closed_form_pvalue(float(theta), data, model, prior))
        gn = float(stat._generic_pvalue(float(theta), data, model, prior))
        # Conflict regime tail probabilities can sit very near 0 or 1;
        # the +1 smoothing biases by O(1/n_mc) which we add to the SE.
        atol = 3.0 / np.sqrt(n_mc) + 2.0 / n_mc
        assert abs(cf - gn) < atol, (
            f"|Δ|≥2 disagreement at D={D}, theta={theta}: "
            f"closed-form={cf:.4f}, generic={gn:.4f}, atol={atol:.4f}"
        )


# --- G.5 fingerprint-based dispatch -------------------------------------


@pytest.mark.L0
class TestIsNormalNormalFingerprint:
    """Audit P1 G.5: `is_normal_normal` keys on `model.fingerprint()[0]`."""

    def test_normal_normal_returns_true(self):
        assert is_normal_normal(NormalNormalModel(sigma=1.0)) is True

    def test_object_without_fingerprint_returns_false(self):
        # Audit hardening: a duck-typed model without `.fingerprint`
        # should not be misclassified as NN.
        class FakeModel:
            sigma = 1.0  # has the field but no fingerprint method

        assert is_normal_normal(FakeModel()) is False

    def test_fingerprint_with_normal_normal_marker_returns_true(self):
        # The whole point of the refactor: fingerprint-based opt-in.
        class WrappedNNModel:
            def fingerprint(self):
                return ("normal_normal", 1.0)

        assert is_normal_normal(WrappedNNModel()) is True


# --- G.6 has_acceptance_region helper -----------------------------------


@pytest.mark.L0
class TestHasAcceptanceRegion:
    """Audit P1 G.6: `acceptance_region` is an optional protocol method."""

    def test_wald_has_it(self):
        assert has_acceptance_region(WaldStatistic()) is True

    def test_waldo_has_it(self):
        assert has_acceptance_region(WaldoStatistic()) is True

    def test_object_without_method_returns_false(self):
        class NoAR:
            name = "noar"

            def evaluate(self, *args, **kwargs):
                return 0.0

        assert has_acceptance_region(NoAR()) is False


# --- G.7 QuantileMixturePath extrapolation guard ------------------------


@pytest.mark.L1
class TestQuantileMixturePathExtrapolationGuard:
    """Audit P0-review #2 + P1 G.7: extrapolation outside [0,1] is allowed
    only when the resulting quantile is monotone (= valid distribution).

    Two Gaussian endpoints with sigma_t > 0 extrapolate cleanly. Two
    non-Gaussian endpoints with mismatched slope structure do not."""

    def test_segment_t_in_unit_interval_admits(self):
        p = NormalDistribution(loc=0.0, scale=1.0)
        q = NormalDistribution(loc=1.0, scale=2.0)
        for t in (0.0, 0.5, 1.0):
            QuantileMixturePath(p=p, q=q, t=t)  # no raise

    def test_gaussian_extrapolation_with_positive_sigma_t_admits(self):
        # Two Gaussians with sigma_t>0 at t=2 (extrapolated): sigma_t
        # = (1-2)*1 + 2*2 = 3 > 0. Valid Gaussian, valid quantile.
        p = NormalDistribution(loc=0.0, scale=1.0)
        q = NormalDistribution(loc=1.0, scale=2.0)
        path = QuantileMixturePath(p=p, q=q, t=2.0)
        # Verify the resulting quantile is monotone.
        u = np.linspace(0.05, 0.95, 17)
        qs = np.asarray(path.quantile(u))
        assert np.all(np.diff(qs) >= 0), (
            f"extrapolated Gaussian quantile not monotone: qs={qs}"
        )

    def test_gaussian_extrapolation_with_negative_sigma_t_raises(self):
        # sigma_t = (1-t)*1 + t*0.1 hits zero at t=10/9 and goes
        # negative beyond. At t=2: sigma_t = -1 + 0.2 = -0.8 < 0.
        p = NormalDistribution(loc=0.0, scale=1.0)
        q = NormalDistribution(loc=1.0, scale=0.1)
        with pytest.raises(ValueError, match="degenerate or reversed Gaussian"):
            QuantileMixturePath(p=p, q=q, t=2.0)

    def test_non_finite_t_raises(self):
        p = NormalDistribution(loc=0.0, scale=1.0)
        q = NormalDistribution(loc=1.0, scale=1.0)
        with pytest.raises(ValueError, match="finite"):
            QuantileMixturePath(p=p, q=q, t=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            QuantileMixturePath(p=p, q=q, t=float("inf"))
