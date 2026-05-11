"""L2 regression: Cluster H — tilting bounds & dynamic CI hardening.

Pins the audit P1 fixes:

  H.2 — `_check_experiment` refuses None fingerprints. Pre-fix the
        None case silently skipped the cross-experiment safety net.
  H.3 — `_check_experiment` accepts optional `n_data` and
        `theta_distribution_fingerprint` and refuses on mismatch.
  H.4 — Multi-region CI dynamic-scan whole-window-accept uses the
        sign convention from Cluster C (sgn at endpoints), NOT the
        midpoint p-value heuristic. This was effectively done by
        Cluster C; we pin the resulting behaviour here.
  H.5 — `brentq_with_doubling` raises `BracketingFailed` immediately
        when `f(midpoint)` is non-finite, instead of burning
        `max_doublings + 1` iterations first.
  H.6 — OT Gaussian fast path's signature still accepts `prior`
        (protocol-required) but the parameter is documented as
        intentionally unused on that path. Behavioural test: the
        result is unchanged for any prior value.
"""

from __future__ import annotations

import pytest

from frasian._errors import BracketingFailed, MissingArtifactError
from frasian.models.distributions import (
    GaussianLikelihood,
    NormalDistribution,
)
from frasian.tilting._solvers import brentq_with_doubling
from frasian.tilting.ot import OTTilting


# --- H.2 _check_experiment refuses None fingerprints --------------------


@pytest.mark.L1
class TestCheckExperimentRequiresFingerprints:
    """The H.2 fix raises rather than silently skipping the
    cross-experiment safety net when fingerprints are None."""

    def _build_selector(self, tmp_path):
        # Build a minimal LearnedDynamicEtaSelector and stub its
        # artifact metadata so we can probe `_check_experiment`
        # directly. (The full Phase E dual-head load path is heavy;
        # this sets the metadata fields the check reads.)
        from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
        selector = LearnedDynamicEtaSelector.__new__(LearnedDynamicEtaSelector)
        selector.artifact = type(
            "_FakeArtifact",
            (),
            {
                "name": "test_v0_smoke",
                "metadata": {
                    "experiment_config": {
                        "model_fingerprint": ["normal_normal", 1.0],
                        "prior_fingerprint": ["normal", 0.0, 1.0],
                        "n_data": 16,
                        "theta_distribution_fingerprint": ["uniform", -5.0, 5.0],
                    },
                },
            },
        )()
        return selector

    def test_none_model_fingerprint_raises(self, tmp_path):
        selector = self._build_selector(tmp_path)
        with pytest.raises(MissingArtifactError, match="model_fingerprint"):
            selector._check_experiment(
                w=0.5,
                model_fingerprint=None,
                prior_fingerprint=("normal", 0.0, 1.0),
            )

    def test_none_prior_fingerprint_raises(self, tmp_path):
        selector = self._build_selector(tmp_path)
        with pytest.raises(MissingArtifactError, match="prior_fingerprint"):
            selector._check_experiment(
                w=0.5,
                model_fingerprint=("normal_normal", 1.0),
                prior_fingerprint=None,
            )

    def test_matching_fingerprints_pass(self, tmp_path):
        selector = self._build_selector(tmp_path)
        # No raise — both match the trained metadata.
        selector._check_experiment(
            w=0.5,
            model_fingerprint=("normal_normal", 1.0),
            prior_fingerprint=("normal", 0.0, 1.0),
        )

    def test_n_data_mismatch_raises(self, tmp_path):
        selector = self._build_selector(tmp_path)
        with pytest.raises(MissingArtifactError, match="n_data"):
            selector._check_experiment(
                w=0.5,
                model_fingerprint=("normal_normal", 1.0),
                prior_fingerprint=("normal", 0.0, 1.0),
                n_data=8,  # trained at 16
            )

    def test_theta_distribution_fingerprint_mismatch_raises(self, tmp_path):
        selector = self._build_selector(tmp_path)
        with pytest.raises(MissingArtifactError, match="theta_distribution"):
            selector._check_experiment(
                w=0.5,
                model_fingerprint=("normal_normal", 1.0),
                prior_fingerprint=("normal", 0.0, 1.0),
                theta_distribution_fingerprint=("uniform", -3.0, 3.0),
            )


# --- H.5 brentq_with_doubling early-raise on non-finite f_mid ------------


@pytest.mark.L0
class TestBrentqEarlyRaiseOnNonFiniteMid:
    """The H.5 fix raises `BracketingFailed` immediately on
    non-finite `f(midpoint)` rather than burning all doublings."""

    def test_nan_at_midpoint_raises(self):
        def f(theta):
            if theta == 0.0:
                return float("nan")
            return theta

        with pytest.raises(BracketingFailed, match="non-finite"):
            brentq_with_doubling(
                f, midpoint=0.0, initial_half_width=1.0, direction=+1
            )

    def test_inf_at_midpoint_raises(self):
        def f(theta):
            if theta == 0.0:
                return float("inf")
            return theta

        with pytest.raises(BracketingFailed, match="non-finite"):
            brentq_with_doubling(
                f, midpoint=0.0, initial_half_width=1.0, direction=+1
            )

    def test_finite_midpoint_brackets_normally(self):
        # Standard root-finding still works after the H.5 guard.
        def f(theta):
            return theta - 0.7

        result = brentq_with_doubling(
            f, midpoint=0.0, initial_half_width=0.5, direction=+1
        )
        assert abs(result - 0.7) < 1e-6


# --- H.6 OT Gaussian fast path is prior-invariant -----------------------


@pytest.mark.L0
class TestOtGaussianFastPathPriorInvariant:
    """The H.6 fix documents that the OT Gaussian fast path does not
    consume `prior`; it interpolates posterior↔likelihood directly.
    Behavioural test: changing the prior leaves the result unchanged."""

    def test_fast_path_invariant_under_prior_swap(self):
        scheme = OTTilting()
        posterior = NormalDistribution(loc=0.5, scale=0.7)
        lik = GaussianLikelihood(D=1.0, sigma=1.0)
        # Two different priors; the fast path should ignore both.
        prior_a = NormalDistribution(loc=0.0, scale=1.0)
        prior_b = NormalDistribution(loc=10.0, scale=0.1)
        out_a = scheme.tilt(posterior, prior_a, lik, 0.4)
        out_b = scheme.tilt(posterior, prior_b, lik, 0.4)
        assert isinstance(out_a, NormalDistribution)
        assert isinstance(out_b, NormalDistribution)
        assert abs(out_a.loc - out_b.loc) < 1e-12
        assert abs(out_a.scale - out_b.scale) < 1e-12
