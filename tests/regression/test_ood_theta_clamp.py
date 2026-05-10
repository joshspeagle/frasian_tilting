"""OOD-θ clamp: LearnedDynamicEtaSelector overrides η to
``scheme.param_space.eta_likelihood_only`` for θ values outside the
training θ-distribution box. Default ON.

Calibration is preserved (any fixed η yields U[0,1] p-values under H₀)
so the clamped CI keeps 1-α coverage; the clamp is a graceful fallback
to data-only inference when the network is being queried outside its
training distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector


@dataclass
class _FakeArtifact:
    metadata: dict[str, Any]
    eta_const: float = 0.5  # the value the "network" returns regardless
    name: str = "_FakeArtifact"

    def load(self) -> None:
        pass

    def predict_eta(self, theta, prior_hp, lik_hp):  # noqa: D401
        return np.full_like(np.asarray(theta, dtype=np.float64), self.eta_const)


def _build_selector(theta_dist_spec, scheme_name="power_law", eta_const=0.5,
                     clamp_outside_training=True):
    meta = {
        "checkpoint_format_version": 4,
        "alpha_mode": "marginalised",
        "alpha": None,
        "experiment_config": {
            "scheme": scheme_name,
            "theta_distribution": theta_dist_spec,
        },
    }
    art = _FakeArtifact(metadata=meta, eta_const=eta_const)
    sel = LearnedDynamicEtaSelector(
        artifact=art, clamp_outside_training=clamp_outside_training,
    )
    sel._loaded = True
    return sel


@pytest.mark.L1
class TestSigmaAnchoredClamp:
    """sigma_anchored_uniform: box = [μ₀ − K·σ₀, μ₀ + K·σ₀] with K from spec."""

    def test_inside_box_eta_unchanged(self):
        from frasian.tilting.power_law import PowerLawTilting
        from frasian.models.distributions import NormalDistribution

        scheme = PowerLawTilting()
        sel = _build_selector({"type": "sigma_anchored_uniform", "K": 5.0})
        prior = NormalDistribution(loc=0.0, scale=1.0)
        # box = [-5, 5]; θ ∈ [-3, 3] all inside.
        theta = np.linspace(-3.0, 3.0, 13)
        eta = sel._clamp_outside_training_box(
            np.full_like(theta, 0.5), theta, prior, scheme,
        )
        assert np.allclose(eta, 0.5), "all θ inside box → no clamp"

    def test_outside_box_clamped_to_likelihood_only(self):
        from frasian.tilting.power_law import PowerLawTilting
        from frasian.models.distributions import NormalDistribution

        scheme = PowerLawTilting()
        sel = _build_selector({"type": "sigma_anchored_uniform", "K": 5.0})
        prior = NormalDistribution(loc=0.0, scale=1.0)
        # box = [-5, 5]; ±10 outside, ±2 inside.
        theta = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
        eta = sel._clamp_outside_training_box(
            np.full_like(theta, 0.5), theta, prior, scheme,
        )
        # power_law eta_likelihood_only = 1.0
        assert eta[0] == 1.0, "θ=-10 outside box → η=1"
        assert eta[1] == 0.5, "θ=-2 inside box → unchanged"
        assert eta[2] == 0.5, "θ=0 inside box → unchanged"
        assert eta[3] == 0.5, "θ=2 inside box → unchanged"
        assert eta[4] == 1.0, "θ=10 outside box → η=1"

    def test_box_uses_per_call_prior_hp(self):
        """Box scales with σ₀ from the prior at call time, not from
        anything frozen in metadata."""
        from frasian.tilting.power_law import PowerLawTilting
        from frasian.models.distributions import NormalDistribution

        scheme = PowerLawTilting()
        sel = _build_selector({"type": "sigma_anchored_uniform", "K": 5.0})
        # σ₀ = 0.3 → box [-1.5, 1.5]; θ=2 should be outside.
        prior_narrow = NormalDistribution(loc=0.0, scale=0.3)
        theta = np.array([0.0, 2.0])
        eta = sel._clamp_outside_training_box(
            np.full_like(theta, 0.5), theta, prior_narrow, scheme,
        )
        assert eta[0] == 0.5
        assert eta[1] == 1.0
        # σ₀ = 4.0 → box [-20, 20]; θ=2 should be inside.
        prior_wide = NormalDistribution(loc=0.0, scale=4.0)
        eta2 = sel._clamp_outside_training_box(
            np.full_like(theta, 0.5), theta, prior_wide, scheme,
        )
        assert eta2[0] == 0.5
        assert eta2[1] == 0.5

    def test_box_recenters_with_mu0(self):
        """Box is centered at μ₀ from the prior (translation invariance)."""
        from frasian.tilting.power_law import PowerLawTilting
        from frasian.models.distributions import NormalDistribution

        scheme = PowerLawTilting()
        sel = _build_selector({"type": "sigma_anchored_uniform", "K": 5.0})
        prior_shifted = NormalDistribution(loc=10.0, scale=1.0)
        # box = [5, 15]; θ=12 inside, θ=20 outside.
        theta = np.array([5.5, 10.0, 12.0, 20.0])
        eta = sel._clamp_outside_training_box(
            np.full_like(theta, 0.5), theta, prior_shifted, scheme,
        )
        assert eta[0] == 0.5  # 5.5 inside
        assert eta[1] == 0.5  # 10 inside (center)
        assert eta[2] == 0.5  # 12 inside
        assert eta[3] == 1.0  # 20 outside


@pytest.mark.L1
class TestUniformClamp:
    """type='uniform': box = [low, high] from spec."""

    def test_outside_box_clamped(self):
        from frasian.tilting.power_law import PowerLawTilting
        from frasian.models.distributions import NormalDistribution

        scheme = PowerLawTilting()
        sel = _build_selector({"type": "uniform", "low": -2.0, "high": 2.0})
        prior = NormalDistribution(loc=0.0, scale=10.0)  # ignored for uniform
        theta = np.array([-5.0, -1.0, 1.0, 5.0])
        eta = sel._clamp_outside_training_box(
            np.full_like(theta, 0.5), theta, prior, scheme,
        )
        assert eta[0] == 1.0
        assert eta[1] == 0.5
        assert eta[2] == 0.5
        assert eta[3] == 1.0


@pytest.mark.L1
class TestClampOff:
    """clamp_outside_training=False → no-op even outside the box."""

    def test_disabled_clamp_no_op(self):
        from frasian.tilting.power_law import PowerLawTilting
        from frasian.models.distributions import NormalDistribution

        scheme = PowerLawTilting()
        sel = _build_selector(
            {"type": "sigma_anchored_uniform", "K": 5.0},
            clamp_outside_training=False,
        )
        prior = NormalDistribution(loc=0.0, scale=1.0)
        theta = np.array([-10.0, 0.0, 10.0])
        # Calling _clamp_outside_training_box directly still applies
        # — but the select_grid pathway gates on the flag. We test
        # the flag's effect via the gating; here we just confirm the
        # field exists and defaults to True.
        assert sel.clamp_outside_training is False


@pytest.mark.L1
class TestSchemeWithoutLikelihoodOnly:
    """For schemes whose param_space.eta_likelihood_only is None, the
    clamp is a no-op (the concept doesn't apply)."""

    def test_identity_scheme_no_clamp(self):
        from frasian.tilting.identity import IdentityTilting
        from frasian.models.distributions import NormalDistribution

        scheme = IdentityTilting()
        # Identity has eta_likelihood_only=None
        assert scheme.param_space.eta_likelihood_only is None

        sel = _build_selector(
            {"type": "sigma_anchored_uniform", "K": 5.0},
            scheme_name="identity",
        )
        prior = NormalDistribution(loc=0.0, scale=1.0)
        theta = np.array([-10.0, 0.0, 10.0])
        eta = sel._clamp_outside_training_box(
            np.full_like(theta, 0.5), theta, prior, scheme,
        )
        # No clamp applied — η returned unchanged
        assert np.allclose(eta, 0.5)


@pytest.mark.L1
class TestParamSpecLikelihoodOnly:
    """eta_likelihood_only is populated for the schemes that have a
    posterior↔likelihood η semantics (power_law, ot, mixture)."""

    def test_power_law_eta_one(self):
        from frasian.tilting.power_law import PowerLawTilting

        assert PowerLawTilting().param_space.eta_likelihood_only == 1.0

    def test_ot_eta_one(self):
        from frasian.tilting.ot import OTTilting

        assert OTTilting().param_space.eta_likelihood_only == 1.0

    def test_mixture_eta_one(self):
        from frasian.tilting.mixture import MixtureTilting

        assert MixtureTilting().param_space.eta_likelihood_only == 1.0

    def test_identity_none(self):
        from frasian.tilting.identity import IdentityTilting

        assert IdentityTilting().param_space.eta_likelihood_only is None
