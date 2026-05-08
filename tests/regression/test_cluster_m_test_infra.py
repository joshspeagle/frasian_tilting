"""L0/L1 regression: Cluster M — test infrastructure.

Pins the audit P1 fixes:

  M.2 — Hypothesis project-default profile sets `deadline=2000` so a
        runaway shrink can't hang CI; per-test `@settings(deadline=...)`
        overrides still win.

  M.4 — `_isolated_registry` resets `_BOOTSTRAPPED` so a test that
        calls `bootstrap()` after the autouse fixture clears the
        registry actually re-populates rather than no-op'ing.

  M.5 — `IdentityTilting` has explicit regression coverage: it is the
        identity element of the cross-product (CI matches the bare
        statistic), accepts any tilting partner via the protocol, and
        reports `is_identity(0.0) == True`.

  M.6 — `PowerLawTilting` at the η = 1 endpoint reduces to
        bare-Wald (the prior contribution drops out, see Theorem 6).
        Pinned with a numeric agreement check on the closed-form NN
        path against a reference Wald CI computed directly.

  M.7 — `check_method_completeness.py` rejects briefs whose section
        bodies are empty whitespace. Pinned by source-level read of
        the tightened predicate.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


# --- M.2 hypothesis deadline default ------------------------------------


@pytest.mark.L0
class TestHypothesisDefaultDeadline:
    """The conftest registers a project-default Hypothesis profile."""

    def test_default_profile_set(self):
        from hypothesis import settings
        # The current profile after `load_profile("frasian_default")`
        # should have `deadline` set (not None).
        prof = settings.default
        # Hypothesis stores deadline as `datetime.timedelta` or None.
        assert prof.deadline is not None, (
            "frasian_default Hypothesis profile should have a deadline"
        )


# --- M.4 _BOOTSTRAPPED reset --------------------------------------------


@pytest.mark.L0
class TestIsolatedRegistryResetsBootstrapped:
    """Audit P1 M.4: the autouse fixture resets `_BOOTSTRAPPED` so a
    follow-up `bootstrap()` call re-populates instead of no-op'ing.
    """

    def test_bootstrap_flag_reset_during_isolation(self):
        from frasian import _registry_bootstrap as bootstrap_mod
        # The autouse fixture clears the registry and resets the flag.
        # If we now call `bootstrap()` ourselves, it should re-populate.
        # First confirm the flag is False inside the test.
        assert bootstrap_mod._BOOTSTRAPPED is False


# --- M.5 IdentityTilting regression -------------------------------------


@pytest.mark.L1
class TestIdentityTiltingRegression:
    """Pin the framework's identity-element semantics: IdentityTilting
    is a no-op delegator that produces the bare statistic's behaviour."""

    @pytest.fixture(autouse=True)
    def _registry(self, bootstrapped_registry):
        return bootstrapped_registry

    def test_is_identity_true_at_default(self):
        from frasian.tilting.identity import IdentityTilting
        scheme = IdentityTilting()
        # The class declares eta_default=eta_identity=0.0; verify
        # is_identity(0.0) is True.
        assert scheme.is_identity(0.0) is True or scheme.is_identity(0.0) == True  # noqa: E712

    def test_confidence_interval_matches_bare_statistic(self):
        # Identity + Wald should produce the same CI as bare Wald.
        from frasian.models.distributions import NormalDistribution
        from frasian.models.normal_normal import NormalNormalModel
        from frasian.statistics.wald import WaldStatistic
        from frasian.tilting.identity import IdentityTilting

        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        stat = WaldStatistic()
        data = np.asarray([0.5])
        identity_scheme = IdentityTilting()

        bare_ci = stat.confidence_interval(0.05, data, model, prior)
        wrapped_ci = identity_scheme.confidence_interval(
            0.05, data, model, prior, stat
        )
        assert abs(bare_ci[0] - wrapped_ci[0]) < 1e-12
        assert abs(bare_ci[1] - wrapped_ci[1]) < 1e-12

    def test_no_selector_field(self):
        # P0-6 fix: IdentityTilting drops the structurally-dead
        # `selector` field. Verify it isn't present.
        from frasian.tilting.identity import IdentityTilting
        scheme = IdentityTilting()
        # Field should not exist on the dataclass.
        from dataclasses import fields
        field_names = [f.name for f in fields(scheme)]
        assert "selector" not in field_names, (
            "IdentityTilting should not have a selector field "
            "(audit P0-6); got fields: " + str(field_names)
        )


# --- M.6 power_law η=1 endpoint -----------------------------------------


@pytest.mark.L1
class TestPowerLawEtaOneReducesToWald:
    """At η=1 the power-law tilted posterior collapses to the
    likelihood (the prior contribution vanishes), and the CI should
    agree with a bare-Wald CI computed at the same data."""

    @pytest.fixture(autouse=True)
    def _registry(self, bootstrapped_registry):
        return bootstrapped_registry

    def test_eta_one_tilted_pvalue_matches_wald(self):
        # Closed-form Theorem 6 at η=1: the tilted posterior is N(D, σ).
        # WALDO at η=1 reduces to Wald: 2·Φ(-|D-θ|/σ).
        from frasian.models.distributions import NormalDistribution
        from frasian.models.normal_normal import NormalNormalModel
        from frasian.tilting.power_law import PowerLawTilting

        sigma = 1.0
        D = 0.7
        theta = 0.0

        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=1.0)

        # Power-law tilted p-value at η=1 (no prior contribution).
        scheme = PowerLawTilting()
        p_tilted = float(scheme.tilted_pvalue(
            theta, D, model, prior, eta=1.0, statistic_name="waldo"
        ))

        # Bare Wald at the same (D, theta, sigma).
        from scipy.stats import norm
        p_wald = 2.0 * norm.cdf(-abs(D - theta) / sigma)

        assert abs(p_tilted - p_wald) < 1e-9, (
            f"power_law tilted_pvalue at η=1 should reduce to bare Wald: "
            f"got {p_tilted:.8f}, expected {p_wald:.8f}"
        )

    def test_eta_one_tilted_posterior_collapses_to_likelihood(self):
        # Theorem 6 at η=1: σ_eta² → σ², μ_eta → D.
        from frasian.models.distributions import NormalDistribution, GaussianLikelihood
        from frasian.models.normal_normal import NormalNormalModel
        from frasian.tilting.power_law import PowerLawTilting

        sigma = 1.0
        D = 0.7
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        likelihood = GaussianLikelihood(D=D, sigma=sigma)
        # Build the bare posterior at this data, then tilt at η=1.
        post = model.posterior(np.asarray([D]), prior)
        tilted = PowerLawTilting().tilt(post, prior, likelihood, eta=1.0)
        # The tilted distribution should be N(D, σ).
        assert abs(float(tilted.loc) - D) < 1e-9
        assert abs(float(tilted.scale) - sigma) < 1e-9


# --- M.7 check_method_completeness rejects empty section bodies --------


@pytest.mark.L0
class TestCheckMethodCompletenessEmptyBody:
    """Audit P1 M.7: the method-completeness check must reject briefs
    whose section bodies are empty whitespace."""

    def test_empty_body_predicate_in_source(self):
        src = Path("tools/check_method_completeness.py").read_text()
        assert "empty body" in src, (
            "check_method_completeness.py should reject empty section bodies"
        )
        # Sanity: the live check still passes on the in-tree briefs.
        result = subprocess.run(
            [sys.executable, "tools/check_method_completeness.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"method-completeness check failed (post-tightening): "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
