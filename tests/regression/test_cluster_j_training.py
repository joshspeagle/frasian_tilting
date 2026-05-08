"""L1/L2 regression: Cluster J — learned-η training hardening.

Pins the audit P1 fixes:

  J.1 — `cd_density_jax` returns NaN pdf when the trapezoidal Z hits
        the floor (constant-p sample); pre-fix it returned a
        finite-but-huge pdf that silently contaminated
        `cd_variance_loss` without tripping the `_masked_mean`
        non-finite filter.

  J.2 — `_W_EPS = 1e-3` is documented in `docs/methods/learned_eta.md`
        with both training-time and inference-time call sites and the
        rationale (JAX clamp distortion). Smoke-checked here by
        verifying the documentation contains the call-site
        cross-references.

  J.4 — `ExperimentConfig.__post_init__` refuses NN+n_data>1 (the JAX
        closed-form pvalue port assumes single observation D per θ).
        Pre-fix this would silently mismatch sqrt(n_data) at
        inference-time CI inversion.

  J.8 — `_training_step` and `_evaluate_epoch` catch `ArithmeticError`
        (parent of `FloatingPointError` / `OverflowError` /
        `ZeroDivisionError`) in addition to `ValueError` /
        `RuntimeError`. Pinned by reading the source.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


# --- J.1 cd_density_jax NaN guard ----------------------------------------


@pytest.mark.L1
class TestCdDensityJaxNaNGuardOnZFloor:
    """When the p-value is constant on the grid (Z → 0), the resulting
    pdf is NaN-marked rather than finite-but-huge. Downstream
    `_masked_mean` then masks the sample out instead of letting it
    contaminate the batch-mean variance."""

    def test_constant_p_returns_nan_pdf(self):
        from frasian.learned.training.cd_jax import cd_density_jax

        # All-zero p-values: gradient is identically zero → Z = 0 → pdf
        # is NaN per row.
        p_theta = jnp.zeros((3, 50))
        theta_grid = jnp.linspace(-2.0, 2.0, 50)
        pdf = cd_density_jax(p_theta, theta_grid)
        assert pdf.shape == (3, 50)
        assert bool(jnp.all(jnp.isnan(pdf)))

    def test_normal_p_returns_finite_pdf(self):
        from frasian.learned.training.cd_jax import cd_density_jax

        theta_grid = jnp.linspace(-3.0, 3.0, 101)
        # Tent-shaped p (peak in the middle): non-degenerate, gives
        # a clean finite pdf.
        p_theta = jnp.maximum(0.0, 1.0 - jnp.abs(theta_grid)[None, :] / 2.0)
        p_theta = jnp.broadcast_to(p_theta, (2, 101))
        pdf = cd_density_jax(p_theta, theta_grid)
        assert bool(jnp.all(jnp.isfinite(pdf)))

    def test_cd_variance_masks_z_floor_samples(self):
        from frasian.learned.training.cd_jax import cd_density_jax
        from frasian.learned.training.losses import cd_variance_loss

        theta_grid = jnp.linspace(-3.0, 3.0, 101)
        # Mix: row 0 is degenerate (constant p), row 1 is a normal
        # tent. The loss should reflect only row 1 — pre-fix row 0
        # produced a huge variance that swamped the mean.
        p_zero = jnp.zeros((1, 101))
        p_tent = jnp.maximum(
            0.0, 1.0 - jnp.abs(theta_grid)[None, :] / 2.0
        )
        p_theta = jnp.concatenate([p_zero, p_tent], axis=0)
        loss = float(cd_variance_loss(p_theta, theta_grid))
        # The tent's variance is bounded (theta in [-2, 2]; var ≤ 4).
        # Pre-fix the constant-p row would dominate via 1/(2*1e-12)
        # huge factor; the post-fix mean reflects only the tent.
        assert np.isfinite(loss)
        assert 0.0 < loss < 4.0


# --- J.2 _W_EPS documentation -------------------------------------------


@pytest.mark.L0
class TestWEpsDocumentation:
    """Audit P1 J.2: `_W_EPS = 1e-3` is documented in the brief
    with both training and inference call-site refs."""

    def test_brief_mentions_both_call_sites(self):
        brief = Path("docs/methods/learned_eta.md").read_text()
        assert "_losses_compose.py:97" in brief, (
            "brief should cite training-time _W_EPS site with line ref"
        )
        assert "eta_selectors.py:839" in brief, (
            "brief should cite inference-time _W_EPS site with line ref"
        )
        assert "_W_EPS = 1e-3" in brief

    def test_brief_explains_rationale(self):
        brief = Path("docs/methods/learned_eta.md").read_text()
        # Should explain WHY the guard exists (clamp distortion).
        assert "1e-6" in brief or "clamp" in brief.lower(), (
            "brief should explain the JAX clamp distortion that "
            "motivates the _W_EPS guard"
        )


# --- J.4 ExperimentConfig pre-flight n_data + NN check ------------------


@pytest.mark.L1
class TestExperimentConfigNnNdata:
    """Audit P1 J.4: NN model + n_data > 1 is rejected at config
    construction (the JAX closed-form pvalue port assumes single
    observation per θ)."""

    @pytest.fixture(autouse=True)
    def _registry(self, bootstrapped_registry):
        # ExperimentConfig.__post_init__ looks up scheme/statistic in
        # the live registry; we need the concrete implementations
        # bootstrapped for these tests.
        return bootstrapped_registry

    def _build(self, model, n_data):
        from frasian.learned.training.sampling import (
            ExperimentConfig,
            UniformThetaDistribution,
        )
        from frasian.models.distributions import NormalDistribution
        return ExperimentConfig(
            scheme_name="power_law",
            statistic_name="waldo",
            prior=NormalDistribution(loc=0.0, scale=1.0),
            model=model,
            theta_distribution=UniformThetaDistribution(low=-3.0, high=3.0),
            n_data=n_data,
        )

    def test_nn_n_data_eq_1_passes(self):
        from frasian.models.normal_normal import NormalNormalModel
        # n_data=1 is the framework's sandbox; passes.
        cfg = self._build(NormalNormalModel(sigma=1.0), n_data=1)
        assert cfg.n_data == 1

    def test_nn_n_data_gt_1_raises(self):
        from frasian.models.normal_normal import NormalNormalModel
        with pytest.raises(ValueError, match="n_data == 1"):
            self._build(NormalNormalModel(sigma=1.0), n_data=4)

    def test_bernoulli_n_data_gt_1_passes(self):
        # Non-NN: n_data > 1 IS supported (the generic MC path uses
        # n_obs correctly). Bernoulli + Beta is the headline example.
        from frasian.learned.training.sampling import (
            ExperimentConfig,
            UniformThetaDistribution,
        )
        from frasian.models.bernoulli import BernoulliModel
        from frasian.models.distributions import BetaDistribution
        cfg = ExperimentConfig(
            scheme_name="power_law",
            statistic_name="waldo",
            prior=BetaDistribution(alpha=2.0, beta=2.0),
            model=BernoulliModel(),
            theta_distribution=UniformThetaDistribution(low=0.05, high=0.95),
            # n_data=16 is realistic for Bernoulli; bound by n_lhs=20
            # minimum (configured via default 10000) and unrelated.
            n_data=16,
        )
        assert cfg.n_data == 16


# --- J.8 catch ArithmeticError in training step --------------------------


@pytest.mark.L0
class TestTrainingStepCatchesArithmeticError:
    """Audit P1 J.8: `_training_step` and `_evaluate_epoch` catch
    `ArithmeticError` (parent of FloatingPointError, etc.) in addition
    to ValueError/RuntimeError."""

    def test_training_step_except_clause_includes_arithmetic_error(self):
        from frasian.learned.training import _train_loop
        src = inspect.getsource(_train_loop)
        # Both call sites should now catch ArithmeticError. Count >= 2
        # (training step and eval epoch).
        n_occ = src.count("ArithmeticError")
        assert n_occ >= 2, (
            f"_train_loop.py should catch ArithmeticError at the "
            f"training-step and eval-epoch call sites; found "
            f"{n_occ} occurrences."
        )
