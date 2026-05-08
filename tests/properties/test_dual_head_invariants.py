"""Phase E dual-head architecture invariants (JAX / Equinox port).

Phase F port commit 2 rewrites these invariants for the
``equinox.Module``-based ``EtaNet`` / ``ValidityNet``. Skeptic-driven
invariants stay; the torch-specific assertions become Equinox /
JAX equivalents:

- ``EtaNet`` / ``ValidityNet`` forward shapes for theta_dim ∈ {1, 3}.
- ``boundary_penalty_from_validity`` value + gradient behaviour.
- Detachment: Head B's params do *not* receive gradient from Head A's
  loss (via ``eqx.partition`` + ``stop_gradient``), but Head A's
  params *do* (gradient flows through the input).
- Detachment: Head A's params do *not* receive gradient from Head B's
  BCE loss when ``eta_pred`` is detached at the input.
- ``is_pair_valid`` / ``validity_mask`` / ``compute_pvalues_per_sample``
  per-point semantics + NaN-on-failure handling (numpy-side, unchanged).
- ``ExperimentConfig`` round-trip through dict + YAML.
- ``Prior.fingerprint`` / ``Model.fingerprint`` hashability.

The legacy torch-specific tests (``torch.func.functional_call``,
``torch.autograd.gradcheck``, ``ot_torch_pvalue_smooth_*``,
``test_width_loss_averages_over_d_batch`` against the torch path) are
either rewritten in JAX terms or migrated to commit 3's orchestrator
re-port. The handful of orchestrator-touching tests
(``test_phase_e_selector_rejects_cross_experiment_use``,
``test_alpha_required_to_be_none_for_marginalised_loss``,
``test_extract_normal_normal_params_rejects_degenerate_w``,
``test_width_loss_averages_over_d_batch``) reach into ``train.py``
which still imports torch — they are expected to error at runtime
during commit 2 and are restored in commit 3 as part of the
orchestrator port.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Equinox + JAX are framework dependencies; no importorskip needed.
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from frasian.learned.training.architecture import EtaNet, ValidityNet
from frasian.learned.training._losses_compose import compose_boundary_penalty
from frasian.learned.training.losses import boundary_penalty_from_validity
from frasian.learned.training.sampling import ExperimentConfig, UniformThetaDistribution, lhs_1d
from frasian.learned.training.validity import (
    compute_pvalues_per_sample,
    is_pair_valid,
    validity_mask,
)
from frasian.models.bernoulli import BernoulliModel
from frasian.models.distributions import BetaDistribution, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.power_law import PowerLawTilting


def _make_keys(n: int = 2, seed: int = 0) -> list[jax.Array]:
    """Convenience: split PRNGKey(seed) into ``n`` subkeys."""
    return list(jax.random.split(jax.random.PRNGKey(seed), n))


# ---------------------------------------------------------------------------
# Architecture forward shapes
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_forward_1d():
    key = jax.random.PRNGKey(0)
    net = EtaNet(theta_dim=1, key=key)
    theta = jnp.linspace(-3.0, 3.0, 11)
    out = net(theta)
    assert out.shape == (11,)
    # 1D-input convenience: also accept (N, 1).
    out2 = net(theta[..., None])
    assert out2.shape == (11,)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out2), atol=0.0)


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_forward_vector_theta():
    key = jax.random.PRNGKey(1)
    net = EtaNet(theta_dim=3, key=key)
    theta = jax.random.normal(jax.random.PRNGKey(2), (7, 3))
    out = net(theta)
    assert out.shape == (7,)


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_rejects_wrong_dim():
    key = jax.random.PRNGKey(0)
    net = EtaNet(theta_dim=2, key=key)
    with pytest.raises(ValueError, match="theta_dim=2"):
        net(jax.random.normal(jax.random.PRNGKey(1), (5, 3)))
    # 1D input on theta_dim=2 net is also rejected.
    with pytest.raises(ValueError, match="theta_dim=2"):
        net(jax.random.normal(jax.random.PRNGKey(2), (5,)))


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_forward():
    key = jax.random.PRNGKey(0)
    net = ValidityNet(theta_dim=1, key=key)
    inputs = jax.random.normal(jax.random.PRNGKey(1), (13, 2))
    out = net(inputs)
    assert out.shape == (13,)


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_forward_vector():
    key = jax.random.PRNGKey(0)
    net = ValidityNet(theta_dim=4, key=key)
    inputs = jax.random.normal(jax.random.PRNGKey(1), (8, 5))  # theta_dim + 1
    out = net(inputs)
    assert out.shape == (8,)


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_rejects_wrong_input_shape():
    key = jax.random.PRNGKey(0)
    net = ValidityNet(theta_dim=1, key=key)
    with pytest.raises(ValueError, match="theta_dim=1"):
        net(jax.random.normal(jax.random.PRNGKey(1), (5, 3)))


# ---------------------------------------------------------------------------
# Boundary penalty
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_boundary_penalty_value_behavior():
    """Penalty is ~|logit| for very negative (linear), ~0 for very positive."""
    very_invalid = jnp.array([-100.0])
    very_valid = jnp.array([+100.0])
    p_inv = float(boundary_penalty_from_validity(very_invalid))
    p_val = float(boundary_penalty_from_validity(very_valid))
    assert 99.0 < p_inv < 101.0, f"penalty for very-invalid logit should be ~100, got {p_inv}"
    assert p_val < 1e-6, f"penalty for very-valid logit should be ~0, got {p_val}"


@pytest.mark.L1
@pytest.mark.properties
def test_boundary_penalty_gradient_alive_at_extremes():
    """Gradient must NOT vanish for very negative logits.

    The wrong-side gradient saturates at -1 (since
    ``d(-log_sigmoid(x))/dx = -sigmoid(-x) → -1`` as x → -∞), so
    the boundary signal stays alive even when ValidityNet is
    extremely confident the point is invalid.
    """
    grad_fn = jax.grad(boundary_penalty_from_validity)
    # Single-element batch so the mean derivative equals the per-element one.
    g_neg = grad_fn(jnp.array([-50.0]))
    np.testing.assert_allclose(np.asarray(g_neg), [-1.0], atol=1e-6)
    # Right side: gradient → 0 as logit → +∞ (no penalty to apply).
    g_pos = grad_fn(jnp.array([+50.0]))
    np.testing.assert_allclose(np.asarray(g_pos), [0.0], atol=1e-6)


@pytest.mark.L1
@pytest.mark.properties
def test_boundary_penalty_finite_difference_check():
    """Finite-difference gradient agrees with autodiff on small input.

    JAX equivalent of the legacy ``torch.autograd.gradcheck``: compare
    the analytic ``jax.grad`` to a centred finite-difference estimate.
    """
    x = jax.random.normal(jax.random.PRNGKey(0), (5,))
    g_auto = jax.grad(boundary_penalty_from_validity)(x)
    eps = 1e-6
    g_fd = np.zeros_like(np.asarray(x))
    for i in range(x.shape[0]):
        xp = x.at[i].add(eps)
        xm = x.at[i].add(-eps)
        g_fd[i] = (
            float(boundary_penalty_from_validity(xp))
            - float(boundary_penalty_from_validity(xm))
        ) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(g_auto), g_fd, atol=1e-5)


@pytest.mark.L1
@pytest.mark.properties
def test_boundary_penalty_finite_difference_at_large_logits():
    """Finite-difference gradient agreement at logits well outside the
    previously-clamped range.

    Catches a regression where a clamp was reintroduced and gradient
    silently zeroed at extreme inputs. Larger atol since
    log_sigmoid is asymptotically linear with finite-difference
    truncation error growing with |x|.
    """
    x = jnp.linspace(-30.0, 30.0, 7)
    g_auto = jax.grad(boundary_penalty_from_validity)(x)
    eps = 1e-5
    g_fd = np.zeros_like(np.asarray(x))
    for i in range(x.shape[0]):
        xp = x.at[i].add(eps)
        xm = x.at[i].add(-eps)
        g_fd[i] = (
            float(boundary_penalty_from_validity(xp))
            - float(boundary_penalty_from_validity(xm))
        ) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(g_auto), g_fd, atol=1e-3)


# ---------------------------------------------------------------------------
# Detachment in the dual-head training loop
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_head_b_params_no_grad_from_head_a_loss():
    """Boundary penalty via eqx.partition + stop_gradient must not gradient Head B."""
    k1, k2 = _make_keys(2, seed=0)
    eta_net = EtaNet(theta_dim=1, key=k1)
    val_net = ValidityNet(theta_dim=1, key=k2)
    theta = jnp.linspace(-2.0, 2.0, 11)

    en_params, en_static = eqx.partition(eta_net, eqx.is_array)
    vn_params, vn_static = eqx.partition(val_net, eqx.is_array)

    def loss_only_eta(en_p):
        en2 = eqx.combine(en_p, en_static)
        eta_pred = en2(theta)
        return compose_boundary_penalty(
            val_net=val_net, theta_batch_t=theta, eta_pred=eta_pred
        )

    g_en = jax.grad(loss_only_eta)(en_params)
    en_norms = [jnp.abs(g).sum() for g in jtu.tree_leaves(g_en)]
    assert any(float(n) > 0 for n in en_norms), (
        "EtaNet did not receive boundary-penalty gradient"
    )

    def loss_only_val(vn_p):
        vn2 = eqx.combine(vn_p, vn_static)
        eta_pred = eta_net(theta)
        return compose_boundary_penalty(
            val_net=vn2, theta_batch_t=theta, eta_pred=eta_pred
        )

    g_vn = jax.grad(loss_only_val)(vn_params)
    vn_norms = [jnp.abs(g).sum() for g in jtu.tree_leaves(g_vn)]
    assert all(float(n) == 0.0 for n in vn_norms), (
        "ValidityNet erroneously received gradient through detached call"
    )


@pytest.mark.L1
@pytest.mark.properties
def test_head_a_params_no_grad_from_head_b_loss():
    """When eta_pred is stopped at the input, Head B's BCE must not
    flow gradient back into EtaNet.
    """
    k1, k2 = _make_keys(2, seed=1)
    eta_net = EtaNet(theta_dim=1, key=k1)
    val_net = ValidityNet(theta_dim=1, key=k2)
    theta = jnp.linspace(-2.0, 2.0, 11)
    target = jnp.zeros((11,))  # arbitrary BCE target

    en_params, en_static = eqx.partition(eta_net, eqx.is_array)
    vn_params, vn_static = eqx.partition(val_net, eqx.is_array)

    def bce_loss_through_eta(en_p):
        en2 = eqx.combine(en_p, en_static)
        eta_pred = jax.lax.stop_gradient(en2(theta))
        inputs = jnp.stack([theta, eta_pred], axis=-1)
        logits = val_net(inputs)
        # BCE with logits, mean reduction.
        return -(
            target * jax.nn.log_sigmoid(logits)
            + (1.0 - target) * jax.nn.log_sigmoid(-logits)
        ).mean()

    g_en = jax.grad(bce_loss_through_eta)(en_params)
    en_norms = [jnp.abs(g).sum() for g in jtu.tree_leaves(g_en)]
    assert all(float(n) == 0.0 for n in en_norms), (
        "EtaNet erroneously received gradient via Head B loss"
    )

    def bce_loss_through_val(vn_p):
        vn2 = eqx.combine(vn_p, vn_static)
        eta_pred = jax.lax.stop_gradient(eta_net(theta))
        inputs = jnp.stack([theta, eta_pred], axis=-1)
        logits = vn2(inputs)
        return -(
            target * jax.nn.log_sigmoid(logits)
            + (1.0 - target) * jax.nn.log_sigmoid(-logits)
        ).mean()

    g_vn = jax.grad(bce_loss_through_val)(vn_params)
    vn_norms = [jnp.abs(g).sum() for g in jtu.tree_leaves(g_vn)]
    assert any(float(n) > 0 for n in vn_norms), "ValidityNet did not receive BCE gradient"


# ---------------------------------------------------------------------------
# Validity helpers (numpy-side, unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_is_pair_valid_predicate():
    assert is_pair_valid(0.5)
    assert is_pair_valid(0.0)
    assert is_pair_valid(1.0)
    # Slack
    assert is_pair_valid(-1e-10)
    assert is_pair_valid(1.0 + 1e-10)
    # Out-of-range
    assert not is_pair_valid(-0.001)
    assert not is_pair_valid(1.001)
    # NaN / Inf
    assert not is_pair_valid(float("nan"))
    assert not is_pair_valid(float("inf"))
    assert not is_pair_valid(float("-inf"))


@pytest.mark.L1
@pytest.mark.properties
def test_validity_mask_vectorised():
    arr = np.array([0.5, np.nan, -0.5, 1.0001, 0.999])
    mask = validity_mask(arr)
    np.testing.assert_array_equal(mask, [True, False, False, False, True])


@pytest.mark.L1
@pytest.mark.properties
def test_compute_pvalues_per_sample_nan_on_invalid_eta():
    """power_law denom = 1 - eta(1-w); for w=0.5 invalid for eta >= 2."""
    scheme = PowerLawTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)  # → w = 0.5
    theta = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    D = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    eta = np.array([0.5, 1.0, 1.9, 2.5, 10.0])
    p = compute_pvalues_per_sample(scheme, theta, D, model, prior, eta, "waldo")
    mask = validity_mask(p)
    # First three valid, last two NaN.
    assert mask[0] and mask[1] and mask[2]
    assert not mask[3] and not mask[4]


@pytest.mark.L1
@pytest.mark.properties
def test_compute_pvalues_per_sample_2d_D_routes_per_sample():
    """``D.shape == (N, n_data)`` (Phase 4c-1) routes through the
    per-sample loop; output shape stays ``(N,)`` and each row's D is
    seen by ``scheme.tilted_pvalue`` as a 1-D dataset.

    For Bernoulli + power_law (no closed-form), the loop catches the
    ``NotImplementedError`` per sample and yields all-NaN — pinning
    that the 2D plumbing reaches the loop without a shape error."""
    from frasian.models.bernoulli import BernoulliModel
    from frasian.models.distributions import BetaDistribution
    from frasian.tilting.power_law import PowerLawTilting

    scheme = PowerLawTilting()
    model = BernoulliModel()
    prior = BetaDistribution(alpha=2.0, beta=2.0)
    theta = np.array([0.3, 0.5, 0.7])
    eta = np.array([0.2, 0.2, 0.2])
    # Each row is a 4-flip Bernoulli dataset.
    D = np.array([[1.0, 0.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0]])
    p = compute_pvalues_per_sample(scheme, theta, D, model, prior, eta, "waldo")
    assert p.shape == (3,)
    # Phase 4c-2: non-NN models route through the scheme's generic MC
    # tilted-pvalue. Output is finite, in [0, 1], and the validity
    # mask flips True (Head B can now train on Bernoulli).
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    from frasian.learned.training.validity import validity_mask
    assert validity_mask(p).all()


@pytest.mark.L1
@pytest.mark.properties
def test_compute_pvalues_per_sample_shape_mismatch():
    scheme = PowerLawTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    with pytest.raises(ValueError, match="agree on first axis"):
        compute_pvalues_per_sample(
            scheme,
            np.array([0.0, 1.0]),
            np.array([0.0]),
            model,
            prior,
            np.array([0.5, 0.5]),
            "waldo",
        )


# ---------------------------------------------------------------------------
# Fingerprints + ExperimentConfig
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_fingerprints_hashable():
    """Fingerprints must be tuples (hashable, comparable)."""
    fps = [
        NormalDistribution(0.0, 1.0).fingerprint(),
        BetaDistribution(2.0, 5.0).fingerprint(),
        NormalNormalModel(sigma=1.0).fingerprint(),
        BernoulliModel().fingerprint(),
        UniformThetaDistribution(low=-5.0, high=5.0).fingerprint(),
    ]
    for fp in fps:
        assert isinstance(fp, tuple)
        hash(fp)  # raises if unhashable


@pytest.mark.L1
@pytest.mark.properties
def test_experiment_config_dict_round_trip(bootstrapped_registry):
    # Audit P1 J.4 (commit 1f6ff8b): NormalNormalModel + n_data > 1 is
    # rejected at __post_init__ because the JAX closed-form pvalue port
    # would silently mismatch the closed form by sqrt(n_data). Use the
    # Bernoulli + Beta combo to exercise a non-default n_data round-trip.
    cfg = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=BetaDistribution(alpha=2.0, beta=2.0),
        model=BernoulliModel(),
        theta_distribution=UniformThetaDistribution(low=0.01, high=0.99),
        n_grid=101,
        n_lhs=200,
        n_data=4,
        eta_explore_box=(-3.0, 3.0),
        seed=7,
        name="test",
    )
    d = cfg.to_dict()
    # JSON-serialisable.
    json.dumps(d)
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.prior.fingerprint() == cfg.prior.fingerprint()
    assert cfg2.model.fingerprint() == cfg.model.fingerprint()
    assert cfg2.theta_distribution.fingerprint() == cfg.theta_distribution.fingerprint()
    assert cfg2.n_grid == cfg.n_grid
    assert cfg2.n_lhs == cfg.n_lhs
    assert cfg2.n_data == cfg.n_data
    assert cfg2.eta_explore_box == cfg.eta_explore_box
    assert cfg2.seed == cfg.seed
    assert cfg2.name == cfg.name


@pytest.mark.L1
@pytest.mark.properties
def test_experiment_config_yaml_round_trip(bootstrapped_registry):
    yaml = pytest.importorskip("yaml")
    cfg = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(0.0, 1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-5.0, high=5.0),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg.to_dict(), f)
        cfg2 = ExperimentConfig.from_yaml(path)
    assert cfg2.to_dict() == cfg.to_dict()


@pytest.mark.L1
@pytest.mark.properties
def test_experiment_config_yaml_shorthand_keys(bootstrapped_registry):
    """YAML may use ``scheme:`` / ``statistic:`` shorthand."""
    pytest.importorskip("yaml")
    text = """
scheme: power_law
statistic: waldo
prior: {type: normal, loc: 0.0, scale: 1.0}
model: {type: normal_normal, sigma: 1.0}
theta_distribution: {type: uniform, low: -5.0, high: 5.0}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.yaml"
        path.write_text(text)
        cfg = ExperimentConfig.from_yaml(path)
    assert cfg.scheme_name == "power_law"
    assert cfg.statistic_name == "waldo"


@pytest.mark.L1
@pytest.mark.properties
def test_experiment_config_rejects_unexpected_kwargs(bootstrapped_registry):
    """YAML must not be able to override class-level fields.

    `NormalNormalModel.name` / `.param_dim` are class-level
    dataclass defaults; we filter them out so a YAML config
    cannot rename a model and confuse the fingerprint check.
    """
    bad = {
        "scheme_name": "power_law",
        "statistic_name": "waldo",
        "prior": {"type": "normal", "loc": 0.0, "scale": 1.0},
        "model": {"type": "normal_normal", "sigma": 1.0, "name": "evil"},
        "theta_distribution": {"type": "uniform", "low": -5.0, "high": 5.0},
    }
    with pytest.raises(ValueError, match="Unexpected model kwargs"):
        ExperimentConfig.from_dict(bad)

    bad_prior = dict(bad)
    bad_prior["model"] = {"type": "normal_normal", "sigma": 1.0}
    bad_prior["prior"] = {"type": "normal", "loc": 0.0, "scale": 1.0, "junk": 5}
    with pytest.raises(ValueError, match="Unexpected prior kwargs"):
        ExperimentConfig.from_dict(bad_prior)


@pytest.mark.L1
@pytest.mark.properties
def test_experiment_config_rejects_unknown_scheme(bootstrapped_registry):
    with pytest.raises(ValueError, match="not in registry"):
        ExperimentConfig(
            scheme_name="not_a_real_scheme",
            statistic_name="waldo",
            prior=NormalDistribution(0.0, 1.0),
            model=NormalNormalModel(sigma=1.0),
            theta_distribution=UniformThetaDistribution(low=-5.0, high=5.0),
        )


@pytest.mark.L1
@pytest.mark.properties
def test_lhs_1d_stratification():
    """Each of the n strata in [low, high] receives exactly one sample."""
    td = UniformThetaDistribution(low=-3.0, high=3.0)
    n = 100
    samples = lhs_1d(td, n=n, seed=42)
    assert samples.shape == (n,)
    assert samples.min() >= -3.0
    assert samples.max() <= 3.0
    # Each stratum [lo + i/n*(hi-lo), lo + (i+1)/n*(hi-lo)) gets one sample.
    strata = np.floor((samples + 3.0) / 6.0 * n).astype(int)
    strata = np.clip(strata, 0, n - 1)
    assert len(np.unique(strata)) == n, (
        f"LHS not stratified: {n} samples but {len(np.unique(strata))} " f"distinct strata."
    )


# ---------------------------------------------------------------------------
# Model identity invariants (skeptic block #2)
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_model_name_not_constructor_kwarg():
    """`name` and `param_dim` are class-level constants, not kwargs.

    Stops a YAML / Python caller from constructing a "lying" model
    whose `.name` differs from its fingerprint's class string.
    """
    with pytest.raises(TypeError):
        BernoulliModel(name="evil")
    with pytest.raises(TypeError):
        NormalNormalModel(sigma=1.0, name="evil")
    with pytest.raises(TypeError):
        NormalNormalModel(sigma=1.0, param_dim=999)


@pytest.mark.L1
@pytest.mark.properties
def test_fingerprint_equality_invariant():
    """For our concrete models / priors / θ-dists, ``a == b`` ↔
    ``a.fingerprint() == b.fingerprint()``.

    Skeptic concern (#2): the selector validates by fingerprint, so a
    pair that's fingerprint-equal but `==`-unequal would be silently
    accepted. Class-level `name`/`param_dim` (now ClassVar) closes
    this gap; this test pins it.
    """
    pairs = [
        (NormalNormalModel(sigma=1.0), NormalNormalModel(sigma=1.0), True),
        (NormalNormalModel(sigma=1.0), NormalNormalModel(sigma=2.0), False),
        (BernoulliModel(), BernoulliModel(), True),
        (NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0), True),
        (NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 2.0), False),
        (
            UniformThetaDistribution(low=-1.0, high=1.0),
            UniformThetaDistribution(low=-1.0, high=1.0),
            True,
        ),
    ]
    for a, b, expected_eq in pairs:
        assert (a == b) == expected_eq
        assert (a.fingerprint() == b.fingerprint()) == expected_eq


# ---------------------------------------------------------------------------
# Skeptic block #4: scheme without tilted_pvalue
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_compute_pvalues_per_sample_without_tilted_pvalue_raises():
    """Stub schemes without ``tilted_pvalue`` raise loudly, not crash."""

    class NoTiltedPvalueScheme:
        name = "no_tp"

    with pytest.raises(AttributeError, match="does not implement"):
        compute_pvalues_per_sample(
            NoTiltedPvalueScheme(),
            np.array([0.0]),
            np.array([0.0]),
            NormalNormalModel(sigma=1.0),
            NormalDistribution(0.0, 1.0),
            np.array([0.5]),
            "waldo",
        )


# ---------------------------------------------------------------------------
# Skeptic block #5: OT raises on η below ``-w/(1-w)`` (closed-form ``s_t > 0``
# bound, audit P0-4) or non-finite. The W2 displacement line itself extends
# in both directions past the [0, 1] geodesic segment; only the s_t<=0 region
# is excluded for the closed-form WALDO p-value.
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_ot_tilted_pvalue_rejects_eta_out_of_admissible_range(
    bootstrapped_registry,
):
    from frasian._errors import TiltingDomainError
    from frasian.tilting.ot import OTTilting

    scheme = OTTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(0.0, 1.0)  # → w = 0.5, eta_lower = -1.0
    # Admissible covers the full open ray (-w/(1-w), ∞); spot-check that
    # the segment endpoints AND a point past the upper segment endpoint
    # AND a point in the (now-admissible) negative-extrapolation band
    # all return finite p ∈ [0, 1].
    for good_eta in (-0.5, 0.0, 0.5, 1.0, 1.5, 5.0):
        p = scheme.tilted_pvalue(
            np.array([0.0]), 0.0, model, prior, good_eta, "waldo"
        )
        p_scalar = float(np.asarray(p).reshape(-1)[0])
        assert np.isfinite(p_scalar) and 0.0 <= p_scalar <= 1.0, (
            f"OT η={good_eta} should be admissible (eta_lower=-1.0); got p={p_scalar}"
        )
    # Reject η at/below -w/(1-w) = -1.0 and non-finite η.
    for bad_eta in (-1.0 - 1e-6, -2.0, -10.0, float("inf"), float("nan")):
        with pytest.raises(TiltingDomainError, match=r"requires eta >"):
            scheme.tilted_pvalue(np.array([0.0]), 0.0, model, prior, bad_eta, "waldo")


@pytest.mark.L1
@pytest.mark.properties
def test_ot_tilted_pvalue_invalid_eta_yields_nan_in_validity_helper(
    bootstrapped_registry,
):
    """The validity helper sees NaN (not a fake-valid p) for OT η<0."""
    from frasian.tilting.ot import OTTilting

    scheme = OTTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(0.0, 1.0)
    p = compute_pvalues_per_sample(
        scheme,
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        model,
        prior,
        np.array([0.5, -2.0]),
        "waldo",
    )
    mask = validity_mask(p)
    assert mask[0]
    assert not mask[1], f"OT η=-2 should be invalid; helper returned p={p[1]}"


# ---------------------------------------------------------------------------
# Skeptic block #9 + #12: ExperimentConfig.__post_init__ guards
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_experiment_config_rejects_incompatible_scheme_statistic(
    bootstrapped_registry,
):
    """Wald only accepts identity tilting; a (power_law, wald) cell
    must be refused at construct time."""
    with pytest.raises(ValueError, match="does not accept"):
        ExperimentConfig(
            scheme_name="power_law",
            statistic_name="wald",
            prior=NormalDistribution(0.0, 1.0),
            model=NormalNormalModel(sigma=1.0),
            theta_distribution=UniformThetaDistribution(low=-5.0, high=5.0),
        )


@pytest.mark.L1
@pytest.mark.properties
def test_experiment_config_rejects_unbounded_theta_distribution(
    bootstrapped_registry,
):
    """Infinite-support θ-distribution → finiteness error at construct time."""

    class UnboundedTheta:
        name = "unbounded"

        def sample(self, n, rng):
            return rng.standard_normal(n)

        def support(self):
            return (-np.inf, np.inf)

        def fingerprint(self):
            return ("unbounded",)

    with pytest.raises(ValueError, match="must be finite"):
        ExperimentConfig(
            scheme_name="power_law",
            statistic_name="waldo",
            prior=NormalDistribution(0.0, 1.0),
            model=NormalNormalModel(sigma=1.0),
            theta_distribution=UnboundedTheta(),
        )


@pytest.mark.L1
@pytest.mark.properties
def test_experiment_config_rejects_unknown_top_level_keys(bootstrapped_registry):
    """A YAML typo (e.g., n_gird) must raise, not silently default."""
    bad = {
        "scheme_name": "power_law",
        "statistic_name": "waldo",
        "prior": {"type": "normal", "loc": 0.0, "scale": 1.0},
        "model": {"type": "normal_normal", "sigma": 1.0},
        "theta_distribution": {"type": "uniform", "low": -5.0, "high": 5.0},
        "n_gird": 401,  # typo
    }
    with pytest.raises(ValueError, match="Unexpected ExperimentConfig keys"):
        ExperimentConfig.from_dict(bad)


# ---------------------------------------------------------------------------
# E.2 round-2 invariants — width-loss + extract-NN params
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_extract_normal_normal_params_rejects_degenerate_w(bootstrapped_registry):
    """w → 0 (delta prior) and w → 1 (improper) raise.

    Validates the JAX-side compose-layer entry point. The
    legacy back-compat alias ``train._extract_normal_normal_params``
    still exists but ``train.py`` itself is rewired in commit 3.
    """
    from frasian.learned.training._losses_compose import extract_normal_normal_params

    # w very close to 0: prior.scale << model.sigma
    bad_prior_tight = NormalDistribution(0.0, 0.001)
    model = NormalNormalModel(sigma=1.0)
    with pytest.raises(ValueError, match="data weight"):
        extract_normal_normal_params(model, bad_prior_tight)

    # w very close to 1: prior.scale >> model.sigma
    bad_prior_wide = NormalDistribution(0.0, 1000.0)
    with pytest.raises(ValueError, match="data weight"):
        extract_normal_normal_params(model, bad_prior_wide)

    # Reasonable case still works.
    ok_prior = NormalDistribution(0.0, 1.0)
    w, mu0, sigma = extract_normal_normal_params(model, ok_prior)
    assert abs(w - 0.5) < 1e-9
    assert mu0 == 0.0
    assert sigma == 1.0


@pytest.mark.L1
@pytest.mark.properties
def test_ot_jax_pvalue_smooth_and_finite_inside_admissible_range():
    """OT JAX port returns finite values across the admissible range,
    plus a clamped surface for slightly-invalid η.

    The numpy ``OTTilting.tilted_pvalue`` raises ``TiltingDomainError``
    for η outside [0, 1] (driving the validity helper's labelling);
    the JAX port instead clamps ``s_t`` so the width-loss surface
    stays smooth and gradient-bearing even at slightly-invalid η.
    The boundary penalty (Head B) is what enforces admissibility on
    the trained EtaNet, not the JAX port.
    """
    from frasian.learned.training.pvalue_jax import ot_tilted_pvalue_jax

    theta = jnp.array([[0.0]])
    D = jnp.array([[0.0]])
    w = jnp.asarray(0.5)
    mu0 = jnp.asarray(0.0)
    sigma = jnp.asarray(1.0)

    # Inside admissible range: finite, in [0, 1].
    for good_eta in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = ot_tilted_pvalue_jax(
            theta, D, w, mu0, sigma, jnp.array([[good_eta]]), "waldo"
        )
        p_val = float(p.reshape(-1)[0])
        assert bool(jnp.isfinite(p).all())
        assert -1e-6 <= p_val <= 1.0 + 1e-6

    # Outside (slightly): the JAX port produces a finite gradient-
    # bearing surface (no NaN), so Head A's width loss can descend.
    # The validity helper raises in numpy-land; that's what gates
    # Head B's labels.
    for bad_eta in (-0.1, 1.1):
        p = ot_tilted_pvalue_jax(
            theta, D, w, mu0, sigma, jnp.array([[bad_eta]]), "waldo"
        )
        p_val = float(p.reshape(-1)[0])
        assert bool(jnp.isfinite(p).all()), (
            f"OT JAX port at η={bad_eta} returned non-finite p={p_val}; "
            "the clamp was supposed to keep it finite."
        )


@pytest.mark.L1
@pytest.mark.properties
def test_width_loss_averages_over_d_batch(bootstrapped_registry):
    """compose_width_loss takes a (B,) D array and returns a scalar.

    Skeptic block #1: the previous single-D estimator had high
    per-step variance, causing val_width to oscillate. Verify the
    new code accepts batched D and that gradient flows back into
    EtaNet's parameters.
    """
    from frasian.learned.training._losses_compose import compose_width_loss

    cfg = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(0.0, 1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-5.0, high=5.0),
        n_grid=51,
        n_lhs=20,
        eta_explore_box=(-2.0, 2.0),
    )
    eta_net = EtaNet(theta_dim=1, key=jax.random.PRNGKey(0))
    theta_grid_t = jnp.asarray(cfg.theta_grid)

    # Single D: 1-element loss.
    loss_1 = compose_width_loss(
        eta_net=eta_net,
        theta_grid_t=theta_grid_t,
        D_batch_t=jnp.array([0.0]),
        config=cfg,
        loss_kind="integrated_p",
        alpha=None,
    )
    assert loss_1.ndim == 0
    # Batch of 8 D: scalar (mean over batch).
    loss_8 = compose_width_loss(
        eta_net=eta_net,
        theta_grid_t=theta_grid_t,
        D_batch_t=jnp.linspace(-2.0, 2.0, 8),
        config=cfg,
        loss_kind="integrated_p",
        alpha=None,
    )
    assert loss_8.ndim == 0

    # Gradient flows back into EtaNet via the loss.
    en_params, en_static = eqx.partition(eta_net, eqx.is_array)

    def loss_fn(p):
        en2 = eqx.combine(p, en_static)
        return compose_width_loss(
            eta_net=en2,
            theta_grid_t=theta_grid_t,
            D_batch_t=jnp.linspace(-2.0, 2.0, 8),
            config=cfg,
            loss_kind="integrated_p",
            alpha=None,
        )

    g = jax.grad(loss_fn)(en_params)
    g_norms = [float(jnp.abs(x).sum()) for x in jtu.tree_leaves(g)]
    assert any(n > 0 for n in g_norms)


# ---------------------------------------------------------------------------
# Skeptic block #13: tighten "no grad to ValidityNet" check (JAX flavour)
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_params_get_zero_grad_from_head_a():
    """After Head A's loss backward, ValidityNet params must receive
    exactly zero gradient — not "small but nonzero" — under the
    ``eqx.partition`` + ``stop_gradient`` boundary.

    JAX has no `.grad is None` analogue (gradients are returned
    structurally, not attached to leaves), so the Equinox-faithful
    check is "every leaf is exactly zero". This is what
    ``stop_gradient`` guarantees algebraically.
    """
    k1, k2 = _make_keys(2, seed=2)
    eta_net = EtaNet(theta_dim=1, key=k1)
    val_net = ValidityNet(theta_dim=1, key=k2)
    theta = jnp.linspace(-2.0, 2.0, 11)
    eta_pred = eta_net(theta)

    vn_params, vn_static = eqx.partition(val_net, eqx.is_array)

    def loss_fn(vn_p):
        vn2 = eqx.combine(vn_p, vn_static)
        return compose_boundary_penalty(
            val_net=vn2, theta_batch_t=theta, eta_pred=eta_pred
        )

    g_vn = jax.grad(loss_fn)(vn_params)
    for leaf in jtu.tree_leaves(g_vn):
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(np.asarray(leaf)))
