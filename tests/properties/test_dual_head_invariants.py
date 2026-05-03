"""Phase E dual-head architecture invariants.

Covers:
- ``EtaNet`` / ``ValidityNet`` forward shapes for theta_dim ∈ {1, 3}.
- ``boundary_penalty_from_validity`` gradcheck and clamping bounds.
- Detachment: Head B's params do *not* receive gradient from Head A's
  loss, but Head A's params *do* (gradient flows through the input).
- Detachment: Head A's params do *not* receive gradient from Head B's
  BCE loss when ``eta_pred`` is detached at the input.
- ``is_pair_valid`` / ``validity_mask`` / ``compute_pvalues_per_sample``
  per-point semantics + NaN-on-failure handling.
- ``ExperimentConfig`` round-trip through dict + YAML.
- ``Prior.fingerprint`` / ``Model.fingerprint`` hashability.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from frasian.learned.training.architecture import EtaNet, ValidityNet
from frasian.learned.training.losses import boundary_penalty_from_validity
from frasian.learned.training.sampling import (
    ExperimentConfig,
    UniformThetaDistribution,
    lhs_1d,
)
from frasian.learned.training.validity import (
    compute_pvalues_per_sample,
    is_pair_valid,
    validity_mask,
)
from frasian.models.distributions import (
    BetaDistribution,
    NormalDistribution,
)
from frasian.models.normal_normal import NormalNormalModel
from frasian.models.bernoulli import BernoulliModel
from frasian.tilting.power_law import PowerLawTilting


# ---------------------------------------------------------------------------
# Architecture forward shapes
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_forward_1d():
    net = EtaNet(theta_dim=1)
    theta = torch.linspace(-3.0, 3.0, 11)
    out = net(theta)
    assert out.shape == (11,)
    # 1D-input convenience: also accept (N, 1).
    out2 = net(theta.unsqueeze(-1))
    assert out2.shape == (11,)
    torch.testing.assert_close(out, out2)


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_forward_vector_theta():
    net = EtaNet(theta_dim=3)
    theta = torch.randn(7, 3)
    out = net(theta)
    assert out.shape == (7,)


@pytest.mark.L1
@pytest.mark.properties
def test_eta_net_rejects_wrong_dim():
    net = EtaNet(theta_dim=2)
    with pytest.raises(ValueError, match="theta_dim=2"):
        net(torch.randn(5, 3))
    # 1D input on theta_dim=2 net is also rejected.
    with pytest.raises(ValueError, match="theta_dim=2"):
        net(torch.randn(5))


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_forward():
    net = ValidityNet(theta_dim=1)
    inputs = torch.randn(13, 2)
    out = net(inputs)
    assert out.shape == (13,)


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_forward_vector():
    net = ValidityNet(theta_dim=4)
    inputs = torch.randn(8, 5)  # theta_dim + 1
    out = net(inputs)
    assert out.shape == (8,)


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_rejects_wrong_input_shape():
    net = ValidityNet(theta_dim=1)
    with pytest.raises(ValueError, match="theta_dim=1"):
        net(torch.randn(5, 3))


# ---------------------------------------------------------------------------
# Boundary penalty
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_boundary_penalty_value_behavior():
    """Penalty is ~|logit| for very negative (linear), ~0 for very positive."""
    very_invalid = torch.tensor([-100.0])
    very_valid = torch.tensor([+100.0])
    p_inv = boundary_penalty_from_validity(very_invalid).item()
    p_val = boundary_penalty_from_validity(very_valid).item()
    assert 99.0 < p_inv < 101.0, (
        f"penalty for very-invalid logit should be ~100, got {p_inv}"
    )
    assert p_val < 1e-6, (
        f"penalty for very-valid logit should be ~0, got {p_val}"
    )


@pytest.mark.L1
@pytest.mark.properties
def test_boundary_penalty_gradient_alive_at_extremes():
    """Gradient must NOT vanish for very negative logits.

    The wrong-side gradient saturates at -1 (since
    ``d(-logsigmoid(x))/dx = -sigmoid(-x) → -1`` as x → -∞), so
    the boundary signal stays alive even when ValidityNet is
    extremely confident the point is invalid.
    """
    # Single-element batch so the mean derivative equals the per-element one.
    very_neg = torch.tensor([-50.0], requires_grad=True)
    boundary_penalty_from_validity(very_neg).backward()
    assert very_neg.grad is not None
    # Expected: -sigmoid(50) ≈ -1.0
    assert torch.allclose(
        very_neg.grad,
        torch.tensor([-1.0]),
        atol=1e-6,
    ), f"wrong-side gradient should saturate at -1, got {very_neg.grad}"

    # Right side: gradient → 0 as logit → +∞ (no penalty to apply).
    very_pos = torch.tensor([+50.0], requires_grad=True)
    boundary_penalty_from_validity(very_pos).backward()
    assert very_pos.grad is not None
    assert torch.allclose(
        very_pos.grad,
        torch.tensor([0.0]),
        atol=1e-6,
    ), f"right-side gradient should saturate at 0, got {very_pos.grad}"


@pytest.mark.L1
@pytest.mark.properties
def test_boundary_penalty_gradcheck():
    """gradcheck against torch.autograd.gradcheck on small input."""
    x = torch.randn(5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        boundary_penalty_from_validity, (x,), eps=1e-6, atol=1e-5
    )


@pytest.mark.L1
@pytest.mark.properties
def test_boundary_penalty_gradcheck_at_large_logits():
    """gradcheck at logits well outside the previously-clamped range.

    Catches a regression where a clamp was reintroduced and gradient
    silently zeroed at extreme inputs. Uses larger atol since
    logsigmoid is asymptotically linear with finite-difference
    truncation error growing with |x|.
    """
    x = torch.linspace(-30.0, 30.0, 7, dtype=torch.float64)
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(
        boundary_penalty_from_validity, (x,), eps=1e-5, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Detachment in the dual-head training loop
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_head_b_params_no_grad_from_head_a_loss():
    """Boundary penalty via functional_call must not gradient Head B."""
    eta_net = EtaNet(theta_dim=1)
    val_net = ValidityNet(theta_dim=1)
    # Forward Head A, then evaluate Head B with detached params.
    theta = torch.linspace(-2.0, 2.0, 11)
    eta_pred = eta_net(theta)
    v_p = {k: v.detach() for k, v in val_net.named_parameters()}
    v_b = {k: v.detach() for k, v in val_net.named_buffers()}
    inputs = torch.stack([theta, eta_pred], dim=-1)
    logits = torch.func.functional_call(val_net, (v_p, v_b), (inputs,))
    penalty = boundary_penalty_from_validity(logits)
    penalty.backward()
    # EtaNet params should have grad (Head A is what we're training).
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in eta_net.parameters()
    ), "EtaNet did not receive boundary-penalty gradient"
    # ValidityNet params should NOT have grad — functional_call detach.
    assert all(
        p.grad is None or p.grad.abs().sum().item() == 0
        for p in val_net.parameters()
    ), "ValidityNet erroneously received gradient through detached call"


@pytest.mark.L1
@pytest.mark.properties
def test_head_a_params_no_grad_from_head_b_loss():
    """When eta_pred is detached, Head B's BCE must not gradient EtaNet."""
    eta_net = EtaNet(theta_dim=1)
    val_net = ValidityNet(theta_dim=1)
    theta = torch.linspace(-2.0, 2.0, 11)
    eta_pred = eta_net(theta)
    inputs = torch.stack([theta, eta_pred.detach()], dim=-1)
    logits = val_net(inputs)
    target = torch.zeros_like(logits)  # arbitrary BCE target
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    assert all(
        p.grad is None or p.grad.abs().sum().item() == 0
        for p in eta_net.parameters()
    ), "EtaNet erroneously received gradient via Head B loss"
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in val_net.parameters()
    ), "ValidityNet did not receive BCE gradient"


# ---------------------------------------------------------------------------
# Validity helpers
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
def test_compute_pvalues_per_sample_shape_mismatch():
    scheme = PowerLawTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    with pytest.raises(ValueError, match="share shape"):
        compute_pvalues_per_sample(
            scheme,
            np.array([0.0, 1.0]),
            np.array([0.0]),
            model, prior,
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
    cfg = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(0.0, 1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-5.0, high=5.0),
        n_grid=101,
        n_lhs=200,
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
    assert (
        cfg2.theta_distribution.fingerprint()
        == cfg.theta_distribution.fingerprint()
    )
    assert cfg2.n_grid == cfg.n_grid
    assert cfg2.n_lhs == cfg.n_lhs
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
    yaml = pytest.importorskip("yaml")
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
def test_lhs_1d_covers_support():
    td = UniformThetaDistribution(low=-3.0, high=3.0)
    samples = lhs_1d(td, n=200, seed=42)
    # LHS should cover both halves of the support.
    assert (samples < 0).any() and (samples > 0).any()
    assert samples.min() >= -3.0
    assert samples.max() <= 3.0
    assert samples.shape == (200,)
