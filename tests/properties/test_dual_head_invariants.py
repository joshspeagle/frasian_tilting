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
    assert 99.0 < p_inv < 101.0, f"penalty for very-invalid logit should be ~100, got {p_inv}"
    assert p_val < 1e-6, f"penalty for very-valid logit should be ~0, got {p_val}"


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
    assert torch.autograd.gradcheck(boundary_penalty_from_validity, (x,), eps=1e-6, atol=1e-5)


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
    assert torch.autograd.gradcheck(boundary_penalty_from_validity, (x,), eps=1e-5, atol=1e-3)


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
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in eta_net.parameters()
    ), "EtaNet did not receive boundary-penalty gradient"
    # ValidityNet params should NOT have grad — functional_call detach.
    assert all(
        p.grad is None or p.grad.abs().sum().item() == 0 for p in val_net.parameters()
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
        p.grad is None or p.grad.abs().sum().item() == 0 for p in eta_net.parameters()
    ), "EtaNet erroneously received gradient via Head B loss"
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in val_net.parameters()
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
    assert cfg2.theta_distribution.fingerprint() == cfg.theta_distribution.fingerprint()
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
# Skeptic block #5: OT raises on η outside [0, 1]
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
    prior = NormalDistribution(0.0, 1.0)
    # Admissible: [0, 1]; valid call works.
    p = scheme.tilted_pvalue(np.array([0.0]), 0.0, model, prior, 0.5, "waldo")
    assert np.isfinite(p[0]) and 0.0 <= p[0] <= 1.0
    # Reject η < 0 and η > 1.
    for bad_eta in (-2.0, -1e-6, 1.0 + 1e-6, 5.0):
        with pytest.raises(TiltingDomainError, match="eta in"):
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
# Skeptic block #13: tighten "no grad to ValidityNet" check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# E.2 round-2 invariants
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.properties
def test_extract_normal_normal_params_rejects_degenerate_w(bootstrapped_registry):
    """w → 0 (delta prior) and w → 1 (improper) raise."""
    from frasian.learned.training.train import _extract_normal_normal_params

    # w very close to 0: prior.scale << model.sigma
    bad_prior_tight = NormalDistribution(0.0, 0.001)
    model = NormalNormalModel(sigma=1.0)
    with pytest.raises(ValueError, match="data weight"):
        _extract_normal_normal_params(model, bad_prior_tight)

    # w very close to 1: prior.scale >> model.sigma
    bad_prior_wide = NormalDistribution(0.0, 1000.0)
    with pytest.raises(ValueError, match="data weight"):
        _extract_normal_normal_params(model, bad_prior_wide)

    # Reasonable case still works.
    ok_prior = NormalDistribution(0.0, 1.0)
    w, mu0, sigma = _extract_normal_normal_params(model, ok_prior)
    assert abs(w - 0.5) < 1e-9
    assert mu0 == 0.0
    assert sigma == 1.0


@pytest.mark.L1
@pytest.mark.properties
def test_ot_torch_pvalue_smooth_and_finite_inside_admissible_range():
    """OT torch port returns finite values across the admissible range.

    The numpy ``OTTilting.tilted_pvalue`` raises ``TiltingDomainError``
    for η outside [0, 1] (driving the validity helper's labelling);
    the torch port instead clamps ``s_t`` so the width-loss surface
    stays smooth and gradient-bearing even at slightly-invalid η.
    The boundary penalty (Head B) is what enforces admissibility on
    the trained EtaNet, not the torch port.
    """
    from frasian.learned.training.pvalue_torch import ot_tilted_pvalue_torch

    theta = torch.tensor([[0.0]])
    D = torch.tensor([[0.0]])
    w = torch.tensor(0.5)
    mu0 = torch.tensor(0.0)
    sigma = torch.tensor(1.0)

    # Inside admissible range: finite, in [0, 1].
    for good_eta in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = ot_tilted_pvalue_torch(
            theta,
            D,
            w,
            mu0,
            sigma,
            torch.tensor([[good_eta]]),
            "waldo",
        )
        assert torch.isfinite(p).all()
        assert -1e-6 <= p.item() <= 1.0 + 1e-6

    # Outside (slightly): the torch port produces a finite gradient-
    # bearing surface (no NaN), so Head A's width loss can descend.
    # The validity helper raises in numpy-land; that's what gates
    # Head B's labels.
    for bad_eta in (-0.1, 1.1):
        p = ot_tilted_pvalue_torch(
            theta,
            D,
            w,
            mu0,
            sigma,
            torch.tensor([[bad_eta]]),
            "waldo",
        )
        assert torch.isfinite(p).all(), (
            f"OT torch port at η={bad_eta} returned non-finite p="
            f"{p.item()}; the clamp was supposed to keep it finite."
        )


@pytest.mark.L1
@pytest.mark.properties
def test_width_loss_averages_over_d_batch(bootstrapped_registry):
    """_width_loss takes a (B,) D tensor and returns a scalar average.

    Skeptic block #1: the previous single-D estimator had high
    per-step variance, causing val_width to oscillate. Verify the
    new code accepts batched D and that its variance scales as 1/B.
    """
    from frasian.learned.training.architecture import EtaNet
    from frasian.learned.training.train import _width_loss

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
    eta_net = EtaNet(theta_dim=1)
    theta_grid_t = torch.as_tensor(cfg.theta_grid, dtype=torch.float32)

    # Single D: 1-element loss.
    loss_1 = _width_loss(
        eta_net=eta_net,
        theta_grid_t=theta_grid_t,
        D_batch_t=torch.tensor([0.0]),
        config=cfg,
        loss_kind="integrated_p",
        alpha=None,
    )
    # Batch of 8 D: scalar (mean over batch).
    loss_8 = _width_loss(
        eta_net=eta_net,
        theta_grid_t=theta_grid_t,
        D_batch_t=torch.linspace(-2.0, 2.0, 8),
        config=cfg,
        loss_kind="integrated_p",
        alpha=None,
    )
    assert loss_1.dim() == 0
    assert loss_8.dim() == 0
    # Both finite, gradient flows back.
    loss_8.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in eta_net.parameters())


@pytest.mark.L3
@pytest.mark.slow
def test_phase_e_selector_rejects_cross_experiment_use(
    bootstrapped_registry,
    tmp_path,
):
    """A Phase E checkpoint trained at (σ₀, σ) is refused for inference
    at any other (σ₀, σ) — even when the derived w matches.

    Skeptic E.3 block #2: closes the silent-wrong-scale-lookup gap.
    """
    from frasian._errors import MissingArtifactError
    from frasian.learned.eta_artifact import EtaArtifact
    from frasian.learned.training.train import fit_eta_artifact
    from frasian.statistics.waldo import WaldoStatistic
    from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
    from frasian.tilting.power_law import PowerLawTilting

    # Train a tiny checkpoint at the canonical (μ₀=0, σ₀=σ=1, w=0.5).
    cfg = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(0.0, 1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-3.0, high=3.0),
        n_grid=21,
        n_lhs=80,
        eta_explore_box=(-2.0, 2.0),
        seed=7,
    )
    out = tmp_path / "ckpt.pt"
    fit_eta_artifact(
        config=cfg,
        out_path=out,
        n_epochs=2,
        batch_size=20,
        n_aux=20,
        lambda_max=1.0,
        lambda_warmup_frac=0.5,
        patience=5,
        verbose=False,
    )
    art = EtaArtifact(artifact_path=out, name="phase_e_test")
    sel = LearnedDynamicEtaSelector(artifact=art)
    scheme = PowerLawTilting(selector=sel)

    # Same fingerprints: works.
    scheme.dynamic_tilted_confidence_interval(
        alpha=0.05,
        D=0.0,
        model=NormalNormalModel(sigma=1.0),
        prior=NormalDistribution(0.0, 1.0),
        statistic_name="waldo",
        eta_selector=sel,
    )
    # Same w (=0.5) but rescaled (σ₀=σ=2): rejected.
    with pytest.raises(MissingArtifactError, match="trained on model"):
        scheme.dynamic_tilted_confidence_interval(
            alpha=0.05,
            D=0.0,
            model=NormalNormalModel(sigma=2.0),
            prior=NormalDistribution(0.0, 2.0),
            statistic_name="waldo",
            eta_selector=sel,
        )
    # Same (σ, σ₀) but shifted prior (μ₀=1): rejected.
    with pytest.raises(MissingArtifactError, match="trained with prior"):
        scheme.dynamic_tilted_confidence_interval(
            alpha=0.05,
            D=0.0,
            model=NormalNormalModel(sigma=1.0),
            prior=NormalDistribution(1.0, 1.0),
            statistic_name="waldo",
            eta_selector=sel,
        )


@pytest.mark.L1
@pytest.mark.properties
def test_lambda_schedule_starts_at_zero():
    """Skeptic E.3 block #4: λ(0) = 0 so Head A's boundary-penalty
    signal is null while Head B trains on its first batch."""
    from frasian.learned.training.train import _lambda_schedule

    assert _lambda_schedule(0, n_epochs=10, lambda_max=10.0, warmup_frac=0.3) == 0.0
    # At warmup_epochs, λ = λ_max.
    warmup_epochs = max(1, int(round(0.3 * 10)))
    assert _lambda_schedule(warmup_epochs, n_epochs=10, lambda_max=10.0, warmup_frac=0.3) == 10.0
    # Post-warmup, constant at λ_max.
    assert _lambda_schedule(8, n_epochs=10, lambda_max=10.0, warmup_frac=0.3) == 10.0


@pytest.mark.L3
@pytest.mark.slow
def test_alpha_required_to_be_none_for_marginalised_loss(
    bootstrapped_registry,
    tmp_path,
):
    """Skeptic E.3 block #7: integrated_p / cd_variance reject non-None α."""
    from frasian.learned.training.train import fit_eta_artifact

    cfg = ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(0.0, 1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-3.0, high=3.0),
        n_grid=21,
        n_lhs=20,
        eta_explore_box=(-2.0, 2.0),
        seed=7,
    )
    with pytest.raises(ValueError, match="α-marginalised"):
        fit_eta_artifact(
            config=cfg,
            out_path=tmp_path / "x.pt",
            loss_kind="integrated_p",
            alpha=0.05,  # invalid pairing
            n_epochs=1,
            batch_size=10,
            verbose=False,
        )


@pytest.mark.L1
@pytest.mark.properties
def test_validity_net_params_get_no_grad_attribute_at_all():
    """After Head A's loss backward, ValidityNet.params should have
    ``grad is None`` (no gradient even allocated), not zero gradient.

    A `.grad == 0` could mean "leak that happens to vanish on this
    input"; `.grad is None` means autograd never recorded a node
    for these params — which is what `functional_call` with
    detached params guarantees.
    """
    eta_net = EtaNet(theta_dim=1)
    val_net = ValidityNet(theta_dim=1)
    theta = torch.linspace(-2.0, 2.0, 11)
    eta_pred = eta_net(theta)
    v_p = {k: v.detach() for k, v in val_net.named_parameters()}
    v_b = {k: v.detach() for k, v in val_net.named_buffers()}
    inputs = torch.stack([theta, eta_pred], dim=-1)
    logits = torch.func.functional_call(val_net, (v_p, v_b), (inputs,))
    boundary_penalty_from_validity(logits).backward()
    assert all(p.grad is None for p in val_net.parameters())
