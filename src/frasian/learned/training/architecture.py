"""Two-head architecture for the Phase E learned-η selector.

- ``EtaNet``: smooth GELU-MLP from θ ∈ R^p to a raw real η. No
  monotonicity prior, no bounded output. Smoothness comes from
  GELU activations + finite parameter count; validity (η inside
  the admissible region of the tilting scheme) is enforced via a
  loss penalty driven by Head B, not by architecture.

- ``ValidityNet``: MLP from (θ, η) to a single logit. Trained on
  validity labels collected during training (whether
  ``scheme.tilted_pvalue`` returned a finite-and-in-[0,1] scalar
  for that (θ, η) pair). Provides ``-log P(valid | θ, η)`` as a
  boundary penalty for Head A's loss.

Both default to ``theta_dim=1`` (today's Normal-Normal setting)
but accept any ``theta_dim ≥ 1`` so future vector-θ models do not
require a refactor.

Implementation note (Phase F port commit 2). The two heads are now
``equinox.Module`` subclasses (PyTrees of ``jax.Array`` leaves) rather
than ``torch.nn.Module``. Forward semantics, shape rules, and
architecture_kwargs() are unchanged. Weights are Xavier-normal +
zero biases, mirroring the original ``nn.init.xavier_normal_`` /
``nn.init.zeros_``; the per-layer init is implemented explicitly via
``eqx.tree_at`` since ``eqx.nn.MLP`` defaults to LeCun-uniform.
"""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ... import _jax_setup as _x64  # noqa: F401  — ensure float64 active

# Architecture version, single source of truth in ``_checkpoint.py``
# (kept torch-free so ``arch_spec_sha`` is importable without torch).
# Bump rule on the canonical version string is documented at
# ``_checkpoint._architecture_version``.
from ._checkpoint import _architecture_version as _read_arch_version

_FORCE_X64 = _x64  # keep static-analysis from stripping the import

__version__: str = _read_arch_version()


def _xavier_normal_layers(
    mlp: eqx.nn.MLP,
    key: jax.Array,
) -> eqx.nn.MLP:
    """Replace each ``Linear`` layer's weight with Xavier-normal samples,
    biases with zero. Returns a new MLP (Equinox modules are immutable).

    Mirrors ``torch.nn.init.xavier_normal_`` (gain=1, fan_avg) and
    ``torch.nn.init.zeros_``: ``W ~ N(0, 2 / (fan_in + fan_out))`` and
    ``b = 0``. The torch and JAX RNG primitives differ even at the
    same seed, so retrained checkpoints have different weights — see
    the Phase F port commit message for the headline-narrowness
    drift expectations.
    """
    keys = jax.random.split(key, len(mlp.layers))
    new_layers: list[eqx.nn.Linear] = []
    for layer, sub_key in zip(mlp.layers, keys):
        if not isinstance(layer, eqx.nn.Linear):
            new_layers.append(layer)
            continue
        out_features, in_features = layer.weight.shape
        std = math.sqrt(2.0 / (in_features + out_features))
        new_weight = jax.random.normal(sub_key, (out_features, in_features)) * std
        new_bias = (
            jnp.zeros_like(layer.bias) if layer.bias is not None else None
        )
        # Replace weight + bias on this Linear layer via eqx.tree_at.
        if new_bias is None:
            layer = eqx.tree_at(lambda m: m.weight, layer, new_weight)
        else:
            layer = eqx.tree_at(
                lambda m: (m.weight, m.bias),
                layer,
                (new_weight, new_bias),
            )
        new_layers.append(layer)
    return eqx.tree_at(lambda m: m.layers, mlp, tuple(new_layers))


def _build_mlp(
    in_features: int,
    hidden_sizes: tuple[int, ...],
    out_features: int,
    key: jax.Array,
) -> eqx.nn.MLP:
    """GELU-activated MLP: in → h0 → ... → out (linear final), with
    Xavier-normal weights and zero biases.

    The original torch path used a non-uniform-hidden-sizes-friendly
    ``nn.Sequential`` of ``Linear → GELU`` blocks. ``eqx.nn.MLP``
    requires a uniform ``width_size`` and ``depth``; the existing
    Phase E architecture uses ``hidden_sizes = (64, 64)`` (uniform),
    so we delegate to ``eqx.nn.MLP`` and reject non-uniform widths
    explicitly. Callers who need non-uniform widths must extend this
    helper to build a custom MLP from explicit ``eqx.nn.Linear``
    layers.
    """
    if len(hidden_sizes) == 0:
        raise ValueError("hidden_sizes must be non-empty")
    width = hidden_sizes[0]
    if any(h != width for h in hidden_sizes):
        raise ValueError(
            "non-uniform hidden_sizes not supported by the equinox port; "
            f"got {hidden_sizes}. Extend _build_mlp to assemble explicit "
            "eqx.nn.Linear layers if a future variant needs this."
        )
    init_key, build_key = jax.random.split(key)
    mlp = eqx.nn.MLP(
        in_size=in_features,
        out_size=out_features,
        width_size=width,
        depth=len(hidden_sizes),
        activation=jax.nn.gelu,
        # final activation defaults to identity
        key=build_key,
    )
    return _xavier_normal_layers(mlp, init_key)


class EtaNet(eqx.Module):
    """Conditional MLP: (θ, prior_hp, lik_hp) → η.

    Phase G three-block input. The MLP learns η_φ(θ | prior_hp, lik_hp)
    so a single checkpoint covers any (prior, likelihood) configuration
    in the trained hyperparameter ranges. No bounded output, no
    monotonicity constraint; validity enforced via the Head-B
    boundary penalty.

    Equivariance not exploited: for Normal-Normal, the loss is
    invariant under translation/scale of (θ, μ₀, σ₀, σ); the network
    learns it implicitly. A regression test in
    ``tests/regression/test_conditional_eta_equivariance.py`` checks
    this on the trained checkpoint.

    Parameters
    ----------
    theta_dim : int
        Dimension of θ (always 1 for current scalar models).
    prior_dim : int
        ``prior.hyperparam_dim`` — e.g. 2 for NormalDistribution.
    lik_dim : int
        ``model.hyperparam_dim`` — 1 for NormalNormalModel,
        0 for BernoulliModel.
    hidden_sizes : tuple of int
        Hidden layer widths. Defaults to ``(128, 128, 128)``.
    key : jax.Array
        PRNG key for weight init. Required.
    """

    mlp: eqx.nn.MLP
    theta_dim: int = eqx.field(static=True)
    prior_dim: int = eqx.field(static=True)
    lik_dim: int = eqx.field(static=True)
    hidden_sizes: tuple[int, ...] = eqx.field(static=True)
    feature_loc: tuple[float, ...] = eqx.field(static=True)
    feature_scale: tuple[float, ...] = eqx.field(static=True)
    feature_log: tuple[bool, ...] = eqx.field(static=True)

    def __init__(
        self,
        theta_dim: int,
        prior_dim: int,
        lik_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128, 128),
        feature_loc: tuple[float, ...] | None = None,
        feature_scale: tuple[float, ...] | None = None,
        feature_log: tuple[bool, ...] | None = None,
        *,
        key: jax.Array,
    ):
        if theta_dim < 1:
            raise ValueError(f"theta_dim must be >= 1, got {theta_dim}")
        if prior_dim < 0:
            raise ValueError(f"prior_dim must be >= 0, got {prior_dim}")
        if lik_dim < 0:
            raise ValueError(f"lik_dim must be >= 0, got {lik_dim}")
        self.theta_dim = int(theta_dim)
        self.prior_dim = int(prior_dim)
        self.lik_dim = int(lik_dim)
        self.hidden_sizes = tuple(hidden_sizes)
        in_features = self.theta_dim + self.prior_dim + self.lik_dim
        self.mlp = _build_mlp(in_features, self.hidden_sizes, 1, key)
        # Normalization defaults: identity (loc=0, scale=1, no log) so any
        # caller that omits them gets the pre-Phase-G-fix behavior.
        if feature_loc is None:
            feature_loc = (0.0,) * in_features
        if feature_scale is None:
            feature_scale = (1.0,) * in_features
        if feature_log is None:
            feature_log = (False,) * in_features
        if len(feature_loc) != in_features:
            raise ValueError(
                f"feature_loc must have len {in_features}; got {len(feature_loc)}."
            )
        if len(feature_scale) != in_features:
            raise ValueError(
                f"feature_scale must have len {in_features}; got {len(feature_scale)}."
            )
        if len(feature_log) != in_features:
            raise ValueError(
                f"feature_log must have len {in_features}; got {len(feature_log)}."
            )
        self.feature_loc = tuple(float(v) for v in feature_loc)
        self.feature_scale = tuple(float(v) for v in feature_scale)
        self.feature_log = tuple(bool(v) for v in feature_log)

    def __call__(
        self,
        theta: jax.Array,
        prior_hp: jax.Array,
        lik_hp: jax.Array,
    ) -> jax.Array:
        """Forward. All inputs share batch size N on axis 0.

        theta    : ``(N,)`` for ``theta_dim==1``, or ``(N, theta_dim)``.
        prior_hp : ``(N, prior_dim)``.
        lik_hp   : ``(N, lik_dim)``.
        returns  : ``(N,)`` raw η values.
        """
        if theta.ndim == 1:
            if self.theta_dim != 1:
                raise ValueError(
                    f"EtaNet(theta_dim={self.theta_dim}) requires (N, "
                    f"{self.theta_dim}) theta input; got 1D shape "
                    f"{tuple(theta.shape)}."
                )
            theta_2d = theta[:, None]
        elif theta.ndim == 2:
            if theta.shape[-1] != self.theta_dim:
                raise ValueError(
                    f"EtaNet(theta_dim={self.theta_dim}) expected theta "
                    f"shape (N, {self.theta_dim}); got {tuple(theta.shape)}."
                )
            theta_2d = theta
        else:
            raise ValueError(
                f"EtaNet expects 1D or 2D theta; got shape {tuple(theta.shape)}."
            )
        N = theta_2d.shape[0]
        if prior_hp.shape != (N, self.prior_dim):
            raise ValueError(
                f"EtaNet(prior_dim={self.prior_dim}) expected prior_hp "
                f"shape ({N}, {self.prior_dim}); got {tuple(prior_hp.shape)}."
            )
        if lik_hp.shape != (N, self.lik_dim):
            raise ValueError(
                f"EtaNet(lik_dim={self.lik_dim}) expected lik_hp shape "
                f"({N}, {self.lik_dim}); got {tuple(lik_hp.shape)}."
            )
        x = jnp.concatenate([theta_2d, prior_hp, lik_hp], axis=-1)
        loc = jnp.asarray(self.feature_loc)
        scale = jnp.asarray(self.feature_scale)
        log_mask = jnp.asarray(self.feature_log)
        x_log = jnp.log(jnp.maximum(x, 1e-12))
        x = jnp.where(log_mask, x_log, x)
        x = (x - loc) / scale
        out = jax.vmap(self.mlp)(x)
        return out[..., 0]

    def architecture_kwargs(self) -> dict[str, Any]:
        return {
            "theta_dim": self.theta_dim,
            "prior_dim": self.prior_dim,
            "lik_dim": self.lik_dim,
            "hidden_sizes": self.hidden_sizes,
            "feature_loc": list(self.feature_loc),
            "feature_scale": list(self.feature_scale),
            "feature_log": list(self.feature_log),
        }


class ValidityNet(eqx.Module):
    """Conditional MLP: (θ, prior_hp, lik_hp, η) → logit.

    Phase G four-block input. Trained on validity labels via
    BCE-with-logits. The logit output is intentional — sigmoid is
    applied at loss time. ``log_sigmoid`` is numerically stable for any
    finite input.
    """

    mlp: eqx.nn.MLP
    theta_dim: int = eqx.field(static=True)
    prior_dim: int = eqx.field(static=True)
    lik_dim: int = eqx.field(static=True)
    hidden_sizes: tuple[int, ...] = eqx.field(static=True)
    feature_loc: tuple[float, ...] = eqx.field(static=True)
    feature_scale: tuple[float, ...] = eqx.field(static=True)
    feature_log: tuple[bool, ...] = eqx.field(static=True)

    def __init__(
        self,
        theta_dim: int,
        prior_dim: int,
        lik_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128, 128),
        feature_loc: tuple[float, ...] | None = None,
        feature_scale: tuple[float, ...] | None = None,
        feature_log: tuple[bool, ...] | None = None,
        *,
        key: jax.Array,
    ):
        if theta_dim < 1:
            raise ValueError(f"theta_dim must be >= 1, got {theta_dim}")
        if prior_dim < 0 or lik_dim < 0:
            raise ValueError(
                f"prior_dim and lik_dim must be >= 0; got {prior_dim}, {lik_dim}"
            )
        self.theta_dim = int(theta_dim)
        self.prior_dim = int(prior_dim)
        self.lik_dim = int(lik_dim)
        self.hidden_sizes = tuple(hidden_sizes)
        in_features = self.theta_dim + self.prior_dim + self.lik_dim + 1
        self.mlp = _build_mlp(in_features, self.hidden_sizes, 1, key)
        if feature_loc is None:
            feature_loc = (0.0,) * in_features
        if feature_scale is None:
            feature_scale = (1.0,) * in_features
        if feature_log is None:
            feature_log = (False,) * in_features
        for nm, val in (("loc", feature_loc), ("scale", feature_scale), ("log", feature_log)):
            if len(val) != in_features:
                raise ValueError(
                    f"feature_{nm} must have len {in_features} (theta+prior+lik+eta); "
                    f"got {len(val)}."
                )
        self.feature_loc = tuple(float(v) for v in feature_loc)
        self.feature_scale = tuple(float(v) for v in feature_scale)
        self.feature_log = tuple(bool(v) for v in feature_log)

    def __call__(
        self,
        theta: jax.Array,
        prior_hp: jax.Array,
        lik_hp: jax.Array,
        eta: jax.Array,
    ) -> jax.Array:
        if theta.ndim == 1:
            if self.theta_dim != 1:
                raise ValueError(
                    f"ValidityNet(theta_dim={self.theta_dim}) requires "
                    f"(N, {self.theta_dim}) theta input; got 1D."
                )
            theta_2d = theta[:, None]
        elif theta.ndim == 2 and theta.shape[-1] == self.theta_dim:
            theta_2d = theta
        else:
            raise ValueError(
                f"ValidityNet expected theta (N,) or (N, {self.theta_dim}); "
                f"got {tuple(theta.shape)}."
            )
        N = theta_2d.shape[0]
        if prior_hp.shape != (N, self.prior_dim):
            raise ValueError(
                f"ValidityNet expected prior_hp ({N}, {self.prior_dim}); "
                f"got {tuple(prior_hp.shape)}."
            )
        if lik_hp.shape != (N, self.lik_dim):
            raise ValueError(
                f"ValidityNet expected lik_hp ({N}, {self.lik_dim}); "
                f"got {tuple(lik_hp.shape)}."
            )
        if eta.shape != (N,):
            raise ValueError(
                f"ValidityNet expected eta ({N},); got {tuple(eta.shape)}."
            )
        eta_2d = eta[:, None]
        x = jnp.concatenate([theta_2d, prior_hp, lik_hp, eta_2d], axis=-1)
        loc = jnp.asarray(self.feature_loc)
        scale = jnp.asarray(self.feature_scale)
        log_mask = jnp.asarray(self.feature_log)
        x_log = jnp.log(jnp.maximum(x, 1e-12))
        x = jnp.where(log_mask, x_log, x)
        x = (x - loc) / scale
        out = jax.vmap(self.mlp)(x)
        return out[..., 0]

    def architecture_kwargs(self) -> dict[str, Any]:
        return {
            "theta_dim": self.theta_dim,
            "prior_dim": self.prior_dim,
            "lik_dim": self.lik_dim,
            "hidden_sizes": self.hidden_sizes,
            "feature_loc": list(self.feature_loc),
            "feature_scale": list(self.feature_scale),
            "feature_log": list(self.feature_log),
        }


__all__ = ["EtaNet", "ValidityNet"]
