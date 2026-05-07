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
    """Smooth MLP from θ ∈ R^p to raw η ∈ R.

    No bounded output, no monotonicity constraint. Validity is
    enforced by the boundary penalty on Head B's prediction —
    see ``losses.boundary_penalty_from_validity``.

    Parameters
    ----------
    theta_dim : int
        Dimension of θ. Defaults to 1 (Normal-Normal). Vector θ is
        accepted for future model extensions; the existing
        ``power_law`` / ``ot`` schemes are scalar today.
    hidden_sizes : tuple of int
        Hidden layer widths. Must be uniform (same width for every
        layer) for the equinox port; see ``_build_mlp``.
    key : jax.Array
        PRNG key for weight initialisation. Required.
    """

    mlp: eqx.nn.MLP
    theta_dim: int = eqx.field(static=True)
    hidden_sizes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        theta_dim: int = 1,
        hidden_sizes: tuple[int, ...] = (64, 64),
        *,
        key: jax.Array,
    ):
        if theta_dim < 1:
            raise ValueError(f"theta_dim must be >= 1, got {theta_dim}")
        self.theta_dim = int(theta_dim)
        self.hidden_sizes = tuple(hidden_sizes)
        self.mlp = _build_mlp(self.theta_dim, self.hidden_sizes, 1, key)

    def __call__(self, theta: jax.Array) -> jax.Array:
        """Forward pass.

        theta : ``(N,)`` (when ``theta_dim==1``) or ``(N, theta_dim)``.
        returns : ``(N,)`` raw η values.
        """
        if theta.ndim == 1:
            if self.theta_dim != 1:
                raise ValueError(
                    f"EtaNet(theta_dim={self.theta_dim}) requires (N, "
                    f"{self.theta_dim}) input; got 1D shape "
                    f"{tuple(theta.shape)}."
                )
            x = theta[..., None]  # (N, 1)
        elif theta.ndim == 2:
            if theta.shape[-1] != self.theta_dim:
                raise ValueError(
                    f"EtaNet(theta_dim={self.theta_dim}) expected "
                    f"input shape (N, {self.theta_dim}); got "
                    f"{tuple(theta.shape)}."
                )
            x = theta
        else:
            raise ValueError(
                f"EtaNet expects 1D or 2D input; got shape {tuple(theta.shape)}."
            )
        # eqx.nn.MLP expects a single sample (in_size,) → (out_size,).
        # vmap over the leading batch axis to get (N, 1), then squeeze.
        out = jax.vmap(self.mlp)(x)  # (N, 1)
        return out[..., 0]  # (N,)

    def architecture_kwargs(self) -> dict[str, Any]:
        """Kwargs to re-instantiate this exact architecture."""
        return {
            "theta_dim": self.theta_dim,
            "hidden_sizes": self.hidden_sizes,
        }


class ValidityNet(eqx.Module):
    """MLP from (θ, η) to a single logit.

    Trained on validity labels (whether ``scheme.tilted_pvalue`` at
    (θ, η) is finite and in [0, 1]) via binary-cross-entropy-with-logits.
    The logit output is intentional — sigmoid is applied at loss time
    via the BCE-with-logits formulation (numerically stable) and the
    boundary penalty uses ``jax.nn.log_sigmoid`` directly.

    No clamp on logits. ``log_sigmoid`` is numerically stable for any
    finite input, and clamping kills the wrong-side gradient — see
    ``losses.boundary_penalty_from_validity`` for the rationale.

    Input convention: a single ``(N, theta_dim + 1)`` tensor with θ
    in the leading ``theta_dim`` columns and η in the last column.
    """

    mlp: eqx.nn.MLP
    theta_dim: int = eqx.field(static=True)
    hidden_sizes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        theta_dim: int = 1,
        hidden_sizes: tuple[int, ...] = (64, 64),
        *,
        key: jax.Array,
    ):
        if theta_dim < 1:
            raise ValueError(f"theta_dim must be >= 1, got {theta_dim}")
        self.theta_dim = int(theta_dim)
        self.hidden_sizes = tuple(hidden_sizes)
        self.mlp = _build_mlp(self.theta_dim + 1, self.hidden_sizes, 1, key)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Forward pass.

        inputs : ``(N, theta_dim + 1)`` array — concat of θ and η.
        returns : ``(N,)`` logits.
        """
        if inputs.ndim != 2 or inputs.shape[-1] != self.theta_dim + 1:
            raise ValueError(
                f"ValidityNet(theta_dim={self.theta_dim}) expects "
                f"input shape (N, {self.theta_dim + 1}); got "
                f"{tuple(inputs.shape)}."
            )
        out = jax.vmap(self.mlp)(inputs)  # (N, 1)
        return out[..., 0]  # (N,)

    def architecture_kwargs(self) -> dict[str, Any]:
        return {
            "theta_dim": self.theta_dim,
            "hidden_sizes": self.hidden_sizes,
        }


__all__ = ["EtaNet", "ValidityNet"]
