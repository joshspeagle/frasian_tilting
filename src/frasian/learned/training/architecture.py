"""Two-head architecture for the learned-η selector (Phase E rewrite).

The Phase E rewrite replaces the Normal-Normal-specific
``MonotonicEtaNet`` (input ``(w, |Δ'|)``, monotone-in-|Δ'| MLP with
bounded sigmoid output) with a *model-agnostic* dual-head design.

``MonotonicEtaNet`` itself is kept here as a transitional shim while
``train.py`` (rewritten in E.2) still imports it; the shim is removed
in the same commit that lands the new training loop.

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
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_UNIFORM_MEAN = 0.5
_UNIFORM_STD = 1.0 / math.sqrt(12.0)


def _standardise_uniform(x: torch.Tensor) -> torch.Tensor:
    return (x - _UNIFORM_MEAN) / _UNIFORM_STD


class MonotonicEtaNet(nn.Module):
    """LEGACY (Phase D). Removed in E.2 when train.py is rewritten.

    See git history for the original docstring; kept here only so the
    pre-E.2 ``train.py`` and ``LearnedDynamicEtaSelector`` continue to
    import this module without breaking during the E.1 commit. Do not
    use in new code.
    """

    def __init__(
        self,
        shared_sizes: Tuple[int, ...] = (64, 64),
        mono_sizes: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        self.shared_sizes = tuple(shared_sizes)
        self.mono_sizes = tuple(mono_sizes)

        self.shared_layers = nn.ModuleList()
        in_features = 1
        for hidden_size in shared_sizes:
            self.shared_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        shared_out = in_features

        self.mono_layers = nn.ModuleList()
        in_features = 1
        for hidden_size in mono_sizes:
            self.mono_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        mono_out = in_features

        self.output_base = nn.Linear(shared_out, 1)
        self.output_scale = nn.Linear(shared_out, 1)
        self.output_mono = nn.Linear(mono_out, 1)

        self._init_weights_legacy()

    def _init_weights_legacy(self) -> None:
        for layer in self.shared_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for layer in self.mono_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            layer.weight.data.abs_()
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_base.weight)
        nn.init.xavier_normal_(self.output_scale.weight)
        nn.init.xavier_normal_(self.output_mono.weight)
        self.output_mono.weight.data.abs_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(-1) != 2:
            raise ValueError(
                f"MonotonicEtaNet expects (N, 2) input [w, Δ']; got {tuple(x.shape)}"
            )
        x_std = _standardise_uniform(x)
        w = x_std[:, 0:1]
        delta_prime = x_std[:, 1:2]

        h_shared = w
        for layer in self.shared_layers:
            h_shared = F.gelu(layer(h_shared))

        h_mono = delta_prime
        for layer in self.mono_layers:
            h_mono = F.relu(F.linear(h_mono, layer.weight.abs(), layer.bias))

        base = self.output_base(h_shared)
        scale = F.softplus(self.output_scale(h_shared))
        mono = F.linear(
            h_mono, self.output_mono.weight.abs(), self.output_mono.bias
        )

        z = base + scale * mono
        return 0.01 + 0.98 * torch.sigmoid(z)

    def architecture_kwargs(self) -> dict:
        return {
            "shared_sizes": self.shared_sizes,
            "mono_sizes": self.mono_sizes,
        }


def _build_mlp(
    in_features: int,
    hidden_sizes: Tuple[int, ...],
    out_features: int,
) -> nn.Sequential:
    """GELU-activated MLP: in → h0 → ... → out (linear final)."""
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.GELU())
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


class EtaNet(nn.Module):
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
        Hidden layer widths.
    """

    def __init__(
        self,
        theta_dim: int = 1,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        if theta_dim < 1:
            raise ValueError(f"theta_dim must be >= 1, got {theta_dim}")
        self.theta_dim = int(theta_dim)
        self.hidden_sizes = tuple(hidden_sizes)
        self.mlp = _build_mlp(self.theta_dim, self.hidden_sizes, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        theta : ``(N,)`` (when ``theta_dim==1``) or ``(N, theta_dim)``.
        returns : ``(N,)`` raw η values.
        """
        if theta.dim() == 1:
            if self.theta_dim != 1:
                raise ValueError(
                    f"EtaNet(theta_dim={self.theta_dim}) requires (N, "
                    f"{self.theta_dim}) input; got 1D shape "
                    f"{tuple(theta.shape)}."
                )
            x = theta.unsqueeze(-1)                              # (N, 1)
        elif theta.dim() == 2:
            if theta.size(-1) != self.theta_dim:
                raise ValueError(
                    f"EtaNet(theta_dim={self.theta_dim}) expected "
                    f"input shape (N, {self.theta_dim}); got "
                    f"{tuple(theta.shape)}."
                )
            x = theta
        else:
            raise ValueError(
                f"EtaNet expects 1D or 2D input; got shape "
                f"{tuple(theta.shape)}."
            )
        return self.mlp(x).squeeze(-1)                           # (N,)

    def architecture_kwargs(self) -> dict:
        """Kwargs to re-instantiate this exact architecture."""
        return {
            "theta_dim": self.theta_dim,
            "hidden_sizes": self.hidden_sizes,
        }


class ValidityNet(nn.Module):
    """MLP from (θ, η) to a single logit.

    Trained on validity labels (whether ``scheme.tilted_pvalue`` at
    (θ, η) is finite and in [0, 1]) via ``BCEWithLogitsLoss``. The
    logit output is intentional — sigmoid is applied at loss time
    via ``BCEWithLogitsLoss`` (numerically stable) and the boundary
    penalty uses ``F.logsigmoid`` directly.

    No clamp on logits. ``logsigmoid`` and ``BCEWithLogitsLoss`` are
    both numerically stable for any finite input, and clamping kills
    the wrong-side gradient — see
    ``losses.boundary_penalty_from_validity`` for the rationale.

    Input convention: a single ``(N, theta_dim + 1)`` tensor with θ
    in the leading ``theta_dim`` columns and η in the last column.
    """

    def __init__(
        self,
        theta_dim: int = 1,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        if theta_dim < 1:
            raise ValueError(f"theta_dim must be >= 1, got {theta_dim}")
        self.theta_dim = int(theta_dim)
        self.hidden_sizes = tuple(hidden_sizes)
        self.mlp = _build_mlp(self.theta_dim + 1, self.hidden_sizes, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        inputs : ``(N, theta_dim + 1)`` tensor — concat of θ and η.
        returns : ``(N,)`` logits.
        """
        if inputs.dim() != 2 or inputs.size(-1) != self.theta_dim + 1:
            raise ValueError(
                f"ValidityNet(theta_dim={self.theta_dim}) expects "
                f"input shape (N, {self.theta_dim + 1}); got "
                f"{tuple(inputs.shape)}."
            )
        return self.mlp(inputs).squeeze(-1)                      # (N,)

    def architecture_kwargs(self) -> dict:
        return {
            "theta_dim": self.theta_dim,
            "hidden_sizes": self.hidden_sizes,
        }


__all__ = ["EtaNet", "ValidityNet"]
