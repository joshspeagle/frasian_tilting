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
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _build_mlp(
    in_features: int,
    hidden_sizes: tuple[int, ...],
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


class EtaNet(nn.Module):  # type: ignore[misc,unused-ignore]
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
        hidden_sizes: tuple[int, ...] = (64, 64),
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
            x = theta.unsqueeze(-1)  # (N, 1)
        elif theta.dim() == 2:
            if theta.size(-1) != self.theta_dim:
                raise ValueError(
                    f"EtaNet(theta_dim={self.theta_dim}) expected "
                    f"input shape (N, {self.theta_dim}); got "
                    f"{tuple(theta.shape)}."
                )
            x = theta
        else:
            raise ValueError(f"EtaNet expects 1D or 2D input; got shape " f"{tuple(theta.shape)}.")
        return self.mlp(x).squeeze(-1)  # (N,)

    def architecture_kwargs(self) -> dict[str, Any]:
        """Kwargs to re-instantiate this exact architecture."""
        return {
            "theta_dim": self.theta_dim,
            "hidden_sizes": self.hidden_sizes,
        }


class ValidityNet(nn.Module):  # type: ignore[misc,unused-ignore]
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
        hidden_sizes: tuple[int, ...] = (64, 64),
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
        return self.mlp(inputs).squeeze(-1)  # (N,)

    def architecture_kwargs(self) -> dict[str, Any]:
        return {
            "theta_dim": self.theta_dim,
            "hidden_sizes": self.hidden_sizes,
        }


__all__ = ["EtaNet", "ValidityNet"]
