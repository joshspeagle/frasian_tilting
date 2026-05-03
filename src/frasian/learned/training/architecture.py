"""MonotonicEtaNet — partial-monotonicity neural network for η*(|Δ|; w).

Ported from `legacy/src/frasian/simulations/mlp_monotonic.py:25-167`,
with α dropped from the inputs (the new training objective is α-free
or α-conditioned via the loss, not the architecture).

Architecture
------------
Inputs: standardised `(w, Δ')` in `[0, 1]²`.
Output: `η' ∈ [0.01, 0.99]` (bounded sigmoid; the selector applies
`eta_inverse` per-scheme to recover η).

```
output = 0.01 + 0.98 · sigmoid( base(w) + softplus(scale(w)) · mono(Δ') )
```

where:
- `base(w)`, `scale(w)`: unconstrained MLPs (GELU activations) on the
  shared `w` pathway.
- `mono(Δ')`: positive-weight + ReLU MLP, structurally monotone non-
  decreasing in `Δ'`.
- `softplus` keeps the scale strictly positive.
- `sigmoid` is monotone increasing and bounded.

Combining: `∂output/∂Δ' ≥ 0` by construction, so the resulting
η*(|Δ|; w) curve is monotone — the load-bearing structural prior that
prevents the lower-clamp kink in the dynamic-procedure loss landscape.

Standardisation: inputs are uniform on `[0, 1]`; we use the analytic
mean and std (`0.5`, `1/√12`) so the model is independent of any
particular sample.
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
    """Standardise inputs assumed uniform on [0, 1]."""
    return (x - _UNIFORM_MEAN) / _UNIFORM_STD


class MonotonicEtaNet(nn.Module):
    """Partial-monotonicity MLP for η'*(w, Δ').

    Forward input is a `(N, 2)` tensor `[w, Δ']`. Output is `(N, 1)`
    with values in `[0.01, 0.99]`.

    Parameters
    ----------
    shared_sizes : tuple of int
        Hidden layer sizes for the unconstrained `(w)` pathway.
    mono_sizes : tuple of int
        Hidden layer sizes for the monotonic `(Δ')` pathway.

    The output transformation `0.01 + 0.98·sigmoid(...)` keeps η' away
    from the strict 0/1 boundaries, equivalent to keeping η a small
    distance inside `[η_min(w), 1]` for power_law (or `[0, 1]` for OT).
    """

    def __init__(
        self,
        shared_sizes: Tuple[int, ...] = (64, 64),
        mono_sizes: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        self.shared_sizes = tuple(shared_sizes)
        self.mono_sizes = tuple(mono_sizes)

        # Shared pathway: 1D input (w) → shared_sizes
        self.shared_layers = nn.ModuleList()
        in_features = 1
        for hidden_size in shared_sizes:
            self.shared_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        shared_out = in_features

        # Monotonic pathway: 1D input (Δ') → mono_sizes (positive weights)
        self.mono_layers = nn.ModuleList()
        in_features = 1
        for hidden_size in mono_sizes:
            self.mono_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        mono_out = in_features

        # Combiners
        self.output_base = nn.Linear(shared_out, 1)
        self.output_scale = nn.Linear(shared_out, 1)
        self.output_mono = nn.Linear(mono_out, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.shared_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for layer in self.mono_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            # |W| → ensure positivity for monotonicity (forward also abs()'s).
            layer.weight.data.abs_()
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_base.weight)
        nn.init.xavier_normal_(self.output_scale.weight)
        nn.init.xavier_normal_(self.output_mono.weight)
        self.output_mono.weight.data.abs_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x : (N, 2) tensor of `[w, Δ']` in [0, 1].
        returns : (N, 1) tensor of η' in (0.01, 0.99).
        """
        if x.dim() != 2 or x.size(-1) != 2:
            raise ValueError(
                f"MonotonicEtaNet expects (N, 2) input [w, Δ']; got {tuple(x.shape)}"
            )
        x_std = _standardise_uniform(x)
        w = x_std[:, 0:1]
        delta_prime = x_std[:, 1:2]

        # Shared (w)
        h_shared = w
        for layer in self.shared_layers:
            h_shared = F.gelu(layer(h_shared))

        # Monotonic (Δ'): use absolute weights at every layer + ReLU.
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
        """Kwargs needed to re-instantiate this exact architecture.

        Stored in the checkpoint so loading rebuilds the same module
        shape that the trained weights were learned for.
        """
        return {
            "shared_sizes": self.shared_sizes,
            "mono_sizes": self.mono_sizes,
        }
