"""Training pipeline for the learned dynamic-η selector.

Imports torch only when this subpackage is loaded; the rest of the
framework stays torch-free.

Public entry points:
- `MonotonicEtaNet` (`.architecture`) — the partial-monotonicity NN.
- `TORCH_TILTED_PVALUE` (`.pvalue_torch`) — registry of differentiable
  tilted-WALDO p-value implementations keyed on scheme name.
- `cd_density_torch` (`.cd_torch`) — Schweder-Hjort CD pdf, no
  signed-confidence (skipping the non-differentiable argmax).
- `integrated_pvalue_loss`, `cd_variance_loss`, `static_width_loss`
  (`.losses`) — differentiable losses for training.
- `TrainingDistribution` (`.sampling`) — configurable π over
  `(w, θ_true)`; `normal_normal_default()` is the canonical default.
- `fit_monotonic_eta_artifact` (`.train`) — top-level training entry.
"""

from __future__ import annotations
