"""Phase E training pipeline for the learned dynamic-η selector.

After Phase F port commit 3 the orchestrator is JAX/Equinox/Optax;
torch is no longer a dependency.

Public entry points:
- `EtaNet`, `ValidityNet` (`.architecture`) — the dual-head MLPs
  (``equinox.Module`` subclasses).
- `JAX_TILTED_PVALUE` / `get_jax_tilted_pvalue` (`.pvalue_jax`) —
  registry of differentiable tilted-WALDO p-value implementations.
- `cd_density_jax` (`.cd_jax`) — Schweder-Hjort CD pdf, no
  signed-confidence (skipping the non-differentiable argmax).
- `integrated_pvalue_loss`, `cd_variance_loss`, `static_width_loss`,
  `boundary_penalty_from_validity` (`.losses`) — differentiable
  losses for training.
- `is_pair_valid`, `validity_mask`, `compute_pvalues_per_sample`
  (`.validity`) — per-sample validity helpers.
- `ExperimentConfig`, `ThetaDistribution`, `UniformThetaDistribution`,
  `lhs_1d` (`.sampling`) — experiment specification + LHS sampling.
- `fit_eta_artifact` (`.train`) — top-level training entry.

The internal modules `_train_loop`, `_validity_data`,
`_losses_compose`, and `_checkpoint` are not part of the public
surface; consumers go through the entries above.
"""

from __future__ import annotations
