"""Offline-trained artifacts (Phase E dual-head learned selector).

The protocol in `base.py` codifies the explicit-lifecycle pattern that
replaces the legacy module-level MLP cache. Concrete artifacts must
`load()` before `predict_eta()` / `predict_validity()` and expose a
`fingerprint()` so the simulation cache can invalidate when the
artifact's training data or weights change.

Phase E ships two production artifacts:
- ``EtaArtifact`` (in ``eta_artifact.py``): wraps a dual-head
  ``EtaNet`` (θ → η, GELU MLP, no monotonicity prior, no bounded
  sigmoid) plus a ``ValidityNet`` (θ, η) → P(valid). Trained
  per-experiment and refuses cross-experiment use via fingerprint
  compare.
- ``NullArtifact`` (in ``null.py``): trivial constant-η stub used
  by tests.

The Phase D ``MonotonicEtaNet`` cache predictor was retired in
Phase E and is no longer present.
"""

from .base import LearnedArtifact
from .null import NullArtifact

__all__ = ["LearnedArtifact", "NullArtifact"]
