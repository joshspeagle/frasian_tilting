"""Offline-trained artifacts (e.g. monotonic eta* MLP).

The protocol in `base.py` codifies the explicit-lifecycle pattern that
replaces the legacy module-level MLP cache. Concrete artifacts must
`load()` before `predict()` and expose a `fingerprint()` so the
simulation cache can invalidate when the artifact's training data
or weights change.
"""

from .base import LearnedArtifact
from .null import NullArtifact

__all__ = ["LearnedArtifact", "NullArtifact"]
