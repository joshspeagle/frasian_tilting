"""Framework-level exceptions."""

from __future__ import annotations


class FrasianError(Exception):
    """Base for all framework exceptions."""


class EmptyRegistryError(FrasianError):
    """Raised when run_experiment is called against an empty registry slice."""


class TiltingDomainError(FrasianError):
    """Raised when a tilting parameter falls outside the admissible range."""


class RegistryConflictError(FrasianError):
    """Raised when two implementations register under the same name."""


class MissingArtifactError(FrasianError):
    """Raised when a LearnedArtifact is requested but not loaded."""


class BracketingFailed(FrasianError):
    """Raised when a CI search cannot find a valid bracket within the
    fixed search window (e.g. dynamic CI extends past the ±search_mult·σ
    box). Better to surface than to silently truncate the interval at the
    box boundary. Tier 1.5-O6 in the audit.
    """
