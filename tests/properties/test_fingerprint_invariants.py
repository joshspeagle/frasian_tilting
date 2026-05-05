"""Model / Prior fingerprint purity invariants.

Fingerprints must be a pure function of public dataclass state (no
hidden state, no time-of-construction, no module-load order). The
EtaArtifact strict-tuple-equal compare relies on this — and so does
the cache layer indirectly via ``persist_cell``.

Tier 1.7-C11 in the audit.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from frasian.models.distributions import BetaDistribution, NormalDistribution
from frasian.models.normal_normal import NormalNormalModel


@pytest.mark.L1
def test_normal_normal_fingerprint_is_pure() -> None:
    """Two NormalNormalModel(sigma) constructed with equal sigma must
    produce equal fingerprints (tuple-equal, not just hash-equal).
    """
    a = NormalNormalModel(sigma=1.5)
    b = NormalNormalModel(sigma=1.5)
    assert a.fingerprint() == b.fingerprint()
    # Differing sigma -> differing fingerprint.
    c = NormalNormalModel(sigma=1.6)
    assert a.fingerprint() != c.fingerprint()


@pytest.mark.L1
def test_normal_distribution_fingerprint_is_pure() -> None:
    """Same loc + scale -> equal fingerprint."""
    a = NormalDistribution(loc=0.5, scale=1.0)
    b = NormalDistribution(loc=0.5, scale=1.0)
    assert a.fingerprint() == b.fingerprint()
    # Differing loc OR scale -> differing fingerprint.
    assert a.fingerprint() != NormalDistribution(loc=0.6, scale=1.0).fingerprint()
    assert a.fingerprint() != NormalDistribution(loc=0.5, scale=1.1).fingerprint()


@pytest.mark.L1
def test_beta_distribution_fingerprint_is_pure() -> None:
    a = BetaDistribution(alpha=2.0, beta=3.0)
    b = BetaDistribution(alpha=2.0, beta=3.0)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != BetaDistribution(alpha=2.5, beta=3.0).fingerprint()


@pytest.mark.L1
def test_fingerprint_is_deterministic_across_constructions() -> None:
    """Repeated construction with identical inputs yields identical
    fingerprints (no time-of-construction state).

    The original audit recommendation included an importlib.reload
    check, but reloading a model module corrupts the isinstance()
    chain in tilting schemes that share the test session — so we
    verify the same property via deterministic re-construction.
    """
    a = NormalNormalModel(sigma=2.0).fingerprint()
    b = NormalNormalModel(sigma=2.0).fingerprint()
    c = NormalNormalModel(sigma=2.0).fingerprint()
    assert a == b == c

    # Same property for distributions used as Priors.
    da = NormalDistribution(loc=0.5, scale=1.0).fingerprint()
    db = NormalDistribution(loc=0.5, scale=1.0).fingerprint()
    assert da == db


@pytest.mark.L1
def test_model_fingerprint_stable_across_processes() -> None:
    """Tier 1.7-C11: ``Model.fingerprint()`` must be stable across
    independent Python processes — not just within one interpreter.

    Mirrors ``test_digest_stable_across_processes`` in
    ``test_cache_invariants.py``: spawn a child via ``sys.executable``
    (no ``PYTHONHASHSEED`` inherited; child gets a fresh random seed),
    construct the same model, print its fingerprint. Catches
    regressions where the fingerprint embeds e.g. ``id(self)``,
    process-local hash randomisation, or ``json.dumps`` without
    ``sort_keys=True``.

    The cache layer relies on this property; without it, two machines
    or two CI runs could compute incompatible cache keys for the same
    Config.
    """
    parent_fp = NormalNormalModel(sigma=1.5).fingerprint()
    code = textwrap.dedent(
        """
        from frasian.models.normal_normal import NormalNormalModel
        print(NormalNormalModel(sigma=1.5).fingerprint())
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    child_fp = proc.stdout.strip()
    # Compare via str() to avoid the tuple/str ambiguity if a future
    # fingerprint impl returns tuples.
    assert str(child_fp) == str(parent_fp), (child_fp, parent_fp)


@pytest.mark.L1
def test_prior_fingerprint_stable_across_processes() -> None:
    """Same cross-process property as the Model test, for
    ``NormalDistribution`` (used as Prior).
    """
    parent_fp = NormalDistribution(loc=0.25, scale=1.5).fingerprint()
    code = textwrap.dedent(
        """
        from frasian.models.distributions import NormalDistribution
        print(NormalDistribution(loc=0.25, scale=1.5).fingerprint())
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    child_fp = proc.stdout.strip()
    assert str(child_fp) == str(parent_fp), (child_fp, parent_fp)
