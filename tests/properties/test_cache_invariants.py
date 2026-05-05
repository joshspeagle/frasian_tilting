"""Cache-key invariants beyond the basic regression suite.

These pin properties that are easy to regress under refactoring:

- C8: digest stable across processes (rules out hash-randomisation /
  sort-order dependence).
- C9: selector kind embedded in the ``tilting`` slot distinguishes
  digests (`power_law` vs `power_law[learned_dynamic]`).
- C10: dirty-tree digest differs from clean-tree digest.

Tier 1.7-C8 / C9 / C10 in the audit.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from frasian.simulation.cache import CacheKey


def _make_key(**overrides) -> CacheKey:
    base = dict(
        experiment="coverage",
        tilting="power_law",
        statistic="waldo",
        config_fingerprint="cfg0",
        git_sha="abc1234",
        raw_fingerprint="raw0",
        extra={},
    )
    base.update(overrides)
    return CacheKey(**base)


@pytest.mark.L1
def test_digest_stable_across_processes() -> None:
    """C8: spawning a subprocess that builds the same key must yield the
    same digest. Catches regressions where digest depends on dict
    ordering, hash randomisation, or float repr.
    """
    parent_digest = _make_key().digest()
    code = textwrap.dedent(
        """
        from frasian.simulation.cache import CacheKey
        k = CacheKey(
            experiment="coverage",
            tilting="power_law",
            statistic="waldo",
            config_fingerprint="cfg0",
            git_sha="abc1234",
            raw_fingerprint="raw0",
            extra={},
        )
        print(k.digest())
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    child_digest = proc.stdout.strip()
    assert child_digest == parent_digest, (child_digest, parent_digest)


@pytest.mark.L1
def test_selector_kind_distinguishes_digest() -> None:
    """C9: the runner writes ``tilting=cell_name`` so the selector kind
    flows into the cache key. Two cells with the same scheme but
    different selectors must digest differently.
    """
    plain = _make_key(tilting="power_law").digest()
    learned = _make_key(tilting="power_law[learned_dynamic]").digest()
    numerical = _make_key(tilting="power_law[dynamic_numerical]").digest()
    assert plain != learned
    assert plain != numerical
    assert learned != numerical


@pytest.mark.L1
def test_dirty_tree_digest_differs_from_clean() -> None:
    """C10: dirty-tree git_sha (``dirty:<hash>``) must produce a different
    digest from a clean-tree sha. Otherwise dirty-tree results could
    silently reuse clean-tree cache entries.
    """
    clean = _make_key(git_sha="abc1234").digest()
    dirty = _make_key(git_sha="dirty:deadbeef").digest()
    nogit = _make_key(git_sha="dirty:nogit").digest()
    assert clean != dirty
    assert clean != nogit
    assert dirty != nogit
    # And the is_dirty predicate agrees.
    assert _make_key(git_sha="dirty:deadbeef").is_dirty()
    assert _make_key(git_sha="dirty:nogit").is_dirty()
    assert not _make_key(git_sha="abc1234").is_dirty()
