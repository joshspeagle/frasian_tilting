"""Config.fingerprint() must change when any cache-relevant field changes.

The dynamic-CI scan parameters (``dynamic_n_grid`` / ``dynamic_coarse_n``
/ ``dynamic_search_mult``) used to live as function-default values inside
``tilting/_dynamic.dynamic_ci_scan``; the cache could not see them. They
moved into ``Config`` as part of Tier 1.7-C2; this test pins the new
contract.
"""

from __future__ import annotations

import pytest

from frasian import Config
from frasian.simulation.cache import CacheKey


def _key_with(cfg: Config) -> str:
    """Build a CacheKey using ``cfg.fingerprint()`` and return its digest."""
    return CacheKey(
        experiment="coverage",
        tilting="power_law[dynamic_numerical]",
        statistic="waldo",
        config_fingerprint=cfg.fingerprint(),
        git_sha="abc1234",
        raw_fingerprint="raw0",
        extra={},
    ).digest()


@pytest.mark.L1
def test_dynamic_n_grid_changes_fingerprint() -> None:
    a = Config.default()
    b = Config.default().from_overrides(dynamic_n_grid=801)
    assert a.fingerprint() != b.fingerprint()


@pytest.mark.L1
def test_dynamic_coarse_n_changes_fingerprint() -> None:
    a = Config.default()
    b = Config.default().from_overrides(dynamic_coarse_n=51)
    assert a.fingerprint() != b.fingerprint()


@pytest.mark.L1
def test_dynamic_search_mult_changes_fingerprint() -> None:
    a = Config.default()
    b = Config.default().from_overrides(dynamic_search_mult=12.0)
    assert a.fingerprint() != b.fingerprint()


@pytest.mark.L1
def test_dynamic_params_distinguish_cache_keys() -> None:
    """Two Configs differing only in ``dynamic_n_grid`` must yield
    different cache digests, so cached results don't leak between
    runs at different scan resolutions.
    """
    a = Config.default()
    b = Config.default().from_overrides(dynamic_n_grid=801)
    assert _key_with(a) != _key_with(b)


@pytest.mark.L1
def test_default_dynamic_params_match_documented_values() -> None:
    """Pin the documented defaults so an accidental edit shows up here
    rather than as a silent cache invalidation.
    """
    cfg = Config.default()
    assert cfg.dynamic_n_grid == 401
    assert cfg.dynamic_coarse_n == 25
    assert cfg.dynamic_search_mult == 8.0
