"""Config.fingerprint() must change when any cache-relevant field changes.

The dynamic-CI scan parameters (``dynamic_n_grid`` / ``dynamic_coarse_n``
/ ``dynamic_search_mult``) used to live as function-default values inside
``tilting/_dynamic.dynamic_ci_scan``; the cache could not see them. They
moved into ``Config`` as part of Tier 1.7-C2; this test pins the new
contract.

In addition (Phase 5 skeptic vector #2), the Config fields must drive
the runtime path — not just the cache fingerprint. The
``test_dynamic_*_changes_runtime_resolution`` tests below pin that
contract by passing two different Configs through
``power_law.confidence_regions(..., config=cfg)`` and asserting the
``dynamic_ci_scan`` is invoked with the correct ``n_grid`` value.
"""

from __future__ import annotations

import numpy as np
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


@pytest.mark.L2
def test_dynamic_n_grid_drives_runtime_scan_resolution() -> None:
    """Phase 5 skeptic vector #2: ``Config.dynamic_n_grid`` must
    actually drive the dynamic-CI scan resolution, not just the cache
    fingerprint. Spy on ``dynamic_ci_scan`` and assert the ``n_grid``
    kwarg matches the Config value.
    """
    from unittest.mock import patch

    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel
    from frasian.statistics.waldo import WaldoStatistic
    from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
    from frasian.tilting.power_law import PowerLawTilting

    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    statistic = WaldoStatistic()
    scheme = PowerLawTilting(selector=DynamicNumericalEtaSelector())

    cfg_lo = Config.default().from_overrides(dynamic_n_grid=201)
    cfg_hi = Config.default().from_overrides(dynamic_n_grid=801)

    captured: dict[str, list[int]] = {"n_grid": []}

    # Wrap the real scan so the call still produces regions (no mock
    # return). We just want to inspect ``n_grid``.
    from frasian.tilting import _dynamic as dyn_mod

    real_scan = dyn_mod.dynamic_ci_scan

    def spy_scan(*args, n_grid: int, **kwargs):
        captured["n_grid"].append(n_grid)
        return real_scan(*args, n_grid=n_grid, **kwargs)

    with patch("frasian.tilting._dynamic.dynamic_ci_scan", side_effect=spy_scan):
        scheme.confidence_regions(
            0.05, np.array([0.5]), model, prior, statistic, config=cfg_lo
        )
        scheme.confidence_regions(
            0.05, np.array([0.5]), model, prior, statistic, config=cfg_hi
        )

    assert captured["n_grid"] == [201, 801], (
        "Config.dynamic_n_grid must drive dynamic_ci_scan(n_grid=); "
        f"got {captured['n_grid']!r}"
    )


@pytest.mark.L2
def test_dynamic_search_mult_drives_runtime_scan_resolution() -> None:
    """Companion: ``dynamic_search_mult`` flows through to the scan."""
    from unittest.mock import patch

    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel
    from frasian.statistics.waldo import WaldoStatistic
    from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
    from frasian.tilting.power_law import PowerLawTilting

    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    statistic = WaldoStatistic()
    scheme = PowerLawTilting(selector=DynamicNumericalEtaSelector())

    cfg = Config.default().from_overrides(dynamic_search_mult=12.0)

    captured: dict[str, list[float]] = {"search_mult": []}
    from frasian.tilting import _dynamic as dyn_mod

    real_scan = dyn_mod.dynamic_ci_scan

    def spy_scan(*args, search_mult: float, **kwargs):
        captured["search_mult"].append(search_mult)
        return real_scan(*args, search_mult=search_mult, **kwargs)

    with patch("frasian.tilting._dynamic.dynamic_ci_scan", side_effect=spy_scan):
        scheme.confidence_regions(
            0.05, np.array([0.5]), model, prior, statistic, config=cfg
        )

    assert captured["search_mult"][0] == 12.0


@pytest.mark.L2
def test_no_config_falls_back_to_selector_defaults() -> None:
    """Backward compatibility: callers that don't pass ``config`` get
    the selector-derived defaults (the pre-fix behaviour).
    """
    from unittest.mock import patch

    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel
    from frasian.statistics.waldo import WaldoStatistic
    from frasian.tilting.eta_selectors import DynamicNumericalEtaSelector
    from frasian.tilting.power_law import PowerLawTilting

    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    statistic = WaldoStatistic()
    selector = DynamicNumericalEtaSelector(n_grid=303)
    scheme = PowerLawTilting(selector=selector)

    captured: dict[str, list[int]] = {"n_grid": []}
    from frasian.tilting import _dynamic as dyn_mod

    real_scan = dyn_mod.dynamic_ci_scan

    def spy_scan(*args, n_grid: int, **kwargs):
        captured["n_grid"].append(n_grid)
        return real_scan(*args, n_grid=n_grid, **kwargs)

    with patch("frasian.tilting._dynamic.dynamic_ci_scan", side_effect=spy_scan):
        # No config kwarg => selector default (303).
        scheme.confidence_regions(0.05, np.array([0.5]), model, prior, statistic)

    assert captured["n_grid"] == [303]
