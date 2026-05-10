"""Audit P0-10: runner gates registry stubs as `incompatible`, not `error`.

Pre-fix: stub tilting schemes (`MixtureTilting`, `FisherRaoTilting`)
declared only `tilt`, `path`, `is_identity` — calling
`confidence_interval` raised AttributeError. Stub statistics
(`LRTStatistic`, `SignedRootStatistic`, `BartlettCorrectedLRT`) raised
NotImplementedError. Both were caught by the runner's per-cell
exception handler and recorded as `status="error"` with a noisy
traceback — even though their stub status was statically declared at
registration.

Cluster D (audit fix):
1. Added explicit NotImplementedError stubs to MixtureTilting and
   FisherRaoTilting for the missing protocol methods.
2. Added a `_is_stub(obj, kind)` helper to the runner; gates stubs
   *before* invocation as `status="incompatible"` with a stub-pointing
   reason. Same status as `accepts_tilting=False` cells, so downstream
   manifest consumers don't need to special-case stubs.

This test pins both halves: stubs raise the right exception type, and
the runner gates them cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian._runner import _is_stub
from frasian.statistics.bartlett import BartlettCorrectedLRT
from frasian.statistics.lrt import LRTStatistic
from frasian.statistics.signed_root import SignedRootStatistic
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.fisher_rao import FisherRaoTilting
from frasian.tilting.identity import IdentityTilting
from frasian.tilting.mixture import MixtureTilting


@pytest.mark.L0
@pytest.mark.usefixtures("bootstrapped_registry")
class TestIsStubHelper:
    """Pin `_is_stub` against the registry's declared status."""

    def test_stub_tiltings_recognised(self):
        # MixtureTilting promoted stub -> implemented in 2026-05-09 (Stage A
        # of mixture-tilting plan). Only FisherRaoTilting remains a stub.
        assert _is_stub(FisherRaoTilting(), "tilting") is True

    def test_implemented_tiltings_not_stub(self):
        assert _is_stub(IdentityTilting(), "tilting") is False
        assert _is_stub(MixtureTilting(), "tilting") is False

    def test_stub_statistics_recognised(self):
        assert _is_stub(LRTStatistic(), "statistic") is True
        assert _is_stub(SignedRootStatistic(), "statistic") is True
        assert _is_stub(BartlettCorrectedLRT(), "statistic") is True

    def test_implemented_statistics_not_stub(self):
        assert _is_stub(WaldoStatistic(), "statistic") is False

    def test_class_input_also_works(self):
        """Helper accepts both classes and instances (cell-runner symmetry)."""
        assert _is_stub(FisherRaoTilting, "tilting") is True
        assert _is_stub(WaldoStatistic, "statistic") is False


@pytest.mark.L0
class TestStubsRaiseCleanlyAtProtocolSurface:
    """Direct protocol-method calls raise NotImplementedError, never
    AttributeError. The audit's specific complaint was that stub
    tiltings *missing* the methods raised AttributeError.
    """

    def _fixtures(self):
        from frasian.models.distributions import GaussianLikelihood, NormalDistribution
        from frasian.models.normal_normal import NormalNormalModel

        model = NormalNormalModel(sigma=1.0)
        prior = NormalDistribution(loc=0.0, scale=1.0)
        lik = GaussianLikelihood(D=1.0, sigma=1.0)
        post = model.posterior(np.asarray([1.0]), prior)
        return model, prior, lik, post

    def test_fisher_rao_pvalue_raises_notimplemented(self):
        model, prior, _, _ = self._fixtures()
        with pytest.raises(NotImplementedError):
            FisherRaoTilting().pvalue(
                np.asarray([0.5]), np.asarray([1.0]), model, prior, WaldoStatistic()
            )
