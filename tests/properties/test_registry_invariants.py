"""Registry invariants for concrete classes.

These tests pin the contract that the canonical identity (the ``name``
attribute used as the registry key + cell label) cannot be silently
overridden via a constructor kwarg. The Phase 2 refactor migrated the
``Protocol`` declaration to ``@property def name(self) -> str``; this
test file pins the dataclass-side mirror as ``ClassVar[str]`` so that
``Cls(name="bogus")`` raises ``TypeError`` instead of minting an
identity-mismatched instance.

If a future contributor reverts ``ClassVar`` to a regular field, these
tests catch it before it leaks into a manifest / cache key.
"""

from __future__ import annotations

import pytest

from frasian.diagnostics.coverage_table import CoverageRateDiagnostic
from frasian.experiments.coverage import CoverageExperiment
from frasian.statistics.wald import WaldStatistic
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L1
@pytest.mark.parametrize(
    "cls",
    [PowerLawTilting, WaldStatistic, CoverageExperiment, CoverageRateDiagnostic],
)
def test_name_kwarg_rejected(cls: type) -> None:
    """``Cls(name='bogus')`` must raise ``TypeError`` (``name`` is a ClassVar)."""
    with pytest.raises(TypeError, match="name"):
        cls(name="bogus")
