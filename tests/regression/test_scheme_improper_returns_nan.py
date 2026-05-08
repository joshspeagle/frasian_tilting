"""Per-scheme audit: ``tilted_pvalue`` raises / returns NaN outside its
admissible η-range.

The Phase E learned-η selector relies on the validity helper
(``compute_pvalues_per_sample`` + ``validity_mask``) detecting bad η
samples by either:
  - the scheme raising ``TiltingDomainError`` / ``ValueError`` /
    ``RuntimeError`` / ``NotImplementedError`` / ``FloatingPointError``
    (caught and converted to NaN by the helper); or
  - the scheme returning a value outside ``[0, 1]`` that the validity
    mask rejects.

A future scheme whose ``tilted_pvalue`` returns a finite-in-[0, 1]
value despite an improper underlying tilted distribution would let
ValidityNet learn the wrong boundary and let EtaNet's boundary
penalty push it deeper into bogus territory. This audit pins each
scheme's behaviour at the boundary so that gap is caught at CI time.

Adding a new scheme: add a parametrize entry below and verify the
audit passes for the documented improper-η region.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian._errors import TiltingDomainError
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting


def _improper_etas_power_law(w: float) -> list[float]:
    """power_law admissible: 1 - η(1-w) > 0 ⇔ η < 1/(1-w).

    For w=0.5 → η < 2 admissible; pick η values past the boundary.
    """
    eta_max = 1.0 / (1.0 - w)
    return [eta_max + 1e-3, eta_max + 0.5, eta_max + 5.0]


def _improper_etas_ot(w: float) -> list[float]:
    """OT admissible (closed-form WALDO): s_t > 0 ⇔ η > -w/(1-w).

    Audit P0-4: OT is well-defined on the full W2 displacement line,
    not just the geodesic segment [0, 1]. ``OTTilting.tilted_pvalue``
    now raises only for non-finite η or η at/below ``-w/(1-w)`` (the
    bound where ``s_t = (w + η(1-w))*σ`` collapses to zero). For
    w=0.5 → η > -1 admissible; pick η values strictly below to
    exercise the raise path.
    """
    eta_lower = -w / (1.0 - w)
    return [eta_lower - 1e-3, eta_lower - 0.5, eta_lower - 5.0]


@pytest.mark.L2
@pytest.mark.parametrize("eta", _improper_etas_power_law(0.5))
def test_power_law_tilted_pvalue_raises_on_improper_eta(eta):
    """power_law: ``tilted_pvalue`` raises ``TiltingDomainError`` outside
    the variance-positivity range."""
    scheme = PowerLawTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)  # → w = 0.5
    with pytest.raises(TiltingDomainError):
        scheme.tilted_pvalue(
            np.array([0.0]),
            0.0,
            model,
            prior,
            eta,
            "waldo",
        )


@pytest.mark.L2
@pytest.mark.parametrize("eta", _improper_etas_ot(0.5))
def test_ot_tilted_pvalue_raises_on_improper_eta(eta):
    """OT: ``tilted_pvalue`` raises for η at or below ``-w/(1-w)``
    (the closed-form ``s_t > 0`` bound, audit P0-4)."""
    scheme = OTTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    with pytest.raises(TiltingDomainError):
        scheme.tilted_pvalue(
            np.array([0.0]),
            0.0,
            model,
            prior,
            eta,
            "waldo",
        )


@pytest.mark.L2
@pytest.mark.parametrize("eta", _improper_etas_power_law(0.5))
def test_power_law_validity_helper_rejects_improper_eta(eta):
    """End-to-end: ``compute_pvalues_per_sample`` returns NaN for invalid η.

    The numpy `tilted_pvalue` raises `TiltingDomainError`, the helper
    catches it and writes NaN, the validity mask drops the sample.
    This is the path used to label Head B's BCE — it must be correct
    regardless of what the torch port does for the width loss.
    """
    from frasian.learned.training.validity import compute_pvalues_per_sample, validity_mask

    scheme = PowerLawTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)  # → w = 0.5
    p = compute_pvalues_per_sample(
        scheme,
        np.array([0.0]),
        np.array([0.0]),
        model,
        prior,
        np.array([float(eta)]),
        "waldo",
    )
    assert not validity_mask(p)[0], (
        f"power_law validity helper accepted η={eta} as valid; " f"got p={p[0]}."
    )


@pytest.mark.L2
@pytest.mark.parametrize("eta", _improper_etas_ot(0.5))
def test_ot_validity_helper_rejects_improper_eta(eta):
    """End-to-end: same as above for OT (numpy path raises, helper → NaN)."""
    from frasian.learned.training.validity import compute_pvalues_per_sample, validity_mask

    scheme = OTTilting()
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=1.0)
    p = compute_pvalues_per_sample(
        scheme,
        np.array([0.0]),
        np.array([0.0]),
        model,
        prior,
        np.array([float(eta)]),
        "waldo",
    )
    assert not validity_mask(p)[0], f"OT validity helper accepted η={eta} as valid; got p={p[0]}."
