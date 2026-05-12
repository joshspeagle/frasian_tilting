"""Trinity collapse for (PL/OT/FR × {waldo, lrto, scoreo}) on NN+Normal.

q_η is a single Gaussian for power_law, ot, fisher_rao on NN+Normal
(see docs/notes/2026-05-12-tilted-trinity-derivation.md), so the
tilted-LRTO and tilted-SCOREO p-values must coincide with the
tilted-WALDO p-value to ~machine precision on the closed-form path.

Mixture is NOT covered here — q_η,mix is a 2-Gaussian mixture and the
three statistics genuinely differ.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.models.normal_normal import NormalNormalModel
from frasian.models.distributions import NormalDistribution
from frasian.tilting.power_law import PowerLawTilting
from frasian.tilting.ot import OTTilting
from frasian.tilting.fisher_rao import FisherRaoTilting
from frasian.tilting.eta_selectors import FixedEtaSelector


_TRIPLES = [
    (PowerLawTilting, "power_law"),
    (OTTilting, "ot"),
    (FisherRaoTilting, "fisher_rao"),
]


@pytest.mark.L2
@pytest.mark.parametrize("tilt_cls,name", _TRIPLES)
@pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("theta", [-0.5, 0.0, 0.7, 1.5])
def test_tilted_trinity_collapse_closed_form(tilt_cls, name, eta, theta):
    model = NormalNormalModel(sigma=1.0)
    prior = NormalDistribution(loc=0.0, scale=2.0)
    data = np.array([0.7])
    tilt = tilt_cls(selector=FixedEtaSelector(eta=eta))
    p_waldo = float(np.asarray(tilt.tilted_pvalue(
        theta, data, model, prior, eta, statistic_name="waldo"
    )))
    p_lrto = float(np.asarray(tilt.tilted_pvalue(
        theta, data, model, prior, eta, statistic_name="lrto"
    )))
    p_scoreo = float(np.asarray(tilt.tilted_pvalue(
        theta, data, model, prior, eta, statistic_name="scoreo"
    )))
    assert p_lrto == pytest.approx(p_waldo, rel=1e-12, abs=1e-12), (
        f"{name} eta={eta} theta={theta}: p_lrto={p_lrto} vs p_waldo={p_waldo}"
    )
    assert p_scoreo == pytest.approx(p_waldo, rel=1e-12, abs=1e-12), (
        f"{name} eta={eta} theta={theta}: p_scoreo={p_scoreo} vs p_waldo={p_waldo}"
    )
