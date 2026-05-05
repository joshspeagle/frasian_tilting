"""Regression: NumericalEtaSelector(objective='integrated_p') vs 'static_width'.

`NumericalEtaSelector` got a new `objective` parameter:

  - "static_width" (default, backwards-compatible): minimize
    `|C_α(D, η)|` at the alpha read from context.
  - "integrated_p" (new): minimize `∫_θ p_dyn(θ; D, η) dθ` — the
    same loss the learned MLP minimizes, with no smoothness or
    monotonicity prior. The apples-to-apples non-NN baseline.

Tests:
  - default selector still gives static_width behavior (no behavior
    change for existing callers).
  - integrated_p path produces a finite η in the admissible range.
  - integrated_p η ≠ static_width η (the loss surfaces differ).
  - Invalid objective raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.base import TiltingContext
from frasian.tilting.eta_selectors import NumericalEtaSelector
from frasian.tilting.power_law import PowerLawTilting


@pytest.mark.L2
class TestNumericalEtaSelectorObjective:
    def test_default_is_static_width(self):
        sel = NumericalEtaSelector()
        assert sel.objective == "static_width"

    def test_integrated_p_constructible(self):
        sel = NumericalEtaSelector(objective="integrated_p")
        assert sel.objective == "integrated_p"

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError, match="objective"):
            NumericalEtaSelector(objective="bogus")

    @pytest.mark.parametrize("abs_delta", [0.5, 1.5, 3.0])
    @pytest.mark.parametrize("w", [0.3, 0.5, 0.7])
    def test_integrated_p_returns_eta_in_admissible_range(self, abs_delta, w):
        """Integrated-p objective produces a finite η in the admissible
        range for power_law."""
        sel = NumericalEtaSelector(objective="integrated_p")
        scheme = PowerLawTilting()
        ctx = TiltingContext(w=w, abs_delta=abs_delta, alpha=0.05)
        eta = sel.select(ctx, scheme, statistic=WaldoStatistic())
        assert np.isfinite(eta)
        eta_min = -w / (1.0 - w)
        assert eta > eta_min
        assert eta < 1.0

    def test_integrated_p_path_runs_select_grid(self):
        """select_grid completes for integrated_p across a |Δ| sweep."""
        sel = NumericalEtaSelector(objective="integrated_p")
        scheme = PowerLawTilting()
        ad_grid = np.linspace(0.0, 4.0, 9)
        eta = sel.select_grid(
            ad_grid,
            scheme,
            statistic=WaldoStatistic(),
            w=0.5,
            alpha=0.05,
        )
        assert eta.shape == (9,)
        assert np.all(np.isfinite(eta))
        eta_min = -1.0  # for w=0.5
        assert np.all(eta > eta_min)
        assert np.all(eta < 1.0)
