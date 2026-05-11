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
        range for power_law.

        Updated 2026-05-11: spurious -w/(1-w) lower bound removed per
        deriver A.3 (PL admissibility on NN is upper-only η < 1/(1-w);
        no lower bound). Asserts only finiteness + the true upper bound.
        """
        from frasian.models.distributions import NormalDistribution
        from frasian.models.normal_normal import NormalNormalModel

        sel = NumericalEtaSelector(objective="integrated_p")
        scheme = PowerLawTilting()
        # Phase 3a-1: TiltingContext no longer carries `abs_delta`; build
        # the equivalent (model, prior, data) so the new selector
        # signature has a concrete D.
        sigma = 1.0
        mu0 = 0.0
        sigma0 = float(np.sqrt(w / (1.0 - w)) * sigma)
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        D = mu0 - abs_delta * sigma / max(1.0 - w, 1e-12)
        eta = sel.select(
            scheme,
            data=np.asarray([D]),
            model=model,
            prior=prior,
            alpha=0.05,
            statistic=WaldoStatistic(),
        )
        assert np.isfinite(eta)
        # PL admissibility on NN: upper-only η < 1/(1-w) (deriver A.3).
        eta_max = 1.0 / (1.0 - w)
        assert eta < eta_max

    def test_integrated_p_path_runs_select_grid(self):
        """select_grid completes for integrated_p across a θ sweep
        (Phase 3a-1.5: θ-keyed signature; the legacy |Δ| path is
        dropped).

        Updated 2026-05-11: spurious -w/(1-w) lower bound removed per
        deriver A.3. Asserts only finiteness + the true PL upper bound.
        """
        from frasian.models.distributions import NormalDistribution
        from frasian.models.normal_normal import NormalNormalModel

        sel = NumericalEtaSelector(objective="integrated_p")
        scheme = PowerLawTilting()
        # w=0.5 → sigma0 == sigma == 1.
        sigma, mu0 = 1.0, 0.0
        sigma0 = 1.0
        model = NormalNormalModel(sigma=sigma)
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        # |Δ| grid spans 0..4 in 9 steps; map to θ via the legacy
        # inversion D = mu0 - |Δ|·σ/(1-w).
        ad_grid = np.linspace(0.0, 4.0, 9)
        theta_grid = mu0 - ad_grid * sigma / 0.5
        eta = sel.select_grid(
            theta_grid,
            scheme,
            statistic=WaldoStatistic(),
            model=model,
            prior=prior,
            alpha=0.05,
        )
        assert eta.shape == (9,)
        assert np.all(np.isfinite(eta))
        # PL admissibility on NN: upper-only η < 1/(1-w) = 2 for w=0.5.
        assert np.all(eta < 2.0)
