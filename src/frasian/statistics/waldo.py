"""WALDO statistic — Bayesian-frequentist hybrid via the posterior mean.

The p-value formula (Theorem 3 in the legacy derivations, ported verbatim
from `legacy/src/frasian/waldo.py:115`):

  a(theta) = |mu_n - theta| / (w * sigma)
  b(theta) = (1 - w) * (mu0 - theta) / (w * sigma)
  p(theta) = Phi(b - a) + Phi(-a - b)

Specializes on `NormalNormalModel` and a `NormalDistribution` prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .._registry import register_statistic
from ..models._dispatch import require_model, require_prior
from ..models.base import Model, Prior
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel, posterior_params
from .base import AsymptoticDistribution

if TYPE_CHECKING:
    from ..tilting.base import TiltingScheme


def _require_normal_normal(
    model: Model, prior: Prior | None
) -> tuple[NormalNormalModel, NormalDistribution]:
    m = require_model(model, NormalNormalModel, caller="WaldoStatistic")
    p = require_prior(prior, NormalDistribution, caller="WaldoStatistic")
    return m, p


def _pvalue_components(
    theta: NDArray[np.float64], mu_n: float, mu0: float, w: float, sigma: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """a(theta), b(theta) from the WALDO p-value formula."""
    a = np.abs(mu_n - theta) / (w * sigma)
    b = (1.0 - w) * (mu0 - theta) / (w * sigma)
    return a, b


@register_statistic(name="waldo", brief="docs/methods/waldo.md")
@dataclass(frozen=True)
class WaldoStatistic:
    """WALDO statistic for the Normal-Normal model."""

    name: str = "waldo"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="weighted_chi2",
        df=1,
        scale=1.0,
        description="(mu_n - theta)^2 / sigma_n^2 ~ w * ncx2_1(lambda(theta)).",
    )

    def evaluate(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> NDArray[np.float64]:
        m, pi = _require_normal_normal(model, prior)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, sigma_n, _ = posterior_params(D, pi.loc, m.sigma, pi.scale)
        diff = np.asarray(mu_n - np.asarray(theta0, dtype=np.float64), dtype=np.float64)
        return np.asarray(diff * diff / (sigma_n**2), dtype=np.float64)

    def pvalue(
        self, theta0: ArrayLike, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> NDArray[np.float64]:
        m, pi = _require_normal_normal(model, prior)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, _, w = posterior_params(D, pi.loc, m.sigma, pi.scale)
        theta_arr = np.asarray(theta0, dtype=np.float64)
        a, b = _pvalue_components(theta_arr, float(mu_n), pi.loc, w, m.sigma)
        return np.asarray(stats.norm.cdf(b - a) + stats.norm.cdf(-a - b), dtype=np.float64)

    def acceptance_region(
        self, alpha: float, theta0: ArrayLike, model: Model, prior: Prior | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Acceptance region in `D`-space at fixed theta0.

        Numerical inversion of the WALDO p-value: find D values for which
        the test fails to reject. We bracket on each side of the
        prior-data conflict zero (D = mu0) and use brentq_with_doubling.
        """
        m, pi = _require_normal_normal(model, prior)
        theta_arr = np.atleast_1d(np.asarray(theta0, dtype=np.float64))

        from ..tilting._solvers import brentq_with_doubling

        D_lo = np.empty_like(theta_arr)
        D_hi = np.empty_like(theta_arr)
        for i, theta_val in enumerate(theta_arr):

            def f(D_val: float, _theta=theta_val) -> float:
                return float(self.pvalue(float(_theta), np.asarray([D_val]), m, pi)) - alpha

            # The p-value at D = theta is 1 (mu_n = theta when prior is at D);
            # bracket outward from there.
            half = 4.0 * m.sigma
            D_lo[i] = brentq_with_doubling(
                f,
                midpoint=float(theta_val),
                initial_half_width=half,
                direction=-1,
            )
            D_hi[i] = brentq_with_doubling(
                f,
                midpoint=float(theta_val),
                initial_half_width=half,
                direction=+1,
            )
        return (
            D_lo if D_lo.size > 1 else D_lo.reshape(()),
            D_hi if D_hi.size > 1 else D_hi.reshape(()),
        )

    def confidence_interval(
        self, alpha: float, data: NDArray[np.float64], model: Model, prior: Prior | None = None
    ) -> tuple[float, float]:
        """Numerical CI inversion of the WALDO p-value via brentq.

        Solves `p(theta; D) = alpha` for theta on each side of `mu_n` (where
        `p(mu_n) = 1`). Bracket-doubling handles weak priors that produce
        wide intervals.
        """
        m, pi = _require_normal_normal(model, prior)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, _, _ = posterior_params(D, pi.loc, m.sigma, pi.scale)

        from ..tilting._solvers import brentq_with_doubling

        def f(theta: float) -> float:
            return float(self.pvalue(theta, np.asarray([D]), m, pi)) - alpha

        half = 4.0 * m.sigma
        lower = brentq_with_doubling(f, midpoint=float(mu_n), initial_half_width=half, direction=-1)
        upper = brentq_with_doubling(f, midpoint=float(mu_n), initial_half_width=half, direction=+1)
        return (lower, upper)

    def accepts_tilting(self, tilting: TiltingScheme) -> bool:
        """WALDO is selector-aware and accepts any tilting scheme."""
        return True
